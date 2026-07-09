#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string escape_json(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char ch : value) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    out += buf;
                } else {
                    out += ch;
                }
        }
    }
    return out;
}

static bool parse_u32(const char * text, uint32_t & out) {
    char * end = nullptr;
    errno = 0;
    unsigned long value = std::strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
        return false;
    }
    out = static_cast<uint32_t>(value);
    return true;
}

static bool parse_i32(const char * text, int32_t & out) {
    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < INT32_MIN || value > INT32_MAX) {
        return false;
    }
    out = static_cast<int32_t>(value);
    return true;
}

static std::vector<float> read_f32_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("failed to open input activation file: " + path);
    }
    const std::streamsize size = file.tellg();
    if (size < 0 || size % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error("input activation file is not aligned to f32 values: " + path);
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<size_t>(size) / sizeof(float));
    if (!values.empty() && !file.read(reinterpret_cast<char *>(values.data()), size)) {
        throw std::runtime_error("failed to read input activation file: " + path);
    }
    return values;
}

static void write_f32_file(const std::string & path, const float * values, size_t count) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("failed to open output activation file: " + path);
    }
    if (count > 0) {
        file.write(reinterpret_cast<const char *>(values), static_cast<std::streamsize>(count * sizeof(float)));
    }
    if (!file) {
        throw std::runtime_error("failed to write output activation file: " + path);
    }
}

static std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & prompt) {
    int n = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    if (n >= 0) {
        throw std::runtime_error("tokenizer unexpectedly accepted a null output buffer");
    }
    std::vector<llama_token> tokens(static_cast<size_t>(-n));
    n = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (n < 0) {
        throw std::runtime_error("failed to tokenize prompt");
    }
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

static std::string token_to_piece(const llama_vocab * vocab, llama_token token) {
    std::string piece(128, '\0');
    int n = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
    if (n < 0) {
        piece.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
    }
    if (n < 0) {
        throw std::runtime_error("failed to detokenize sampled token");
    }
    piece.resize(static_cast<size_t>(n));
    return piece;
}

static std::string format_chat_prompt(const llama_model * model, const std::string & prompt) {
    const char * chat_template = llama_model_chat_template(model, nullptr);
    if (!chat_template) {
        return prompt;
    }

    const llama_chat_message message = { "user", prompt.c_str() };
    std::vector<char> formatted(prompt.size() * 4 + 1024);
    int32_t length = llama_chat_apply_template(
        chat_template,
        &message,
        1,
        true,
        formatted.data(),
        static_cast<int32_t>(formatted.size()));
    if (length < 0) {
        throw std::runtime_error("failed to apply the model's chat template");
    }
    if (static_cast<size_t>(length) > formatted.size()) {
        formatted.resize(static_cast<size_t>(length));
        length = llama_chat_apply_template(
            chat_template,
            &message,
            1,
            true,
            formatted.data(),
            static_cast<int32_t>(formatted.size()));
        if (length < 0 || static_cast<size_t>(length) > formatted.size()) {
            throw std::runtime_error("failed to resize the formatted chat prompt");
        }
    }
    return std::string(formatted.data(), static_cast<size_t>(length));
}

static void print_usage(const char * argv0) {
    std::cerr
        << "usage: " << argv0 << " --model file.gguf --layer-start N --layer-end N --hidden-size N --threads N --prompt text [--gpu-layers N] [--max-context N] [--full-model] [--input activation.bin] [--output activation.bin]\n"
        << "\n"
        << "Runs a patched llama.cpp layer-range graph. If --input is omitted, tokens are embedded and execution starts at layer 0.\n"
        << "If --layer-end is less than the model layer count, hidden activations are written to --output.\n"
        << "If --layer-end reaches the model layer count, the final shard generates up to 32 greedy tokens and returns them as JSON.\n";
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt;
    std::string input_path;
    std::string output_path;
    uint32_t layer_start = 0;
    uint32_t layer_end = 0;
    uint32_t hidden_size = 0;
    uint32_t threads = 4;
    uint32_t max_context_tokens = 8192;
    int32_t gpu_layers = -1;
    bool full_model = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--model") {
            model_path = need_value("--model");
        } else if (arg == "--prompt") {
            prompt = need_value("--prompt");
        } else if (arg == "--input") {
            input_path = need_value("--input");
        } else if (arg == "--output") {
            output_path = need_value("--output");
        } else if (arg == "--layer-start") {
            if (!parse_u32(need_value("--layer-start"), layer_start)) {
                throw std::runtime_error("invalid --layer-start");
            }
        } else if (arg == "--layer-end") {
            if (!parse_u32(need_value("--layer-end"), layer_end)) {
                throw std::runtime_error("invalid --layer-end");
            }
        } else if (arg == "--hidden-size") {
            if (!parse_u32(need_value("--hidden-size"), hidden_size)) {
                throw std::runtime_error("invalid --hidden-size");
            }
        } else if (arg == "--threads") {
            if (!parse_u32(need_value("--threads"), threads) || threads == 0 || threads > 64) {
                throw std::runtime_error("invalid --threads");
            }
        } else if (arg == "--gpu-layers") {
            if (!parse_i32(need_value("--gpu-layers"), gpu_layers)) {
                throw std::runtime_error("invalid --gpu-layers");
            }
        } else if (arg == "--max-context") {
            if (!parse_u32(need_value("--max-context"), max_context_tokens) || max_context_tokens == 0) {
                throw std::runtime_error("invalid --max-context");
            }
        } else if (arg == "--full-model") {
            full_model = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    try {
        if (model_path.empty() || prompt.empty() || layer_end <= layer_start || hidden_size == 0) {
            print_usage(argv[0]);
            return 2;
        }

        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        // llama.cpp treats a negative value as all available layers. CPU-only
        // builds safely keep them on CPU; CUDA/Metal builds offload the shard.
        model_params.n_gpu_layers = gpu_layers;
        model_params.use_mmap = true;
        model_params.infernet_partial = !full_model;
        if (!full_model) {
            model_params.infernet_layer_start = layer_start;
            model_params.infernet_layer_end = layer_end;
        }

        llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!model) {
            throw std::runtime_error("failed to load model shard tensors from " + model_path);
        }

        const uint32_t model_layers = static_cast<uint32_t>(llama_model_n_layer(model));
        const uint32_t model_hidden = static_cast<uint32_t>(llama_model_n_embd(model));
        if (hidden_size != model_hidden) {
            std::ostringstream error;
            error << "hidden size mismatch: request=" << hidden_size << " model=" << model_hidden;
            throw std::runtime_error(error.str());
        }
        if (layer_end > model_layers) {
            std::ostringstream error;
            error << "layer range " << layer_start << ":" << layer_end << " exceeds model layers " << model_layers;
            throw std::runtime_error(error.str());
        }
        if (full_model && (layer_start != 0 || layer_end != model_layers)) {
            throw std::runtime_error("--full-model requires the complete 0:N layer range");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const std::string formatted_prompt = format_chat_prompt(model, prompt);
        std::vector<llama_token> tokens = tokenize_prompt(vocab, formatted_prompt);
        if (tokens.empty()) {
            throw std::runtime_error("prompt produced no tokens");
        }
        if (tokens.size() > 2048) {
            throw std::runtime_error("prompt exceeds Infernet's 2048-token safety limit");
        }

        std::vector<float> input_activation;
        if (!input_path.empty()) {
            input_activation = read_f32_file(input_path);
            const size_t expected = tokens.size() * static_cast<size_t>(hidden_size);
            if (input_activation.size() != expected) {
                std::ostringstream error;
                error << "activation shape mismatch: got " << input_activation.size()
                      << " f32 values, expected " << expected
                      << " (" << tokens.size() << " tokens x " << hidden_size << ")";
                throw std::runtime_error(error.str());
            }
        } else if (layer_start != 0) {
            throw std::runtime_error("non-zero layer_start requires --input activation");
        }

        constexpr uint32_t max_generated_tokens = 32;
        llama_context_params ctx_params = llama_context_default_params();
        const size_t context_tokens = tokens.size() + (layer_end == model_layers ? max_generated_tokens : 0);
        if (context_tokens > max_context_tokens) {
            std::ostringstream error;
            error << "request needs " << context_tokens
                  << " context tokens, exceeding the Infernet launch cap of " << max_context_tokens;
            throw std::runtime_error(error.str());
        }
        ctx_params.n_ctx = static_cast<uint32_t>(std::max<size_t>(context_tokens, 1));
        ctx_params.n_batch = static_cast<uint32_t>(tokens.size());
        ctx_params.n_ubatch = static_cast<uint32_t>(tokens.size());
        ctx_params.n_seq_max = 1;
        ctx_params.n_threads = static_cast<int32_t>(threads);
        ctx_params.n_threads_batch = static_cast<int32_t>(threads);
        ctx_params.no_perf = false;
        ctx_params.embeddings = layer_end < model_layers;

        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            throw std::runtime_error("failed to create llama context");
        }
        if (!full_model) {
            llama_infernet_set_layer_range(ctx, layer_start, layer_end);
        }

        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), input_activation.empty() ? 0 : static_cast<int32_t>(hidden_size), 1);
        if (input_activation.empty()) {
            std::copy(tokens.begin(), tokens.end(), batch.token);
        } else {
            // Middle/final Gemma-family shards still need prompt tokens for per-layer token embeddings.
            batch.token = static_cast<llama_token *>(std::malloc(sizeof(llama_token) * tokens.size()));
            if (!batch.token) {
                throw std::runtime_error("failed to allocate token side input");
            }
            std::copy(tokens.begin(), tokens.end(), batch.token);
            std::copy(input_activation.begin(), input_activation.end(), batch.embd);
        }
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.pos[i] = static_cast<llama_pos>(i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i + 1 == tokens.size()) ? 1 : 0;
        }

        const int64_t start_us = ggml_time_us();
        const int decode_status = llama_decode(ctx, batch);
        if (decode_status != 0) {
            std::ostringstream error;
            error << "llama_decode failed with status " << decode_status;
            throw std::runtime_error(error.str());
        }

        const size_t output_values = tokens.size() * static_cast<size_t>(hidden_size);

        if (layer_end < model_layers) {
            if (output_path.empty()) {
                throw std::runtime_error("non-final shard requires --output activation path");
            }
            float * embeddings = llama_get_embeddings(ctx);
            if (!embeddings) {
                throw std::runtime_error("llama_get_embeddings returned null for shard output");
            }
            write_f32_file(output_path, embeddings, output_values);
            const int64_t end_us = ggml_time_us();
            const double timing_ms = static_cast<double>(end_us - start_us) / 1000.0;
            std::cout
                << "{\"ok\":true"
                << ",\"n_tokens\":" << tokens.size()
                << ",\"hidden_size\":" << hidden_size
                << ",\"output_f32_count\":" << output_values
                << ",\"output_text\":null"
                << ",\"timing_ms\":" << timing_ms
                << "}\n";
        } else {
            llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
            llama_sampler * sampler = llama_sampler_chain_init(sparams);
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
            llama_token token = llama_sampler_sample(sampler, ctx, -1);
            std::string generated;
            uint32_t generated_tokens = 0;
            while (generated_tokens < max_generated_tokens && !llama_vocab_is_eog(vocab, token)) {
                generated += token_to_piece(vocab, token);
                ++generated_tokens;
                llama_sampler_accept(sampler, token);
                if (generated_tokens >= max_generated_tokens) {
                    break;
                }
                llama_batch next_batch = llama_batch_get_one(&token, 1);
                const int next_status = llama_decode(ctx, next_batch);
                if (next_status != 0) {
                    std::ostringstream error;
                    error << "llama_decode failed while generating token " << generated_tokens
                          << " with status " << next_status;
                    throw std::runtime_error(error.str());
                }
                token = llama_sampler_sample(sampler, ctx, -1);
            }
            llama_sampler_free(sampler);
            const int64_t end_us = ggml_time_us();
            const double timing_ms = static_cast<double>(end_us - start_us) / 1000.0;
            std::cout
                << "{\"ok\":true"
                << ",\"n_tokens\":" << tokens.size()
                << ",\"hidden_size\":" << hidden_size
                << ",\"output_f32_count\":0"
                << ",\"generated_tokens\":" << generated_tokens
                << ",\"output_text\":\"" << escape_json(generated) << "\""
                << ",\"timing_ms\":" << timing_ms
                << "}\n";
        }

        if (!input_activation.empty() && batch.token) {
            std::free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    } catch (const std::exception & error) {
        std::cout
            << "{\"ok\":false"
            << ",\"error\":\"" << escape_json(error.what()) << "\""
            << "}\n";
        return 1;
    }
}
