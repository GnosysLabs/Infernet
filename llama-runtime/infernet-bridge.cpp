#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

static bool parse_u32(const std::string & text, uint32_t & out) {
    char * end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' || value > UINT32_MAX) {
        return false;
    }
    out = static_cast<uint32_t>(value);
    return true;
}

static bool parse_i32(const std::string & text, int32_t & out) {
    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' || value < INT32_MIN || value > INT32_MAX) {
        return false;
    }
    out = static_cast<int32_t>(value);
    return true;
}

static std::vector<std::string> split_tabs(const std::string & line) {
    std::vector<std::string> fields;
    size_t start = 0;
    for (;;) {
        const size_t end = line.find('\t', start);
        fields.push_back(line.substr(start, end == std::string::npos ? end : end - start));
        if (end == std::string::npos) {
            return fields;
        }
        start = end + 1;
    }
}

static uint8_t hex_nibble(char ch) {
    if (ch >= '0' && ch <= '9') return static_cast<uint8_t>(ch - '0');
    if (ch >= 'a' && ch <= 'f') return static_cast<uint8_t>(ch - 'a' + 10);
    if (ch >= 'A' && ch <= 'F') return static_cast<uint8_t>(ch - 'A' + 10);
    throw std::runtime_error("invalid hex command field");
}

static std::string hex_decode(const std::string & value) {
    if (value.size() % 2 != 0) {
        throw std::runtime_error("invalid hex command field length");
    }
    std::string out(value.size() / 2, '\0');
    for (size_t i = 0; i < value.size(); i += 2) {
        out[i / 2] = static_cast<char>((hex_nibble(value[i]) << 4) | hex_nibble(value[i + 1]));
    }
    return out;
}

static std::vector<float> read_f32_file(const std::string & path) {
    if (path.empty()) return {};
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("failed to open input activation file: " + path);
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
    if (path.empty()) throw std::runtime_error("non-final worker requires an output activation path");
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("failed to open output activation file: " + path);
    if (count > 0) {
        file.write(reinterpret_cast<const char *>(values), static_cast<std::streamsize>(count * sizeof(float)));
    }
    if (!file) throw std::runtime_error("failed to write output activation file: " + path);
}

static std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & prompt) {
    int n = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    if (n >= 0) throw std::runtime_error("tokenizer unexpectedly accepted a null output buffer");
    std::vector<llama_token> tokens(static_cast<size_t>(-n));
    n = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (n < 0) throw std::runtime_error("failed to tokenize prompt");
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
    if (n < 0) throw std::runtime_error("failed to detokenize sampled token");
    piece.resize(static_cast<size_t>(n));
    return piece;
}

static std::string format_chat_prompt(const llama_model * model, const std::string & prompt) {
    const char * chat_template = llama_model_chat_template(model, nullptr);
    if (!chat_template) return prompt;
    const llama_chat_message message = { "user", prompt.c_str() };
    std::vector<char> formatted(prompt.size() * 4 + 1024);
    const char * applied_template = chat_template;
    int32_t length = llama_chat_apply_template(applied_template, &message, 1, true, formatted.data(), static_cast<int32_t>(formatted.size()));
    if (length < 0) {
        applied_template = "gemma";
        length = llama_chat_apply_template(applied_template, &message, 1, true, formatted.data(), static_cast<int32_t>(formatted.size()));
    }
    if (length < 0) throw std::runtime_error("failed to apply the model chat template");
    if (static_cast<size_t>(length) > formatted.size()) {
        formatted.resize(static_cast<size_t>(length));
        length = llama_chat_apply_template(applied_template, &message, 1, true, formatted.data(), static_cast<int32_t>(formatted.size()));
        if (length < 0 || static_cast<size_t>(length) > formatted.size()) {
            throw std::runtime_error("failed to resize the formatted chat prompt");
        }
    }
    return std::string(formatted.data(), static_cast<size_t>(length));
}

class InfernetWorker {
public:
    static constexpr uint32_t MAX_GENERATED_TOKENS = 4096;

    InfernetWorker(const std::string & model_path, uint32_t layer_start, uint32_t layer_end,
                   uint32_t hidden_size, uint32_t threads, uint32_t max_context_tokens,
                   int32_t gpu_layers)
        : layer_start_(layer_start), layer_end_(layer_end), hidden_size_(hidden_size),
          max_context_tokens_(max_context_tokens) {
        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = gpu_layers;
        model_params.use_mmap = true;
        model_params.infernet_partial = true;
        model_params.infernet_layer_start = layer_start;
        model_params.infernet_layer_end = layer_end;
        model_ = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!model_) throw std::runtime_error("failed to load assigned model layers from " + model_path);

        model_layers_ = static_cast<uint32_t>(llama_model_n_layer(model_));
        const uint32_t model_hidden = static_cast<uint32_t>(llama_model_n_embd(model_));
        if (hidden_size_ != model_hidden || layer_end_ <= layer_start_ || layer_end_ > model_layers_) {
            throw std::runtime_error("worker layer range or hidden size does not match the local model");
        }
        final_worker_ = layer_end_ == model_layers_;
        vocab_ = llama_model_get_vocab(model_);

        llama_context_params params = llama_context_default_params();
        params.n_ctx = max_context_tokens_;
        params.n_batch = std::min<uint32_t>(2048, max_context_tokens_);
        params.n_ubatch = std::min<uint32_t>(2048, max_context_tokens_);
        params.n_seq_max = 1;
        params.infernet_layer_start = layer_start_;
        params.infernet_layer_end = layer_end_;
        params.infernet_partial = true;
        params.n_threads = static_cast<int32_t>(threads);
        params.n_threads_batch = static_cast<int32_t>(threads);
        params.no_perf = false;
        params.embeddings = !final_worker_;
        ctx_ = llama_init_from_model(model_, params);
        if (!ctx_) throw std::runtime_error("failed to create resident worker context");
        llama_infernet_set_layer_range(ctx_, layer_start_, layer_end_);

        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        sampler_ = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());
    }

    ~InfernetWorker() {
        if (sampler_) llama_sampler_free(sampler_);
        if (ctx_) llama_free(ctx_);
        if (model_) llama_model_free(model_);
    }

    void prefill(const std::string & trace_id, const std::string & prompt,
                 const std::string & input_path, const std::string & output_path) {
        llama_memory_clear(llama_get_memory(ctx_), true);
        llama_sampler_reset(sampler_);
        active_trace_ = trace_id;
        next_position_ = 0;
        const std::string formatted = format_chat_prompt(model_, prompt);
        std::vector<llama_token> tokens = tokenize_prompt(vocab_, formatted);
        if (tokens.empty()) throw std::runtime_error("prompt produced no tokens");
        if (tokens.size() > 2048 || tokens.size() + MAX_GENERATED_TOKENS > max_context_tokens_) {
            throw std::runtime_error("prompt exceeds the Infernet context safety limit");
        }
        const std::vector<float> activation = read_f32_file(input_path);
        decode(trace_id, tokens, activation, 0, output_path);
        next_position_ = static_cast<uint32_t>(tokens.size());
    }

    void token(const std::string & trace_id, llama_token token_id, uint32_t position,
               const std::string & input_path, const std::string & output_path) {
        if (active_trace_ != trace_id) throw std::runtime_error("worker has no resident KV session for this trace");
        if (position != next_position_) throw std::runtime_error("token position does not match resident KV state");
        const std::vector<llama_token> tokens = { token_id };
        const std::vector<float> activation = read_f32_file(input_path);
        decode(trace_id, tokens, activation, position, output_path);
        next_position_ += 1;
    }

private:
    void decode(const std::string &, const std::vector<llama_token> & tokens,
                const std::vector<float> & activation, uint32_t start_position,
                const std::string & output_path) {
        const size_t expected_values = tokens.size() * static_cast<size_t>(hidden_size_);
        if (!activation.empty() && activation.size() != expected_values) {
            throw std::runtime_error("activation shape does not match token count and hidden size");
        }
        if (activation.empty() && layer_start_ != 0) {
            throw std::runtime_error("non-entry worker requires an input activation");
        }

        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), activation.empty() ? 0 : static_cast<int32_t>(hidden_size_), 1);
        bool allocated_side_tokens = false;
        if (activation.empty()) {
            std::copy(tokens.begin(), tokens.end(), batch.token);
        } else {
            batch.token = static_cast<llama_token *>(std::malloc(sizeof(llama_token) * tokens.size()));
            if (!batch.token) {
                llama_batch_free(batch);
                throw std::runtime_error("failed to allocate token side input");
            }
            allocated_side_tokens = true;
            std::copy(tokens.begin(), tokens.end(), batch.token);
            std::copy(activation.begin(), activation.end(), batch.embd);
        }
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.pos[i] = static_cast<llama_pos>(start_position + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            // Boundary workers must return every prompt token so the next
            // range can build its own KV cache. Only the sampling worker needs
            // the final-token logits.
            batch.logits[i] = final_worker_ ? ((i + 1 == tokens.size()) ? 1 : 0) : 1;
        }

        const int64_t started = ggml_time_us();
        const int status = llama_decode(ctx_, batch);
        if (allocated_side_tokens) {
            std::free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
        if (status != 0) {
            std::ostringstream error;
            error << "llama_decode failed with status " << status;
            throw std::runtime_error(error.str());
        }

        const double timing_ms = static_cast<double>(ggml_time_us() - started) / 1000.0;
        if (!final_worker_) {
            float * embeddings = llama_get_embeddings(ctx_);
            if (!embeddings) throw std::runtime_error("worker produced no boundary activation");
            write_f32_file(output_path, embeddings, expected_values);
            emit_success(tokens.size(), expected_values, timing_ms, -1, "", false);
            return;
        }

        const llama_token sampled = llama_sampler_sample(sampler_, ctx_, -1);
        const bool complete = llama_vocab_is_eog(vocab_, sampled);
        const std::string piece = complete ? std::string() : token_to_piece(vocab_, sampled);
        llama_sampler_accept(sampler_, sampled);
        emit_success(tokens.size(), 0, timing_ms, sampled, piece, complete);
    }

    void emit_success(size_t n_tokens, size_t output_count, double timing_ms,
                      llama_token sampled, const std::string & piece, bool complete) const {
        std::cout << "{\"ok\":true"
                  << ",\"n_tokens\":" << n_tokens
                  << ",\"hidden_size\":" << hidden_size_
                  << ",\"output_f32_count\":" << output_count
                  << ",\"sampled_token_id\":";
        if (sampled < 0) std::cout << "null"; else std::cout << sampled;
        std::cout << ",\"generation_complete\":" << (complete ? "true" : "false")
                  << ",\"next_sequence_position\":" << (next_position_ + static_cast<uint32_t>(n_tokens))
                  << ",\"output_text\":\"" << escape_json(piece) << "\""
                  << ",\"timing_ms\":" << timing_ms << "}\n" << std::flush;
    }

    uint32_t layer_start_ = 0;
    uint32_t layer_end_ = 0;
    uint32_t hidden_size_ = 0;
    uint32_t max_context_tokens_ = 0;
    uint32_t model_layers_ = 0;
    uint32_t next_position_ = 0;
    bool final_worker_ = false;
    std::string active_trace_;
    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    llama_sampler * sampler_ = nullptr;
};

static void print_usage(const char * argv0) {
    std::cerr << "usage: " << argv0
              << " --model file.gguf --layer-start N --layer-end N --hidden-size N --threads N [--gpu-layers N] [--max-context N] --server\n";
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::string model_path;
    uint32_t layer_start = 0, layer_end = 0, hidden_size = 0, threads = 4, max_context = 8192;
    int32_t gpu_layers = -1;
    bool server = false;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&](const char * name) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
                return argv[++i];
            };
            if (arg == "--model") model_path = value("--model");
            else if (arg == "--layer-start") { if (!parse_u32(value("--layer-start"), layer_start)) throw std::runtime_error("invalid layer start"); }
            else if (arg == "--layer-end") { if (!parse_u32(value("--layer-end"), layer_end)) throw std::runtime_error("invalid layer end"); }
            else if (arg == "--hidden-size") { if (!parse_u32(value("--hidden-size"), hidden_size)) throw std::runtime_error("invalid hidden size"); }
            else if (arg == "--threads") { if (!parse_u32(value("--threads"), threads) || threads == 0 || threads > 64) throw std::runtime_error("invalid threads"); }
            else if (arg == "--gpu-layers") { if (!parse_i32(value("--gpu-layers"), gpu_layers)) throw std::runtime_error("invalid gpu layers"); }
            else if (arg == "--max-context") { if (!parse_u32(value("--max-context"), max_context) || max_context == 0) throw std::runtime_error("invalid max context"); }
            else if (arg == "--server") server = true;
            else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
            else throw std::runtime_error("unknown argument: " + arg);
        }
        if (!server || model_path.empty() || layer_end <= layer_start || hidden_size == 0) {
            print_usage(argv[0]);
            return 2;
        }

        InfernetWorker worker(model_path, layer_start, layer_end, hidden_size, threads, max_context, gpu_layers);
        std::cout << "{\"ready\":true,\"layer_start\":" << layer_start
                  << ",\"layer_end\":" << layer_end << "}\n" << std::flush;

        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                const auto fields = split_tabs(line);
                if (fields.size() == 5 && fields[0] == "PREFILL") {
                    worker.prefill(fields[1], hex_decode(fields[2]), hex_decode(fields[3]), hex_decode(fields[4]));
                } else if (fields.size() == 6 && fields[0] == "TOKEN") {
                    int32_t token = 0;
                    uint32_t position = 0;
                    if (!parse_i32(fields[2], token) || !parse_u32(fields[3], position)) throw std::runtime_error("invalid token command");
                    worker.token(fields[1], token, position, hex_decode(fields[4]), hex_decode(fields[5]));
                } else {
                    throw std::runtime_error("invalid worker command");
                }
            } catch (const std::exception & error) {
                std::cout << "{\"ok\":false,\"error\":\"" << escape_json(error.what()) << "\"}\n" << std::flush;
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cout << "{\"ok\":false,\"error\":\"" << escape_json(error.what()) << "\"}\n" << std::flush;
        return 1;
    }
}
