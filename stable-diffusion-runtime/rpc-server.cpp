#include "ggml-rpc.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#ifndef INFERNET_SD_RPC_REVISION
#define INFERNET_SD_RPC_REVISION "unknown"
#endif

struct ServerParams {
    std::string host = "127.0.0.1";
    int port = 50053;
    int threads = std::max(1U, std::thread::hardware_concurrency() / 2);
    std::vector<std::string> devices;
};

static void print_usage(const char* program, const ServerParams& params) {
    std::fprintf(stderr, "Usage: %s [options]\n\n", program);
    std::fprintf(stderr, "options:\n");
    std::fprintf(stderr, "  -h, --help                       show this help message and exit\n");
    std::fprintf(stderr, "  -t, --threads N                  CPU worker threads (default: %d)\n", params.threads);
    std::fprintf(stderr, "  -d, --device <dev1,dev2,...>     comma-separated device list\n");
    std::fprintf(stderr, "  -H, --host HOST                  loopback host (default: %s)\n", params.host.c_str());
    std::fprintf(stderr, "  -p, --port PORT                  loopback port (default: %d)\n", params.port);
}

static bool parse_args(int argc, char** argv, ServerParams& params) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "-h" || argument == "--help") {
            print_usage(argv[0], params);
            std::exit(0);
        }
        if (argument == "--version") {
            std::printf("Infernet Image RPC stable-diffusion.cpp %s\n", INFERNET_SD_RPC_REVISION);
            std::exit(0);
        }
        if (argument == "-H" || argument == "--host") {
            if (++index >= argc) return false;
            params.host = argv[index];
        } else if (argument == "-p" || argument == "--port") {
            if (++index >= argc) return false;
            params.port = std::stoi(argv[index]);
            if (params.port <= 0 || params.port > 65535) return false;
        } else if (argument == "-t" || argument == "--threads") {
            if (++index >= argc) return false;
            params.threads = std::stoi(argv[index]);
            if (params.threads <= 0) return false;
        } else if (argument == "-d" || argument == "--device") {
            if (++index >= argc) return false;
            const std::string value = argv[index];
            const std::regex separator{R"([,/]+)"};
            std::sregex_token_iterator current(value.begin(), value.end(), separator, -1);
            const std::sregex_token_iterator end;
            for (; current != end; ++current) {
                if (!current->str().empty()) params.devices.push_back(current->str());
            }
        } else if (argument == "-c" || argument == "--cache") {
            std::fprintf(stderr, "Infernet Image RPC does not enable an unverified tensor cache\n");
            return false;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argument.c_str());
            return false;
        }
    }
    return true;
}

static std::vector<ggml_backend_dev_t> select_devices(const ServerParams& params) {
    std::vector<ggml_backend_dev_t> devices;
    for (const auto& name : params.devices) {
        auto* device = ggml_backend_dev_by_name(name.c_str());
        if (device == nullptr) {
            std::fprintf(stderr, "unknown device: %s\n", name.c_str());
            return {};
        }
        devices.push_back(device);
    }
    if (devices.empty()) {
        for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
            auto* device = ggml_backend_dev_get(index);
            if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                devices.push_back(device);
            }
        }
    }
    if (devices.empty()) {
        if (auto* cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
            devices.push_back(cpu);
        }
    }
    return devices;
}

int main(int argc, char** argv) {
    std::setlocale(LC_NUMERIC, "C");
    ggml_backend_load_all();

    ServerParams params;
    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0], params);
        return 1;
    }
    if (params.host != "127.0.0.1") {
        std::fprintf(stderr, "Infernet Image RPC must remain bound to IPv4 loopback\n");
        return 1;
    }
    auto devices = select_devices(params);
    if (devices.empty()) {
        std::fprintf(stderr, "no GGML devices are available\n");
        return 1;
    }
    auto* registry = ggml_backend_reg_by_name("RPC");
    if (registry == nullptr) {
        std::fprintf(stderr, "RPC backend is unavailable\n");
        return 1;
    }
    auto start = reinterpret_cast<decltype(ggml_backend_rpc_start_server)*>(
        ggml_backend_reg_get_proc_address(registry, "ggml_backend_rpc_start_server"));
    if (start == nullptr) {
        std::fprintf(stderr, "RPC server entry point is unavailable\n");
        return 1;
    }

    const std::string endpoint = params.host + ":" + std::to_string(params.port);
    start(endpoint.c_str(), nullptr, params.threads, devices.size(), devices.data());
    return 0;
}
