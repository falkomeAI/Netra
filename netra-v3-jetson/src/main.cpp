/*
 * NETRA v3 — Native C++ Pipeline for Jetson Orin Nano
 *
 * Single binary, no Python, no Ollama, no HTTP overhead.
 * Direct C/C++ calls to llama.cpp, whisper.cpp, FAISS, Piper, Tesseract.
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make -j$(nproc)
 *
 * Run:
 *   ./netra --llm models/qwen2.5-3b-q4_k_m.gguf \
 *           --whisper models/ggml-small.bin \
 *           --port 5050
 */

#include "server.h"
#include "utils.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <csignal>

static netra::Server* g_server = nullptr;

static void signal_handler(int sig) {
    netra::log(netra::LogLevel::INFO, "[MAIN] Signal %d received, shutting down...", sig);
    if (g_server) g_server->stop();
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "\n"
        "  NETRA v3 — Native C++ Pipeline for Jetson Orin Nano\n"
        "  ===================================================\n"
        "\n"
        "  Usage: %s [options]\n"
        "\n"
        "  Options:\n"
        "    --llm PATH        GGUF model path         (default: models/qwen2.5-3b-q4_k_m.gguf)\n"
        "    --whisper PATH    Whisper GGML model       (default: models/ggml-small.bin)\n"
        "    --embedding PATH  ONNX embedding model     (default: models/bge-small-en-v1.5.onnx)\n"
        "    --voices PATH     Piper voice directory    (default: models/piper)\n"
        "    --policies PATH   Policy documents dir     (default: knowledge_base/policies)\n"
        "    --index PATH      Vector store dir         (default: knowledge_base/vector_store)\n"
        "    --host ADDR       Bind address             (default: 0.0.0.0)\n"
        "    --port PORT       Listen port              (default: 5050)\n"
        "    --max-tokens N    Max LLM output tokens    (default: 256)\n"
        "    --temperature F   LLM temperature          (default: 0.3)\n"
        "    --help            Show this help\n"
        "\n", prog);
}

int main(int argc, char** argv) {
    netra::ServerConfig cfg;

    // Defaults
    cfg.llm_model      = "models/qwen2.5-3b-q4_k_m.gguf";
    cfg.whisper_model   = "models/ggml-small.bin";
    cfg.embedding_model = "models/bge-small-en-v1.5.onnx";
    cfg.voice_dir       = "models/piper";
    cfg.policies_dir    = "knowledge_base/policies";
    cfg.index_dir       = "knowledge_base/vector_store";
    cfg.host            = "0.0.0.0";
    cfg.port            = 5050;
    cfg.max_tokens      = 256;
    cfg.temperature     = 0.3f;

    // Parse CLI args
    for (int i = 1; i < argc; ++i) {
        auto arg = std::string(argv[i]);
        auto next = [&]() -> const char* {
            return (i + 1 < argc) ? argv[++i] : nullptr;
        };

        if (arg == "--llm")         { auto v = next(); if (v) cfg.llm_model = v; }
        else if (arg == "--whisper")    { auto v = next(); if (v) cfg.whisper_model = v; }
        else if (arg == "--embedding")  { auto v = next(); if (v) cfg.embedding_model = v; }
        else if (arg == "--voices")     { auto v = next(); if (v) cfg.voice_dir = v; }
        else if (arg == "--policies")   { auto v = next(); if (v) cfg.policies_dir = v; }
        else if (arg == "--index")      { auto v = next(); if (v) cfg.index_dir = v; }
        else if (arg == "--host")       { auto v = next(); if (v) cfg.host = v; }
        else if (arg == "--port")       { auto v = next(); if (v) cfg.port = atoi(v); }
        else if (arg == "--max-tokens") { auto v = next(); if (v) cfg.max_tokens = atoi(v); }
        else if (arg == "--temperature"){ auto v = next(); if (v) cfg.temperature = atof(v); }
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
        else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // Signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    using netra::log;
    using netra::LogLevel;

    log(LogLevel::INFO, "============================================================");
    log(LogLevel::INFO, "  NETRA v3 — Native C++ Pipeline for Jetson Orin Nano");
    log(LogLevel::INFO, "============================================================");
    log(LogLevel::INFO, "  LLM:       %s", cfg.llm_model.c_str());
    log(LogLevel::INFO, "  Whisper:   %s", cfg.whisper_model.c_str());
    log(LogLevel::INFO, "  Embedding: %s", cfg.embedding_model.c_str());
    log(LogLevel::INFO, "  Voices:    %s", cfg.voice_dir.c_str());
    log(LogLevel::INFO, "  Policies:  %s", cfg.policies_dir.c_str());
    log(LogLevel::INFO, "  Listen:    %s:%d", cfg.host.c_str(), cfg.port);
    log(LogLevel::INFO, "============================================================");

    try {
        netra::Server server(cfg);
        g_server = &server;
        server.start();
    } catch (const std::exception& e) {
        log(LogLevel::ERR, "Fatal: %s", e.what());
        return 1;
    }

    log(LogLevel::INFO, "NETRA v3 shutdown complete.");
    return 0;
}
