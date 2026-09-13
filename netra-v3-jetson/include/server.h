#pragma once

#include <memory>
#include <string>

namespace netra {

class LLMEngine;
class ASREngine;
class RAGEngine;
class TTSEngine;
class OCREngine;

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 5050;
    int workers = 4;

    // Engine configs embedded
    std::string llm_model;
    std::string whisper_model;
    std::string embedding_model;
    std::string voice_dir;
    std::string policies_dir;
    std::string index_dir;

    int max_tokens    = 256;
    float temperature = 0.3f;
};

class Server {
public:
    explicit Server(const ServerConfig& config);
    ~Server();

    void start();  // blocking
    void stop();

private:
    ServerConfig config_;

    std::unique_ptr<LLMEngine> llm_;
    std::unique_ptr<ASREngine> asr_;
    std::unique_ptr<RAGEngine> rag_;
    std::unique_ptr<TTSEngine> tts_;
    std::unique_ptr<OCREngine> ocr_;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    void init_engines();
    void auto_ingest_policies();
    void register_routes();

    // Shared pipeline: RAG search → LLM → TTS
    std::string run_pipeline(const std::string& question,
                              const std::string& language);
};

} // namespace netra
