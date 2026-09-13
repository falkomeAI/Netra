#pragma once

#include <string>
#include <mutex>
#include <memory>

struct llama_model;
struct llama_context;

namespace netra {

struct LLMResult {
    std::string response;
    int prompt_tokens   = 0;
    int completion_tokens = 0;
    double prompt_eval_ms  = 0;
    double generation_ms   = 0;
    double tokens_per_sec  = 0;
};

struct LLMConfig {
    std::string model_path;
    int n_gpu_layers = -1;   // all layers on GPU
    int n_ctx        = 2048;
    int n_batch      = 512;
    int n_threads    = 4;
    int max_tokens   = 256;
    float temperature = 0.3f;
    float top_p       = 0.9f;
};

class LLMEngine {
public:
    explicit LLMEngine(const LLMConfig& config);
    ~LLMEngine();

    LLMEngine(const LLMEngine&) = delete;
    LLMEngine& operator=(const LLMEngine&) = delete;

    LLMResult generate(const std::string& prompt,
                        const std::string& system_prompt = "",
                        int max_tokens = -1,
                        float temperature = -1.0f) const;

    bool is_loaded() const { return model_ != nullptr; }
    const std::string& model_path() const { return config_.model_path; }

private:
    LLMConfig config_;
    llama_model* model_     = nullptr;
    llama_context* ctx_     = nullptr;
    mutable std::mutex mtx_;
};

} // namespace netra
