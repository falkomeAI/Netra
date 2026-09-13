#pragma once

#include <string>
#include <mutex>
#include <vector>

struct whisper_context;

namespace netra {

struct ASRResult {
    std::string text;
    std::string language;
    float language_probability = 0.0f;
    double duration_s      = 0.0;
    double processing_ms   = 0.0;
};

struct ASRConfig {
    std::string model_path = "models/ggml-small.bin";
    int n_threads   = 4;
    bool use_gpu    = true;
    int beam_size   = 1;
    bool translate  = false;
};

class ASREngine {
public:
    explicit ASREngine(const ASRConfig& config);
    ~ASREngine();

    ASREngine(const ASREngine&) = delete;
    ASREngine& operator=(const ASREngine&) = delete;

    ASRResult transcribe(const std::string& audio_path,
                          const std::string& language = "") const;

    ASRResult transcribe(const std::vector<float>& pcm_16khz,
                          const std::string& language = "") const;

    bool is_loaded() const { return ctx_ != nullptr; }

private:
    ASRConfig config_;
    whisper_context* ctx_ = nullptr;
    mutable std::mutex mtx_;

    ASRResult run_inference(const float* samples, int n_samples,
                             const std::string& language) const;
    static std::vector<float> load_wav(const std::string& path);
};

} // namespace netra
