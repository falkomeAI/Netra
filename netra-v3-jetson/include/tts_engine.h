#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace netra {

struct TTSConfig {
    std::string voice_dir = "models/piper";
    std::string default_lang = "hi";
    int sample_rate = 22050;
};

class TTSEngine {
public:
    explicit TTSEngine(const TTSConfig& config);
    ~TTSEngine();

    TTSEngine(const TTSEngine&) = delete;
    TTSEngine& operator=(const TTSEngine&) = delete;

    // Returns WAV file bytes (complete with header)
    std::vector<uint8_t> synthesize(const std::string& text,
                                     const std::string& language = "hi") const;

    void synthesize_to_file(const std::string& text,
                             const std::string& output_path,
                             const std::string& language = "hi") const;

    bool is_loaded() const { return !voices_.empty(); }

private:
    TTSConfig config_;

    struct Voice {
        std::string model_path;
        std::string config_path;
        // Piper uses espeak-ng + ONNX — we shell out to the piper binary
    };

    std::unordered_map<std::string, Voice> voices_;
    mutable std::mutex mtx_;

    static std::vector<uint8_t> make_wav(const std::vector<int16_t>& samples,
                                          int sample_rate);
};

} // namespace netra
