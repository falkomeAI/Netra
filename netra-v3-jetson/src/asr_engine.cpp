#include "asr_engine.h"
#include "utils.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstdint>

#ifdef NETRA_HAS_ASR

#include "whisper.h"

namespace netra {

// ── Constructor ────────────────────────────────────────────────────────────────

ASREngine::ASREngine(const ASRConfig& config) : config_(config) {
    if (!file_exists(config_.model_path)) {
        LOG_ERR("ASR model not found: %s", config_.model_path.c_str());
        throw std::runtime_error("ASR model file not found: " + config_.model_path);
    }

    Timer t;
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = config_.use_gpu;

    ctx_ = whisper_init_from_file_with_params(config_.model_path.c_str(), cparams);
    if (!ctx_) {
        LOG_ERR("Failed to initialise whisper context from: %s",
                config_.model_path.c_str());
        throw std::runtime_error("Failed to load whisper model: " + config_.model_path);
    }

    LOG_INFO("ASR model loaded in %.1f ms  [gpu=%s, threads=%d, beam=%d]",
             t.elapsed_ms(),
             config_.use_gpu ? "yes" : "no",
             config_.n_threads,
             config_.beam_size);
}

// ── Destructor ─────────────────────────────────────────────────────────────────

ASREngine::~ASREngine() {
    if (ctx_) {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
}

// ── WAV loader ─────────────────────────────────────────────────────────────────

struct WavHeader {
    char     riff[4];       // "RIFF"
    uint32_t file_size;
    char     wave[4];       // "WAVE"
    char     fmt_id[4];     // "fmt "
    uint32_t fmt_size;
    uint16_t audio_format;  // 1 = PCM
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

std::vector<float> ASREngine::load_wav(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open WAV file: " + path);
    }

    WavHeader hdr{};
    file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!file || std::memcmp(hdr.riff, "RIFF", 4) != 0 ||
        std::memcmp(hdr.wave, "WAVE", 4) != 0) {
        throw std::runtime_error("Invalid WAV header: " + path);
    }
    if (hdr.audio_format != 1) {
        throw std::runtime_error("Only PCM WAV is supported (got format " +
                                 std::to_string(hdr.audio_format) + "): " + path);
    }
    if (hdr.bits_per_sample != 16) {
        throw std::runtime_error("Only 16-bit PCM WAV is supported (got " +
                                 std::to_string(hdr.bits_per_sample) + "-bit): " + path);
    }

    // Skip any extra fmt bytes and non-data chunks to find the "data" chunk.
    if (hdr.fmt_size > 16) {
        file.seekg(hdr.fmt_size - 16, std::ios::cur);
    }

    char chunk_id[4];
    uint32_t chunk_size = 0;
    while (file.read(chunk_id, 4) && file.read(reinterpret_cast<char*>(&chunk_size), 4)) {
        if (std::memcmp(chunk_id, "data", 4) == 0) break;
        file.seekg(chunk_size, std::ios::cur);
    }
    if (!file) {
        throw std::runtime_error("No data chunk found in WAV: " + path);
    }

    const uint32_t num_samples = chunk_size / (hdr.bits_per_sample / 8);
    std::vector<int16_t> raw(num_samples);
    file.read(reinterpret_cast<char*>(raw.data()),
              static_cast<std::streamsize>(num_samples * sizeof(int16_t)));

    // Convert interleaved channels to mono float32 normalised to [-1, 1].
    const uint16_t channels = hdr.num_channels;
    const uint32_t frames   = num_samples / channels;
    std::vector<float> mono(frames);
    for (uint32_t i = 0; i < frames; ++i) {
        float sum = 0.0f;
        for (uint16_t c = 0; c < channels; ++c) {
            sum += static_cast<float>(raw[i * channels + c]);
        }
        mono[i] = (sum / channels) / 32768.0f;
    }

    // Resample to 16 kHz if necessary (simple linear interpolation).
    if (hdr.sample_rate != 16000) {
        const double ratio = static_cast<double>(hdr.sample_rate) / 16000.0;
        const auto out_len = static_cast<uint32_t>(frames / ratio);
        std::vector<float> resampled(out_len);
        for (uint32_t i = 0; i < out_len; ++i) {
            double src_idx  = i * ratio;
            auto   idx0     = static_cast<uint32_t>(src_idx);
            float  frac     = static_cast<float>(src_idx - idx0);
            uint32_t idx1   = std::min(idx0 + 1, frames - 1);
            resampled[i]    = mono[idx0] * (1.0f - frac) + mono[idx1] * frac;
        }
        return resampled;
    }

    return mono;
}

// ── Public transcribe overloads ────────────────────────────────────────────────

ASRResult ASREngine::transcribe(const std::string& audio_path,
                                const std::string& language) const {
    if (!ctx_) {
        LOG_ERR("ASR engine not loaded – cannot transcribe");
        return {};
    }

    Timer t;
    auto pcm = load_wav(audio_path);
    double duration_s = static_cast<double>(pcm.size()) / 16000.0;
    LOG_INFO("Loaded %s  (%.2f s, %zu samples) in %.1f ms",
             audio_path.c_str(), duration_s, pcm.size(), t.elapsed_ms());

    auto result = run_inference(pcm.data(), static_cast<int>(pcm.size()), language);
    result.duration_s = duration_s;
    return result;
}

ASRResult ASREngine::transcribe(const std::vector<float>& pcm_16khz,
                                const std::string& language) const {
    if (!ctx_) {
        LOG_ERR("ASR engine not loaded – cannot transcribe");
        return {};
    }

    double duration_s = static_cast<double>(pcm_16khz.size()) / 16000.0;
    auto result = run_inference(pcm_16khz.data(),
                                static_cast<int>(pcm_16khz.size()), language);
    result.duration_s = duration_s;
    return result;
}

// ── Core inference ─────────────────────────────────────────────────────────────

ASRResult ASREngine::run_inference(const float* samples, int n_samples,
                                   const std::string& language) const {
    std::lock_guard<std::mutex> lock(mtx_);
    Timer t;

    whisper_full_params wparams =
        whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    wparams.n_threads      = config_.n_threads;
    wparams.no_timestamps  = true;
    wparams.single_segment = false;
    wparams.print_special  = false;
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.translate      = config_.translate;
    wparams.greedy.best_of = config_.beam_size;

    if (!language.empty()) {
        wparams.language = language.c_str();
    } else {
        wparams.language  = nullptr;
        wparams.detect_language = true;
    }

    int rc = whisper_full(ctx_, wparams, samples, n_samples);
    if (rc != 0) {
        LOG_ERR("whisper_full() failed with code %d", rc);
        return {};
    }

    ASRResult result;

    // Concatenate all recognised segments.
    int n_segments = whisper_full_n_segments(ctx_);
    std::string text;
    text.reserve(512);
    for (int i = 0; i < n_segments; ++i) {
        const char* seg = whisper_full_get_segment_text(ctx_, i);
        if (seg) text += seg;
    }

    // Trim leading/trailing whitespace.
    auto start = text.find_first_not_of(" \t\n\r");
    auto end   = text.find_last_not_of(" \t\n\r");
    result.text = (start == std::string::npos)
                      ? ""
                      : text.substr(start, end - start + 1);

    // Detected language.
    int lang_id = whisper_full_lang_id(ctx_);
    if (lang_id >= 0) {
        result.language            = whisper_lang_str(lang_id);
        result.language_probability = whisper_full_lang_id(ctx_) >= 0 ? 1.0f : 0.0f;
    }

    result.processing_ms = t.elapsed_ms();

    LOG_INFO("ASR done in %.1f ms  [lang=%s, %d segments, %zu chars]",
             result.processing_ms,
             result.language.c_str(),
             n_segments,
             result.text.size());

    return result;
}

} // namespace netra

#else // !NETRA_HAS_ASR – stub implementation

namespace netra {

ASREngine::ASREngine(const ASRConfig& config) : config_(config) {
    LOG_WARN("ASR support not compiled (NETRA_HAS_ASR not defined)");
}

ASREngine::~ASREngine() {}

std::vector<float> ASREngine::load_wav(const std::string& /*path*/) {
    return {};
}

ASRResult ASREngine::transcribe(const std::string& /*audio_path*/,
                                const std::string& /*language*/) const {
    LOG_WARN("ASR transcribe called but ASR support is not compiled");
    return {};
}

ASRResult ASREngine::transcribe(const std::vector<float>& /*pcm_16khz*/,
                                const std::string& /*language*/) const {
    LOG_WARN("ASR transcribe called but ASR support is not compiled");
    return {};
}

ASRResult ASREngine::run_inference(const float* /*samples*/, int /*n_samples*/,
                                   const std::string& /*language*/) const {
    return {};
}

} // namespace netra

#endif // NETRA_HAS_ASR
