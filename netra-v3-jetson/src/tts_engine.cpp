#include "tts_engine.h"
#include "utils.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <array>
#include <unistd.h>

namespace fs = std::filesystem;

namespace netra {

// ──────────────────────────────────────────────
// Constructor
// ──────────────────────────────────────────────

TTSEngine::TTSEngine(const TTSConfig& config) : config_(config) {
    Timer t;

    // Register available voices
    auto hi_model  = config_.voice_dir + "/hi_IN-medium.onnx";
    auto hi_config = config_.voice_dir + "/hi_IN-medium.onnx.json";
    if (file_exists(hi_model)) {
        voices_["hi"] = {hi_model, hi_config};
        LOG_INFO("[TTS] Hindi voice registered: %s", hi_model.c_str());
    }

    auto en_model  = config_.voice_dir + "/en_US-lessac-medium.onnx";
    auto en_config = config_.voice_dir + "/en_US-lessac-medium.onnx.json";
    if (file_exists(en_model)) {
        voices_["en"] = {en_model, en_config};
        LOG_INFO("[TTS] English voice registered: %s", en_model.c_str());
    }

    if (voices_.empty()) {
        LOG_WARN("[TTS] No voice models found in %s", config_.voice_dir.c_str());
    }

    LOG_INFO("[TTS] Ready in %.0fms (%zu voices)", t.elapsed_ms(), voices_.size());
}

TTSEngine::~TTSEngine() = default;

// ──────────────────────────────────────────────
// Synthesize using piper CLI binary
// ──────────────────────────────────────────────

std::vector<uint8_t> TTSEngine::synthesize(const std::string& text,
                                            const std::string& language) const {
    std::lock_guard<std::mutex> lock(mtx_);
    Timer t;

    std::string lang = language;
    if (voices_.find(lang) == voices_.end()) {
        lang = "hi";  // fallback to Hindi
    }
    if (voices_.find(lang) == voices_.end()) {
        LOG_ERR("[TTS] No voice available for language: %s", language.c_str());
        return {};
    }

    const auto& voice = voices_.at(lang);

    // Use piper CLI: echo "text" | piper --model model.onnx --output_file output.wav
    // This is the most reliable way to call Piper from C++ — the piper binary
    // is a self-contained ONNX+espeak-ng runtime, ~5MB, and is pre-built for aarch64.

    char tmp_path[] = "/tmp/netra_tts_XXXXXX.wav";
    int fd = mkstemps(tmp_path, 4);
    if (fd < 0) {
        LOG_ERR("[TTS] Failed to create temp file");
        return {};
    }
    close(fd);

    // Build command — pipe text through piper
    std::string escaped_text = text;
    // Escape single quotes for shell
    size_t pos = 0;
    while ((pos = escaped_text.find('\'', pos)) != std::string::npos) {
        escaped_text.replace(pos, 1, "'\\''");
        pos += 4;
    }

    std::string cmd = "echo '" + escaped_text + "' | piper"
                      " --model " + voice.model_path +
                      " --config " + voice.config_path +
                      " --output_file " + tmp_path +
                      " 2>/dev/null";

    int ret = system(cmd.c_str());
    if (ret != 0) {
        LOG_ERR("[TTS] Piper command failed (exit %d)", ret);
        unlink(tmp_path);
        return {};
    }

    // Read the WAV file
    std::ifstream wav_file(tmp_path, std::ios::binary);
    if (!wav_file.is_open()) {
        LOG_ERR("[TTS] Cannot read output WAV: %s", tmp_path);
        unlink(tmp_path);
        return {};
    }

    std::vector<uint8_t> wav_data(
        (std::istreambuf_iterator<char>(wav_file)),
        std::istreambuf_iterator<char>());
    wav_file.close();
    unlink(tmp_path);

    LOG_INFO("[TTS] Synthesized %zu chars → %zu bytes in %.0fms (lang=%s)",
             text.size(), wav_data.size(), t.elapsed_ms(), lang.c_str());
    return wav_data;
}

void TTSEngine::synthesize_to_file(const std::string& text,
                                    const std::string& output_path,
                                    const std::string& language) const {
    auto wav = synthesize(text, language);
    if (wav.empty()) return;

    fs::create_directories(fs::path(output_path).parent_path());
    write_file(output_path, wav.data(), wav.size());
    LOG_INFO("[TTS] Saved WAV to %s", output_path.c_str());
}

// ──────────────────────────────────────────────
// WAV builder (for future direct ONNX integration)
// ──────────────────────────────────────────────

std::vector<uint8_t> TTSEngine::make_wav(const std::vector<int16_t>& samples,
                                          int sample_rate) {
    uint32_t data_size = static_cast<uint32_t>(samples.size() * 2);
    uint32_t file_size = 36 + data_size;
    uint16_t channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sample_rate * channels * bits_per_sample / 8;
    uint16_t block_align = channels * bits_per_sample / 8;

    std::vector<uint8_t> wav;
    wav.reserve(44 + data_size);

    auto write_u32 = [&](uint32_t v) {
        wav.push_back(v & 0xFF);
        wav.push_back((v >> 8) & 0xFF);
        wav.push_back((v >> 16) & 0xFF);
        wav.push_back((v >> 24) & 0xFF);
    };
    auto write_u16 = [&](uint16_t v) {
        wav.push_back(v & 0xFF);
        wav.push_back((v >> 8) & 0xFF);
    };
    auto write_str = [&](const char* s) {
        while (*s) wav.push_back(*s++);
    };

    write_str("RIFF");
    write_u32(file_size);
    write_str("WAVE");
    write_str("fmt ");
    write_u32(16);
    write_u16(1);
    write_u16(channels);
    write_u32(sample_rate);
    write_u32(byte_rate);
    write_u16(block_align);
    write_u16(bits_per_sample);
    write_str("data");
    write_u32(data_size);

    const auto* raw = reinterpret_cast<const uint8_t*>(samples.data());
    wav.insert(wav.end(), raw, raw + data_size);

    return wav;
}

} // namespace netra
