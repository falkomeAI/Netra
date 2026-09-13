#pragma once

#include <string>
#include <chrono>
#include <vector>
#include <cstdint>

namespace netra {

// High-resolution timer
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    void reset() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    double elapsed_s() const { return elapsed_ms() / 1000.0; }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Base64 encode/decode
std::string base64_encode(const uint8_t* data, size_t len);
std::string base64_encode(const std::vector<uint8_t>& data);
std::vector<uint8_t> base64_decode(const std::string& encoded);

// File I/O
std::string read_file(const std::string& path);
bool write_file(const std::string& path, const void* data, size_t len);
bool file_exists(const std::string& path);
std::vector<std::string> list_files(const std::string& dir,
                                     const std::vector<std::string>& extensions);

// Language detection (heuristic: Unicode script block analysis for 22 Indian languages)
std::string detect_language(const std::string& text);
std::string language_name(const std::string& code);

// Logging
enum class LogLevel { DEBUG, INFO, WARN, ERR };
void log(LogLevel level, const char* fmt, ...);

#define LOG_DEBUG(...) netra::log(netra::LogLevel::DEBUG, __VA_ARGS__)
#define LOG_INFO(...)  netra::log(netra::LogLevel::INFO,  __VA_ARGS__)
#define LOG_WARN(...)  netra::log(netra::LogLevel::WARN,  __VA_ARGS__)
#define LOG_ERR(...)   netra::log(netra::LogLevel::ERR,   __VA_ARGS__)

} // namespace netra
