#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <mutex>

namespace netra {

struct OCRResult {
    std::string text;
    int pages = 0;
    std::string method;    // "text" | "ocr" | "error"
    std::string language;
    double processing_ms = 0.0;
};

struct OCRConfig {
    std::string tesseract_lang = "hin+eng";
    bool use_gpu = true;
};

class OCREngine {
public:
    explicit OCREngine(const OCRConfig& config);
    ~OCREngine();

    OCREngine(const OCREngine&) = delete;
    OCREngine& operator=(const OCREngine&) = delete;

    OCRResult extract_text(const std::string& file_path,
                            const std::string& file_type = "pdf") const;

    OCRResult extract_from_bytes(const std::vector<uint8_t>& data,
                                  const std::string& file_type = "pdf") const;

private:
    OCRConfig config_;
    mutable std::mutex mtx_;

    OCRResult extract_pdf(const std::string& path) const;
    OCRResult extract_image(const std::string& path) const;
    static std::string detect_language(const std::string& text);
    static std::string normalize_text(const std::string& text);
};

} // namespace netra
