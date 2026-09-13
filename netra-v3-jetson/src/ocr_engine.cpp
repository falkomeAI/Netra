#include "ocr_engine.h"
#include "utils.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstring>
#include <unistd.h>

namespace fs = std::filesystem;

namespace netra {

OCREngine::OCREngine(const OCRConfig& config) : config_(config) {
    LOG_INFO("[OCR] Initialized (tesseract_lang=%s, gpu=%s)",
             config_.tesseract_lang.c_str(), config_.use_gpu ? "yes" : "no");
}

OCREngine::~OCREngine() = default;

// ──────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────

OCRResult OCREngine::extract_text(const std::string& file_path,
                                   const std::string& file_type) const {
    std::lock_guard<std::mutex> lock(mtx_);
    Timer t;

    OCRResult result;
    try {
        if (file_type == "pdf") {
            result = extract_pdf(file_path);
        } else {
            result = extract_image(file_path);
        }
    } catch (const std::exception& e) {
        LOG_ERR("[OCR] Extraction failed: %s", e.what());
        result.method = "error";
    }

    result.processing_ms = t.elapsed_ms();
    return result;
}

OCRResult OCREngine::extract_from_bytes(const std::vector<uint8_t>& data,
                                         const std::string& file_type) const {
    // Write bytes to temp file, then extract
    std::string suffix = (file_type == "pdf") ? ".pdf" : ".png";
    char tmp_path[64];
    snprintf(tmp_path, sizeof(tmp_path), "/tmp/netra_ocr_XXXXXX%s", suffix.c_str());
    int fd = mkstemps(tmp_path, static_cast<int>(suffix.size()));
    if (fd < 0) {
        return {"", 0, "error", "en", 0.0};
    }
    write(fd, data.data(), data.size());
    close(fd);

    auto result = extract_text(tmp_path, file_type);
    unlink(tmp_path);
    return result;
}

// ──────────────────────────────────────────────
// PDF text extraction using poppler (pdftotext)
// ──────────────────────────────────────────────

OCRResult OCREngine::extract_pdf(const std::string& path) const {
    OCRResult result;
    result.method = "pdftotext";

    // Try pdftotext (from poppler-utils) — fast, no GPU needed
    char tmp_txt[] = "/tmp/netra_pdf_XXXXXX.txt";
    int fd = mkstemps(tmp_txt, 4);
    if (fd < 0) {
        result.method = "error";
        return result;
    }
    close(fd);

    std::string cmd = "pdftotext -layout \"" + path + "\" " + tmp_txt + " 2>/dev/null";
    int ret = system(cmd.c_str());

    if (ret == 0 && file_exists(tmp_txt)) {
        result.text = read_file(tmp_txt);
        result.text = normalize_text(result.text);

        // Count pages with pdfinfo
        std::string page_cmd = "pdfinfo \"" + path + "\" 2>/dev/null | grep -i 'pages'";
        FILE* pipe = popen(page_cmd.c_str(), "r");
        if (pipe) {
            char buf[128];
            if (fgets(buf, sizeof(buf), pipe)) {
                // Parse "Pages:          5"
                const char* p = strstr(buf, ":");
                if (p) result.pages = atoi(p + 1);
            }
            pclose(pipe);
        }
    }

    unlink(tmp_txt);

    // If pdftotext got too little text, fall back to OCR
    if (result.text.size() < 50 && result.pages > 0) {
        LOG_INFO("[OCR] PDF text too short (%zu chars), falling back to tesseract",
                 result.text.size());
        result = extract_image(path);  // tesseract can handle PDFs
        result.method = "ocr";
    }

    result.language = detect_language(result.text);
    return result;
}

// ──────────────────────────────────────────────
// Image OCR using tesseract CLI
// ──────────────────────────────────────────────

OCRResult OCREngine::extract_image(const std::string& path) const {
    OCRResult result;
    result.method = "ocr";
    result.pages = 1;

    // Use tesseract CLI: tesseract input.png output -l hin+eng
    char tmp_base[] = "/tmp/netra_ocr_out_XXXXXX";
    int fd = mkstemp(tmp_base);
    if (fd < 0) {
        result.method = "error";
        return result;
    }
    close(fd);

    std::string cmd = "tesseract \"" + path + "\" " + tmp_base +
                      " -l " + config_.tesseract_lang +
                      " --psm 6 2>/dev/null";
    int ret = system(cmd.c_str());

    std::string out_path = std::string(tmp_base) + ".txt";
    if (ret == 0 && file_exists(out_path)) {
        result.text = read_file(out_path);
        result.text = normalize_text(result.text);
    } else {
        LOG_WARN("[OCR] Tesseract failed on %s (exit %d)", path.c_str(), ret);
    }

    unlink(tmp_base);
    unlink(out_path.c_str());

    result.language = detect_language(result.text);
    return result;
}

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────

std::string OCREngine::detect_language(const std::string& text) {
    if (text.empty()) return "en";
    int devanagari = 0;
    int total = 0;
    for (unsigned char c : text) {
        if (c > 127) {
            // Check for Devanagari Unicode range (UTF-8 encoded)
            // Devanagari: U+0900–U+097F → UTF-8: 0xE0 0xA4 0x80 – 0xE0 0xA5 0xBF
            ++total;
            if (c >= 0xE0) ++devanagari;  // Rough heuristic for 3-byte UTF-8
        }
    }
    return (total > 0 && devanagari > total / 3) ? "hi" : "en";
}

std::string OCREngine::normalize_text(const std::string& text) {
    // Simple loop-based normalization (avoids std::regex stack overflow on large text)
    std::string result;
    result.reserve(text.size());

    bool in_tag = false;
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (c == '<') { in_tag = true; continue; }
        if (c == '>') { in_tag = false; result += ' '; continue; }
        if (in_tag) continue;
        result += c;
    }

    // Collapse spaces/tabs and excess newlines in a single pass
    std::string cleaned;
    cleaned.reserve(result.size());
    int newline_count = 0;
    bool prev_space = false;
    for (char c : result) {
        if (c == '\n') {
            ++newline_count;
            prev_space = false;
            if (newline_count <= 2) cleaned += '\n';
        } else if (c == ' ' || c == '\t') {
            newline_count = 0;
            if (!prev_space) { cleaned += ' '; prev_space = true; }
        } else {
            newline_count = 0;
            prev_space = false;
            cleaned += c;
        }
    }

    // Trim
    size_t start = cleaned.find_first_not_of(" \t\n\r");
    size_t end = cleaned.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    return cleaned.substr(start, end - start + 1);
}

} // namespace netra
