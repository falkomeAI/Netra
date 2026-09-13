#include "utils.h"

#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

namespace netra {

// ──────────────────────────────────────────────
// Base64
// ──────────────────────────────────────────────

static const char B64_TABLE[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(const uint8_t* data, size_t len) {
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = static_cast<uint32_t>(data[i]) << 16;
        if (i + 1 < len) n |= static_cast<uint32_t>(data[i + 1]) << 8;
        if (i + 2 < len) n |= static_cast<uint32_t>(data[i + 2]);
        out.push_back(B64_TABLE[(n >> 18) & 0x3F]);
        out.push_back(B64_TABLE[(n >> 12) & 0x3F]);
        out.push_back((i + 1 < len) ? B64_TABLE[(n >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < len) ? B64_TABLE[n & 0x3F] : '=');
    }
    return out;
}

std::string base64_encode(const std::vector<uint8_t>& data) {
    return base64_encode(data.data(), data.size());
}

std::vector<uint8_t> base64_decode(const std::string& encoded) {
    static const int DECODE_TABLE[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
    };

    std::vector<uint8_t> out;
    out.reserve(encoded.size() * 3 / 4);
    uint32_t buf = 0;
    int bits = 0;
    for (char c : encoded) {
        if (c == '=' || c == '\n' || c == '\r') continue;
        int val = DECODE_TABLE[static_cast<unsigned char>(c)];
        if (val < 0) continue;
        buf = (buf << 6) | val;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buf >> bits) & 0xFF));
        }
    }
    return out;
}

// ──────────────────────────────────────────────
// File I/O
// ──────────────────────────────────────────────

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool write_file(const std::string& path, const void* data, size_t len) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.write(static_cast<const char*>(data), len);
    return f.good();
}

bool file_exists(const std::string& path) {
    return fs::exists(path);
}

std::vector<std::string> list_files(const std::string& dir,
                                     const std::vector<std::string>& extensions) {
    std::vector<std::string> result;
    if (!fs::is_directory(dir)) return result;

    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        for (const auto& e : extensions) {
            if (ext == e) {
                result.push_back(entry.path().string());
                break;
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

// ──────────────────────────────────────────────
// Language detection
// ──────────────────────────────────────────────

std::string detect_language(const std::string& text) {
    if (text.empty()) return "hi";

    // Count characters per Indic script block via UTF-8 byte patterns
    // 3-byte Indic scripts all start with E0, second byte identifies the block
    int script_counts[16] = {};  // indexed by (second_byte - 0xA4) / 2
    int arabic = 0;   // Urdu/Sindhi/Kashmiri: 2-byte D8-DB
    int ascii = 0;
    int other = 0;

    const auto* p = reinterpret_cast<const unsigned char*>(text.c_str());
    size_t len = text.size();
    for (size_t i = 0; i < len; ) {
        unsigned char c = p[i];
        if (c < 0x80) {
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) ++ascii;
            ++i;
        } else if (c == 0xE0 && i + 2 < len) {
            unsigned char b1 = p[i + 1];
            if (b1 >= 0xA4 && b1 <= 0xB5) {
                int idx = (b1 - 0xA4) / 2;
                if (idx >= 0 && idx < 16) ++script_counts[idx];
            }
            i += 3;
        } else if (c >= 0xD8 && c <= 0xDB && i + 1 < len) {
            ++arabic;
            i += 2;
        } else if (c >= 0xC0 && c < 0xE0) {
            ++other; i += 2;
        } else if (c >= 0xE0 && c < 0xF0) {
            ++other; i += 3;
        } else if (c >= 0xF0) {
            ++other; i += 4;
        } else {
            ++i;
        }
    }

    // Map script indices to language codes
    // E0 A4/A5 = Devanagari(hi), E0 A6/A7 = Bengali(bn), E0 A8/A9 = Gurmukhi(pa)
    // E0 AA/AB = Gujarati(gu), E0 AC/AD = Oriya(or), E0 AE/AF = Tamil(ta)
    // E0 B0/B1 = Telugu(te), E0 B2/B3 = Kannada(kn), E0 B4/B5 = Malayalam(ml)
    struct { int idx; const char* code; } script_map[] = {
        {0, "hi"}, {1, "bn"}, {2, "pa"}, {3, "gu"}, {4, "or"},
        {5, "ta"}, {6, "te"}, {7, "kn"}, {8, "ml"},
    };

    int best_count = 0;
    const char* best_code = "en";

    for (auto& s : script_map) {
        if (script_counts[s.idx] > best_count) {
            best_count = script_counts[s.idx];
            best_code = s.code;
        }
    }

    if (arabic > best_count) {
        best_count = arabic;
        best_code = "ur";
    }

    if (best_count == 0 && ascii > 0) return "en";
    if (best_count == 0) return "hi";
    return best_code;
}

std::string language_name(const std::string& code) {
    static const struct { const char* code; const char* name; } langs[] = {
        {"hi", "Hindi"}, {"en", "English"}, {"ta", "Tamil"}, {"te", "Telugu"},
        {"bn", "Bengali"}, {"mr", "Marathi"}, {"gu", "Gujarati"}, {"kn", "Kannada"},
        {"ml", "Malayalam"}, {"pa", "Punjabi"}, {"or", "Odia"}, {"as", "Assamese"},
        {"ur", "Urdu"}, {"sa", "Sanskrit"}, {"sd", "Sindhi"}, {"ne", "Nepali"},
        {"kok", "Konkani"}, {"mni", "Manipuri"}, {"brx", "Bodo"}, {"doi", "Dogri"},
        {"mai", "Maithili"}, {"sat", "Santhali"}, {"ks", "Kashmiri"},
    };
    for (auto& l : langs) {
        if (code == l.code) return l.name;
    }
    return "Hindi";
}

// ──────────────────────────────────────────────
// Logging
// ──────────────────────────────────────────────

void log(LogLevel level, const char* fmt, ...) {
    const char* prefix;
    switch (level) {
        case LogLevel::DEBUG: prefix = "DEBUG"; break;
        case LogLevel::INFO:  prefix = "INFO";  break;
        case LogLevel::WARN:  prefix = "WARN";  break;
        case LogLevel::ERR:   prefix = "ERROR"; break;
        default:              prefix = "???";   break;
    }

    // Timestamp
    time_t now = time(nullptr);
    struct tm tm_buf;
    localtime_r(&now, &tm_buf);
    char ts[32];
    strftime(ts, sizeof(ts), "%H:%M:%S", &tm_buf);

    fprintf(stderr, "%s [%s] ", ts, prefix);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");
}

} // namespace netra
