#include "server.h"
#include "llm_engine.h"
#include "asr_engine.h"
#include "rag_engine.h"
#include "tts_engine.h"
#include "ocr_engine.h"
#include "utils.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <set>

namespace fs = std::filesystem;
using json = nlohmann::json;

static constexpr const char* NETRA_VERSION = "3.0.0";

namespace netra {

// ── Impl (pimpl for httplib::Server) ────────────────────────────────────────

struct Server::Impl {
    httplib::Server svr;
};

// ── Constructor / Destructor ────────────────────────────────────────────────

Server::Server(const ServerConfig& config)
    : config_(config), impl_(std::make_unique<Impl>()) {}

Server::~Server() { stop(); }

// ── Engine Initialisation ───────────────────────────────────────────────────

void Server::init_engines() {
    Timer t;

    LOG_INFO("Initialising LLM engine  — %s", config_.llm_model.c_str());
    LLMConfig lc;
    lc.model_path  = config_.llm_model;
    lc.max_tokens  = config_.max_tokens;
    lc.temperature = config_.temperature;
    lc.n_threads   = config_.workers;
    llm_ = std::make_unique<LLMEngine>(lc);

    try {
        LOG_INFO("Initialising ASR engine  — %s", config_.whisper_model.c_str());
        ASRConfig ac;
        ac.model_path = config_.whisper_model;
        ac.n_threads  = config_.workers;
        asr_ = std::make_unique<ASREngine>(ac);
    } catch (const std::exception& e) {
        LOG_WARN("ASR engine unavailable: %s", e.what());
    }

    try {
        LOG_INFO("Initialising RAG engine  — %s", config_.embedding_model.c_str());
        RAGConfig rc;
        rc.onnx_model_path = config_.embedding_model;
        rc.index_dir       = config_.index_dir;
        rag_ = std::make_unique<RAGEngine>(rc);
    } catch (const std::exception& e) {
        LOG_WARN("RAG engine unavailable: %s", e.what());
    }

    try {
        LOG_INFO("Initialising TTS engine  — %s", config_.voice_dir.c_str());
        TTSConfig tc;
        tc.voice_dir = config_.voice_dir;
        tts_ = std::make_unique<TTSEngine>(tc);
    } catch (const std::exception& e) {
        LOG_WARN("TTS engine unavailable: %s", e.what());
    }

    LOG_INFO("Initialising OCR engine");
    OCRConfig oc;
    ocr_ = std::make_unique<OCREngine>(oc);

    LOG_INFO("All engines loaded in %.1f ms", t.elapsed_ms());
}

// ── Auto-ingest policy documents ────────────────────────────────────────────

void Server::auto_ingest_policies() {
    if (!rag_) {
        LOG_WARN("RAG engine not available — skipping policy ingest");
        return;
    }

    // Collect policy files from all configured directories
    std::vector<std::string> all_files;
    auto scan_dir = [&](const std::string& dir) {
        if (dir.empty() || !fs::is_directory(dir)) return;
        auto files = list_files(dir, {".txt", ".pdf"});
        all_files.insert(all_files.end(), files.begin(), files.end());
    };

    scan_dir(config_.policies_dir);

    // Also scan the polyic directory if it exists (relative to working dir or project root)
    for (const auto& alt : {"polyic", "../polyic", "../../polyic"}) {
        if (fs::is_directory(alt) && std::string(alt) != config_.policies_dir) {
            LOG_INFO("Also scanning policy dir: %s", alt);
            scan_dir(alt);
        }
    }

    if (all_files.empty()) {
        LOG_WARN("No policy files found — skipping ingest");
        return;
    }

    // Build set of already-ingested sources to avoid duplicates
    std::set<std::string> existing_sources;
    auto meta_path = config_.index_dir + "/metadata.txt";
    if (file_exists(meta_path)) {
        std::ifstream in(meta_path);
        std::string line;
        while (std::getline(in, line)) {
            auto tab = line.find('\t');
            if (tab != std::string::npos) {
                existing_sources.insert(line.substr(0, tab));
            }
        }
    }

    Timer t;
    int ingested = 0;
    int skipped = 0;

    for (size_t fi = 0; fi < all_files.size(); ++fi) {
        const auto& path = all_files[fi];
        std::string filename = fs::path(path).filename().string();

        // Skip if already ingested
        if (existing_sources.count(filename)) {
            ++skipped;
            continue;
        }

        try {
            std::string ext = fs::path(path).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            std::string content;
            if (ext == ".txt") {
                content = read_file(path);
            } else if (ext == ".pdf") {
                auto ocr_result = ocr_->extract_text(path, "pdf");
                content = ocr_result.text;
            }

            if (content.empty()) {
                LOG_WARN("Empty content from %s — skipping", filename.c_str());
                continue;
            }

            int chunks = rag_->ingest(content, filename);
            LOG_INFO("Ingested %s  → %d chunks", filename.c_str(), chunks);
            ++ingested;
        } catch (const std::exception& e) {
            LOG_ERR("Failed to ingest %s: %s", path.c_str(), e.what());
        }
    }

    if (ingested > 0) {
        rag_->save();
    }
    LOG_INFO("Policy ingest: %d new, %d skipped, %d total chunks in %.1f ms",
             ingested, skipped, rag_->index_size(), t.elapsed_ms());
}

// ── Shared pipeline: RAG → LLM → TTS ───────────────────────────────────────

std::string Server::run_pipeline(const std::string& question,
                                  const std::string& language) {
    json timing = json::object();
    Timer total;

    // RAG retrieval — for non-English queries, also search with an English translation
    // prompt so the keyword-overlap matcher can find English policy documents
    Timer stage;
    RAGSearchResult rag_result;
    if (rag_) {
        rag_result = rag_->search(question, 2);

        // If non-English query returned no results, ask the LLM for a quick English translation
        // then re-search. This ensures Hindi/Tamil/etc. queries find English policy documents.
        if (rag_result.chunks.empty() && language != "en" && !question.empty()) {
            std::string translate_prompt =
                "Translate the following text to English in one line. "
                "Output ONLY the translation, nothing else.\n\n" + question;
            auto tr = llm_->generate(translate_prompt, "", 50, 0.1f);
            std::string en_query = tr.response;
            // Strip think blocks from translation
            auto ts = en_query.find("<think>");
            if (ts != std::string::npos) {
                auto te = en_query.find("</think>", ts);
                if (te != std::string::npos) en_query.erase(ts, te + 8 - ts);
                else en_query.erase(ts);
            }
            auto wp = en_query.find_first_not_of(" \t\n\r");
            if (wp != std::string::npos && wp > 0) en_query = en_query.substr(wp);
            auto nl = en_query.find('\n');
            if (nl != std::string::npos) en_query = en_query.substr(0, nl);

            if (!en_query.empty() && en_query.size() > 5) {
                LOG_INFO("RAG re-search with EN translation: %s", en_query.c_str());
                rag_result = rag_->search(en_query, 2);
            }
        }
    }
    timing["rag_ms"] = stage.elapsed_ms();

    // Build prompts
    std::string lang_name = language_name(language);

    std::string system_prompt =
        "You are NETRA, a helpful government scheme assistant for Indian citizens.\n"
        "IMPORTANT RULES:\n"
        "- Your ENTIRE response must be in " + lang_name + " language ONLY.\n"
        "- Do NOT write ANY English text (unless the language requested is English).\n"
        "- Do NOT explain your reasoning. Do NOT describe what you are doing.\n"
        "- Answer the question directly using the provided context.\n"
        "- If the context does not contain relevant information, say so briefly in " + lang_name + ".\n"
        "- Be concise and factual.";

    std::string user_prompt;
    if (!rag_result.context_text.empty()) {
        user_prompt = "Context:\n" + rag_result.context_text + "\n\n";
    }
    user_prompt += "Question: " + question;

    // LLM generation
    stage.reset();
    auto llm_result = llm_->generate(user_prompt, system_prompt,
                                      config_.max_tokens, config_.temperature);
    timing["llm_ms"] = stage.elapsed_ms();
    timing["tokens_per_sec"] = llm_result.tokens_per_sec;

    // Sanitize: trim trailing incomplete UTF-8 sequences (LLM may cut mid-character)
    std::string answer = llm_result.response;
    while (!answer.empty()) {
        auto c = static_cast<unsigned char>(answer.back());
        if (c < 0x80) break;               // ASCII — valid end
        if ((c & 0xC0) != 0x80) {           // leading byte without continuation
            answer.pop_back();
            continue;
        }
        // continuation byte — walk back to find leading byte
        size_t pos = answer.size() - 1;
        while (pos > 0 && (static_cast<unsigned char>(answer[pos]) & 0xC0) == 0x80) --pos;
        auto lead = static_cast<unsigned char>(answer[pos]);
        int expected = (lead >= 0xF0) ? 4 : (lead >= 0xE0) ? 3 : (lead >= 0xC0) ? 2 : 1;
        int actual = (int)(answer.size() - pos);
        if (actual >= expected) break;      // complete sequence
        answer.resize(pos);                 // trim incomplete
    }

    // Strip any <think>...</think> blocks from model output
    {
        auto think_start = answer.find("<think>");
        while (think_start != std::string::npos) {
            auto think_end = answer.find("</think>", think_start);
            if (think_end != std::string::npos) {
                answer.erase(think_start, think_end + 8 - think_start);
            } else {
                answer.erase(think_start);
                break;
            }
            think_start = answer.find("<think>");
        }
        auto s = answer.find_first_not_of(" \t\n\r");
        if (s != std::string::npos && s > 0) answer = answer.substr(s);
    }

    // Strip meta-commentary from LLM output
    {
        auto strip_ws = [](std::string& s) {
            auto p = s.find_first_not_of(" \t\n\r");
            if (p != std::string::npos && p > 0) s = s.substr(p);
            while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r' || s.back() == '\t'))
                s.pop_back();
        };

        if (language != "en" && !answer.empty()) {
            // For non-English: find the first substantial non-ASCII (Indic script) block
            // The model often reasons in English before the actual target-language answer
            const auto* p = reinterpret_cast<const unsigned char*>(answer.c_str());
            size_t len = answer.size();
            size_t best_start = std::string::npos;

            // Scan for first position where non-ASCII chars appear densely
            for (size_t i = 0; i < len; ++i) {
                if (p[i] >= 0xC0) {  // start of a multi-byte UTF-8 char
                    // Check if this is part of a substantial non-ASCII block (not just a quoted word)
                    int non_ascii_count = 0;
                    size_t scan = i;
                    while (scan < len && scan < i + 100) {
                        if (p[scan] >= 0xC0) ++non_ascii_count;
                        ++scan;
                    }
                    if (non_ascii_count >= 5) {
                        // Found a block with 5+ non-ASCII chars in 100 bytes — this is real content
                        // Back up to the start of the line or sentence containing this
                        best_start = i;
                        while (best_start > 0 && answer[best_start - 1] != '\n') --best_start;
                        break;
                    }
                }
            }

            if (best_start != std::string::npos && best_start > 0) {
                answer = answer.substr(best_start);
            }
        }

        // For all languages: strip common English preamble markers from the start
        static const char* markers[] = {
            "Hmm", "Let me", "I need", "I'll", "Wait", "But ",
            "The user", "Looking at", "Based on", "I should",
            "Let's", "OK,", "Ok,", "Okay,", "First,", "Now,",
            "So,", "So the", "Actually", "The context",
            "From the", "According", "Checking", "The provided",
            "This seems", "First document", "Second document",
            "The question", "I see", "Here ", "Alright",
            "However,", "NETRA", "Important:", "Note:",
            "The problem", "In this case", "Since ",
            "The term", "The rules", "The answer",
            "The first", "The second", "The third",
            "Hindi ", "Tamil ", "Telugu ", "Bengali ", "Marathi ",
            "Gujarati ", "Kannada ", "Malayalam ", "Punjabi ",
            "Odia ", "Assamese ", "Urdu ", "Sanskrit ",
            "Nepali ", "Bodo ", "Dogri ", "Maithili ",
            "Sindhi ", "Konkani ", "Manipuri ", "Santhali ",
            nullptr
        };

        bool did_strip = true;
        int rounds = 15;
        while (did_strip && !answer.empty() && --rounds > 0) {
            did_strip = false;
            strip_ws(answer);
            for (int m = 0; markers[m]; ++m) {
                size_t mlen = strlen(markers[m]);
                if (answer.size() >= mlen &&
                    answer.compare(0, mlen, markers[m]) == 0) {
                    auto nl = answer.find('\n');
                    auto dot = answer.find(". ");
                    size_t cut = std::string::npos;
                    if (nl != std::string::npos && nl < 500) cut = nl + 1;
                    else if (dot != std::string::npos && dot < 500) cut = dot + 2;
                    else if (answer.size() < 300) {
                        // Strip entire short answer if it's mostly ASCII (English meta-commentary)
                        // Allow non-ASCII in quoted words (e.g. quoting the original question)
                        int non_ascii = 0;
                        for (auto ch : answer) if (static_cast<unsigned char>(ch) >= 0x80) ++non_ascii;
                        if (non_ascii * 3 < (int)answer.size()) cut = answer.size();
                    }
                    if (cut != std::string::npos) {
                        answer = answer.substr(cut);
                        did_strip = true;
                        break;
                    }
                }
            }
        }
        strip_ws(answer);

        // If after all stripping the answer is too short or empty, use a native-language fallback
        if (answer.size() < 20) {
            static const struct { const char* code; const char* msg; } fallbacks[] = {
                {"hi", "यह जानकारी वर्तमान दस्तावेजों में उपलब्ध नहीं है।"},
                {"ta", "இந்தத் தகவல் தற்போதைய ஆவணங்களில் கிடைக்கவில்லை."},
                {"te", "ఈ సమాచారం ప్రస్తుత పత్రాలలో అందుబాటులో లేదు."},
                {"bn", "এই তথ্য বর্তমান নথিতে পাওয়া যায়নি।"},
                {"mr", "ही माहिती सध्याच्या दस्तऐवजांमध्ये उपलब्ध नाही."},
                {"gu", "આ માહિતી હાલના દસ્તાવેજોમાં ઉપલબ્ધ નથી."},
                {"kn", "ಈ ಮಾಹಿತಿ ಪ್ರಸ್ತುತ ದಾಖಲೆಗಳಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ."},
                {"ml", "ഈ വിവരം നിലവിലെ രേഖകളിൽ ലഭ്യമല്ല."},
                {"pa", "ਇਹ ਜਾਣਕਾਰੀ ਮੌਜੂਦਾ ਦਸਤਾਵੇਜ਼ਾਂ ਵਿੱਚ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।"},
                {"or", "ଏହି ତଥ୍ୟ ବର୍ତ୍ତମାନ ଉପಲବ୍ଧ ଦଲିଲରେ ନାହିଁ।"},
                {"as", "এই তথ্য বৰ্তমান নথিত উপলব্ধ নহয়।"},
                {"ur", "یہ معلومات موجودہ دستاویزات میں دستیاب نہیں ہے۔"},
                {"sa", "एतत् सूचना वर्तमानदस्तावेजेषु उपलब्धा नास्ति।"},
                {"ne", "यो जानकारी हालका कागजातहरूमा उपलब्ध छैन।"},
                {"sd", "هي ڄاڻ موجوده دستاويزن ۾ دستياب ناهي."},
                {"kok", "ही म्हायती सध्याच्या दस्तऐवजांनी उपलब्ध ना."},
                {"mni", "ꯃꯁꯤ ꯂꯩꯕꯥ ꯃꯑꯣꯡ ꯑꯗꯨꯗꯒꯤ ꯐꯪꯕꯤꯔꯣꯏ."},
                {"brx", "बे जानाय बिदांनि फाइलनि गेजेराव मोनसे गैया।"},
                {"doi", "एह् जानकारी मौजूदा दस्तावेजें च उपलब्ध नेईं ऐ।"},
                {"mai", "ई जानकारी वर्तमान दस्तावेज मे उपलब्ध नहि अछि।"},
                {"sat", "ᱱᱚᱶᱟ ᱡᱟᱱᱟᱣ ᱱᱤᱛᱚᱜ ᱫᱚᱞᱤᱞ ᱨᱮ ᱵᱟᱝ ᱧᱟᱢᱚᱜᱼᱟ।"},
                {"en", "This information is not available in the current documents."},
            };
            bool found = false;
            for (auto& fb : fallbacks) {
                if (language == fb.code) {
                    answer = fb.msg;
                    found = true;
                    break;
                }
            }
            if (!found) answer = "This information is not available in the current documents.";
        }
    }

    // TTS synthesis + base64 encoding
    stage.reset();
    std::string audio_b64;
    if (tts_) {
        auto wav_bytes = tts_->synthesize(answer, language);
        audio_b64 = base64_encode(wav_bytes);
    }
    timing["tts_ms"] = stage.elapsed_ms();

    timing["total_ms"] = total.elapsed_ms();

    // Collect source names
    json sources = json::array();
    for (const auto& chunk : rag_result.chunks) {
        sources.push_back({{"source", chunk.source},
                           {"score",  chunk.score},
                           {"rank",   chunk.rank}});
    }

    json response = {
        {"answer",   answer},
        {"audio",    audio_b64},
        {"language", language},
        {"sources",  sources},
        {"timing",   timing}
    };

    LOG_INFO("Pipeline completed in %.1f ms  (RAG=%.1f  LLM=%.1f  TTS=%.1f)",
             timing["total_ms"].get<double>(),
             timing["rag_ms"].get<double>(),
             timing["llm_ms"].get<double>(),
             timing["tts_ms"].get<double>());

    return response.dump();
}

// ── Helpers ─────────────────────────────────────────────────────────────────

static void set_cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

static void json_error(httplib::Response& res, int status, const std::string& msg) {
    set_cors(res);
    res.status = status;
    res.set_content(json({{"error", msg}}).dump(), "application/json");
}

static std::string save_temp_file(const httplib::FormData& file,
                                  const std::string& suffix) {
    std::string tmp = "/tmp/netra_" +
                      std::to_string(std::chrono::steady_clock::now()
                                         .time_since_epoch().count()) + suffix;
    std::ofstream ofs(tmp, std::ios::binary);
    ofs.write(file.content.data(), static_cast<std::streamsize>(file.content.size()));
    ofs.close();
    return tmp;
}

// ── Route Registration ──────────────────────────────────────────────────────

void Server::register_routes() {
    auto& svr = impl_->svr;

    // Global CORS pre-flight
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        set_cors(res);
        res.status = 204;
    });

    // ── GET /api/health ─────────────────────────────────────────────────
    svr.Get("/api/health", [this](const httplib::Request&, httplib::Response& res) {
        set_cors(res);
        json body = {
            {"status",  "ok"},
            {"version", NETRA_VERSION},
            {"engines", {
                {"llm", llm_ && llm_->is_loaded()},
                {"asr", asr_ && asr_->is_loaded()},
                {"rag", rag_ != nullptr},
                {"tts", tts_ && tts_->is_loaded()},
                {"ocr", ocr_ != nullptr}
            }},
            {"index_size", rag_ ? rag_->index_size() : 0}
        };
        res.set_content(body.dump(), "application/json");
    });

    // ── POST /api/query ─────────────────────────────────────────────────
    svr.Post("/api/query", [this](const httplib::Request& req,
                                   httplib::Response& res) {
        Timer t;
        set_cors(res);

        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            json_error(res, 400, "Invalid JSON body");
            return;
        }

        if (!body.contains("question") || !body["question"].is_string() ||
            body["question"].get<std::string>().empty()) {
            json_error(res, 422, "Missing or empty 'question' field");
            return;
        }

        std::string question = body["question"].get<std::string>();
        std::string language = body.value("language", "");
        if (language.empty()) {
            language = detect_language(question);
        }

        try {
            std::string result = run_pipeline(question, language);
            LOG_INFO("POST /api/query  — %.1f ms", t.elapsed_ms());
            res.set_content(result, "application/json");
        } catch (const std::exception& e) {
            LOG_ERR("POST /api/query failed: %s", e.what());
            json_error(res, 500, std::string("Pipeline error: ") + e.what());
        }
    });

    // ── POST /api/audio ─────────────────────────────────────────────────
    svr.Post("/api/audio", [this](const httplib::Request& req,
                                   httplib::Response& res) {
        Timer t;
        set_cors(res);

        if (!asr_) {
            json_error(res, 503, "ASR engine not available");
            return;
        }

        if (!req.form.has_file("file")) {
            json_error(res, 400, "Missing 'file' in multipart upload");
            return;
        }

        auto file = req.form.get_file("file");
        std::string tmp_path = save_temp_file(file, ".wav");

        try {
            Timer asr_timer;
            auto asr_result = asr_->transcribe(tmp_path);
            double asr_ms = asr_timer.elapsed_ms();

            std::remove(tmp_path.c_str());

            if (asr_result.text.empty()) {
                json_error(res, 422, "Could not transcribe audio");
                return;
            }

            std::string language = asr_result.language;
            if (language.empty()) {
                language = detect_language(asr_result.text);
            }

            std::string result_str = run_pipeline(asr_result.text, language);
            json result = json::parse(result_str);
            result["transcription"] = asr_result.text;
            result["timing"]["asr_ms"] = asr_ms;

            LOG_INFO("POST /api/audio  — %.1f ms", t.elapsed_ms());
            res.set_content(result.dump(), "application/json");
        } catch (const std::exception& e) {
            std::remove(tmp_path.c_str());
            LOG_ERR("POST /api/audio failed: %s", e.what());
            json_error(res, 500, std::string("Audio pipeline error: ") + e.what());
        }
    });

    // ── POST /api/pdf ───────────────────────────────────────────────────
    svr.Post("/api/pdf", [this](const httplib::Request& req,
                                 httplib::Response& res) {
        Timer t;
        set_cors(res);

        if (!req.form.has_file("file")) {
            json_error(res, 400, "Missing 'file' in multipart upload");
            return;
        }

        auto file = req.form.get_file("file");
        std::string tmp_path = save_temp_file(file, ".pdf");

        try {
            Timer ocr_timer;
            auto ocr_result = ocr_->extract_text(tmp_path, "pdf");
            double ocr_ms = ocr_timer.elapsed_ms();

            std::remove(tmp_path.c_str());

            if (ocr_result.text.empty()) {
                json_error(res, 422, "Could not extract text from PDF");
                return;
            }

            std::string language = detect_language(ocr_result.text);
            std::string question = ocr_result.text.substr(
                0, std::min<size_t>(ocr_result.text.size(), 2000));

            std::string result_str = run_pipeline(question, language);
            json result = json::parse(result_str);
            result["extracted_text"] = ocr_result.text;
            result["timing"]["ocr_ms"] = ocr_ms;

            LOG_INFO("POST /api/pdf  — %.1f ms", t.elapsed_ms());
            res.set_content(result.dump(), "application/json");
        } catch (const std::exception& e) {
            std::remove(tmp_path.c_str());
            LOG_ERR("POST /api/pdf failed: %s", e.what());
            json_error(res, 500, std::string("PDF pipeline error: ") + e.what());
        }
    });

    // ── POST /api/image ─────────────────────────────────────────────────
    svr.Post("/api/image", [this](const httplib::Request& req,
                                   httplib::Response& res) {
        Timer t;
        set_cors(res);

        if (!req.form.has_file("file")) {
            json_error(res, 400, "Missing 'file' in multipart upload");
            return;
        }

        auto file = req.form.get_file("file");
        std::string ext = fs::path(file.filename).extension().string();
        if (ext.empty()) ext = ".png";
        std::string tmp_path = save_temp_file(file, ext);

        try {
            Timer ocr_timer;
            auto ocr_result = ocr_->extract_text(tmp_path, "image");
            double ocr_ms = ocr_timer.elapsed_ms();

            std::remove(tmp_path.c_str());

            if (ocr_result.text.empty()) {
                json_error(res, 422, "Could not extract text from image");
                return;
            }

            std::string language = detect_language(ocr_result.text);

            std::string result_str = run_pipeline(ocr_result.text, language);
            json result = json::parse(result_str);
            result["extracted_text"] = ocr_result.text;
            result["timing"]["ocr_ms"] = ocr_ms;

            LOG_INFO("POST /api/image  — %.1f ms", t.elapsed_ms());
            res.set_content(result.dump(), "application/json");
        } catch (const std::exception& e) {
            std::remove(tmp_path.c_str());
            LOG_ERR("POST /api/image failed: %s", e.what());
            json_error(res, 500, std::string("Image pipeline error: ") + e.what());
        }
    });

    // ── POST /api/ingest ────────────────────────────────────────────────
    svr.Post("/api/ingest", [this](const httplib::Request& req,
                                    httplib::Response& res) {
        Timer t;
        set_cors(res);

        if (!req.form.has_file("file")) {
            json_error(res, 400, "Missing 'file' in multipart upload");
            return;
        }

        auto file = req.form.get_file("file");
        std::string ext = fs::path(file.filename).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext.empty()) ext = ".pdf";
        std::string tmp_path = save_temp_file(file, ext);

        try {
            std::string content;
            if (ext == ".txt") {
                content = read_file(tmp_path);
            } else {
                auto ocr_result = ocr_->extract_text(tmp_path,
                    (ext == ".pdf") ? "pdf" : "image");
                content = ocr_result.text;
            }

            std::remove(tmp_path.c_str());

            if (content.empty()) {
                json_error(res, 422, "Could not extract text from uploaded file");
                return;
            }

            std::string source = file.filename.empty() ? "upload" : file.filename;
            int chunks = rag_->ingest(content, source);
            rag_->save();

            json result = {
                {"status",     "ingested"},
                {"source",     source},
                {"chunks",     chunks},
                {"index_size", rag_->index_size()},
                {"timing",     {{"total_ms", t.elapsed_ms()}}}
            };

            LOG_INFO("POST /api/ingest  — %s  → %d chunks  (%.1f ms)",
                     source.c_str(), chunks, t.elapsed_ms());
            res.set_content(result.dump(), "application/json");
        } catch (const std::exception& e) {
            std::remove(tmp_path.c_str());
            LOG_ERR("POST /api/ingest failed: %s", e.what());
            json_error(res, 500, std::string("Ingest error: ") + e.what());
        }
    });

    // ── POST /api/pipeline/text — compatibility with NETRA v2 UI ─────
    svr.Post("/api/pipeline/text", [this](const httplib::Request& req,
                                          httplib::Response& res) {
        Timer t;
        set_cors(res);

        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            json_error(res, 400, "Invalid JSON body");
            return;
        }

        std::string text = body.value("text", "");
        if (text.empty()) {
            json_error(res, 422, "Missing or empty 'text' field");
            return;
        }

        std::string language = body.value("language", "");
        if (language.empty()) language = detect_language(text);

        try {
            std::string result = run_pipeline(text, language);
            LOG_INFO("POST /api/pipeline/text  — %.1f ms", t.elapsed_ms());
            res.set_content(result, "application/json");
        } catch (const std::exception& e) {
            LOG_ERR("POST /api/pipeline/text failed: %s", e.what());
            json_error(res, 500, std::string("Pipeline error: ") + e.what());
        }
    });

    // ── POST /api/pipeline/run — compatibility for file uploads ──────
    svr.Post("/api/pipeline/run", [this](const httplib::Request& req,
                                          httplib::Response& res) {
        Timer t;
        set_cors(res);

        if (req.form.has_file("audio")) {
            auto file = req.form.get_file("audio");
            if (!asr_) {
                json_error(res, 503, "ASR engine not available");
                return;
            }
            std::string tmp_path = save_temp_file(file, ".wav");
            try {
                auto asr_result = asr_->transcribe(tmp_path);
                std::remove(tmp_path.c_str());
                if (asr_result.text.empty()) {
                    json_error(res, 422, "Could not transcribe audio");
                    return;
                }
                std::string lang = asr_result.language.empty()
                    ? detect_language(asr_result.text) : asr_result.language;
                std::string result_str = run_pipeline(asr_result.text, lang);
                json result = json::parse(result_str);
                result["transcription"] = asr_result.text;
                res.set_content(result.dump(), "application/json");
            } catch (const std::exception& e) {
                std::remove(tmp_path.c_str());
                json_error(res, 500, std::string("Pipeline error: ") + e.what());
            }
        } else if (req.form.has_file("image")) {
            auto file = req.form.get_file("image");
            std::string ext = fs::path(file.filename).extension().string();
            if (ext.empty()) ext = ".png";
            std::string tmp_path = save_temp_file(file, ext);
            try {
                auto ocr_result = ocr_->extract_text(tmp_path, "image");
                std::remove(tmp_path.c_str());
                if (ocr_result.text.empty()) {
                    json_error(res, 422, "Could not extract text from image");
                    return;
                }
                std::string lang = detect_language(ocr_result.text);
                std::string result_str = run_pipeline(ocr_result.text, lang);
                json result = json::parse(result_str);
                result["extracted_text"] = ocr_result.text;
                res.set_content(result.dump(), "application/json");
            } catch (const std::exception& e) {
                std::remove(tmp_path.c_str());
                json_error(res, 500, std::string("Pipeline error: ") + e.what());
            }
        } else if (req.form.has_file("file")) {
            auto file = req.form.get_file("file");
            std::string ext = fs::path(file.filename).extension().string();
            std::string tmp_path = save_temp_file(file, ext.empty() ? ".pdf" : ext);
            try {
                auto ocr_result = ocr_->extract_text(tmp_path, "pdf");
                std::remove(tmp_path.c_str());
                if (ocr_result.text.empty()) {
                    json_error(res, 422, "Could not extract text");
                    return;
                }
                std::string lang = detect_language(ocr_result.text);
                std::string result_str = run_pipeline(ocr_result.text, lang);
                json result = json::parse(result_str);
                result["extracted_text"] = ocr_result.text;
                res.set_content(result.dump(), "application/json");
            } catch (const std::exception& e) {
                std::remove(tmp_path.c_str());
                json_error(res, 500, std::string("Pipeline error: ") + e.what());
            }
        } else {
            json_error(res, 400, "Missing file in multipart upload");
        }

        LOG_INFO("POST /api/pipeline/run  — %.1f ms", t.elapsed_ms());
    });

    // ── GET /api/ready — compatibility ───────────────────────────────
    svr.Get("/api/ready", [this](const httplib::Request&, httplib::Response& res) {
        set_cors(res);
        json body = {
            {"ready", llm_ && llm_->is_loaded()},
            {"status", (llm_ && llm_->is_loaded()) ? "ready" : "not_ready"}
        };
        res.set_content(body.dump(), "application/json");
    });

    // ── GET /api/config — stub ───────────────────────────────────────
    svr.Get("/api/config", [](const httplib::Request&, httplib::Response& res) {
        set_cors(res);
        json body = {{"version", NETRA_VERSION}, {"backend", "cpp"}};
        res.set_content(body.dump(), "application/json");
    });

    LOG_INFO("Registered API routes");
}

// ── Start / Stop ────────────────────────────────────────────────────────────

void Server::start() {
    LOG_INFO("═══════════════════════════════════════════════════");
    LOG_INFO("  NETRA v%s — Government Scheme Voice Assistant", NETRA_VERSION);
    LOG_INFO("═══════════════════════════════════════════════════");

    Timer t;
    init_engines();
    auto_ingest_policies();
    register_routes();

    LOG_INFO("Server ready in %.1f ms — listening on %s:%d",
             t.elapsed_ms(), config_.host.c_str(), config_.port);

    impl_->svr.listen(config_.host, config_.port);
}

void Server::stop() {
    if (impl_) {
        impl_->svr.stop();
        LOG_INFO("Server stopped");
    }
}

} // namespace netra
