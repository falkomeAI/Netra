#include "rag_engine.h"
#include "utils.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

#ifdef NETRA_HAS_FAISS
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#ifdef NETRA_HAS_CUDA
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuAutoTune.h>
#endif
#endif

#ifdef NETRA_HAS_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace netra {

// ──────────────────────────────────────────────
// Internal storage
// ──────────────────────────────────────────────

struct RAGEngine::Impl {
    std::vector<std::string> texts;
    std::vector<std::string> sources;

#ifdef NETRA_HAS_FAISS
    std::unique_ptr<faiss::IndexFlatIP> cpu_index;
#ifdef NETRA_HAS_CUDA
    std::unique_ptr<faiss::gpu::StandardGpuResources> gpu_res;
    std::unique_ptr<faiss::gpu::GpuIndexFlatIP> gpu_index;
#endif
    faiss::Index* active_index = nullptr;
#endif

#ifdef NETRA_HAS_ORT
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "netra_embed"};
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
#endif
};

// ──────────────────────────────────────────────
// Constructor
// ──────────────────────────────────────────────

RAGEngine::RAGEngine(const RAGConfig& config)
    : config_(config), impl_(std::make_unique<Impl>())
{
    Timer t;

    // ONNX embedding model
#ifdef NETRA_HAS_ORT
    if (!config_.onnx_model_path.empty() && file_exists(config_.onnx_model_path)) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(2);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Try CUDA provider
        if (config_.use_gpu) {
            try {
                OrtCUDAProviderOptions cuda_opts{};
                cuda_opts.device_id = 0;
                opts.AppendExecutionProvider_CUDA(cuda_opts);
                LOG_INFO("[RAG] ONNX Runtime using CUDA");
            } catch (...) {
                LOG_WARN("[RAG] CUDA provider unavailable, using CPU");
            }
        }

        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, config_.onnx_model_path.c_str(), opts);
        LOG_INFO("[RAG] Embedding model loaded: %s (%.1fms)",
                 config_.onnx_model_path.c_str(), t.elapsed_ms());
    } else {
        LOG_WARN("[RAG] No ONNX embedding model at: %s", config_.onnx_model_path.c_str());
    }
#endif

    // FAISS index
#ifdef NETRA_HAS_FAISS
    impl_->cpu_index = std::make_unique<faiss::IndexFlatIP>(config_.embedding_dim);

#ifdef NETRA_HAS_CUDA
    if (config_.use_gpu) {
        try {
            impl_->gpu_res = std::make_unique<faiss::gpu::StandardGpuResources>();
            impl_->gpu_res->setTempMemory(64 * 1024 * 1024);
            impl_->gpu_index = std::make_unique<faiss::gpu::GpuIndexFlatIP>(
                impl_->gpu_res.get(), config_.embedding_dim);
            impl_->active_index = impl_->gpu_index.get();
            LOG_INFO("[RAG] FAISS GPU index created (dim=%d)", config_.embedding_dim);
        } catch (const std::exception& e) {
            LOG_WARN("[RAG] FAISS GPU failed (%s), using CPU", e.what());
            impl_->active_index = impl_->cpu_index.get();
        }
    } else
#endif
    {
        impl_->active_index = impl_->cpu_index.get();
        LOG_INFO("[RAG] FAISS CPU index created (dim=%d)", config_.embedding_dim);
    }
#endif

    // Load existing index from disk
    load_from_disk();
    LOG_INFO("[RAG] Ready in %.1fms (index size: %d)", t.elapsed_ms(), index_size());
}

RAGEngine::~RAGEngine() = default;

// ──────────────────────────────────────────────
// Embedding
// ──────────────────────────────────────────────

std::vector<float> RAGEngine::embed(const std::string& text) const {
    std::vector<float> embedding(config_.embedding_dim, 0.0f);

#ifdef NETRA_HAS_ORT
    if (!impl_->session) return embedding;

    // Simple whitespace tokenization (placeholder — real deployment should use
    // the model's actual tokenizer via sentencepiece or tokenizers.cpp).
    // For BGE-small we just feed the raw text via ONNX input_ids.
    // This simplified path works when the ONNX model has been exported with
    // a tokenizer baked in (e.g., optimum export with --task feature-extraction).

    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Encode text as UTF-8 bytes → int64 token IDs
    // Minimal byte-level fallback: each byte becomes a token ID
    std::vector<int64_t> input_ids;
    input_ids.reserve(text.size() + 2);
    input_ids.push_back(101);  // [CLS]
    for (unsigned char c : text) {
        if (input_ids.size() >= 510) break;
        input_ids.push_back(static_cast<int64_t>(c));
    }
    input_ids.push_back(102);  // [SEP]

    std::vector<int64_t> attention_mask(input_ids.size(), 1);
    std::vector<int64_t> token_type_ids(input_ids.size(), 0);

    int64_t seq_len = static_cast<int64_t>(input_ids.size());
    std::array<int64_t, 2> shape = {1, seq_len};

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_info, input_ids.data(), input_ids.size(), shape.data(), 2));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_info, attention_mask.data(), attention_mask.size(), shape.data(), 2));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_info, token_type_ids.data(), token_type_ids.size(), shape.data(), 2));

    // Get input/output names
    auto input_count = impl_->session->GetInputCount();
    std::vector<Ort::AllocatedStringPtr> in_ptrs;
    std::vector<const char*> input_names;
    for (size_t i = 0; i < input_count && i < 3; ++i) {
        in_ptrs.push_back(impl_->session->GetInputNameAllocated(i, impl_->allocator));
        input_names.push_back(in_ptrs.back().get());
    }

    auto output_count = impl_->session->GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> out_ptrs;
    std::vector<const char*> output_names;
    for (size_t i = 0; i < output_count; ++i) {
        out_ptrs.push_back(impl_->session->GetOutputNameAllocated(i, impl_->allocator));
        output_names.push_back(out_ptrs.back().get());
    }

    auto outputs = impl_->session->Run(
        Ort::RunOptions{nullptr},
        input_names.data(), inputs.data(), inputs.size(),
        output_names.data(), output_names.size());

    // Mean-pool the token embeddings
    auto& out_tensor = outputs[0];
    auto out_shape = out_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const float* out_data = out_tensor.GetTensorData<float>();

    int tokens = static_cast<int>(out_shape[1]);
    int dim = static_cast<int>(out_shape[2]);
    if (dim != config_.embedding_dim) {
        LOG_WARN("[RAG] Embedding dim mismatch: expected %d, got %d", config_.embedding_dim, dim);
    }

    for (int d = 0; d < std::min(dim, config_.embedding_dim); ++d) {
        float sum = 0.0f;
        for (int t = 0; t < tokens; ++t) {
            sum += out_data[t * dim + d];
        }
        embedding[d] = sum / static_cast<float>(tokens);
    }

    // L2 normalize
    float norm = 0.0f;
    for (float v : embedding) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-9f) {
        for (float& v : embedding) v /= norm;
    }
#endif

    return embedding;
}

// ──────────────────────────────────────────────
// Chunking
// ──────────────────────────────────────────────

std::vector<std::string> RAGEngine::chunk_text(const std::string& text) const {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) words.push_back(word);

    if (words.empty()) return {};
    if (static_cast<int>(words.size()) <= config_.chunk_size_words) return {text};

    std::vector<std::string> chunks;
    int start = 0;
    while (start < static_cast<int>(words.size())) {
        int end = std::min(start + config_.chunk_size_words,
                           static_cast<int>(words.size()));
        std::ostringstream oss;
        for (int i = start; i < end; ++i) {
            if (i > start) oss << ' ';
            oss << words[i];
        }
        chunks.push_back(oss.str());
        if (end >= static_cast<int>(words.size())) break;
        start += config_.chunk_size_words - config_.chunk_overlap;
    }
    return chunks;
}

// ──────────────────────────────────────────────
// Ingest
// ──────────────────────────────────────────────

int RAGEngine::ingest(const std::string& text, const std::string& source) {
    auto chunks = chunk_text(text);
    if (chunks.empty()) return 0;

    Timer t;
    std::lock_guard<std::mutex> lock(mtx_);

    int added = 0;
    for (auto& chunk : chunks) {
        auto vec = embed(chunk);

#ifdef NETRA_HAS_FAISS
        if (impl_->active_index) {
            impl_->active_index->add(1, vec.data());
        }
#endif
        impl_->texts.push_back(std::move(chunk));
        impl_->sources.push_back(source);
        ++added;
    }

    LOG_INFO("[RAG] Ingested %d chunks from '%s' in %.0fms (total: %d)",
             added, source.c_str(), t.elapsed_ms(), index_size());
    return added;
}

// ──────────────────────────────────────────────
// Search
// ──────────────────────────────────────────────

RAGSearchResult RAGEngine::search(const std::string& query, int top_k,
                                   float min_score) const {
    RAGSearchResult result;
    if (top_k < 0) top_k = config_.top_k;
    if (min_score < 0) min_score = config_.min_score;

#ifdef NETRA_HAS_FAISS
    if (!impl_->active_index || impl_->active_index->ntotal == 0) {
        return result;
    }

    Timer t;
    auto qvec = embed(query);

    int k = std::min(top_k, static_cast<int>(impl_->active_index->ntotal));
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> indices(k);

    {
        std::lock_guard<std::mutex> lock(mtx_);
        impl_->active_index->search(1, qvec.data(), k, distances.data(), indices.data());
    }

    std::ostringstream ctx;
    int rank = 0;
    for (int i = 0; i < k; ++i) {
        if (indices[i] < 0 || distances[i] < min_score) continue;
        auto idx = static_cast<size_t>(indices[i]);
        if (idx >= impl_->texts.size()) continue;

        ++rank;
        ChunkResult cr;
        cr.text   = impl_->texts[idx];
        cr.score  = distances[i];
        cr.source = impl_->sources[idx];
        cr.rank   = rank;
        result.chunks.push_back(cr);

        ctx << "[Source: " << cr.source << " | Score: " << cr.score << "]\n"
            << cr.text << "\n\n---\n\n";
    }
    result.context_text = ctx.str();

    LOG_INFO("[RAG] Search returned %d chunks in %.1fms",
             static_cast<int>(result.chunks.size()), t.elapsed_ms());
#else
    // Fallback: keyword overlap search when FAISS is not available
    if (impl_->texts.empty()) return result;

    Timer t;

    // Tokenize query into lowercase words
    auto to_lower_words = [](const std::string& s) {
        std::vector<std::string> words;
        std::istringstream iss(s);
        std::string w;
        while (iss >> w) {
            std::transform(w.begin(), w.end(), w.begin(), ::tolower);
            if (w.size() > 2) words.push_back(w);
        }
        return words;
    };

    auto query_words = to_lower_words(query);
    if (query_words.empty()) return result;

    struct ScoredIdx { float score; size_t idx; };
    std::vector<ScoredIdx> scored;

    std::lock_guard<std::mutex> lock(mtx_);
    for (size_t i = 0; i < impl_->texts.size(); ++i) {
        auto chunk_words = to_lower_words(impl_->texts[i]);
        if (chunk_words.empty()) continue;

        int hits = 0;
        for (const auto& qw : query_words) {
            for (const auto& cw : chunk_words) {
                if (cw.find(qw) != std::string::npos || qw.find(cw) != std::string::npos) {
                    ++hits;
                    break;
                }
            }
        }
        float score = static_cast<float>(hits) / static_cast<float>(query_words.size());
        if (score > 0.0f) scored.push_back({score, i});
    }

    std::sort(scored.begin(), scored.end(), [](const ScoredIdx& a, const ScoredIdx& b) {
        return a.score > b.score;
    });

    std::ostringstream ctx;
    int rank = 0;
    for (int i = 0; i < std::min(top_k, static_cast<int>(scored.size())); ++i) {
        ++rank;
        ChunkResult cr;
        cr.text   = impl_->texts[scored[i].idx];
        cr.score  = scored[i].score;
        cr.source = impl_->sources[scored[i].idx];
        cr.rank   = rank;
        result.chunks.push_back(cr);

        ctx << "[Source: " << cr.source << " | Score: " << cr.score << "]\n"
            << cr.text << "\n\n---\n\n";
    }
    result.context_text = ctx.str();

    LOG_INFO("[RAG] Keyword search returned %d chunks in %.1fms",
             static_cast<int>(result.chunks.size()), t.elapsed_ms());
#endif

    return result;
}

// ──────────────────────────────────────────────
// Persistence
// ──────────────────────────────────────────────

void RAGEngine::save() const {
    Timer t;
    fs::create_directories(config_.index_dir);

    auto meta_path  = config_.index_dir + "/metadata.txt";

#ifdef NETRA_HAS_FAISS
    if (!impl_->active_index) return;

    auto index_path = config_.index_dir + "/index.faiss";

    faiss::Index* to_save = impl_->cpu_index.get();
#ifdef NETRA_HAS_CUDA
    std::unique_ptr<faiss::Index> cpu_copy;
    if (impl_->gpu_index) {
        cpu_copy.reset(faiss::gpu::index_gpu_to_cpu(impl_->gpu_index.get()));
        to_save = cpu_copy.get();
    }
#endif
    faiss::write_index(to_save, index_path.c_str());
#endif

    // Save texts + sources (always, regardless of FAISS)
    std::ofstream out(meta_path);
    for (size_t i = 0; i < impl_->texts.size(); ++i) {
        std::string escaped = impl_->texts[i];
        for (auto& c : escaped) { if (c == '\n') c = ' '; }
        out << impl_->sources[i] << "\t" << escaped << "\n";
    }

    LOG_INFO("[RAG] Saved index to %s in %.0fms (%d vectors)",
             config_.index_dir.c_str(), t.elapsed_ms(), index_size());
}

void RAGEngine::load_from_disk() {
    auto meta_path  = config_.index_dir + "/metadata.txt";

#ifdef NETRA_HAS_FAISS
    auto index_path = config_.index_dir + "/index.faiss";

    if (file_exists(index_path)) {
        Timer t;

        faiss::Index* loaded = faiss::read_index(index_path.c_str());
        impl_->cpu_index.reset(dynamic_cast<faiss::IndexFlatIP*>(loaded));

#ifdef NETRA_HAS_CUDA
        if (impl_->gpu_res && impl_->cpu_index) {
            try {
                impl_->gpu_index = std::make_unique<faiss::gpu::GpuIndexFlatIP>(
                    impl_->gpu_res.get(), config_.embedding_dim);
                impl_->gpu_index->copyFrom(impl_->cpu_index.get());
                impl_->active_index = impl_->gpu_index.get();
            } catch (...) {
                impl_->active_index = impl_->cpu_index.get();
            }
        } else
#endif
        {
            impl_->active_index = impl_->cpu_index.get();
        }
    }
#endif

    // Load metadata (always, regardless of FAISS)
    if (file_exists(meta_path)) {
        Timer t;
        std::ifstream in(meta_path);
        std::string line;
        while (std::getline(in, line)) {
            auto tab = line.find('\t');
            if (tab != std::string::npos) {
                impl_->sources.push_back(line.substr(0, tab));
                impl_->texts.push_back(line.substr(tab + 1));
            }
        }
        LOG_INFO("[RAG] Loaded metadata from %s in %.0fms (%d entries)",
                 config_.index_dir.c_str(), t.elapsed_ms(), index_size());
    } else {
        LOG_INFO("[RAG] No existing index at %s", config_.index_dir.c_str());
    }
}

int RAGEngine::index_size() const {
#ifdef NETRA_HAS_FAISS
    if (impl_->active_index) return static_cast<int>(impl_->active_index->ntotal);
#endif
    return static_cast<int>(impl_->texts.size());
}

} // namespace netra
