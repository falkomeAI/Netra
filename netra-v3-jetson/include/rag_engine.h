#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <memory>

namespace netra {

struct ChunkResult {
    std::string text;
    float score = 0.0f;
    std::string source;
    int rank = 0;
};

struct RAGSearchResult {
    std::string context_text;          // formatted for LLM prompt injection
    std::vector<ChunkResult> chunks;
};

struct RAGConfig {
    std::string onnx_model_path;       // embedding ONNX model
    std::string index_dir = "knowledge_base/vector_store";
    int embedding_dim     = 384;       // bge-small-en-v1.5
    int chunk_size_words  = 256;
    int chunk_overlap     = 32;
    int top_k             = 2;
    float min_score       = 0.30f;
    bool use_gpu          = true;
};

class RAGEngine {
public:
    explicit RAGEngine(const RAGConfig& config);
    ~RAGEngine();

    RAGEngine(const RAGEngine&) = delete;
    RAGEngine& operator=(const RAGEngine&) = delete;

    int ingest(const std::string& text, const std::string& source = "");
    RAGSearchResult search(const std::string& query, int top_k = -1,
                            float min_score = -1.0f) const;
    void save() const;
    int index_size() const;

private:
    RAGConfig config_;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    mutable std::mutex mtx_;

    std::vector<float> embed(const std::string& text) const;
    std::vector<std::string> chunk_text(const std::string& text) const;
    void load_from_disk();
};

} // namespace netra
