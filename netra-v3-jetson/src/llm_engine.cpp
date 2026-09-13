#include "llm_engine.h"
#include "utils.h"
#include "llama.h"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace netra {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

LLMEngine::LLMEngine(const LLMConfig& config) : config_(config) {
    if (!file_exists(config_.model_path)) {
        LOG_ERR("Model file not found: %s", config_.model_path.c_str());
        throw std::runtime_error("Model file not found: " + config_.model_path);
    }

    Timer load_timer;

    llama_backend_init();
    LOG_INFO("llama.cpp backend initialised");

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config_.n_gpu_layers;

    model_ = llama_model_load_from_file(config_.model_path.c_str(), model_params);
    if (!model_) {
        llama_backend_free();
        LOG_ERR("Failed to load model: %s", config_.model_path.c_str());
        throw std::runtime_error("Failed to load model: " + config_.model_path);
    }
    LOG_INFO("Model loaded in %.1f ms: %s", load_timer.elapsed_ms(),
             config_.model_path.c_str());

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = config_.n_ctx;
    ctx_params.n_batch   = config_.n_batch;
    ctx_params.n_threads = config_.n_threads;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        llama_backend_free();
        LOG_ERR("Failed to create llama context");
        throw std::runtime_error("Failed to create llama context");
    }

    LOG_INFO("LLMEngine ready – ctx=%d  batch=%d  gpu_layers=%d  (%.1f ms total)",
             config_.n_ctx, config_.n_batch, config_.n_gpu_layers,
             load_timer.elapsed_ms());
}

LLMEngine::~LLMEngine() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    llama_backend_free();
    LOG_INFO("LLMEngine destroyed");
}

// ---------------------------------------------------------------------------
// Tokenisation helpers
// ---------------------------------------------------------------------------

static std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                         const std::string& text,
                                         bool add_special,
                                         bool parse_special) {
    const int n_estimate = text.size() + 128;
    std::vector<llama_token> tokens(n_estimate);

    int n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()),
                           add_special, parse_special);

    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()),
                           add_special, parse_special);
        if (n < 0) {
            LOG_ERR("Tokenisation failed for %zu-byte input", text.size());
            return {};
        }
    }
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

static std::string detokenize(const llama_vocab* vocab, llama_token token) {
    char buf[128];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) {
        std::string result(static_cast<size_t>(-n), '\0');
        llama_token_to_piece(vocab, token, result.data(),
                             static_cast<int32_t>(result.size()), 0, true);
        return result;
    }
    return {buf, static_cast<size_t>(n)};
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

LLMResult LLMEngine::generate(const std::string& prompt,
                               const std::string& system_prompt,
                               int max_tokens,
                               float temperature) const {
    std::lock_guard<std::mutex> lock(mtx_);
    Timer total_timer;

    if (max_tokens <= 0)   max_tokens   = config_.max_tokens;
    if (temperature < 0.f) temperature  = config_.temperature;

    // -- Build ChatML prompt ------------------------------------------------
    std::string full_prompt;
    full_prompt.reserve(system_prompt.size() + prompt.size() + 128);

    if (!system_prompt.empty()) {
        full_prompt += "<|im_start|>system\n";
        full_prompt += system_prompt;
        full_prompt += "\n<|im_end|>\n";
    }
    full_prompt += "<|im_start|>user\n";
    full_prompt += prompt;
    full_prompt += "\n<|im_end|>\n";
    full_prompt += "<|im_start|>assistant\n<think>\n</think>\n";

    // -- Tokenize -----------------------------------------------------------
    Timer tok_timer;
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    auto tokens = tokenize(vocab, full_prompt, true, true);
    if (tokens.empty()) {
        LOG_ERR("Empty tokenisation result – aborting generation");
        return {};
    }
    LOG_INFO("Tokenised %zu tokens in %.1f ms", tokens.size(), tok_timer.elapsed_ms());

    const int n_ctx = llama_n_ctx(ctx_);
    if (static_cast<int>(tokens.size()) > n_ctx - 4) {
        LOG_ERR("Prompt too long: %zu tokens vs context %d", tokens.size(), n_ctx);
        return {};
    }

    // -- Clear KV cache -----------------------------------------------------
    llama_memory_clear(llama_get_memory(ctx_), true);

    // -- Prompt evaluation in batches ---------------------------------------
    Timer eval_timer;
    const int n_prompt = static_cast<int>(tokens.size());
    const int batch_sz = config_.n_batch;

    for (int i = 0; i < n_prompt; i += batch_sz) {
        int n_eval = std::min(batch_sz, n_prompt - i);
        llama_batch batch = llama_batch_get_one(tokens.data() + i, n_eval);
        if (llama_decode(ctx_, batch) != 0) {
            LOG_ERR("llama_decode failed during prompt eval at offset %d", i);
            return {};
        }
    }
    const double prompt_eval_ms = eval_timer.elapsed_ms();
    LOG_INFO("Prompt eval: %d tokens in %.1f ms (%.1f t/s)",
             n_prompt, prompt_eval_ms,
             n_prompt / (prompt_eval_ms / 1000.0));

    // -- Set up sampler chain -----------------------------------------------
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = false;

    llama_sampler* sampler = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(config_.top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));

    // -- Auto-regressive generation -----------------------------------------
    Timer gen_timer;
    std::string response;
    response.reserve(max_tokens * 8);

    int n_generated = 0;

    for (int i = 0; i < max_tokens; ++i) {
        llama_token new_token = llama_sampler_sample(sampler, ctx_, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            LOG_INFO("EOS reached after %d tokens", n_generated);
            break;
        }

        response += detokenize(vocab, new_token);
        ++n_generated;

        llama_batch single = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx_, single) != 0) {
            LOG_ERR("llama_decode failed at generation step %d", i);
            break;
        }
    }

    const double generation_ms = gen_timer.elapsed_ms();

    llama_sampler_free(sampler);

    // -- Collect timing from llama.cpp internals ----------------------------
    llama_perf_context_data perf = llama_perf_context(ctx_);

    LLMResult result;
    result.response          = std::move(response);
    result.prompt_tokens     = n_prompt;
    result.completion_tokens = n_generated;
    result.prompt_eval_ms    = perf.t_p_eval_ms;
    result.generation_ms     = perf.t_eval_ms;
    result.tokens_per_sec    = (n_generated > 0 && generation_ms > 0)
                               ? (n_generated / (generation_ms / 1000.0))
                               : 0.0;

    LOG_INFO("Generation complete: %d tokens in %.1f ms (%.1f t/s) – total %.1f ms",
             n_generated, generation_ms, result.tokens_per_sec,
             total_timer.elapsed_ms());

    return result;
}

} // namespace netra
