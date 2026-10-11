#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace specedge {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// llama.cpp is chatty at load; keep only warnings and errors.
void quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
}

}  // namespace

Engine::Engine(const EngineConfig & config) : config_(config) {
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    model_ = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (!model_) throw std::runtime_error("cannot load model " + config.model_path);
    vocab_ = llama_model_get_vocab(model_);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(config.n_ctx);
    ctx_params.n_batch = static_cast<uint32_t>(config.n_batch);
    ctx_params.n_threads = config.n_threads;
    ctx_params.n_threads_batch = config.n_threads;
    ctx_params.no_perf = true;
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) throw std::runtime_error("cannot create a context");
    batch_ = llama_batch_init(config.n_batch, 0, 1);
    have_batch_ = true;

    const int n_vocab = llama_vocab_n_tokens(vocab_);
    control_.resize(static_cast<size_t>(n_vocab));
    for (int token = 0; token < n_vocab; token++)
        control_[static_cast<size_t>(token)] =
            (llama_vocab_get_attr(vocab_, token) & LLAMA_TOKEN_ATTR_CONTROL) && !llama_vocab_is_eog(vocab_, token);

    char buffer[256];
    llama_model_desc(model_, buffer, sizeof buffer);
    description_ = buffer;
}

Engine::~Engine() {
    if (have_batch_) llama_batch_free(batch_);
    if (ctx_) llama_free(ctx_);
    if (model_) llama_model_free(model_);
    llama_backend_free();
}

std::vector<llama_token> Engine::tokenize(const std::string & text) const {
    std::vector<llama_token> tokens(text.size() + 8);
    int count = llama_tokenize(vocab_, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
                               static_cast<int32_t>(tokens.size()), /*add_special=*/false, /*parse_special=*/true);
    if (count < 0) {
        tokens.resize(static_cast<size_t>(-count));
        count = llama_tokenize(vocab_, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
                               static_cast<int32_t>(tokens.size()), false, true);
    }
    if (count < 0) throw std::runtime_error("tokenize failed");
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

std::string Engine::piece(llama_token token) const {
    char buffer[256];
    const int length = llama_token_to_piece(vocab_, token, buffer, sizeof buffer, 0, /*special=*/true);
    return length > 0 ? std::string(buffer, static_cast<size_t>(length)) : std::string();
}

void Engine::decode(const llama_token * tokens, int count, int first_position) {
    llama_batch & batch = batch_;
    for (int done = 0; done < count;) {
        const int chunk = std::min(config_.n_batch, count - done);
        batch.n_tokens = chunk;
        for (int i = 0; i < chunk; i++) {
            batch.token[i] = tokens[done + i];
            batch.pos[i] = first_position + done + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = done + i == count - 1;    // only the last token's output is needed
        }
        const int status = llama_decode(ctx_, batch);
        if (status != 0) throw std::runtime_error("llama_decode failed with status " + std::to_string(status));
        done += chunk;
    }
}

PromptStats Engine::set_prompt(const std::vector<llama_token> & tokens, bool reuse) {
    if (tokens.empty()) throw std::runtime_error("empty prompt");
    if (static_cast<int>(tokens.size()) >= config_.n_ctx)
        throw std::runtime_error("prompt of " + std::to_string(tokens.size()) + " tokens does not fit the context");

    size_t keep = 0;
    if (reuse) {
        while (keep < cached_.size() && keep < tokens.size() && cached_[keep] == tokens[keep]) keep++;
        if (keep == tokens.size()) keep--;       // the last token must be evaluated to get its output
    }
    llama_memory_t memory = llama_get_memory(ctx_);
    if (keep == 0 || !llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(keep), -1)) {
        llama_memory_clear(memory, true);
        keep = 0;
    }
    cached_.resize(keep);

    const auto start = Clock::now();
    decode(tokens.data() + keep, static_cast<int>(tokens.size() - keep), static_cast<int>(keep));
    PromptStats stats;
    stats.ms = ms_since(start);
    cached_ = tokens;
    stats.prompt_tokens = static_cast<int>(tokens.size());
    stats.reused_tokens = static_cast<int>(keep);
    stats.processed_tokens = static_cast<int>(tokens.size() - keep);
    return stats;
}

void Engine::eval_keeping_outputs(const std::vector<llama_token> & tokens) {
    const int count = static_cast<int>(tokens.size());
    if (count < 1 || count > config_.n_batch) throw std::runtime_error("eval_keeping_outputs: need 1 to n_batch tokens");
    if (static_cast<int>(cached_.size()) + count >= config_.n_ctx) throw std::runtime_error("the context is full");
    batch_.n_tokens = count;
    for (int i = 0; i < count; i++) {
        batch_.token[i] = tokens[static_cast<size_t>(i)];
        batch_.pos[i] = static_cast<llama_pos>(cached_.size()) + i;
        batch_.n_seq_id[i] = 1;
        batch_.seq_id[i][0] = 0;
        batch_.logits[i] = 1;                    // the output at every position is needed to check a draft
    }
    const int status = llama_decode(ctx_, batch_);
    if (status != 0) throw std::runtime_error("llama_decode failed with status " + std::to_string(status));
    cached_.insert(cached_.end(), tokens.begin(), tokens.end());
}

llama_token Engine::pick(int index, llama_sampler * rules, int & rejections) const {
    const int n_vocab = llama_vocab_n_tokens(vocab_);
    const float * logits = llama_get_logits_ith(ctx_, index);
    if (!logits) throw std::runtime_error("no output at index " + std::to_string(index));
    llama_token token = -1;
    for (int t = 0; t < n_vocab; t++) {
        if (rules && control_[static_cast<size_t>(t)]) continue;
        if (token < 0 || logits[t] > logits[token]) token = t;
    }
    if (rules) {
        llama_token_data one = {token, logits[token], 0.0f};
        llama_token_data_array single = {&one, 1, -1, false};
        llama_sampler_apply(rules, &single);
        if (std::isinf(one.logit)) {
            rejections++;
            std::vector<llama_token_data> candidates;
            candidates.reserve(static_cast<size_t>(n_vocab));
            for (int t = 0; t < n_vocab; t++)
                if (!control_[static_cast<size_t>(t)]) candidates.push_back({t, logits[t], 0.0f});
            llama_token_data_array all = {candidates.data(), candidates.size(), -1, false};
            llama_sampler_apply(rules, &all);
            token = -1;
            float best = -INFINITY;
            for (const auto & candidate : candidates)
                if (candidate.logit > best) { best = candidate.logit; token = candidate.id; }
            if (token < 0) throw std::runtime_error("the grammar allows no token here");
        }
        llama_sampler_accept(rules, token);
    }
    return token;
}

void Engine::truncate(size_t size) {
    if (size >= cached_.size()) return;
    if (size == 0 || !llama_memory_seq_rm(llama_get_memory(ctx_), 0, static_cast<llama_pos>(size), -1))
        throw std::runtime_error("cannot drop the end of the cache");
    cached_.resize(size);
}

llama_sampler * Engine::new_grammar(const std::string & grammar) const {
    llama_sampler * rules = llama_sampler_init_grammar(vocab_, grammar.c_str(), "root");
    if (!rules) throw std::runtime_error("the grammar did not parse");
    return rules;
}

bool Engine::grammar_accepts(const std::string & grammar, const std::string & text) const {
    llama_sampler * rules = llama_sampler_init_grammar(vocab_, grammar.c_str(), "root");
    if (!rules) throw std::runtime_error("the grammar did not parse");
    std::vector<llama_token> tokens = tokenize(text);
    tokens.push_back(llama_vocab_eot(vocab_));          // and then it must be allowed to stop
    bool accepted = true;
    for (const llama_token token : tokens) {
        llama_token_data one = {token, 0.0f, 0.0f};
        llama_token_data_array single = {&one, 1, -1, false};
        llama_sampler_apply(rules, &single);
        if (std::isinf(one.logit)) { accepted = false; break; }
        llama_sampler_accept(rules, token);
    }
    llama_sampler_free(rules);
    return accepted;
}

GenerateStats Engine::generate(int max_tokens, const std::string & grammar) {
    llama_sampler * rules = nullptr;
    if (!grammar.empty()) {
        rules = llama_sampler_init_grammar(vocab_, grammar.c_str(), "root");
        if (!rules) throw std::runtime_error("the grammar did not parse");
    }
    const int n_vocab = llama_vocab_n_tokens(vocab_);
    std::vector<llama_token_data> candidates;
    GenerateStats stats;
    const auto start = Clock::now();

    while (true) {
        if (stats.n_tokens >= max_tokens || static_cast<int>(cached_.size()) + 1 >= config_.n_ctx) {
            stats.hit_token_limit = true;
            break;
        }
        const auto choosing = Clock::now();
        const float * logits = llama_get_logits_ith(ctx_, -1);
        llama_token token = -1;
        for (int t = 0; t < n_vocab; t++) {
            if (rules && control_[static_cast<size_t>(t)]) continue;   // under a grammar: no <|python_tag|> and the like
            if (token < 0 || logits[t] > logits[token]) token = t;
        }
        if (rules) {
            llama_token_data one = {token, logits[token], 0.0f};
            llama_token_data_array single = {&one, 1, -1, false};
            llama_sampler_apply(rules, &single);
            if (std::isinf(one.logit)) {
                // The model's first choice is not allowed here: filter the whole vocabulary.
                stats.grammar_rejections++;
                candidates.clear();
                for (int t = 0; t < n_vocab; t++)
                    if (!control_[static_cast<size_t>(t)]) candidates.push_back({t, logits[t], 0.0f});
                llama_token_data_array all = {candidates.data(), candidates.size(), -1, false};
                llama_sampler_apply(rules, &all);
                token = -1;
                float best = -INFINITY;
                for (const auto & candidate : candidates)
                    if (candidate.logit > best) { best = candidate.logit; token = candidate.id; }
                if (token < 0) {
                    llama_sampler_free(rules);
                    throw std::runtime_error("the grammar allows no token here");
                }
            }
            llama_sampler_accept(rules, token);
        }
        stats.sampling_ms += ms_since(choosing);

        if (llama_vocab_is_eog(vocab_, token)) {
            stats.stop_token = piece(token);
            break;
        }
        stats.tokens.push_back(token);
        stats.text += piece(token);
        stats.n_tokens++;
        decode(&token, 1, static_cast<int>(cached_.size()));
        cached_.push_back(token);
    }
    stats.ms = ms_since(start);
    if (rules) llama_sampler_free(rules);
    return stats;
}

}  // namespace specedge
