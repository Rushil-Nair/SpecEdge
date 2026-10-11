// The model, through llama.h only (llama.cpp commit f498f864): load, tokenize, keep the
// key-value cache in step with a prompt, and write a reply greedily, optionally under a grammar.
#pragma once

#include "llama.h"

#include <string>
#include <vector>

namespace specedge {

struct EngineConfig {
    std::string model_path;
    int n_ctx = 8192;
    int n_batch = 2048;
    int n_threads = 4;
    int n_gpu_layers = 0;
};

struct PromptStats {
    int prompt_tokens = 0;      // length of the prompt
    int reused_tokens = 0;      // already in the cache from earlier turns
    int processed_tokens = 0;   // evaluated now
    double ms = 0.0;            // time spent evaluating them
};

struct GenerateStats {
    std::string text;
    std::vector<llama_token> tokens;
    int n_tokens = 0;              // tokens written (the end-of-turn token is not counted)
    double ms = 0.0;               // from the first choice to the last, model and sampling
    double sampling_ms = 0.0;      // the part of ms spent choosing tokens (grammar included)
    int grammar_rejections = 0;    // times the model's first choice was not allowed
    bool hit_token_limit = false;
    std::string stop_token;        // "<|eot_id|>", "<|eom_id|>", or "" at the limit
};

class Engine {
public:
    explicit Engine(const EngineConfig & config);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    // Text to tokens. Special tokens written in the text (<|eot_id|> ...) become their ids;
    // nothing is added, so the text must carry its own <|begin_of_text|>.
    std::vector<llama_token> tokenize(const std::string & text) const;
    std::string piece(llama_token token) const;

    // Make the cache hold exactly this prompt. With `reuse`, the longest common start with
    // what the cache already holds is kept and only the rest is evaluated; without it the
    // cache is emptied first.
    PromptStats set_prompt(const std::vector<llama_token> & tokens, bool reuse);

    // Continue from the prompt, always taking the most likely token. `grammar` is GBNF or
    // empty. With a grammar the most likely token is tried first and the full vocabulary is
    // filtered only when it is refused.
    GenerateStats generate(int max_tokens, const std::string & grammar);

    // Would the grammar let the model write exactly this text and then stop? For the tests.
    bool grammar_accepts(const std::string & grammar, const std::string & text) const;

    const std::string & model_description() const { return description_; }
    int n_ctx() const { return config_.n_ctx; }

    // --- for the drafted path (engine_draft.cpp). generate() above does not use these. ---

    // Evaluate these tokens after what the cache holds, in ONE llama_decode call, keeping the
    // output at every position. At most n_batch tokens.
    void eval_keeping_outputs(const std::vector<llama_token> & tokens);

    // The most likely token at output `index` of the last evaluation (-1: the last output).
    // With `rules` the choice is filtered by the grammar and accepted into it; `rejections`
    // counts the times the model's first choice was not allowed.
    llama_token pick(int index, llama_sampler * rules, int & rejections) const;

    // Forget everything in the cache from position `size` on.
    void truncate(size_t size);

    // A grammar sampler for this model's vocabulary; free it with llama_sampler_free.
    llama_sampler * new_grammar(const std::string & grammar) const;

    size_t cached_size() const { return cached_.size(); }
    const std::vector<llama_token> & cached() const { return cached_; }
    bool is_end(llama_token token) const { return llama_vocab_is_eog(vocab_, token); }
    int n_vocab() const { return llama_vocab_n_tokens(vocab_); }

private:
    void decode(const llama_token * tokens, int count, int first_position);

    EngineConfig config_;
    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    llama_batch batch_{};                    // one batch, reused for every evaluation
    bool have_batch_ = false;
    std::vector<llama_token> cached_;        // what the cache holds, in order
    std::vector<bool> control_;              // per token: a control token that is not end-of-turn
    std::string description_;
};

}  // namespace specedge
