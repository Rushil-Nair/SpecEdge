#include "engine_draft.h"

#include <chrono>
#include <stdexcept>

namespace specedge {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

EngineTarget::EngineTarget(Engine & engine, const std::string & grammar) : engine_(engine) {
    if (!grammar.empty()) rules_ = engine.new_grammar(grammar);
}

EngineTarget::~EngineTarget() {
    if (rules_) llama_sampler_free(rules_);
}

Token EngineTarget::first() {
    const auto start = Clock::now();
    const Token token = engine_.pick(-1, rules_, grammar_rejections);
    choosing_ms += ms_since(start);
    return token;
}

void EngineTarget::eval(const std::vector<Token> & tokens) {
    before_ = engine_.cached_size();
    engine_.eval_keeping_outputs(std::vector<llama_token>(tokens.begin(), tokens.end()));
}

Token EngineTarget::choose(int i) {
    // Output i of the batch follows its i-th token. The loop asks in order and stops at the
    // first choice it does not keep, so the grammar has accepted exactly the tokens written
    // plus the one pending.
    const auto start = Clock::now();
    const Token token = engine_.pick(i, rules_, grammar_rejections);
    choosing_ms += ms_since(start);
    return token;
}

void EngineTarget::keep(int n) { engine_.truncate(before_ + static_cast<size_t>(n)); }

bool EngineTarget::is_end(Token token) const { return engine_.is_end(token); }

int EngineTarget::room() const { return engine_.n_ctx() - static_cast<int>(engine_.cached_size()) - 1; }

LlamaDraft::LlamaDraft(const EngineConfig & config, const Engine & target) : engine_(config) {
    if (engine_.n_vocab() != target.n_vocab())
        throw std::runtime_error("the draft model's vocabulary (" + std::to_string(engine_.n_vocab()) +
                                 " tokens) is not the target's (" + std::to_string(target.n_vocab()) + ")");
}

void LlamaDraft::begin(const std::vector<llama_token> & prompt) {
    prompt_ = prompt;
    ms = 0.0;
}

std::vector<Token> LlamaDraft::draft(const std::vector<Token> & written, Token last, int n_max) {
    const auto start = Clock::now();
    // The draft model reads the same text as the target: prompt, reply so far, the new token.
    // Its own cache keeps the common start, so only what is new is evaluated.
    std::vector<llama_token> context = prompt_;
    context.insert(context.end(), written.begin(), written.end());
    context.push_back(last);
    engine_.set_prompt(context, /*reuse=*/true);
    const GenerateStats proposed = engine_.generate(n_max, "");       // greedy, no grammar; stops at an end token
    ms += ms_since(start);
    return std::vector<Token>(proposed.tokens.begin(), proposed.tokens.end());
}

GenerateStats generate_with_policy(Engine & engine, LlamaDraft * drafter, DraftPolicy * policy, int requests_in_flight,
                                   int max_tokens, const std::string & grammar, RequestStats & request) {
    const auto start = Clock::now();
    EngineTarget target(engine, grammar);
    request = run_request(target, drafter, policy, requests_in_flight, max_tokens);

    GenerateStats stats;
    stats.tokens.assign(request.tokens.begin(), request.tokens.end());
    for (const llama_token token : stats.tokens) stats.text += engine.piece(token);
    stats.n_tokens = static_cast<int>(stats.tokens.size());
    stats.hit_token_limit = request.hit_token_limit;
    if (request.ended) stats.stop_token = engine.piece(request.end_token);
    stats.grammar_rejections = target.grammar_rejections;
    stats.sampling_ms = target.choosing_ms;
    stats.ms = ms_since(start);
    return stats;
}

}  // namespace specedge
