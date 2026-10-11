// The drafted path of the runtime: the llama.cpp engine behind the two interfaces of
// runtime/controller/drafting.h, so that the policy (policy.h) decides per request whether a
// reply is written with a draft model or plainly.
//
//   EngineTarget  the served model: an Engine plus, when forced JSON is on, its grammar.
//   LlamaDraft    a second, smaller model with the same vocabulary that proposes tokens
//                 greedily (what llama.cpp calls draft-simple; for Llama 3.2 3B the 1B).
//
// Drafters that need llama.cpp's `common` library (EAGLE-3 heads, DFlash) are not here yet.
// They would be one more DraftSource; nothing else changes.
#pragma once

#include "drafting.h"
#include "engine.h"

#include <memory>
#include <string>
#include <vector>

namespace specedge {

class EngineTarget : public Target {
public:
    // `grammar` is GBNF or empty. The engine must already hold the prompt (set_prompt).
    EngineTarget(Engine & engine, const std::string & grammar);
    ~EngineTarget() override;

    Token first() override;
    void eval(const std::vector<Token> & tokens) override;
    Token choose(int i) override;
    void keep(int n) override;
    bool is_end(Token token) const override;
    int room() const override;

    int grammar_rejections = 0;
    double choosing_ms = 0.0;

private:
    Engine & engine_;
    llama_sampler * rules_ = nullptr;
    size_t before_ = 0;      // cache length before the last eval()
};

class LlamaDraft : public DraftSource {
public:
    // Loads the draft model. Throws when its vocabulary is not the size of the target's.
    LlamaDraft(const EngineConfig & config, const Engine & target);

    // Call before each request with the prompt the target was given.
    void begin(const std::vector<llama_token> & prompt);

    std::vector<Token> draft(const std::vector<Token> & written, Token last, int n_max) override;

    const std::string & model_description() const { return engine_.model_description(); }
    double ms = 0.0;         // time spent in the draft model since begin()

private:
    Engine engine_;
    std::vector<llama_token> prompt_;
};

// Write one reply through the policy. `drafter` or `policy` may be null (plain decoding
// through the same loop). Fills `request` with what the policy decided and what was drafted.
GenerateStats generate_with_policy(Engine & engine, LlamaDraft * drafter, DraftPolicy * policy, int requests_in_flight,
                                   int max_tokens, const std::string & grammar, RequestStats & request);

}  // namespace specedge
