// One request written with or without drafting, as the policy decides. This is the piece
// that connects the policy (policy.h) to an engine: it asks the policy once per request,
// runs the draft-and-verify loop or the plain loop, reports what happened back to the
// policy, and falls back to plain decoding when the drafter fails.
//
// It knows nothing about llama.cpp. The engine is reached through two small interfaces, so
// the loop is tested here with made-up models and used unchanged with the real one
// (runtime/agent/src/engine_draft.*).
//
// What it guarantees: every token written is the target's own choice in the pass that
// checked it. A drafted token is kept only when the target chose the same token at that
// position; the first disagreement is replaced by the target's choice and the rest of the
// draft is thrown away. With a target whose choices do not depend on how tokens are grouped
// the text is therefore the same as plain decoding; llama.cpp's do depend on it slightly
//, so there the match with plain decoding is measured, not assumed.
#pragma once

#include "policy.h"

#include <cstdint>
#include <string>
#include <vector>

namespace specedge {

using Token = std::int32_t;

// The model being served. It holds a context (the prompt, then everything written so far).
class Target {
public:
    virtual ~Target() = default;

    // The target's choice for the next token from the outputs the prompt left behind.
    // Called once, at the start of a request.
    virtual Token first() = 0;

    // Evaluate these tokens after the context, in one batch, keeping the output at every one.
    virtual void eval(const std::vector<Token> & tokens) = 0;

    // The target's choice after the i-th token of the last eval(). The loop calls it with
    // i = 0, 1, 2, ... in order, each at most once, and stops at the first one it does not
    // keep, so an implementation may advance a grammar as it answers.
    virtual Token choose(int i) = 0;

    // Keep only the first n tokens of the last eval() in the context; forget the rest.
    virtual void keep(int n) = 0;

    virtual bool is_end(Token token) const = 0;

    // How many more tokens the context can hold.
    virtual int room() const = 0;
};

// Whatever proposes tokens: a small model, an EAGLE-3 head, a DFlash drafter.
class DraftSource {
public:
    virtual ~DraftSource() = default;

    // Up to n_max tokens that might follow `last`. `written` is this reply so far, `last` the
    // token the target has just chosen (not yet in `written`); the prompt before them is the
    // drafter's own business. May return fewer than n_max, or none.
    // Throws std::exception when it cannot draft; the request then continues without it.
    virtual std::vector<Token> draft(const std::vector<Token> & written, Token last, int n_max) = 0;
};

struct RequestStats {
    std::vector<Token> tokens;      // what was written (the end token is not included)
    Decision decision{0, Reason::QueueDeep};
    bool asked_policy = false;      // false when there was no drafter or no policy
    int rounds = 0;                 // evaluations of the target
    int drafts = 0;                 // times the drafter was asked
    int drafted = 0;                // tokens it proposed
    int accepted = 0;               // of those, kept
    bool hit_token_limit = false;
    bool ended = false;             // the target chose an end token
    Token end_token = -1;
    bool draft_failed = false;      // the drafter threw; the rest was decoded plainly
    std::string draft_error;
};

// Write one reply of at most max_tokens. `source` or `policy` may be null: then it is plain
// decoding. `requests_in_flight` is the queue depth the policy is asked with (this request
// included). The policy is asked once, told the request's drafted and accepted totals at the
// end, and told at once if the drafter fails.
RequestStats run_request(Target & target, DraftSource * source, DraftPolicy * policy, int requests_in_flight, int max_tokens);

}  // namespace specedge
