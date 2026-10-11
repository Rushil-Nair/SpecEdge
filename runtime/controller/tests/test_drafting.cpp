// Unit tests of the draft-and-verify loop and of its connection to the policy, with made-up
// models: no llama.cpp, no model file.     test_drafting
//
// The made-up target chooses each token from the tokens before it, the same way however they
// were grouped, so here drafted output must equal plain output exactly. It also checks that
// the loop keeps to the interface's rules (choose() in order, keep() of what was accepted),
// because the real engine advances a grammar inside choose() and relies on them.
#include "drafting.h"

#include <cstdio>
#include <stdexcept>

using namespace specedge;

static int failures = 0, checks = 0;

#define CHECK(condition)                                                                   \
    do {                                                                                   \
        checks++;                                                                          \
        if (!(condition)) {                                                                \
            failures++;                                                                    \
            std::printf("FAILED %s:%d  %s\n", __FILE__, __LINE__, #condition);            \
        }                                                                                  \
    } while (0)

static const Token END = 0;

class FakeTarget : public Target {
public:
    FakeTarget(int prompt_tokens, int reply_length, int capacity = 100000)
        : reply_length_(reply_length), capacity_(capacity) {
        for (int i = 0; i < prompt_tokens; i++) context_.push_back(1000 + i);
        prompt_ = context_.size();
    }

    // What the target writes after a given context: a fixed function of the last three tokens,
    // and the end token once the reply is long enough.
    Token next_after(const std::vector<Token> & context) const {
        if (static_cast<int>(context.size() - prompt_) >= reply_length_) return END;
        unsigned h = 2166136261u;
        for (size_t i = context.size() >= 3 ? context.size() - 3 : 0; i < context.size(); i++) h = (h ^ static_cast<unsigned>(context[i])) * 16777619u;
        return static_cast<Token>(h % 50u) + 1;
    }

    Token first() override {
        if (asked_first_) broke_rules = true;
        asked_first_ = true;
        return next_after(context_);
    }
    void eval(const std::vector<Token> & tokens) override {
        if (tokens.empty() || static_cast<int>(tokens.size()) > room()) broke_rules = true;
        before_ = context_.size();
        context_.insert(context_.end(), tokens.begin(), tokens.end());
        last_ = static_cast<int>(tokens.size());
        next_choose_ = 0;
        evals++;
        tokens_evaluated += last_;
    }
    Token choose(int i) override {
        if (i != next_choose_ || i >= last_) broke_rules = true;      // in order, each once, inside the batch
        next_choose_ = i + 1;
        chosen++;
        return next_after(std::vector<Token>(context_.begin(), context_.begin() + static_cast<long>(before_) + i + 1));
    }
    void keep(int n) override {
        if (n < 1 || n > last_ || n != next_choose_) broke_rules = true;   // exactly what was accepted, plus the first token
        context_.resize(before_ + static_cast<size_t>(n));
        last_ = n;
    }
    bool is_end(Token token) const override { return token == END; }
    int room() const override { return capacity_ - static_cast<int>(context_.size()); }

    std::vector<Token> reply() const { return std::vector<Token>(context_.begin() + static_cast<long>(prompt_), context_.end()); }
    const std::vector<Token> & context() const { return context_; }

    int evals = 0, tokens_evaluated = 0, chosen = 0;
    bool broke_rules = false;

private:
    std::vector<Token> context_;
    size_t prompt_ = 0, before_ = 0;
    int reply_length_, capacity_, last_ = 0, next_choose_ = 0;
    bool asked_first_ = false;
};

// A drafter that knows the target and gets every `wrong_every`-th token wrong (0: never
// wrong; 1: always wrong), and can be made to fail.
class FakeDraft : public DraftSource {
public:
    FakeDraft(const FakeTarget & target, int wrong_every) : target_(target), wrong_every_(wrong_every) {}

    std::vector<Token> draft(const std::vector<Token> &, Token last, int n_max) override {
        calls++;
        if (fail_on_call > 0 && calls >= fail_on_call) throw std::runtime_error("drafter is down");
        if (n_max > largest_request) largest_request = n_max;
        std::vector<Token> context = target_.context(), out;
        context.push_back(last);
        for (int i = 0; i < n_max + extra; i++) {
            Token token = target_.next_after(context);
            if (token == END && !draft_past_end) break;
            produced++;
            if (wrong_every_ > 0 && produced % wrong_every_ == 0) token = token == 7 ? 8 : 7;
            out.push_back(token);
            context.push_back(token);
        }
        return out;
    }

    int calls = 0, produced = 0, fail_on_call = 0, largest_request = 0, extra = 0;
    bool draft_past_end = false;

private:
    const FakeTarget & target_;
    int wrong_every_;
};

static PolicyConfig always(int n_max) {            // a policy that drafts whatever happens, with this block
    PolicyConfig config;
    config.bands = {{1000000, n_max, -1.0, -1.0}};
    return config;
}

static std::vector<Token> plain_reply(int prompt, int length, int max_tokens) {
    FakeTarget target(prompt, length);
    const RequestStats stats = run_request(target, nullptr, nullptr, 1, max_tokens);
    return stats.tokens;
}

// --- the loop --------------------------------------------------------------------------------

static void test_plain() {
    FakeTarget target(20, 30);
    const RequestStats s = run_request(target, nullptr, nullptr, 1, 100);
    CHECK(s.tokens.size() == 30 && s.ended && s.end_token == END && !s.hit_token_limit && !s.asked_policy);
    CHECK(s.rounds == 30 && target.evals == 30 && target.tokens_evaluated == 30);       // one token per evaluation
    CHECK(s.drafts == 0 && s.drafted == 0 && target.reply() == s.tokens && !target.broke_rules);

    FakeTarget cut(20, 30);
    const RequestStats c = run_request(cut, nullptr, nullptr, 1, 12);                   // the token limit
    CHECK(c.tokens.size() == 12 && c.hit_token_limit && !c.ended && cut.reply() == c.tokens);
    CHECK(std::vector<Token>(s.tokens.begin(), s.tokens.begin() + 12) == c.tokens);

    FakeTarget empty(20, 0);                                                            // the model stops at once
    const RequestStats e = run_request(empty, nullptr, nullptr, 1, 100);
    CHECK(e.tokens.empty() && e.ended && e.rounds == 0);

    FakeTarget full(20, 30, 25);                                                        // the context fills up
    const RequestStats f = run_request(full, nullptr, nullptr, 1, 100);
    CHECK(f.tokens.size() == 5 && f.hit_token_limit && !full.broke_rules);
}

static void test_same_text_as_plain() {
    // Whatever the drafter does, the reply is the plain reply, and the context ends as prompt + reply.
    for (int length : {0, 1, 2, 7, 30, 64}) {
        for (int n_max : {1, 3, 4, 16}) {
            for (int wrong_every : {0, 1, 2, 3, 5}) {
                for (int max_tokens : {1, 10, 100}) {
                    FakeTarget target(15, length);
                    FakeDraft drafter(target, wrong_every);
                    DraftPolicy policy(always(n_max));
                    const RequestStats s = run_request(target, &drafter, &policy, 1, max_tokens);
                    const std::vector<Token> expected = plain_reply(15, length, max_tokens);
                    CHECK(s.tokens == expected);
                    CHECK(target.reply() == expected);
                    CHECK(!target.broke_rules);
                    CHECK(s.accepted <= s.drafted && s.decision.n_max == n_max && s.asked_policy);
                    CHECK(s.hit_token_limit == (length > max_tokens) && s.ended == (length <= max_tokens));
                    CHECK(drafter.largest_request <= n_max);
                }
            }
        }
    }
    // A drafter that returns more than it was asked for, and one that drafts past the end.
    FakeTarget target(15, 20);
    FakeDraft greedy(target, 0);
    greedy.extra = 5;
    greedy.draft_past_end = true;
    DraftPolicy policy(always(4));
    const RequestStats s = run_request(target, &greedy, &policy, 1, 100);
    CHECK(s.tokens == plain_reply(15, 20, 100) && target.reply() == s.tokens && !target.broke_rules && s.ended);
}

static void test_counts() {
    {   // a perfect drafter: everything proposed is kept, and the target is evaluated far less often
        FakeTarget target(15, 40);
        FakeDraft drafter(target, 0);
        DraftPolicy policy(always(4));
        const RequestStats s = run_request(target, &drafter, &policy, 1, 100);
        CHECK(s.drafted == s.accepted && s.accepted == 32);        // 8 rounds of 1 own token + 4 drafted
        CHECK(s.rounds == 8 && target.evals == 8 && target.tokens_evaluated == 40);
        CHECK(policy.rolling_share(4) < 0);                        // 32 drafted tokens: too few to judge yet
    }
    {   // a drafter that is always wrong: nothing kept, one token per round, every draft token evaluated for nothing
        FakeTarget target(15, 40);
        FakeDraft drafter(target, 1);
        DraftPolicy policy(always(4));
        const RequestStats s = run_request(target, &drafter, &policy, 1, 100);
        CHECK(s.accepted == 0 && s.drafted > 100 && s.rounds == 40);
        CHECK(target.tokens_evaluated == 40 + s.drafted);
    }
    {   // every third proposed token wrong: two kept, then the target's own token
        FakeTarget target(15, 60);
        FakeDraft drafter(target, 3);
        DraftPolicy policy(always(8));
        const RequestStats s = run_request(target, &drafter, &policy, 1, 100);
        CHECK(s.accepted > 0 && s.accepted < s.drafted && s.rounds < 60 && s.rounds > 60 / 9);
    }
}

// --- the policy decides ------------------------------------------------------------------------

static RequestStats one(DraftPolicy & policy, int wrong_every, int depth, int length = 192, int fail_on_call = 0) {
    FakeTarget target(15, length);
    FakeDraft drafter(target, wrong_every);
    drafter.fail_on_call = fail_on_call;
    const RequestStats s = run_request(target, &drafter, &policy, depth, 256);
    CHECK(s.tokens == plain_reply(15, length, 256));               // the text never depends on the decision
    CHECK(!target.broke_rules);
    return s;
}

static void test_queue_depth_decides() {
    DraftPolicy policy;                                            // the default bands
    const RequestStats light = one(policy, 0, 1), four = one(policy, 0, 4), six = one(policy, 0, 6), deep = one(policy, 0, 16);
    CHECK(light.decision.n_max == 16 && four.decision.n_max == 16 && light.drafts > 0);
    CHECK(six.decision.n_max == 4 && six.drafts > 0);
    CHECK(deep.decision.reason == Reason::QueueDeep && deep.drafts == 0 && deep.drafted == 0 && deep.rounds == 192);
    CHECK(one(policy, 0, 1).decision.reason == Reason::Drafting);  // and back when the queue is short again
}

static void test_acceptance_decides() {
    PolicyConfig config;
    config.window_drafted = 3000;
    config.probe_every = 4;
    config.probes_to_recover = 1;
    DraftPolicy policy(config);

    // Good drafting for a while: stays on, and the policy's window holds the loop's own counts.
    for (int i = 0; i < 5; i++) CHECK(one(policy, 0, 1).decision.reason == Reason::Drafting);
    CHECK(policy.rolling_share(16) > 0.99 && policy.switches() == 0);

    // The drafter turns bad (always wrong): drafting is switched off within a few requests.
    int until_off = 0;
    while (policy.drafting_enabled(16) && until_off < 50) { one(policy, 1, 1); until_off++; }
    CHECK(until_off >= 1 && until_off <= 3 && policy.switches() == 1);

    // While off: plain requests, never asking the drafter, with a probe every fourth request.
    int probes = 0, plain = 0;
    for (int i = 0; i < 12; i++) {
        const RequestStats s = one(policy, 1, 1);
        if (s.decision.reason == Reason::Probe) { probes++; CHECK(s.drafts > 0 && s.accepted == 0); }
        else { plain++; CHECK(s.decision.reason == Reason::LowAcceptance && s.drafts == 0 && s.rounds == 192); }
    }
    CHECK(probes == 3 && plain == 9 && !policy.drafting_enabled(16));

    // The drafter recovers: the next probe is accepted and drafting comes back.
    int until_on = 0;
    while (!policy.drafting_enabled(16) && until_on < 50) { one(policy, 0, 1); until_on++; }
    CHECK(until_on == 4 && policy.switches() == 2);
    CHECK(one(policy, 0, 1).decision.reason == Reason::Drafting);
}

static void test_error_fallback() {
    PolicyConfig config;
    config.error_cooldown = 5;
    config.errors_to_latch = 3;
    DraftPolicy policy(config);
    CHECK(one(policy, 0, 1).decision.reason == Reason::Drafting);

    // The drafter fails on its third call inside a request: the request still finishes with the
    // right text (checked in one()), what was drafted before the failure is counted, and
    // nothing is drafted after it.
    const RequestStats broke = one(policy, 0, 1, 192, 3);
    CHECK(broke.draft_failed && broke.draft_error == "drafter is down" && broke.drafts == 2 && broke.ended);
    CHECK(broke.rounds == 2 + (192 - 2 * 17));                     // two drafted rounds of 17 tokens, the rest one by one

    // The next five requests do not draft, then drafting is tried again.
    for (int i = 0; i < 5; i++) {
        const RequestStats s = one(policy, 0, 1);
        CHECK(s.decision.reason == Reason::ErrorCooldown && s.drafts == 0);
    }
    CHECK(one(policy, 0, 1).decision.reason == Reason::Drafting && !policy.latched());

    // A drafter that fails at once, twice more: three errors close together turn drafting off for good.
    CHECK(one(policy, 0, 1, 192, 1).draft_failed);
    for (int i = 0; i < 5; i++) one(policy, 0, 1);
    CHECK(one(policy, 0, 1, 192, 1).draft_failed && policy.latched());
    for (int i = 0; i < 30; i++) CHECK(one(policy, 0, 1).decision.reason == Reason::ErrorLatched);
    policy.reset();
    CHECK(one(policy, 0, 1).decision.reason == Reason::Drafting);
}

static void test_no_policy_or_no_drafter() {
    FakeTarget a(15, 40), b(15, 40);
    FakeDraft drafter(a, 0);
    DraftPolicy policy;
    const RequestStats without_policy = run_request(a, &drafter, nullptr, 1, 100);
    const RequestStats without_drafter = run_request(b, nullptr, &policy, 1, 100);
    CHECK(!without_policy.asked_policy && without_policy.drafts == 0 && drafter.calls == 0);
    CHECK(!without_drafter.asked_policy && policy.decisions() == 0);
    CHECK(without_policy.tokens == without_drafter.tokens && without_policy.rounds == 40);
}

int main() {
    test_plain();
    test_same_text_as_plain();
    test_counts();
    test_queue_depth_decides();
    test_acceptance_decides();
    test_error_fallback();
    test_no_policy_or_no_drafter();
    std::printf("%d of %d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
