// Unit tests of the drafting policy. No model and no clock: the rules are tested directly,
// and the behaviour on real traffic by replaying recorded traces (runtime/controller/traces).
//     test_policy <traces directory>
#include "trace.h"

#include <cstdio>

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

static int first_plain(const Replay & r) {
    for (size_t i = 0; i < r.decisions.size(); i++)
        if (!r.decisions[i].draft()) return static_cast<int>(i);
    return -1;
}

// --- the rules, on made-up observations ------------------------------------------------------

static void test_queue_depth() {
    DraftPolicy policy;
    CHECK(policy.decide(1).n_max == 16 && policy.decide(4).n_max == 16);      // light load: the full block
    CHECK(policy.decide(5).n_max == 4 && policy.decide(8).n_max == 4);        // deeper: a shorter block
    const Decision deep = policy.decide(9);
    CHECK(!deep.draft() && deep.reason == Reason::QueueDeep);                 // beyond the last band: plain
    CHECK(policy.decide(16).reason == Reason::QueueDeep && policy.decide(1000).reason == Reason::QueueDeep);
    CHECK(policy.decide(1).reason == Reason::Drafting);                       // and back as soon as the queue is short
    CHECK(policy.count(Reason::QueueDeep) == 3 && policy.decisions() == 8);

    PolicyConfig none;
    none.bands.clear();                                                       // no band at all: never draft
    DraftPolicy off(none);
    CHECK(!off.decide(1).draft());
}

static void test_low_acceptance_and_recovery() {
    PolicyConfig config;
    config.window_drafted = 1000;
    config.min_drafted = 400;
    config.probe_every = 4;
    config.probes_to_recover = 2;
    DraftPolicy policy(config);

    CHECK(policy.rolling_share(16) < 0);                                      // nothing seen yet
    CHECK(policy.decide(1).reason == Reason::Drafting);
    policy.observe(16, 300, 12);                                              // 4% accepted, but only 300 drafted
    CHECK(policy.drafting_enabled(16) && policy.rolling_share(16) < 0);       // too little to judge
    CHECK(policy.decide(1).reason == Reason::Drafting);
    policy.observe(16, 300, 12);                                              // 600 drafted at 4%: under 0.10
    CHECK(!policy.drafting_enabled(16) && policy.switches() == 1);

    // off: three plain requests, then a probe
    CHECK(policy.decide(1).reason == Reason::LowAcceptance && policy.decide(1).reason == Reason::LowAcceptance);
    CHECK(policy.decide(1).reason == Reason::LowAcceptance);
    Decision probe = policy.decide(1);
    CHECK(probe.reason == Reason::Probe && probe.n_max == 16);
    policy.observe(16, 200, 10);                                              // the probe is still bad
    CHECK(!policy.drafting_enabled(16));

    for (int i = 0; i < 3; i++) policy.decide(1);
    CHECK(policy.decide(1).reason == Reason::Probe);
    policy.observe(16, 200, 60);                                              // good probe, one of two
    CHECK(!policy.drafting_enabled(16));
    for (int i = 0; i < 3; i++) policy.decide(1);
    CHECK(policy.decide(1).reason == Reason::Probe);
    policy.observe(16, 10, 0);                                                // a probe too short to count: ignored
    CHECK(!policy.drafting_enabled(16));
    for (int i = 0; i < 3; i++) policy.decide(1);
    CHECK(policy.decide(1).reason == Reason::Probe);
    policy.observe(16, 200, 60);                                              // second good probe in a row
    CHECK(policy.drafting_enabled(16) && policy.switches() == 2);
    CHECK(policy.decide(1).reason == Reason::Drafting);

    // a bad probe between two good ones starts the count again
    DraftPolicy other(config);
    other.decide(1);
    other.observe(16, 600, 6);
    CHECK(!other.drafting_enabled(16));
    for (int round = 0; round < 3; round++) {
        for (int i = 0; i < 4; i++) other.decide(1);
        other.observe(16, 200, round == 1 ? 2 : 60);                          // good, bad, good
    }
    CHECK(!other.drafting_enabled(16));
}

static void test_no_flapping_between_thresholds() {
    PolicyConfig config;
    config.window_drafted = 1000;
    DraftPolicy policy(config);
    // Shares that wander between the two thresholds (0.10 and 0.12) change nothing.
    for (int i = 0; i < 200; i++) {
        CHECK(policy.decide(1).reason == Reason::Drafting);
        policy.observe(16, 300, i % 2 ? 31 : 35);                             // 10.3% and 11.7%
    }
    CHECK(policy.switches() == 0);
    const double share = policy.rolling_share(16);
    CHECK(share > 0.10 && share < 0.12);
}

static void test_window_forgets() {
    PolicyConfig config;
    config.window_drafted = 1000;
    DraftPolicy policy(config);
    for (int i = 0; i < 20; i++) { policy.decide(1); policy.observe(16, 300, 150); }     // a long good stretch
    CHECK(policy.rolling_share(16) > 0.49);
    int bad = 0;
    while (policy.drafting_enabled(16) && bad < 50) { policy.decide(1); policy.observe(16, 300, 6); bad++; }
    CHECK(bad == 4);            // the good stretch is forgotten after about one window of bad drafting, not twenty
}

static void test_block_lengths_are_tracked_apart() {
    DraftPolicy policy;
    policy.decide(1);
    policy.observe(16, 600, 6);                       // block 16 goes off
    CHECK(!policy.drafting_enabled(16) && policy.drafting_enabled(4));
    CHECK(policy.decide(6).n_max == 4);               // the block-4 band still drafts
    policy.observe(4, 600, 300);
    CHECK(policy.rolling_share(4) > 0.49 && !policy.drafting_enabled(16));
    // 30% accepted is fine at block 16 and under the block-4 threshold of 0.40
    policy.decide(6);
    policy.observe(4, 20000, 3000);
    CHECK(!policy.drafting_enabled(4));
}

static void test_thresholds_follow_the_band() {
    // 11% accepted: enough with 1 request in flight (threshold 0.10), not with 4 (0.12).
    DraftPolicy alone, shared;
    for (int i = 0; i < 5; i++) {
        alone.decide(1);
        alone.observe(16, 300, 33);
        shared.decide(4);
        shared.observe(16, 300, 33);
    }
    CHECK(alone.drafting_enabled(16) && !shared.drafting_enabled(16));
}

static void test_errors() {
    PolicyConfig config;
    config.error_cooldown = 5;
    config.errors_to_latch = 3;
    config.error_window = 50;
    DraftPolicy policy(config);
    CHECK(policy.decide(1).draft());
    policy.on_error();
    for (int i = 0; i < 5; i++) {
        const Decision d = policy.decide(1);
        CHECK(!d.draft() && d.reason == Reason::ErrorCooldown);
    }
    CHECK(policy.decide(1).reason == Reason::Drafting && !policy.latched());   // back after the cooldown

    policy.on_error();
    for (int i = 0; i < 6; i++) policy.decide(1);
    policy.on_error();                                                          // third error within 50 requests
    CHECK(policy.latched());
    for (int i = 0; i < 100; i++) CHECK(policy.decide(1).reason == Reason::ErrorLatched);
    policy.reset();
    CHECK(!policy.latched() && policy.decide(1).reason == Reason::Drafting);

    // errors far apart never latch
    DraftPolicy patient(config);
    for (int round = 0; round < 10; round++) {
        patient.on_error();
        for (int i = 0; i < 60; i++) patient.decide(1);
    }
    CHECK(!patient.latched() && patient.decide(1).draft());

    // an error wins over everything else, including a short queue and good acceptance
    DraftPolicy busy(config);
    busy.decide(1);
    busy.observe(16, 1000, 800);
    busy.on_error();
    CHECK(busy.decide(1).reason == Reason::ErrorCooldown);
}

static void test_odd_input() {
    DraftPolicy policy;
    policy.observe(16, 0, 0);            // nothing drafted
    policy.observe(0, 100, 50);          // a plain request reported by mistake
    policy.observe(16, -5, 0);
    CHECK(policy.drafting_enabled(16) && policy.rolling_share(16) < 0 && policy.switches() == 0);
    CHECK(policy.decide(0).draft() && policy.decide(-3).draft());     // a depth below 1 is treated as light load
    CHECK(std::string(reason_name(Reason::QueueDeep)) == "queue_deep");
}

// --- recorded traffic --------------------------------------------------------------------------

static void test_traces(const std::string & dir) {
    const auto u1 = load_trace(dir + "/load_u1.csv"), u4 = load_trace(dir + "/load_u4.csv"), u16 = load_trace(dir + "/load_u16.csv");
    CHECK(u1.size() == 200 && u4.size() == 200 && u16.size() == 200);

    {   // text that drafts well: never switched off
        for (const char * kind : {"code", "tool_calls", "reasoning", "chat"}) {
            DraftPolicy policy;
            const Replay r = replay(policy, of_kind(u1, kind));
            CHECK(r.drafted_requests == 40 && policy.switches() == 0 && r.policy_ms == r.always_ms);
        }
    }
    {   // Chinese only, where drafting was slower than plain on the recording: off after two requests
        DraftPolicy policy;
        const Replay r = replay(policy, of_kind(u1, "chinese"));
        CHECK(r.always_ms > r.never_ms);                        // the recording: drafting loses here
        CHECK(first_plain(r) == 2);
        CHECK(r.policy_ms < r.always_ms);                       // the policy avoids most of the loss
        CHECK(r.policy_ms < r.never_ms * 1.10);                 // and its probes cost under 10% over never drafting
        CHECK(r.drafted_requests <= 12);
    }
    {   // 16 requests in flight: plain decoding throughout, which the recording says is right
        DraftPolicy policy;
        const Replay r = replay(policy, u16);
        CHECK(r.drafted_requests == 0 && policy.count(Reason::QueueDeep) == 200);
        CHECK(r.policy_ms == r.never_ms && r.always_ms > 2.0 * r.never_ms);
    }
    {   // the five kinds mixed, in the order they were sent: drafting pays overall and the
        // policy must not throw that away
        DraftPolicy one, four;
        const Replay r1 = replay(one, u1), r4 = replay(four, u4);
        CHECK(r1.always_ms < r1.never_ms && r4.always_ms < r4.never_ms);
        CHECK(r1.policy_ms <= r1.always_ms * 1.01);
        CHECK(r4.policy_ms <= r4.always_ms * 1.02);
        CHECK(r1.policy_ms < r1.never_ms && r4.policy_ms < r4.never_ms);
    }
    {   // traffic that changes: code, then Chinese, then code again (part 8's test, replayed)
        for (const auto * trace : {&u1, &u4}) {
            const auto rows = joined({of_kind(*trace, "code"), of_kind(*trace, "chinese"), of_kind(*trace, "code")});
            DraftPolicy policy;
            const Replay r = replay(policy, rows);
            const int detected = first_plain(r);
            CHECK(detected >= 40 && detected <= 40 + 16);       // noticed within 16 Chinese requests
            // Off once and on once at 1 user. At 4 users one Chinese probe happens to be accepted
            // above the threshold, so drafting comes back for a moment and goes off again: 4.
            CHECK(policy.switches() == (trace == &u1 ? 2 : 4));
            bool end_drafting = true;                           // drafting again through the last 30 code requests
            for (size_t i = rows.size() - 30; i < rows.size(); i++) end_drafting = end_drafting && r.decisions[i].reason == Reason::Drafting;
            CHECK(end_drafting);
            CHECK(r.policy_ms < r.always_ms && r.policy_ms < r.never_ms);
        }
    }
    {   // a short window notices sooner (and costs more on mixed traffic: see the README)
        PolicyConfig quick;
        quick.window_drafted = 1500;
        DraftPolicy policy(quick);
        const Replay r = replay(policy, joined({of_kind(u1, "code"), of_kind(u1, "chinese"), of_kind(u1, "code")}));
        CHECK(first_plain(r) >= 40 && first_plain(r) <= 43);
    }
    {   // an error in the middle of good traffic: plain for the cooldown, then drafting again
        DraftPolicy policy;
        const auto rows = of_kind(u1, "code");
        int plain = 0;
        for (size_t i = 0; i < rows.size(); i++) {
            const Decision d = policy.decide(1);
            if (i == 10) policy.on_error();
            if (d.draft()) policy.observe(d.n_max, rows[i].draft_n, rows[i].draft_n_accepted);
            else plain++;
        }
        CHECK(plain == policy.config().error_cooldown && policy.decide(1).reason == Reason::Drafting);
    }
}

int main(int argc, char ** argv) {
    test_queue_depth();
    test_low_acceptance_and_recovery();
    test_no_flapping_between_thresholds();
    test_window_forgets();
    test_block_lengths_are_tracked_apart();
    test_thresholds_follow_the_band();
    test_errors();
    test_odd_input();
    test_traces(argc > 1 ? argv[1] : "traces");
    std::printf("%d of %d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
