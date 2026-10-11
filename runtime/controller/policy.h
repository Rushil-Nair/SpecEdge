// The drafting policy: for each request, draft with a block of some length or decode plainly.
// Three rules, in this order:
//
//   1. Errors. After a drafting error the next requests are decoded plainly; after several
//      errors close together drafting stays off until reset().
//   2. Queue depth. The more requests are in flight, the less drafting pays: the deepest
//      band allowed to draft uses a shorter block, and beyond the last band nothing drafts.
//   3. Rolling acceptance (part 8's fallback rule). While drafting, the share of drafted
//      tokens accepted is tracked over a rolling window. When it stays below the band's
//      lower threshold drafting is turned off; every few requests one request is drafted as
//      a probe, and drafting comes back when probes are accepted above the upper threshold.
//
// The module is plain logic: no model, no clock, no llama.cpp. The caller reports what
// happened (observe, on_error) and asks what to do next (decide). "Request" below is whatever
// unit the caller decides on; the recorded traces it is tested against are per request.
//
// Default numbers come from recorded runs of Qwen3-8B 4-bit with the z-lab DFlash drafter on
// an RTX 5090 (runtime/controller/traces/ANALYSIS.md). They belong to that model, drafter and
// build; another setup needs its own.
#pragma once

#include <deque>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace specedge {

// What to do while at most `max_depth` requests are in flight.
struct Band {
    int max_depth;        // requests in flight, this request included
    int n_max;            // draft block length
    double off_below;     // stop drafting when the rolling accepted share is under this
    double on_above;      // resume when probes are accepted at or above this
};

struct PolicyConfig {
    // Block 16, 1 user: of the recorded requests under 0.10 accepted, 43 of 44 were slower
    // than plain; at or above, 1 of 140. Block 16, 4 users: break-even between 0.12 and 0.15.
    // 16 users: plain wins on every kind of text, so nothing drafts beyond the last band.
    // The third band (5 to 8 in flight, block 4) has NO recorded run under load: its
    // thresholds are the one-user block-4 break-even (0.40 to 0.45). 8 is where llama.cpp's
    // CUDA backend changes kernel (MMVQ_MAX_BATCH_SIZE). To be measured before it is relied on.
    std::vector<Band> bands = {{1, 16, 0.10, 0.12}, {4, 16, 0.12, 0.15}, {8, 4, 0.40, 0.45}};

    // The window is a trade. A short one notices a change of traffic within a few requests
    // but is filled by one or two badly drafted requests (a Chinese request drafts about
    // 1,100 tokens for 192 written), so on mixed traffic it turns drafting off when drafting
    // is still paying overall. 20,000 drafted tokens is about 18 such requests. The sweep
    // behind these three numbers is in runtime/controller/README.md; they were chosen on the
    // same recorded traces the tests replay.
    int window_drafted = 20000;   // length of the rolling window, in drafted tokens
    int min_drafted = 400;        // do not judge acceptance on fewer drafted tokens than this
    int min_probe_drafted = 32;   // a probe that drafted fewer tokens says nothing either way
    int probe_every = 4;          // while off for low acceptance: draft one request in this many
    int probes_to_recover = 1;    // good probes in a row needed to resume

    int error_cooldown = 20;      // requests decoded plainly after an error
    int errors_to_latch = 3;      // this many errors within error_window requests: off until reset()
    int error_window = 200;
};

enum class Reason { Drafting, Probe, LowAcceptance, QueueDeep, ErrorCooldown, ErrorLatched };

const char * reason_name(Reason reason);

struct Decision {
    int n_max;            // 0: decode plainly
    Reason reason;
    bool draft() const { return n_max > 0; }
};

class DraftPolicy {
public:
    explicit DraftPolicy(PolicyConfig config = {});

    // What to do with the next request, given how many requests are in flight (it included).
    Decision decide(int requests_in_flight);

    // What a drafted request did: the block length it used, tokens drafted, tokens accepted.
    void observe(int n_max, int drafted, int accepted);

    // Drafting failed (the drafter could not be run, a decode error ...). The caller decodes
    // that request plainly; the policy keeps drafting off for a while.
    void on_error();

    // Forget everything, including a latched error state.
    void reset();

    // The rolling accepted share at this block length, or -1 with too little data.
    double rolling_share(int n_max) const;
    bool drafting_enabled(int n_max) const;      // false while off for low acceptance
    bool latched() const { return latched_; }
    long decisions() const { return decisions_; }
    long count(Reason reason) const;
    int switches() const { return switches_; }   // times acceptance turned drafting off or back on
    const PolicyConfig & config() const { return config_; }

private:
    // Acceptance is tracked per block length: shares at different lengths cannot be compared.
    struct Track {
        std::deque<std::pair<int, int>> window;   // (drafted, accepted), oldest first
        long drafted = 0, accepted = 0;
        bool on = true;
        int since_probe = 0, good_probes = 0;
        double off_below = 0.0, on_above = 0.0;   // thresholds of the band last decided in
    };

    PolicyConfig config_;
    std::map<int, Track> tracks_;
    std::deque<long> errors_;       // decision numbers at which errors happened
    int cooldown_left_ = 0;
    bool latched_ = false;
    long decisions_ = 0;
    std::map<Reason, long> counts_;
    int switches_ = 0;
};

}  // namespace specedge
