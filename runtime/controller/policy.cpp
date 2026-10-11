#include "policy.h"

namespace specedge {

const char * reason_name(Reason reason) {
    switch (reason) {
        case Reason::Drafting: return "drafting";
        case Reason::Probe: return "probe";
        case Reason::LowAcceptance: return "low_acceptance";
        case Reason::QueueDeep: return "queue_deep";
        case Reason::ErrorCooldown: return "error_cooldown";
        case Reason::ErrorLatched: return "error_latched";
    }
    return "unknown";
}

DraftPolicy::DraftPolicy(PolicyConfig config) : config_(std::move(config)) {}

void DraftPolicy::reset() {
    tracks_.clear();
    errors_.clear();
    cooldown_left_ = 0;
    latched_ = false;
}

Decision DraftPolicy::decide(int requests_in_flight) {
    decisions_++;
    const auto decided = [this](int n_max, Reason reason) {
        counts_[reason]++;
        return Decision{n_max, reason};
    };
    if (latched_) return decided(0, Reason::ErrorLatched);
    if (cooldown_left_ > 0) {
        cooldown_left_--;
        return decided(0, Reason::ErrorCooldown);
    }
    const Band * band = nullptr;
    for (const auto & candidate : config_.bands) {
        if (requests_in_flight <= candidate.max_depth) { band = &candidate; break; }
    }
    if (!band || band->n_max <= 0) return decided(0, Reason::QueueDeep);

    Track & track = tracks_[band->n_max];
    track.off_below = band->off_below;
    track.on_above = band->on_above;
    if (track.on) return decided(band->n_max, Reason::Drafting);
    if (++track.since_probe >= config_.probe_every) {
        track.since_probe = 0;
        return decided(band->n_max, Reason::Probe);
    }
    return decided(0, Reason::LowAcceptance);
}

void DraftPolicy::observe(int n_max, int drafted, int accepted) {
    if (n_max <= 0 || drafted <= 0) return;
    Track & track = tracks_[n_max];
    if (!track.on) {
        // This was a probe.
        if (drafted < config_.min_probe_drafted) return;
        const double share = static_cast<double>(accepted) / drafted;
        track.good_probes = share >= track.on_above ? track.good_probes + 1 : 0;
        if (track.good_probes >= config_.probes_to_recover) {
            track.on = true;
            track.good_probes = 0;
            track.window.clear();
            track.drafted = track.accepted = 0;
            switches_++;
        }
        return;
    }
    track.window.emplace_back(drafted, accepted);
    track.drafted += drafted;
    track.accepted += accepted;
    // Drop the oldest entries while the rest still covers the window.
    while (track.window.size() > 1 && track.drafted - track.window.front().first >= config_.window_drafted) {
        track.drafted -= track.window.front().first;
        track.accepted -= track.window.front().second;
        track.window.pop_front();
    }
    if (track.drafted >= config_.min_drafted && static_cast<double>(track.accepted) / track.drafted < track.off_below) {
        track.on = false;
        track.since_probe = 0;
        track.good_probes = 0;
        track.window.clear();
        track.drafted = track.accepted = 0;
        switches_++;
    }
}

void DraftPolicy::on_error() {
    cooldown_left_ = config_.error_cooldown;
    errors_.push_back(decisions_);
    while (!errors_.empty() && decisions_ - errors_.front() > config_.error_window) errors_.pop_front();
    if (static_cast<int>(errors_.size()) >= config_.errors_to_latch) latched_ = true;
}

double DraftPolicy::rolling_share(int n_max) const {
    const auto it = tracks_.find(n_max);
    if (it == tracks_.end() || it->second.drafted < config_.min_drafted) return -1.0;
    return static_cast<double>(it->second.accepted) / it->second.drafted;
}

bool DraftPolicy::drafting_enabled(int n_max) const {
    const auto it = tracks_.find(n_max);
    return it == tracks_.end() || it->second.on;
}

long DraftPolicy::count(Reason reason) const {
    const auto it = counts_.find(reason);
    return it == counts_.end() ? 0 : it->second;
}

}  // namespace specedge
