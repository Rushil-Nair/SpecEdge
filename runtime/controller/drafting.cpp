#include "drafting.h"

#include <algorithm>
#include <exception>

namespace specedge {

RequestStats run_request(Target & target, DraftSource * source, DraftPolicy * policy, int requests_in_flight, int max_tokens) {
    RequestStats stats;
    int n_max = 0;
    if (source && policy) {
        stats.decision = policy->decide(requests_in_flight);
        stats.asked_policy = true;
        n_max = stats.decision.n_max;
    }

    // `pending` is always a token the target has already chosen and that is not yet in the
    // context. Each round puts it in, perhaps with a draft behind it, and ends with the next one.
    Token pending = target.first();
    while (true) {
        if (target.is_end(pending)) {
            stats.ended = true;
            stats.end_token = pending;
            break;
        }
        if (static_cast<int>(stats.tokens.size()) >= max_tokens || target.room() < 1) {
            stats.hit_token_limit = true;
            break;
        }

        // A draft is only worth asking for if there is room to keep some of it.
        const int budget = std::min(max_tokens - static_cast<int>(stats.tokens.size()) - 1, target.room() - 1);
        std::vector<Token> draft;
        if (n_max > 0 && !stats.draft_failed && budget > 0) {
            try {
                draft = source->draft(stats.tokens, pending, std::min(n_max, budget));
                if (static_cast<int>(draft.size()) > std::min(n_max, budget)) draft.resize(static_cast<size_t>(std::min(n_max, budget)));
                stats.drafts++;
                stats.drafted += static_cast<int>(draft.size());
            } catch (const std::exception & problem) {
                // Error fallback: this request goes on without drafting, and the policy keeps
                // drafting off for the requests that follow.
                stats.draft_failed = true;
                stats.draft_error = problem.what();
                policy->on_error();
                draft.clear();
            }
        }

        std::vector<Token> batch;
        batch.reserve(draft.size() + 1);
        batch.push_back(pending);
        batch.insert(batch.end(), draft.begin(), draft.end());
        target.eval(batch);
        stats.rounds++;
        stats.tokens.push_back(pending);

        // Check the draft against the target, position by position.
        int kept = 0;
        bool have_next = false;
        for (size_t i = 0; i < draft.size(); i++) {
            const Token choice = target.choose(static_cast<int>(i));
            if (choice != draft[i] || target.is_end(choice)) {
                pending = choice;                  // the target's own token replaces the draft from here on
                have_next = true;
                break;
            }
            stats.tokens.push_back(choice);
            kept++;
        }
        stats.accepted += kept;
        if (!have_next) pending = target.choose(static_cast<int>(draft.size()));     // the whole draft was right
        else target.keep(1 + kept);                                                   // drop the rejected tail
    }

    if (stats.asked_policy && stats.drafted > 0) policy->observe(stats.decision.n_max, stats.drafted, stats.accepted);
    return stats;
}

}  // namespace specedge
