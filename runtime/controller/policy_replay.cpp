// Replay the policy over the recorded traces and print a table.
//     policy_replay <traces directory> [window_drafted probe_every probes_to_recover]
// Every number is an estimate from recorded per-request times, not a timing run.
#include "trace.h"

#include <cstdio>

using namespace specedge;

static PolicyConfig config;      // defaults, or changed from the command line

static void line(const char * name, const std::vector<TraceRow> & rows) {
    DraftPolicy policy(config);
    const Replay r = replay(policy, rows);
    int first_off = -1, back_on = -1;
    for (size_t i = 0; i < r.decisions.size(); i++) {
        if (first_off < 0 && !r.decisions[i].draft()) first_off = static_cast<int>(i);
        if (first_off >= 0 && r.decisions[i].reason == Reason::Drafting) back_on = static_cast<int>(i);
    }
    std::printf("| %s | %zu | %.1f | %.1f | %.1f | %.1f | %d | %d | %ld | %d |\n", name, rows.size(), r.never_ms / 1000,
                r.always_ms / 1000, r.policy_ms / 1000, r.oracle_ms / 1000, r.drafted_requests, policy.switches(),
                policy.count(Reason::Probe), first_off);
    (void) back_on;
}

int main(int argc, char ** argv) {
    const std::string dir = argc > 1 ? argv[1] : "traces";
    if (argc > 4) {
        config.window_drafted = std::stoi(argv[2]);
        config.probe_every = std::stoi(argv[3]);
        config.probes_to_recover = std::stoi(argv[4]);
        std::printf("window_drafted %d, probe_every %d, probes_to_recover %d\n\n", config.window_drafted, config.probe_every,
                    config.probes_to_recover);
    }
    const auto u1 = load_trace(dir + "/load_u1.csv"), u4 = load_trace(dir + "/load_u4.csv"), u16 = load_trace(dir + "/load_u16.csv");
    std::printf("| Trace | Requests | Never draft, s | Always draft, s | Policy, s | Oracle, s | Requests drafted | Switches | Probes | "
                "First plain request |\n|---|---|---|---|---|---|---|---|---|---|\n");
    line("1 user, recorded order (five kinds mixed)", u1);
    line("4 users, recorded order", u4);
    line("16 users, recorded order", u16);
    for (const char * kind : {"code", "tool_calls", "chat", "chinese", "reasoning"}) {
        const std::string name = std::string("1 user, ") + kind + " only";
        line(name.c_str(), of_kind(u1, kind));
    }
    line("1 user: code, then Chinese, then code", joined({of_kind(u1, "code"), of_kind(u1, "chinese"), of_kind(u1, "code")}));
    line("4 users: code, then Chinese, then code", joined({of_kind(u4, "code"), of_kind(u4, "chinese"), of_kind(u4, "code")}));
    line("1 user: tool calls, then chat, then Chinese, then reasoning",
         joined({of_kind(u1, "tool_calls"), of_kind(u1, "chat"), of_kind(u1, "chinese"), of_kind(u1, "reasoning")}));
    return 0;
}
