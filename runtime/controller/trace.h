// Recorded traces (runtime/controller/traces/*.csv) and a replay of a policy over them.
// Used by the tests and by policy_replay; not part of the policy itself.
#pragma once

#include "policy.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace specedge {

struct TraceRow {
    std::string id, kind;
    int users = 0, tokens = 0;
    double plain_ms = 0, draft_ms = 0;     // generation time of this request, recorded both ways
    int draft_n = 0, draft_n_accepted = 0;
};

inline std::vector<TraceRow> load_trace(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<TraceRow> rows;
    std::string line;
    std::getline(in, line);                // header
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> cell;
        std::istringstream cells(line);
        for (std::string one; std::getline(cells, one, ',');) cell.push_back(one);
        if (cell.size() != 9) throw std::runtime_error("bad trace line: " + line);
        TraceRow row;
        row.id = cell[1];
        row.kind = cell[2];
        row.users = std::stoi(cell[3]);
        row.tokens = std::stoi(cell[4]);
        row.plain_ms = std::stod(cell[5]);
        row.draft_ms = std::stod(cell[6]);
        row.draft_n = std::stoi(cell[7]);
        row.draft_n_accepted = std::stoi(cell[8]);
        rows.push_back(row);
    }
    return rows;
}

struct Replay {
    double policy_ms = 0, always_ms = 0, never_ms = 0, oracle_ms = 0;
    int drafted_requests = 0;
    std::vector<Decision> decisions;
};

// Walk the requests in order. A drafted request costs what drafting took when it was
// recorded and is reported to the policy; a plain one costs what plain decoding took.
// `always` drafts everything, `never` nothing, `oracle` takes the faster of the two for
// every request (it knows the answer in advance; no policy can).
// An estimate from recorded times, not a measurement.
inline Replay replay(DraftPolicy & policy, const std::vector<TraceRow> & rows) {
    Replay out;
    for (const auto & row : rows) {
        const Decision decision = policy.decide(row.users);
        out.decisions.push_back(decision);
        if (decision.draft()) {
            out.policy_ms += row.draft_ms;
            out.drafted_requests++;
            policy.observe(decision.n_max, row.draft_n, row.draft_n_accepted);
        } else {
            out.policy_ms += row.plain_ms;
        }
        out.always_ms += row.draft_ms;
        out.never_ms += row.plain_ms;
        out.oracle_ms += row.draft_ms < row.plain_ms ? row.draft_ms : row.plain_ms;
    }
    return out;
}

inline std::vector<TraceRow> of_kind(const std::vector<TraceRow> & rows, const std::string & kind) {
    std::vector<TraceRow> out;
    for (const auto & row : rows)
        if (row.kind == kind) out.push_back(row);
    return out;
}

inline std::vector<TraceRow> joined(std::initializer_list<std::vector<TraceRow>> parts) {
    std::vector<TraceRow> out;
    for (const auto & part : parts) out.insert(out.end(), part.begin(), part.end());
    return out;
}

}  // namespace specedge
