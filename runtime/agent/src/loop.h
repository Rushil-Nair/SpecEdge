// The tool loop: prompt, reply, parse, run the tools, feed the results back, until the
// model answers in plain text or the turn limit is reached.
#pragma once

#include "engine.h"
#include "engine_draft.h"
#include "prompt.h"
#include "tools.h"
#include "world.h"

#include <string>

namespace specedge {

struct LoopConfig {
    bool cache = true;           // keep the key-value cache between turns (and between tasks)
    bool forced_json = true;     // write replies under the grammar built from the tool schemas
    int max_turns = 10;          // assistant turns per task
    int max_tokens = 256;        // tokens per assistant turn
    std::string date;            // the prompt's "Today Date" line
    std::string system_prompt;
    ToolResultStyle tool_results = ToolResultStyle::Stock;
    ToolPlacement placement = ToolPlacement::FirstUser;

    // Drafting. With a drafter and a policy, every reply goes through the policy: it decides
    // whether to draft and with what block length, and is told what was accepted. Without
    // them replies are written by Engine::generate exactly as before.
    LlamaDraft * drafter = nullptr;
    DraftPolicy * policy = nullptr;
    int requests_in_flight = 1;       // the queue depth the policy is asked with
};

// Run one task: {"id", "user", "initial", "expect", ...}. `grammar` is used when
// config.forced_json is set. Returns the task's record: every turn with its timings, the
// final state and the checker's verdict.
json run_task(Engine & engine, const ToolSet & tools, const std::string & grammar, agent::World & world,
              const json & task, const LoopConfig & config);

}  // namespace specedge
