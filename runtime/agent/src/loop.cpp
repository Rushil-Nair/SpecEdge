#include "loop.h"

#include "parser.h"
#include "prompt.h"

#include <chrono>
#include <exception>
#include <vector>

namespace specedge {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

json run_task(Engine & engine, const ToolSet & tools, const std::string & grammar, agent::World & world,
              const json & task, const LoopConfig & config) {
    const auto task_start = Clock::now();
    world.reset(task.value("initial", json::object()));

    std::vector<Message> messages;
    if (!config.system_prompt.empty()) messages.push_back({"system", config.system_prompt});
    messages.push_back({"user", task.at("user").get<std::string>()});

    json turns = json::array();
    std::string final_answer, ended = "turn_limit", error;
    double prompt_ms = 0, generation_ms = 0, build_ms = 0;
    int prompt_tokens = 0, processed_tokens = 0, written_tokens = 0, calls_made = 0;
    int drafted_tokens = 0, accepted_tokens = 0;
    int first_turn_state_ok = -1;        // the first turn after which the world was as the task expects
    const json expect = task.value("expect", json::object());
    // Is the world as the task expects right now? The final answer is not part of this.
    const auto state_ok = [&]() {
        const json verdict = agent::check(expect, world.state(), "");
        return verdict["events_ok"].get<bool>() && verdict["expenses_ok"].get<bool>() && verdict["messages_ok"].get<bool>() &&
               verdict["tools_ok"].get<bool>();
    };

    try {
        for (int turn = 0; turn < config.max_turns; turn++) {
            const auto build_start = Clock::now();
            const std::vector<llama_token> prompt = engine.tokenize(render_prompt(tools, messages, config.date, config.tool_results, config.placement));
            const double build = ms_since(build_start);

            const PromptStats read = engine.set_prompt(prompt, config.cache);
            GenerateStats reply;
            RequestStats request;
            const bool through_policy = config.drafter && config.policy;
            if (through_policy) {
                config.drafter->begin(prompt);
                reply = generate_with_policy(engine, config.drafter, config.policy, config.requests_in_flight, config.max_tokens,
                                             config.forced_json ? grammar : std::string(), request);
            } else {
                reply = engine.generate(config.max_tokens, config.forced_json ? grammar : std::string());
            }
            const ParseResult parsed = parse_reply(reply.text, &tools);

            build_ms += build;
            prompt_ms += read.ms;
            generation_ms += reply.ms;
            prompt_tokens += read.prompt_tokens;
            processed_tokens += read.processed_tokens;
            written_tokens += reply.n_tokens;

            json record = {{"turn", turn},
                           {"prompt_tokens", read.prompt_tokens},
                           {"reused_tokens", read.reused_tokens},
                           {"processed_tokens", read.processed_tokens},
                           {"build_ms", build},
                           {"prompt_ms", read.ms},
                           {"written_tokens", reply.n_tokens},
                           {"generation_ms", reply.ms},
                           {"sampling_ms", reply.sampling_ms},
                           {"grammar_rejections", reply.grammar_rejections},
                           {"hit_token_limit", reply.hit_token_limit},
                           {"stop_token", reply.stop_token},
                           {"text", reply.text},
                           {"broken_call", parsed.broken},
                           {"documented_format", parsed.documented_format},
                           {"values_converted", parsed.converted},
                           {"calls", json::array()}};

            if (through_policy) {
                record["draft"] = json{{"decision", reason_name(request.decision.reason)}, {"n_max", request.decision.n_max},
                                       {"target_evaluations", request.rounds}, {"drafts", request.drafts},
                                       {"drafted", request.drafted}, {"accepted", request.accepted},
                                       {"failed", request.draft_failed}, {"error", request.draft_error},
                                       {"draft_model_ms", config.drafter->ms}};
                drafted_tokens += request.drafted;
                accepted_tokens += request.accepted;
            }

            // The model's own words go back in as its turn, exactly as written.
            messages.push_back({"assistant", reply.text});

            if (parsed.calls.empty() && !parsed.broken) {
                final_answer = parsed.text;
                ended = "answered";
                record["state_ok"] = state_ok();
                turns.push_back(record);
                break;
            }
            if (parsed.calls.empty()) {
                messages.push_back({"tool", dump_spaced(json{{"error", "the tool call was not valid JSON"}})});
            }
            for (const auto & call : parsed.calls) {
                const json result = world.call(call.name, call.arguments);
                calls_made++;
                record["calls"].push_back(json{{"name", call.name}, {"arguments", call.arguments}, {"result", result}});
                messages.push_back({"tool", dump_spaced(result)});
            }
            record["state_ok"] = state_ok();
            if (first_turn_state_ok < 0 && record["state_ok"].get<bool>()) first_turn_state_ok = turn;
            turns.push_back(record);
        }
    } catch (const std::exception & problem) {
        ended = "error";
        error = problem.what();
    }

    const json state = world.state();
    json verdict = agent::check(expect, state, final_answer);
    if (ended != "answered") {                                // a task that never finished is not correct
        verdict["correct"] = false;
        verdict["correct_strict"] = false;
    }

    return json{{"id", task.value("id", "")},
                {"family", task.value("family", "")},
                {"correct", verdict["correct"]},
                {"correct_strict", verdict["correct_strict"]},
                {"verdict", verdict},
                {"ended", ended},
                {"first_turn_state_ok", first_turn_state_ok},
                {"error", error},
                {"turns_used", turns.size()},
                {"calls_made", calls_made},
                {"wall_ms", ms_since(task_start)},
                {"prompt_ms", prompt_ms},
                {"generation_ms", generation_ms},
                {"build_ms", build_ms},
                {"prompt_tokens", prompt_tokens},
                {"processed_tokens", processed_tokens},
                {"written_tokens", written_tokens},
                {"drafted_tokens", drafted_tokens},
                {"accepted_tokens", accepted_tokens},
                {"final_answer", final_answer},
                {"final_state", state},
                {"turns", turns}};
}

}  // namespace specedge
