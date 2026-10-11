// specedge-agent: the part 6 agent runtime.
//
//   specedge-agent run      --model M --tools DIR --contacts FILE --tasks FILE --out FILE [options]
//                           [--draft-model M2 --policy FILE [--draft-ngl N] [--requests-in-flight N]]
//   specedge-agent toolhost --tools DIR --contacts FILE
//   specedge-agent render   --tools DIR --messages FILE [--date "08 Oct 2026"]
//   specedge-agent grammar  --tools DIR
//   specedge-agent tokenize --model M --file FILE
//   specedge-agent grammar-check --model M --tools DIR --cases FILE
//
// run       runs the tasks through the tool loop and writes one results file.
// toolhost  serves the tools, the parser, the prompt builder and the checker over stdin and
//           stdout, one JSON object per line, so the Python control uses the same code.
// render    prints the prompt for a list of messages; grammar prints the forced-JSON grammar;
//           tokenize prints the token ids of a text file; grammar-check feeds texts through
//           the grammar with the model's vocabulary. These four exist for the tests.
#include "engine.h"
#include "grammar.h"
#include "host.h"
#include "loop.h"
#include "parser.h"
#include "policy_config.h"
#include "prompt.h"
#include "tools.h"
#include "world.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#ifndef SPECEDGE_LLAMA_COMMIT
#define SPECEDGE_LLAMA_COMMIT "unknown"
#endif

using namespace specedge;

namespace {

using Options = std::map<std::string, std::string>;

Options parse_options(int argc, char ** argv, int first) {
    Options options;
    for (int i = first; i < argc; i++) {
        const std::string key = argv[i];
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) throw std::runtime_error("expected --option value, got " + key);
        options[key.substr(2)] = argv[++i];
    }
    return options;
}

std::string need(const Options & options, const std::string & key) {
    const auto it = options.find(key);
    if (it == options.end()) throw std::runtime_error("missing --" + key);
    return it->second;
}

std::string get(const Options & options, const std::string & key, const std::string & fallback) {
    const auto it = options.find(key);
    return it == options.end() ? fallback : it->second;
}

int get_int(const Options & options, const std::string & key, int fallback) {
    const auto it = options.find(key);
    return it == options.end() ? fallback : std::stoi(it->second);
}

std::string first_line_with(const std::string & path, const std::string & key) {
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        if (line.rfind(key, 0) == 0) {
            const size_t colon = line.find(':');
            return colon == std::string::npos ? line : line.substr(colon + 2);
        }
    }
    return "";
}

std::string whole_file_or_empty(const std::string & path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    return line;
}

std::vector<json> read_jsonl(const std::string & path) {
    std::vector<json> rows;
    std::istringstream in(read_file(path));
    for (std::string line; std::getline(in, line);)
        if (line.find_first_not_of(" \t\r\n") != std::string::npos) rows.push_back(json::parse(line));
    return rows;
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

int run(const Options & options) {
    const ToolSet tools = load_tools(need(options, "tools"));
    agent::World world(json::parse(read_file(need(options, "contacts"))));
    std::vector<json> tasks = read_jsonl(need(options, "tasks"));

    // --ids a,b,c keeps only those tasks; --limit N keeps the first N.
    if (options.count("ids")) {
        std::set<std::string> wanted;
        std::istringstream list(options.at("ids"));
        for (std::string id; std::getline(list, id, ',');) wanted.insert(id);
        std::vector<json> kept;
        for (const auto & task : tasks)
            if (wanted.count(task.value("id", ""))) kept.push_back(task);
        tasks = kept;
    }
    const int limit = get_int(options, "limit", 0);
    if (limit > 0 && static_cast<size_t>(limit) < tasks.size()) tasks.resize(static_cast<size_t>(limit));

    LoopConfig loop;
    loop.cache = get_int(options, "cache", 1) != 0;
    loop.forced_json = get_int(options, "forced-json", 1) != 0;
    loop.max_turns = get_int(options, "max-turns", 10);
    loop.max_tokens = get_int(options, "max-tokens", 256);
    loop.date = get(options, "date", today_date());
    loop.tool_results = style_from(get(options, "tool-results", "stock"));
    loop.placement = placement_from(get(options, "tools-in", "user"));
    if (options.count("system")) loop.system_prompt = read_file(options.at("system"));

    EngineConfig engine_config;
    engine_config.model_path = need(options, "model");
    engine_config.n_ctx = get_int(options, "ctx", 8192);
    engine_config.n_batch = get_int(options, "batch", 2048);
    engine_config.n_threads = get_int(options, "threads", 4);
    engine_config.n_gpu_layers = get_int(options, "ngl", 0);

    // Drafting: a second model proposes tokens and the policy decides per reply whether to use it.
    const bool drafting = options.count("draft-model") > 0;
    if (drafting && !options.count("policy"))
        throw std::runtime_error("--draft-model needs --policy FILE: the thresholds belong to a model, a drafter and a machine");
    DraftPolicy policy(drafting ? policy_from_json(json::parse(read_file(options.at("policy")))) : PolicyConfig{});
    loop.requests_in_flight = get_int(options, "requests-in-flight", 1);

    const std::string grammar = build_tool_grammar(tools);
    const double started_unix = static_cast<double>(std::time(nullptr));
    const auto start = std::chrono::steady_clock::now();
    const auto load_start = start;
    Engine engine(engine_config);
    const double load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();

    std::unique_ptr<LlamaDraft> drafter;
    if (drafting) {
        EngineConfig draft_config = engine_config;
        draft_config.model_path = options.at("draft-model");
        draft_config.n_gpu_layers = get_int(options, "draft-ngl", engine_config.n_gpu_layers);
        drafter = std::make_unique<LlamaDraft>(draft_config, engine);
        loop.drafter = drafter.get();
        loop.policy = &policy;
    }

    // One short evaluation before the tasks, so the first task does not pay for first use.
    engine.set_prompt(engine.tokenize("<|begin_of_text|>Hello"), false);
    engine.generate(4, "");

    json rows = json::array();
    std::map<std::string, std::pair<int, int>> by_family;      // family -> (correct, tasks)
    std::vector<double> wall, prompt, generation, build;
    long drafted_total = 0, accepted_total = 0;
    int correct = 0, correct_strict = 0, reached_state = 0, answered = 0;
    for (size_t i = 0; i < tasks.size(); i++) {
        json row = run_task(engine, tools, grammar, world, tasks[i], loop);
        const bool ok = row["correct"].get<bool>();
        correct += ok;
        drafted_total += row["drafted_tokens"].get<long>();
        accepted_total += row["accepted_tokens"].get<long>();
        correct_strict += row["correct_strict"].get<bool>();
        const json & verdict = row["verdict"];
        reached_state += row["first_turn_state_ok"].get<int>() >= 0 ||
                         (verdict["events_ok"].get<bool>() && verdict["expenses_ok"].get<bool>() &&
                          verdict["messages_ok"].get<bool>() && verdict["tools_ok"].get<bool>());
        answered += row["ended"] == "answered";
        auto & family = by_family[row["family"].get<std::string>()];
        family.first += ok;
        family.second += 1;
        wall.push_back(row["wall_ms"].get<double>() / 1000.0);
        prompt.push_back(row["prompt_ms"].get<double>() / 1000.0);
        generation.push_back(row["generation_ms"].get<double>() / 1000.0);
        build.push_back(row["build_ms"].get<double>() / 1000.0);
        std::fprintf(stderr, "[%zu/%zu] %s %s  %s  turns %d  %.1f s\n", i + 1, tasks.size(), row["id"].get<std::string>().c_str(),
                     ok ? "correct" : "wrong", row["ended"].get<std::string>().c_str(), row["turns_used"].get<int>(), wall.back());
        rows.push_back(std::move(row));
    }

    const auto sum = [](const std::vector<double> & values) { double total = 0; for (double v : values) total += v; return total; };
    const double n = tasks.empty() ? 1.0 : static_cast<double>(tasks.size());
    json families = json::object();
    for (const auto & [name, counts] : by_family) families[name] = {{"correct", counts.first}, {"tasks", counts.second}};

    json record = {
        {"run_id", get(options, "name", "run")},
        {"part", 6},
        {"program", "specedge-agent run"},
        {"timing_valid", get_int(options, "timing", 0) != 0},
        {"note", get(options, "note", "")},
        {"config", {{"cache", loop.cache}, {"forced_json", loop.forced_json}, {"max_turns", loop.max_turns},
                    {"max_tokens", loop.max_tokens}, {"decoding", "greedy"}, {"date", loop.date},
                    {"tool_results", get(options, "tool-results", "stock")},
                    {"tools_in", get(options, "tools-in", "user")},
                    {"n_ctx", engine_config.n_ctx}, {"n_batch", engine_config.n_batch},
                    {"threads", engine_config.n_threads}, {"n_gpu_layers", engine_config.n_gpu_layers},
                    {"drafting", drafting}, {"draft_model", get(options, "draft-model", "")},
                    {"draft_n_gpu_layers", get_int(options, "draft-ngl", engine_config.n_gpu_layers)},
                    {"requests_in_flight", loop.requests_in_flight},
                    {"policy_file", get(options, "policy", "")},
                    {"policy", drafting ? policy_to_json(policy.config()) : json()}}},
        {"seed", 0},
        {"model", {{"path", engine_config.model_path},
                   {"file", std::filesystem::path(engine_config.model_path).filename().string()},
                   {"bytes", std::filesystem::file_size(engine_config.model_path)},
                   {"description", engine.model_description()},
                   {"sha256", get(options, "model-sha256", "")},
                   {"source", get(options, "model-source", "")}}},
        {"llama_cpp_commit", SPECEDGE_LLAMA_COMMIT},
        {"tasks_file", need(options, "tasks")},
        {"tasks_sha256", get(options, "tasks-sha256", "")},
        {"tools", tools.tools.size()},
        {"hardware", {{"cpu", first_line_with("/proc/cpuinfo", "model name")},
                      {"cpus_visible", std::thread::hardware_concurrency()},
                      {"cpu_quota", whole_file_or_empty("/sys/fs/cgroup/cpu.max")},
                      {"memory", first_line_with("/proc/meminfo", "MemTotal")},
                      {"gpu", get(options, "gpu", "")}}},
        {"started_unix", started_unix},
        {"model_load_seconds", load_seconds},
        {"wall_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()},
        {"summary", {{"tasks", tasks.size()},
                     {"correct", correct},
                     {"correct_strict", correct_strict},
                     {"reached_expected_state_at_some_turn", reached_state},
                     {"ended_with_an_answer", answered},
                     {"drafting", drafting ? json{{"replies", policy.decisions()},
                                                  {"replies_drafted", policy.count(Reason::Drafting) + policy.count(Reason::Probe)},
                                                  {"by_decision", json{{"drafting", policy.count(Reason::Drafting)},
                                                                       {"probe", policy.count(Reason::Probe)},
                                                                       {"low_acceptance", policy.count(Reason::LowAcceptance)},
                                                                       {"queue_deep", policy.count(Reason::QueueDeep)},
                                                                       {"error_cooldown", policy.count(Reason::ErrorCooldown)},
                                                                       {"error_latched", policy.count(Reason::ErrorLatched)}}},
                                                  {"times_switched", policy.switches()},
                                                  {"drafted_tokens", drafted_total}, {"accepted_tokens", accepted_total}}
                                           : json()},
                     {"by_family", families},
                     {"seconds_per_task", {{"wall_mean", sum(wall) / n}, {"wall_median", median(wall)},
                                           {"prompt_mean", sum(prompt) / n}, {"generation_mean", sum(generation) / n},
                                           {"build_prompt_mean", sum(build) / n}}},
                     {"seconds_total", {{"wall", sum(wall)}, {"prompt", sum(prompt)}, {"generation", sum(generation)}}}}},
        {"tasks", rows}};

    const std::string out = need(options, "out");
    std::filesystem::create_directories(std::filesystem::path(out).parent_path());
    std::ofstream(out) << record.dump() << "\n";
    std::fprintf(stderr, "%d of %zu correct; mean %.2f s per task (prompt %.2f, generation %.2f); wrote %s\n", correct,
                 tasks.size(), sum(wall) / n, sum(prompt) / n, sum(generation) / n, out.c_str());
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: specedge-agent run|toolhost|render|grammar|tokenize --option value ...");
        const std::string command = argv[1];
        const Options options = parse_options(argc, argv, 2);
        if (command == "run") return run(options);
        if (command == "toolhost") return run_toolhost(need(options, "tools"), need(options, "contacts"));
        if (command == "render") {
            const ToolSet tools = options.count("tools") ? load_tools(options.at("tools")) : ToolSet{};
            std::cout << render_prompt(tools, messages_from(json::parse(read_file(need(options, "messages")))),
                                       get(options, "date", today_date()), style_from(get(options, "tool-results", "stock")));
            return 0;
        }
        if (command == "grammar") {
            std::cout << build_tool_grammar(load_tools(need(options, "tools")));
            return 0;
        }
        if (command == "tokenize") {
            EngineConfig config;
            config.model_path = need(options, "model");
            config.n_ctx = 512;
            config.n_batch = 512;
            Engine engine(config);
            std::cout << json(engine.tokenize(read_file(need(options, "file")))).dump() << "\n";
            return 0;
        }
        if (command == "grammar-check") {
            // --cases FILE: [{"text": "...", "accept": true}, ...]. Exit code 1 when any case disagrees.
            EngineConfig config;
            config.model_path = need(options, "model");
            config.n_ctx = 512;
            config.n_batch = 512;
            Engine engine(config);
            const std::string grammar = build_tool_grammar(load_tools(need(options, "tools")));
            int wrong = 0, total = 0;
            for (const auto & one : json::parse(read_file(need(options, "cases")))) {
                const bool accepted = engine.grammar_accepts(grammar, one.at("text").get<std::string>());
                total++;
                if (accepted != one.at("accept").get<bool>()) {
                    wrong++;
                    std::printf("DISAGREES (grammar %s): %s\n", accepted ? "accepts" : "refuses", one["text"].get<std::string>().c_str());
                }
            }
            std::printf("grammar-check: %d of %d cases as expected\n", total - wrong, total);
            return wrong ? 1 : 0;
        }
        throw std::runtime_error("unknown command: " + command);
    } catch (const std::exception & problem) {
        std::fprintf(stderr, "specedge-agent: %s\n", problem.what());
        return 1;
    }
}
