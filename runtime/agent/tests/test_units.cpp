// Tests that need no model: the parser, the prompt builder, the grammar builder's output,
// the tools and the checker.   test_units <agent/tools directory>
#include "grammar.h"
#include "parser.h"
#include "policy_config.h"
#include "prompt.h"
#include "tools.h"
#include "world.h"

#include <cstdio>
#include <string>

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

static bool has(const std::string & text, const std::string & part) { return text.find(part) != std::string::npos; }

static void test_parser(const ToolSet & tools) {
    {   // the documented form
        const auto r = parse_reply(R"({"name": "list_events", "parameters": {"date": "2026-11-12"}})", &tools);
        CHECK(r.calls.size() == 1 && r.calls[0].name == "list_events");
        CHECK(r.calls[0].arguments["date"] == "2026-11-12");
        CHECK(!r.broken && r.documented_format && r.text.empty() && r.converted == 0);
    }
    {   // several calls the way Llama 3.2 joins them, which llama-server refuses
        const auto r = parse_reply(R"({"name": "search_contacts", "parameters": {"query": "Priya"}}; {"name": "list_events", "parameters": {"date": "2026-11-12"}})", &tools);
        CHECK(r.calls.size() == 2 && r.calls[0].name == "search_contacts" && r.calls[1].name == "list_events");
        CHECK(r.text.empty() && !r.broken);
    }
    {   // a leading <|python_tag|> and calls on separate lines
        const auto r = parse_reply("<|python_tag|>{\"name\": \"cancel_event\", \"parameters\": {\"event_id\": \"e_101\"}}\n{\"name\": \"cancel_event\", \"parameters\": {\"event_id\": \"e_102\"}}", &tools);
        CHECK(r.calls.size() == 2 && r.calls[1].arguments["event_id"] == "e_102" && r.text.empty());
    }
    {   // a number written as a string becomes a number when the schema says number
        const auto r = parse_reply(R"({"name": "add_expense", "parameters": {"amount": "12.75", "category": "meals"}})", &tools);
        CHECK(r.calls.size() == 1 && r.calls[0].arguments["amount"].is_number() && r.calls[0].arguments["amount"] == 12.75);
        CHECK(r.calls[0].arguments["category"] == "meals" && r.converted == 1);
    }
    {   // a list written as a string becomes a list
        const auto r = parse_reply(R"({"name": "create_event", "parameters": {"title": "Sync", "date": "2026-11-12", "start_time": "09:00", "duration_hours": "1.5", "attendee_ids": "[\"c_101\", \"c_102\"]"}})", &tools);
        CHECK(r.calls.size() == 1 && r.calls[0].arguments["attendee_ids"].is_array() && r.calls[0].arguments["attendee_ids"].size() == 2);
        CHECK(r.calls[0].arguments["duration_hours"] == 1.5 && r.converted == 2);
    }
    {   // a number where the schema says string becomes a string
        const auto r = parse_reply(R"({"name": "cancel_event", "parameters": {"event_id": 101}})", &tools);
        CHECK(r.calls[0].arguments["event_id"] == "101");
    }
    {   // a string that is not a number stays a string
        const auto r = parse_reply(R"({"name": "add_expense", "parameters": {"amount": "twelve", "category": "meals"}})", &tools);
        CHECK(r.calls[0].arguments["amount"] == "twelve" && r.converted == 0);
    }
    {   // the forms the 1B uses
        const auto a = parse_reply(R"({"type": "function", "function": "list_events", "parameters": {"date": "2026-11-12"}})", &tools);
        CHECK(a.calls.size() == 1 && a.calls[0].name == "list_events" && !a.documented_format);
        const auto b = parse_reply(R"({"function": {"name": "list_events", "arguments": {"date": "2026-11-12"}}})", &tools);
        CHECK(b.calls.size() == 1 && b.calls[0].arguments["date"] == "2026-11-12" && !b.documented_format);
        const auto c = parse_reply(R"({"name": "list_events", "arguments": "{\"date\": \"2026-11-12\"}"})", &tools);
        CHECK(c.calls.size() == 1 && c.calls[0].arguments["date"] == "2026-11-12" && !c.documented_format);
    }
    {   // a call cut off before it closed
        const auto r = parse_reply(R"({"name": "list_events", "parameters": {"date": "2026)", &tools);
        CHECK(r.calls.empty() && r.broken);
    }
    {   // plain text, and text with braces that is not a call
        const auto a = parse_reply("  Your meeting starts at 14:30.  ", &tools);
        CHECK(a.calls.empty() && !a.broken && a.text == "Your meeting starts at 14:30.");
        const auto b = parse_reply("The set {1, 2} is small.", &tools);
        CHECK(b.calls.empty() && !b.broken && b.text == "The set {1, 2} is small.");
    }
    {   // braces and quotes inside a string argument
        const auto r = parse_reply(R"({"name": "send_message", "parameters": {"contact_id": "c_101", "text": "use } and { and \"quotes\"; ok"}})", &tools);
        CHECK(r.calls.size() == 1 && r.calls[0].arguments["text"] == "use } and { and \"quotes\"; ok");
    }
    {   // a call with words around it keeps the words
        const auto r = parse_reply(R"(I will look that up. {"name": "list_events", "parameters": {"date": "2026-11-12"}})", &tools);
        CHECK(r.calls.size() == 1 && r.text == "I will look that up.");
    }
    {   // an unknown tool is still a call; the world answers with an error
        const auto r = parse_reply(R"({"name": "delete_everything", "parameters": {}})", &tools);
        CHECK(r.calls.size() == 1 && r.calls[0].name == "delete_everything");
    }
}

static void test_prompt(const ToolSet & tools) {
    const std::string prompt = render_prompt(tools, {{"system", "  Be brief. "}, {"user", "Hello"}, {"assistant", "{\"name\": \"x\"}"},
                                                     {"tool", "{\"ok\": true}"}}, "08 Oct 2026");
    CHECK(prompt.rfind("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\n"
                       "Cutting Knowledge Date: December 2023\nToday Date: 08 Oct 2026\n\nBe brief.<|eot_id|>"
                       "<|start_header_id|>user<|end_header_id|>\n\nGiven the following functions", 0) == 0);
    CHECK(has(prompt, "Do not use variables.\n\n{\n    \"type\": \"function\",\n    \"function\": {\n        \"name\": \"search_contacts\","));
    CHECK(has(prompt, "}\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{\"name\": \"x\"}<|eot_id|>"
                      "<|start_header_id|>ipython<|end_header_id|>\n\n\"{\\\"ok\\\": true}\"<|eot_id|>"
                      "<|start_header_id|>assistant<|end_header_id|>\n\n"));
    const std::string unquoted = render_prompt(tools, {{"user", "Hello"}, {"assistant", "x"}, {"tool", "{\"ok\": true}"}},
                                               "08 Oct 2026", ToolResultStyle::Plain);
    CHECK(has(unquoted, "<|start_header_id|>ipython<|end_header_id|>\n\n{\"ok\": true}<|eot_id|>"));
    CHECK(dump_spaced(json::parse(R"({"a":[1,2.5,"x"],"b":{"c":true,"d":null},"e":[]})")) ==
          R"({"a": [1, 2.5, "x"], "b": {"c": true, "d": null}, "e": []})");
    const std::string plain = render_prompt(ToolSet{}, {{"user", "Hi"}}, "08 Oct 2026");
    CHECK(plain == "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n"
                   "Today Date: 08 Oct 2026\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
                   "<|start_header_id|>assistant<|end_header_id|>\n\n");
}

static void test_grammar(const ToolSet & tools) {
    const std::string grammar = build_tool_grammar(tools);
    CHECK(has(grammar, "root ::= call (\"; \" call)* | text\n"));
    CHECK(has(grammar, "call ::= t0-call | t1-call | t2-call | t3-call | t4-call | t5-call\n"));
    CHECK(has(grammar, "t4-p0 ::= \"\\\"amount\\\": \" number\n"));                       // a number is a bare number
    CHECK(has(grammar, "t4-p1 ::= \"\\\"category\\\": \" (\"\\\"travel\\\"\" | \"\\\"meals\\\"\""));
    CHECK(has(grammar, "t2-p4-list ::= \"[\" ( string (\", \" string)* )? \"]\"\n"));      // a typed list
    CHECK(has(grammar, "t4-h0 ::= t4-p0 t4-t1\n"));                                       // required first argument
    CHECK(has(grammar, "t4-t2 ::= (\", \" t4-p2)?\n"));                                   // optional last argument
    CHECK(gbnf_literal("a\"b\\c\n") == "\"a\\\"b\\\\c\\n\"");
    const std::string one = build_tool_grammar(tools, {false, false});
    CHECK(has(one, "root ::= call\n"));
    CHECK(has(build_tool_grammar(ToolSet{}), "root ::= text\n"));
}

static void test_world(const std::string & tools_dir) {
    agent::World world(json::parse(read_file(tools_dir + "/contacts.json")));
    const json initial = json::parse(R"({"events": [
        {"event_id": "e_101", "title": "Budget review", "date": "2026-11-12", "start_time": "14:30", "duration_hours": 1.5, "attendee_ids": ["c_107", "c_108"]},
        {"event_id": "e_102", "title": "Team lunch", "date": "2026-11-12", "start_time": "12:00", "duration_hours": 1.0, "attendee_ids": []}]})");
    world.reset(initial);

    CHECK(world.call("search_contacts", {{"query", "priya"}})["contacts"].size() == 1);
    CHECK(world.call("search_contacts", {{"query", "Design team"}})["contacts"].size() == 3);
    CHECK(world.call("search_contacts", {{"query", "nobody"}})["contacts"].empty());
    const json listed = world.call("list_events", {{"date", "2026-11-12"}});
    CHECK(listed["events"].size() == 2 && listed["events"][0]["event_id"] == "e_102");     // sorted by start time
    CHECK(world.call("list_events", {{"date", "12 Nov"}}).contains("error"));
    CHECK(world.call("create_event", {{"title", "X"}, {"date", "2026-11-13"}, {"start_time", "9:00"}, {"duration_hours", 1}}).contains("error"));
    CHECK(world.call("create_event", {{"title", "X"}, {"date", "2026-11-13"}, {"start_time", "09:00"}, {"duration_hours", "1"}}).contains("error"));
    CHECK(world.call("create_event", {{"title", "X"}, {"date", "2026-11-13"}, {"start_time", "09:00"}, {"duration_hours", 1},
                                      {"attendee_ids", json::array({"c_999"})}}).contains("error"));
    const json made = world.call("create_event", {{"title", "Design sync"}, {"date", "2026-11-13"}, {"start_time", "09:00"},
                                                  {"duration_hours", 0.5}, {"attendee_ids", json::array({"c_101"})}});
    CHECK(made["event_id"] == "e_501");
    CHECK(world.call("cancel_event", {{"event_id", "e_102"}})["ok"] == true);
    CHECK(world.call("cancel_event", {{"event_id", "e_102"}}).contains("error"));
    CHECK(world.call("add_expense", {{"amount", 48.6}, {"category", "meals"}, {"note", "Cancellation fee"}})["expense_id"] == "x_1");
    CHECK(world.call("add_expense", {{"amount", 5}, {"category", "lunch"}}).contains("error"));
    CHECK(world.call("send_message", {{"contact_id", "c_107"}, {"text", "The lunch is cancelled."}})["ok"] == true);
    CHECK(world.call("send_message", {{"contact_id", "c_999"}, {"text", "hi"}}).contains("error"));
    CHECK(world.call("no_such_tool", json::object()).contains("error"));

    const json expect = json::parse(R"({
        "events": [
            {"title": "budget review", "date": "2026-11-12", "start_time": "14:30", "duration_hours": 1.5, "attendee_ids": ["c_108", "c_107"]},
            {"title": "Design sync", "date": "2026-11-13", "start_time": "09:00", "duration_hours": 0.5, "attendee_ids": ["c_101"]}],
        "expenses": [{"amount": 48.60, "category": "meals", "note_contains": ["fee"]}],
        "messages": [{"contact_id": "c_107", "contains": ["cancel"]}],
        "answer_contains": [["14:30", "2:30"]],
        "required_tools": ["list_events"]})");
    CHECK(agent::check(expect, world.state(), "It starts at 2:30 PM.")["correct"] == true);
    CHECK(agent::check(expect, world.state(), "Done.")["correct"] == false);               // the answer is checked
    // the stricter rule: the final answer must be plain text, not a tool result given back
    CHECK(agent::check(expect, world.state(), "It starts at 2:30 PM.")["correct_strict"] == true);
    const std::string as_result = "\"{\\\"error\\\": \\\"it is at 14:30\\\"}\"";       // "{\"error\": \"it is at 14:30\"}"
    CHECK(agent::check(expect, world.state(), as_result)["correct"] == true);
    CHECK(agent::check(expect, world.state(), as_result)["correct_strict"] == false);
    CHECK(agent::answer_is_text("The meeting is at 14:30.") && agent::answer_is_text("14:30") && agent::answer_is_text("42"));
    CHECK(agent::answer_is_text("\"Done\"") && agent::answer_is_text("Use {curly} braces."));
    CHECK(!agent::answer_is_text("") && !agent::answer_is_text("   ") && !agent::answer_is_text("{\"ok\": true}"));
    CHECK(!agent::answer_is_text("[1, 2]") && !agent::answer_is_text("{oops") && !agent::answer_is_text("\"[1, 2]\""));
    CHECK(!agent::answer_is_text("  \"{\\\"ok\\\": true}\"  "));
    json other = expect;
    other["expenses"][0]["amount"] = 48.0;
    CHECK(agent::check(other, world.state(), "14:30")["expenses_ok"] == false);            // 48 is not 48.60
    other = expect;
    other["messages"].push_back(json{{"contact_id", "c_108"}, {"contains", json::array()}});
    CHECK(agent::check(other, world.state(), "14:30")["messages_ok"] == false);            // a missing message
    world.call("send_message", {{"contact_id", "c_108"}, {"text", "extra"}});
    CHECK(agent::check(expect, world.state(), "14:30")["messages_ok"] == false);           // an extra message
    world.reset(initial);
    CHECK(agent::check(expect, world.state(), "14:30")["events_ok"] == false);             // nothing was done
    CHECK(agent::check(expect, world.state(), "14:30")["tools_ok"] == false);
}

static void test_policy_config() {
    const PolicyConfig defaults;
    const PolicyConfig same = policy_from_json(policy_to_json(defaults));           // there and back
    CHECK(same.bands.size() == defaults.bands.size() && same.bands[2].n_max == 4 && same.window_drafted == defaults.window_drafted);
    const PolicyConfig one = policy_from_json(json::parse(R"({"bands": [{"max_depth": 2, "n_max": 8, "off_below": 0.2, "on_above": 0.3}], "probe_every": 9})"));
    CHECK(one.bands.size() == 1 && one.bands[0].n_max == 8 && one.probe_every == 9 && one.error_cooldown == defaults.error_cooldown);
    DraftPolicy policy(one);
    CHECK(policy.decide(2).n_max == 8 && !policy.decide(3).draft());
    bool refused = false;
    try { policy_from_json(json::parse(R"({"bands": [{"max_depth": 4, "n_max": 8, "off_below": 0.2, "on_above": 0.3}, {"max_depth": 2, "n_max": 4, "off_below": 0.2, "on_above": 0.3}]})")); }
    catch (const std::exception &) { refused = true; }
    CHECK(refused);                                                                  // bands out of order
    refused = false;
    try { policy_from_json(json::parse(R"({"probe_every": -1})")); } catch (const std::exception &) { refused = true; }
    CHECK(refused);
}

int main(int argc, char ** argv) {
    const std::string tools_dir = argc > 1 ? argv[1] : "../../agent/tools";
    const ToolSet tools = load_tools(tools_dir + "/schemas");
    CHECK(tools.tools.size() == 6 && tools.find("add_expense") && !tools.find("nothing"));
    test_parser(tools);
    test_prompt(tools);
    test_grammar(tools);
    test_world(tools_dir);
    test_policy_config();
    std::printf("%d of %d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
