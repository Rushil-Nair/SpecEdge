// The in-memory world the agent acts on: a company directory, a calendar, an expense log
// and sent messages. Six tools read or change it; the checker compares the state a task
// ended in with the state the task expects.
#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace agent {

using json = nlohmann::ordered_json;

class World {
public:
    explicit World(json contacts);

    // Start a task: the calendar is replaced by initial["events"], everything else is emptied.
    void reset(const json & initial);

    // Run one tool. Never throws: a bad name or bad arguments come back as {"error": "..."},
    // which is what the model is shown.
    json call(const std::string & name, const json & arguments);

    // Events, expenses and messages as they stand, plus the names of the tools called so far.
    json state() const;

private:
    json search_contacts(const json & a) const;
    json list_events(const json & a) const;
    json create_event(const json & a);
    json cancel_event(const json & a);
    json add_expense(const json & a);
    json send_message(const json & a);
    bool contact_exists(const std::string & id) const;

    json contacts_;
    std::vector<json> events_;
    std::vector<json> expenses_;
    std::vector<json> messages_;
    std::vector<std::string> called_;
    int next_event_ = 501;
    int next_expense_ = 1;
};

// Decide whether a task ended correctly. `expect` is the task's "expect" object, `state` is
// World::state() at the end, `final_answer` is the model's last plain-text reply.
// Returns {"correct": bool, "correct_strict": bool, "events_ok": ..., "expenses_ok": ...,
// "messages_ok": ..., "answer_ok": ..., "tools_ok": ..., "answer_is_text": ...}.
//   correct         the current rule: any final text passes once the rest is right.
//   correct_strict  the same, and the final answer must be plain text (answer_is_text).
// Both are always reported; neither replaces the other.
json check(const json & expect, const json & state, const std::string & final_answer);

// Is this final answer plain text written to the user? False when it is empty, starts with
// "{" or "[", is JSON for an object or a list, or is a quoted JSON string holding one (the
// form a tool result has in the prompt). The same rule as runtime/agent/strict.py.
bool answer_is_text(const std::string & final_answer);

}  // namespace agent
