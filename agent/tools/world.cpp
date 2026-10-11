#include "world.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <utility>

namespace agent {

namespace {

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
    return text;
}

std::string trim(const std::string & text) {
    size_t begin = 0, end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) begin++;
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) end--;
    return text.substr(begin, end - begin);
}

json error(const std::string & message) { return json{{"error", message}}; }

// "2026-11-12": four digits, two, two, and a month and day that can exist.
bool is_date(const std::string & text) {
    if (text.size() != 10 || text[4] != '-' || text[7] != '-') return false;
    for (size_t i : {0, 1, 2, 3, 5, 6, 8, 9})
        if (!std::isdigit(static_cast<unsigned char>(text[i]))) return false;
    int month = std::stoi(text.substr(5, 2)), day = std::stoi(text.substr(8, 2));
    return month >= 1 && month <= 12 && day >= 1 && day <= 31;
}

// "09:30" on a 24-hour clock.
bool is_time(const std::string & text) {
    if (text.size() != 5 || text[2] != ':') return false;
    for (size_t i : {0, 1, 3, 4})
        if (!std::isdigit(static_cast<unsigned char>(text[i]))) return false;
    return std::stoi(text.substr(0, 2)) <= 23 && std::stoi(text.substr(3, 2)) <= 59;
}

// Fetch a string argument; `problem` is filled when it is missing or not a string.
bool string_arg(const json & a, const char * key, std::string & out, std::string & problem) {
    if (!a.contains(key)) { problem = std::string("missing required argument: ") + key; return false; }
    if (!a[key].is_string()) { problem = std::string(key) + " must be a string"; return false; }
    out = a[key].get<std::string>();
    return true;
}

long cents(double amount) { return std::lround(amount * 100.0); }

bool contains_all(const std::string & haystack, const json & needles) {
    const std::string text = lower(haystack);
    if (!needles.is_array()) return true;
    for (const auto & needle : needles) {
        if (needle.is_array()) {             // a group: any one of these is enough
            bool any = false;
            for (const auto & option : needle) any = any || text.find(lower(option.get<std::string>())) != std::string::npos;
            if (!any) return false;
        } else if (text.find(lower(needle.get<std::string>())) == std::string::npos) {
            return false;
        }
    }
    return true;
}

// What makes two events the same event for the checker. The id is left out on purpose:
// an event that was cancelled and created again is the same calendar.
std::string event_key(const json & event) {
    std::vector<std::string> attendees;
    if (event.contains("attendee_ids"))
        for (const auto & id : event["attendee_ids"]) attendees.push_back(id.get<std::string>());
    std::sort(attendees.begin(), attendees.end());
    std::string key = lower(trim(event.value("title", ""))) + "|" + event.value("date", "") + "|" +
                      event.value("start_time", "") + "|" + std::to_string(cents(event.value("duration_hours", 0.0)));
    for (const auto & id : attendees) key += "|" + id;
    return key;
}

// Every expected item must be matched by its own actual item, and nothing may be left over.
template <typename Match>
bool same_multiset(const json & expected, const json & actual, Match match) {
    if (expected.size() != actual.size()) return false;
    std::vector<bool> used(actual.size(), false);
    for (const auto & want : expected) {
        bool found = false;
        for (size_t i = 0; i < actual.size() && !found; i++) {
            if (!used[i] && match(want, actual[i])) { used[i] = true; found = true; }
        }
        if (!found) return false;
    }
    return true;
}

}  // namespace

World::World(json contacts) : contacts_(std::move(contacts)) {}

void World::reset(const json & initial) {
    events_.clear();
    expenses_.clear();
    messages_.clear();
    called_.clear();
    next_event_ = 501;
    next_expense_ = 1;
    if (initial.contains("events"))
        for (const auto & event : initial["events"]) events_.push_back(event);
}

bool World::contact_exists(const std::string & id) const {
    for (const auto & contact : contacts_)
        if (contact["contact_id"] == id) return true;
    return false;
}

json World::call(const std::string & name, const json & arguments) {
    called_.push_back(name);
    if (!arguments.is_object()) return error("arguments must be a JSON object");
    if (name == "search_contacts") return search_contacts(arguments);
    if (name == "list_events") return list_events(arguments);
    if (name == "create_event") return create_event(arguments);
    if (name == "cancel_event") return cancel_event(arguments);
    if (name == "add_expense") return add_expense(arguments);
    if (name == "send_message") return send_message(arguments);
    return error("unknown tool: " + name);
}

json World::search_contacts(const json & a) const {
    std::string query, problem;
    if (!string_arg(a, "query", query, problem)) return error(problem);
    query = lower(trim(query));
    if (query.size() > 5 && query.compare(query.size() - 5, 5, " team") == 0) query.resize(query.size() - 5);
    json found = json::array();
    for (const auto & contact : contacts_) {
        const std::string name = lower(contact["name"].get<std::string>());
        if (!query.empty() && (name.find(query) != std::string::npos || contact["team"] == query)) found.push_back(contact);
    }
    return json{{"contacts", found}};
}

json World::list_events(const json & a) const {
    std::string date, problem;
    if (!string_arg(a, "date", date, problem)) return error(problem);
    if (!is_date(date)) return error("date must be written as YYYY-MM-DD");
    std::vector<json> found;
    for (const auto & event : events_)
        if (event["date"] == date) found.push_back(event);
    std::sort(found.begin(), found.end(), [](const json & x, const json & y) {
        return x["start_time"].get<std::string>() < y["start_time"].get<std::string>();
    });
    return json{{"date", date}, {"events", found}};
}

json World::create_event(const json & a) {
    std::string title, date, start, problem;
    if (!string_arg(a, "title", title, problem) || !string_arg(a, "date", date, problem) ||
        !string_arg(a, "start_time", start, problem))
        return error(problem);
    if (trim(title).empty()) return error("title must not be empty");
    if (!is_date(date)) return error("date must be written as YYYY-MM-DD");
    if (!is_time(start)) return error("start_time must be written as HH:MM on a 24-hour clock");
    if (!a.contains("duration_hours")) return error("missing required argument: duration_hours");
    if (!a["duration_hours"].is_number()) return error("duration_hours must be a number");
    const double hours = a["duration_hours"].get<double>();
    if (!(hours > 0.0) || hours > 12.0) return error("duration_hours must be more than 0 and at most 12");
    json attendees = json::array();
    if (a.contains("attendee_ids")) {
        if (!a["attendee_ids"].is_array()) return error("attendee_ids must be a list of contact ids");
        for (const auto & id : a["attendee_ids"]) {
            if (!id.is_string()) return error("attendee_ids must be a list of contact ids");
            if (!contact_exists(id.get<std::string>())) return error("no contact with id " + id.get<std::string>());
            attendees.push_back(id);
        }
    }
    const std::string id = "e_" + std::to_string(next_event_++);
    events_.push_back(json{{"event_id", id}, {"title", trim(title)}, {"date", date}, {"start_time", start},
                           {"duration_hours", hours}, {"attendee_ids", attendees}});
    return json{{"ok", true}, {"event_id", id}};
}

json World::cancel_event(const json & a) {
    std::string id, problem;
    if (!string_arg(a, "event_id", id, problem)) return error(problem);
    for (size_t i = 0; i < events_.size(); i++) {
        if (events_[i]["event_id"] == id) {
            const std::string title = events_[i]["title"].get<std::string>();
            events_.erase(events_.begin() + static_cast<long>(i));
            return json{{"ok", true}, {"cancelled", id}, {"title", title}};
        }
    }
    return error("no event with id " + id);
}

json World::add_expense(const json & a) {
    if (!a.contains("amount")) return error("missing required argument: amount");
    if (!a["amount"].is_number()) return error("amount must be a number");
    const double amount = a["amount"].get<double>();
    if (!(amount > 0.0)) return error("amount must be more than 0");
    std::string category, problem, note;
    if (!string_arg(a, "category", category, problem)) return error(problem);
    static const char * const categories[] = {"travel", "meals", "equipment", "software", "other"};
    if (std::find(std::begin(categories), std::end(categories), category) == std::end(categories))
        return error("category must be one of travel, meals, equipment, software, other");
    if (a.contains("note")) {
        if (!a["note"].is_string()) return error("note must be a string");
        note = a["note"].get<std::string>();
    }
    const std::string id = "x_" + std::to_string(next_expense_++);
    expenses_.push_back(json{{"expense_id", id}, {"amount", amount}, {"category", category}, {"note", note}});
    return json{{"ok", true}, {"expense_id", id}};
}

json World::send_message(const json & a) {
    std::string contact, text, problem;
    if (!string_arg(a, "contact_id", contact, problem) || !string_arg(a, "text", text, problem)) return error(problem);
    if (!contact_exists(contact)) return error("no contact with id " + contact);
    if (trim(text).empty()) return error("text must not be empty");
    messages_.push_back(json{{"contact_id", contact}, {"text", text}});
    return json{{"ok", true}, {"sent_to", contact}};
}

json World::state() const {
    return json{{"events", events_}, {"expenses", expenses_}, {"messages", messages_}, {"called", called_}};
}

bool answer_is_text(const std::string & final_answer) {
    const std::string text = trim(final_answer);
    if (text.empty() || text[0] == '{' || text[0] == '[') return false;
    const json value = json::parse(text, nullptr, false);
    if (value.is_discarded()) return true;
    if (value.is_object() || value.is_array()) return false;
    if (value.is_string()) {
        const std::string inner = trim(value.get<std::string>());
        if (!inner.empty() && (inner[0] == '{' || inner[0] == '[')) {
            const json held = json::parse(inner, nullptr, false);
            return !(held.is_object() || held.is_array());
        }
    }
    return true;
}

json check(const json & expect, const json & state, const std::string & final_answer) {
    std::vector<std::string> want, have;
    for (const auto & event : expect.value("events", json::array())) want.push_back(event_key(event));
    for (const auto & event : state.value("events", json::array())) have.push_back(event_key(event));
    std::sort(want.begin(), want.end());
    std::sort(have.begin(), have.end());
    const bool events_ok = want == have;

    const bool expenses_ok = same_multiset(
        expect.value("expenses", json::array()), state.value("expenses", json::array()),
        [](const json & w, const json & h) {
            return cents(w["amount"].get<double>()) == cents(h["amount"].get<double>()) && w["category"] == h["category"] &&
                   contains_all(h.value("note", ""), w.value("note_contains", json::array()));
        });

    const bool messages_ok = same_multiset(
        expect.value("messages", json::array()), state.value("messages", json::array()),
        [](const json & w, const json & h) {
            return w["contact_id"] == h["contact_id"] && contains_all(h.value("text", ""), w.value("contains", json::array()));
        });

    const bool answer_ok = contains_all(final_answer, expect.value("answer_contains", json::array()));

    bool tools_ok = true;
    const json called = state.value("called", json::array());
    for (const auto & tool : expect.value("required_tools", json::array()))
        tools_ok = tools_ok && std::find(called.begin(), called.end(), tool) != called.end();

    const bool correct = events_ok && expenses_ok && messages_ok && answer_ok && tools_ok;
    const bool is_text = answer_is_text(final_answer);
    return json{{"correct", correct}, {"correct_strict", correct && is_text},
                {"events_ok", events_ok}, {"expenses_ok", expenses_ok}, {"messages_ok", messages_ok},
                {"answer_ok", answer_ok}, {"tools_ok", tools_ok}, {"answer_is_text", is_text}};
}

}  // namespace agent
