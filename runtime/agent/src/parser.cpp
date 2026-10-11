#include "parser.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>

namespace specedge {

namespace {

std::string trim(const std::string & text) {
    size_t begin = 0, end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) begin++;
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) end--;
    return text.substr(begin, end - begin);
}

void remove_all(std::string & text, const std::string & what) {
    for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at)) text.erase(at, what.size());
}

// Index one past the "}" that closes the object opening at `start`, or npos when the text
// ends first. Braces and brackets inside strings do not count.
size_t object_end(const std::string & text, size_t start) {
    int depth = 0;
    bool in_string = false;
    for (size_t i = start; i < text.size(); i++) {
        const char c = text[i];
        if (in_string) {
            if (c == '\\') i++;
            else if (c == '"') in_string = false;
        } else if (c == '"') {
            in_string = true;
        } else if (c == '{' || c == '[') {
            depth++;
        } else if (c == '}' || c == ']') {
            if (--depth == 0) return i + 1;
        }
    }
    return std::string::npos;
}

// The whole string as a number, or false.
bool as_number(const std::string & raw, json & out, bool want_integer) {
    const std::string text = trim(raw);
    if (text.empty() || !(std::isdigit(static_cast<unsigned char>(text[0])) || text[0] == '-' || text[0] == '+' || text[0] == '.'))
        return false;
    errno = 0;
    char * end = nullptr;
    const double number = std::strtod(text.c_str(), &end);
    if (errno != 0 || end != text.c_str() + text.size() || !std::isfinite(number)) return false;
    if (want_integer && number == std::floor(number) && std::fabs(number) < 9e15) out = static_cast<long long>(number);
    else out = number;
    return true;
}

std::string type_of(const json & schema) {
    if (!schema.is_object() || !schema.contains("type")) return "";
    const json & type = schema["type"];
    if (type.is_string()) return type.get<std::string>();
    if (type.is_array())                      // ["number", "null"]: the first type that is not null
        for (const auto & option : type)
            if (option.is_string() && option != "null") return option.get<std::string>();
    return "";
}

// Read one JSON object as a call. `documented` says whether it used the documented keys.
bool as_call(const json & value, ToolCall & call, bool & documented) {
    if (!value.is_object()) return false;
    const json * holder = &value;
    documented = true;
    if (value.contains("name") && value["name"].is_string()) {
        call.name = value["name"].get<std::string>();
    } else if (value.contains("function") && value["function"].is_string()) {
        call.name = value["function"].get<std::string>();
        documented = false;
    } else if (value.contains("function") && value["function"].is_object() && value["function"].contains("name") &&
               value["function"]["name"].is_string()) {
        call.name = value["function"]["name"].get<std::string>();
        documented = false;
        if (value["function"].contains("parameters") || value["function"].contains("arguments")) holder = &value["function"];
    } else {
        return false;
    }
    json arguments;
    if (holder->contains("parameters")) {
        arguments = (*holder)["parameters"];
    } else if (holder->contains("arguments")) {
        arguments = (*holder)["arguments"];
        documented = false;
    } else {
        return false;
    }
    if (arguments.is_string()) {              // the whole argument object written as a string
        json inner = json::parse(arguments.get<std::string>(), nullptr, false);
        if (inner.is_object()) arguments = inner;
    }
    if (arguments.is_null()) arguments = json::object();
    if (!arguments.is_object()) return false;
    call.arguments = arguments;
    return true;
}

bool mentions_call(const std::string & text, size_t start) {
    const std::string head = text.substr(start, 200);
    return head.find("\"name\"") != std::string::npos || head.find("\"function\"") != std::string::npos;
}

}  // namespace

json convert_to_schema(const json & value, const json & schema, int & converted) {
    const std::string type = type_of(schema);
    if (type == "number" || type == "integer") {
        json number;
        if (value.is_string() && as_number(value.get<std::string>(), number, type == "integer")) { converted++; return number; }
        if (type == "integer" && value.is_number_float()) {
            const double d = value.get<double>();
            if (d == std::floor(d) && std::fabs(d) < 9e15) return static_cast<long long>(d);
        }
        return value;
    }
    if (type == "boolean") {
        if (value.is_string()) {
            std::string text = trim(value.get<std::string>());
            for (auto & c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (text == "true" || text == "false") { converted++; return text == "true"; }
        }
        return value;
    }
    if (type == "string") {
        if (value.is_number()) { converted++; return value.dump(); }
        return value;
    }
    if (type == "array") {
        json list = value;
        if (value.is_string()) {
            const std::string text = trim(value.get<std::string>());
            json inner = text.rfind("[", 0) == 0 ? json::parse(text, nullptr, false) : json(json::value_t::discarded);
            if (!inner.is_array()) return value;
            list = inner;
            converted++;
        }
        if (list.is_array() && schema.contains("items"))
            for (auto & item : list) item = convert_to_schema(item, schema["items"], converted);
        return list;
    }
    if (type == "object") {
        json object = value;
        if (value.is_string()) {
            const std::string text = trim(value.get<std::string>());
            json inner = text.rfind("{", 0) == 0 ? json::parse(text, nullptr, false) : json(json::value_t::discarded);
            if (!inner.is_object()) return value;
            object = inner;
            converted++;
        }
        if (object.is_object() && schema.contains("properties") && schema["properties"].is_object())
            for (auto it = object.begin(); it != object.end(); ++it)
                if (schema["properties"].contains(it.key())) it.value() = convert_to_schema(it.value(), schema["properties"][it.key()], converted);
        return object;
    }
    return value;
}

ParseResult parse_reply(const std::string & reply, const ToolSet * tools) {
    std::string text = reply;
    for (const char * marker : {"<|python_tag|>", "<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"}) remove_all(text, marker);

    ParseResult result;
    std::string rest;                 // everything that is not a call
    size_t position = 0;
    while (true) {
        const size_t start = text.find('{', position);
        if (start == std::string::npos) break;
        const size_t end = object_end(text, start);
        json value = end == std::string::npos ? json(json::value_t::discarded)
                                              : json::parse(text.substr(start, end - start), nullptr, false);
        ToolCall call;
        bool documented = true;
        if (value.is_discarded()) {
            // Not JSON from this brace on. If it looks like a call, the reply is broken.
            if (mentions_call(text, start)) result.broken = true;
            rest += text.substr(position, start + 1 - position);
            position = start + 1;
            continue;
        }
        if (!as_call(value, call, documented)) {
            rest += text.substr(position, end - position);      // some other JSON: leave it in the text
            position = end;
            continue;
        }
        rest += text.substr(position, start - position);
        position = end;
        if (tools) {
            if (const json * properties = tools->properties(call.name)) {
                for (auto it = call.arguments.begin(); it != call.arguments.end(); ++it)
                    if (properties->contains(it.key())) it.value() = convert_to_schema(it.value(), (*properties)[it.key()], result.converted);
            }
        }
        result.documented_format = result.documented_format && documented;
        result.calls.push_back(std::move(call));
    }
    rest += text.substr(position);

    // What is left between calls is only separators when the reply was calls and nothing else.
    result.text = trim(rest);
    if (!result.calls.empty()) {
        bool only_separators = true;
        for (const char c : result.text)
            only_separators = only_separators && (c == ';' || c == ',' || std::isspace(static_cast<unsigned char>(c)));
        if (only_separators) result.text.clear();
    }
    return result;
}

}  // namespace specedge
