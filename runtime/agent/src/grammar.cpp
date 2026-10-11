#include "grammar.h"

#include <vector>

namespace specedge {

namespace {

const char * const COMMON_RULES = R"GBNF(string ::= "\"" char* "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt/] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
integer ::= "-"? ("0" | [1-9] [0-9]*)
number ::= integer ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
boolean ::= "true" | "false"
null ::= "null"
value ::= string | number | boolean | null | array | object
array ::= "[" ( value (", " value)* )? "]"
object ::= "{" ( string ": " value (", " string ": " value)* )? "}"
text ::= [^{ \t\n] textchar*
textchar ::= [^\x00]
)GBNF";

class Builder {
public:
    // The rule (a name, or a bracketed expression) matching one value of this schema.
    std::string value_rule(const json & schema, const std::string & hint) {
        if (!schema.is_object()) return "value";
        if (schema.contains("const")) return gbnf_literal(schema["const"].dump());
        if (schema.contains("enum") && schema["enum"].is_array() && !schema["enum"].empty()) {
            std::string options;
            for (const auto & option : schema["enum"]) options += (options.empty() ? "" : " | ") + gbnf_literal(option.dump());
            return "(" + options + ")";
        }
        for (const char * key : {"anyOf", "oneOf"}) {
            if (schema.contains(key) && schema[key].is_array() && !schema[key].empty()) {
                std::string options;
                int index = 0;
                for (const auto & option : schema[key])
                    options += (options.empty() ? "" : " | ") + value_rule(option, hint + "-o" + std::to_string(index++));
                return "(" + options + ")";
            }
        }
        if (!schema.contains("type")) return "value";
        const json & type = schema["type"];
        if (type.is_array()) {
            std::string options;
            for (const auto & one : type) {
                json narrowed = schema;
                narrowed["type"] = one;
                options += (options.empty() ? "" : " | ") + value_rule(narrowed, hint);
            }
            return options.empty() ? "value" : "(" + options + ")";
        }
        if (!type.is_string()) return "value";
        const std::string name = type.get<std::string>();
        if (name == "string" || name == "integer" || name == "number" || name == "boolean" || name == "null") return name;
        if (name == "array") {
            if (!schema.contains("items")) return "array";
            const std::string item = value_rule(schema["items"], hint + "-item");
            return add(hint + "-list", "\"[\" ( " + item + " (\", \" " + item + ")* )? \"]\"");
        }
        if (name == "object") {
            if (!schema.contains("properties") || !schema["properties"].is_object()) return "object";
            return add(hint + "-obj", "\"{\" " + members(schema, hint) + " \"}\"");
        }
        return "value";
    }

    // What may stand between the braces of an object with these properties: each property
    // at most once, in schema order, required ones always.
    std::string members(const json & schema, const std::string & hint) {
        if (!schema.is_object() || !schema.contains("properties") || !schema["properties"].is_object() ||
            schema["properties"].empty())
            return "\"\"";
        const json & properties = schema["properties"];
        std::vector<std::string> pairs;
        std::vector<bool> required;
        int index = 0;
        for (auto it = properties.begin(); it != properties.end(); ++it, ++index) {
            const std::string own = hint + "-p" + std::to_string(index);
            pairs.push_back(add(own, gbnf_literal(json(it.key()).dump() + ": ") + " " + value_rule(it.value(), own)));
            bool needed = false;
            if (schema.contains("required") && schema["required"].is_array())
                for (const auto & key : schema["required"]) needed = needed || key == it.key();
            required.push_back(needed);
        }
        const int count = static_cast<int>(pairs.size());
        // tail[i]: properties i.. when something was already written (each starts with ", ").
        // head[i]: properties i.. when nothing was written yet. "" stands for nothing at all.
        std::vector<std::string> tail(count + 1), head(count + 1);
        for (int i = count - 1; i >= 0; i--) {
            const std::string after = tail[i + 1].empty() ? "" : " " + tail[i + 1];
            tail[i] = add(hint + "-t" + std::to_string(i),
                          required[i] ? "\", \" " + pairs[i] + after : "(\", \" " + pairs[i] + ")?" + after);
            std::string body;
            if (required[i]) body = pairs[i] + after;
            else if (head[i + 1].empty()) body = "(" + pairs[i] + after + ")?";
            else body = pairs[i] + after + " | " + head[i + 1];
            head[i] = add(hint + "-h" + std::to_string(i), body);
        }
        return head[0];
    }

    std::string add(const std::string & name, const std::string & body) {
        rules_ += name + " ::= " + body + "\n";
        return name;
    }

    const std::string & rules() const { return rules_; }

private:
    std::string rules_;
};

}  // namespace

std::string gbnf_literal(const std::string & text) {
    std::string out = "\"";
    for (const char c : text) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out + "\"";
}

std::string build_tool_grammar(const ToolSet & tools, const GrammarOptions & options) {
    if (tools.tools.empty()) return std::string("root ::= text\n") + COMMON_RULES;

    Builder builder;
    std::string calls;
    for (size_t i = 0; i < tools.tools.size(); i++) {
        const json & function = tools.tools[i]["function"];
        const std::string hint = "t" + std::to_string(i);
        const json parameters = function.contains("parameters") ? function["parameters"] : json::object();
        const std::string opening = "{\"name\": " + json(function["name"]).dump() + ", \"parameters\": {";
        builder.add(hint + "-call", gbnf_literal(opening) + " " + builder.members(parameters, hint) + " \"}}\"");
        calls += (calls.empty() ? "" : " | ") + hint + "-call";
    }
    std::string root = options.allow_several ? "call (\"; \" call)*" : "call";
    if (options.allow_text) root += " | text";
    return "root ::= " + root + "\ncall ::= " + calls + "\n" + builder.rules() + COMMON_RULES;
}

}  // namespace specedge
