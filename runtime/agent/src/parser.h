// Reads tool calls out of what Llama 3.2 writes. The documented form is
//     {"name": "<tool>", "parameters": {...}}
// and several calls in one reply are joined with "; " (which llama-server's own parser
// rejects at the pinned commit). This parser accepts any separator, an optional leading
// <|python_tag|>, "arguments" in place of "parameters", and the two forms the 1B uses
// ({"function": "<tool>", ...} and {"function": {"name": ...}}).
//
// The base model writes most numbers as quoted strings and lists as strings. Where the
// tool's schema gives a type, such a value is converted to it ("12.5" -> 12.5, "[1, 2]" ->
// [1, 2], "true" -> true). A value already cut short by the model ("12" for 12.5) cannot be
// recovered here; forced JSON is what prevents that.
#pragma once

#include "tools.h"

#include <string>
#include <vector>

namespace specedge {

struct ToolCall {
    std::string name;
    json arguments;       // always an object
};

struct ParseResult {
    std::vector<ToolCall> calls;
    std::string text;                  // the reply with the calls taken out, trimmed
    bool broken = false;               // a call was started and never became valid JSON
    bool documented_format = true;     // every call used {"name": ..., "parameters": ...}
    int converted = 0;                 // argument values changed to the schema's type
};

// `tools` may be nullptr: then nothing is converted.
ParseResult parse_reply(const std::string & reply, const ToolSet * tools);

// Convert one value to the type its schema names, where that loses nothing. Counts changes.
json convert_to_schema(const json & value, const json & schema, int & converted);

}  // namespace specedge
