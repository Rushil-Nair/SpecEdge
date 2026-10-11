// Forced JSON: a GBNF grammar built from the tool schemas, for llama.cpp's grammar sampler.
//
// Under it a reply is either
//   - one or more calls, each exactly {"name": "<a listed tool>", "parameters": {...}},
//     joined by "; " the way Llama 3.2 writes several calls; or
//   - plain text that does not start with "{" (an answer, a question back, a refusal).
//
// Inside "parameters" only that tool's arguments can appear, in the order the schema lists
// them, each at most once, every required one present, and every value in the JSON type the
// schema gives: a "number" is a bare number, never a quoted string. That last point is what
// stops the base model's habit of writing "12" for 12.5.
//
// Supported schema keywords: type (also a list of types), enum, const, properties, required,
// items, anyOf, oneOf. Anything else is accepted as any JSON value.
#pragma once

#include "tools.h"

#include <string>

namespace specedge {

struct GrammarOptions {
    bool allow_text = true;        // false: the reply must be a call
    bool allow_several = true;     // false: exactly one call
};

std::string build_tool_grammar(const ToolSet & tools, const GrammarOptions & options = {});

// A GBNF string literal matching exactly `text`.
std::string gbnf_literal(const std::string & text);

}  // namespace specedge
