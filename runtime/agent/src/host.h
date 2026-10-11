// The tool host: the tools, the parser, the prompt builder, the grammar and the checker served
// over stdin and stdout, one JSON object per line. It needs no model and no llama.cpp, so the
// Python control and verify_tasks.py use exactly the runtime's code.
#pragma once

#include "prompt.h"
#include "tools.h"

#include <string>
#include <vector>

namespace specedge {

// Requests: {"op": "reset" | "parse" | "call" | "state" | "check" | "render" | "grammar" | "tools", ...}.
int run_toolhost(const std::string & tools_path, const std::string & contacts_path);

std::vector<Message> messages_from(const json & list);
ToolResultStyle style_from(const std::string & name);       // "stock" or "plain"
ToolPlacement placement_from(const std::string & name);     // "user" or "system"

}  // namespace specedge
