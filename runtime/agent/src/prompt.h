// Builds the prompt text in Llama 3.2's own chat format, following the chat template that
// ships with meta-llama/Llama-3.2-3B-Instruct (revision 0cb88a4f): the tool list goes into
// the first user message, tool results are "ipython" turns.
#pragma once

#include "tools.h"

#include <string>
#include <vector>

namespace specedge {

struct Message {
    std::string role;      // "system", "user", "assistant" or "tool"
    std::string content;   // an assistant turn holds the model's reply exactly as it wrote it
};

// How a tool result is written into its "ipython" turn.
//   Stock: as the template writes a string result, and so as stock llama-server does: one
//          JSON string, quoted and escaped ("{\"ok\": true}"). llama-server at the pinned
//          commit refuses a result given as an object, so this is the only form it produces.
//   Plain: the result's JSON as it is ({"ok": true}), which is what the template writes
//          when the result is an object.
enum class ToolResultStyle { Stock, Plain };

// Where the tool list goes. The template has both branches, chosen by its variable
// tools_in_user_message (true unless the caller sets it):
//   FirstUser: in the first user message, under "Given the following functions, please
//              respond with a JSON for a function call ...". The default everywhere else in
//              this project, and what the fine-tuning data uses.
//   System:    in the system message, under "You have access to the following functions. To
//              call a function, please respond with JSON for a function call."
enum class ToolPlacement { FirstUser, System };

// `date` is the template's "Today Date" line, written like "08 Oct 2026". The result starts
// with <|begin_of_text|> and ends with the assistant header, ready for the model to continue.
// Throws std::runtime_error when tools are given and there is no user message to carry them.
std::string render_prompt(const ToolSet & tools, const std::vector<Message> & messages, const std::string & date,
                          ToolResultStyle tool_results = ToolResultStyle::Stock,
                          ToolPlacement placement = ToolPlacement::FirstUser);

// Today's date the way the template writes it ("%d %b %Y").
std::string today_date();

}  // namespace specedge
