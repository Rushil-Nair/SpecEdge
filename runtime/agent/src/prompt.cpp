#include "prompt.h"

#include <cctype>
#include <ctime>
#include <stdexcept>

namespace specedge {

namespace {

// Jinja's trim: whitespace off both ends.
std::string trim(const std::string & text) {
    size_t begin = 0, end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) begin++;
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) end--;
    return text.substr(begin, end - begin);
}

const char * const EOT = "<|eot_id|>";

std::string header(const std::string & role) { return "<|start_header_id|>" + role + "<|end_header_id|>\n\n"; }

}  // namespace

std::string today_date() {
    std::time_t now = std::time(nullptr);
    char buffer[32];
    std::strftime(buffer, sizeof buffer, "%d %b %Y", std::localtime(&now));
    return buffer;
}

std::string render_prompt(const ToolSet & tools, const std::vector<Message> & messages, const std::string & date,
                          ToolResultStyle tool_results, ToolPlacement placement) {
    const bool have_tools = !tools.tools.empty();
    size_t next = 0;
    std::string system_message;
    if (!messages.empty() && messages[0].role == "system") {
        system_message = trim(messages[0].content);
        next = 1;
    }

    std::string out = "<|begin_of_text|>" + header("system");
    if (have_tools) out += "Environment: ipython\n";
    out += "Cutting Knowledge Date: December 2023\n";
    out += "Today Date: " + date + "\n\n";
    if (have_tools && placement == ToolPlacement::System) {
        out += "You have access to the following functions. To call a function, please respond with JSON for a function call."
               "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}."
               "Do not use variables.\n\n";
        for (const auto & tool : tools.tools) out += tool.dump(4) + "\n\n";
    }
    out += system_message;
    out += EOT;

    if (have_tools && placement == ToolPlacement::FirstUser) {
        if (next >= messages.size()) throw std::runtime_error("tools need a first user message to be placed in");
        out += header("user");
        out += "Given the following functions, please respond with a JSON for a function call "
               "with its proper arguments that best answers the given prompt.\n\n"
               "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}."
               "Do not use variables.\n\n";
        for (const auto & tool : tools.tools) out += tool.dump(4) + "\n\n";
        out += trim(messages[next].content);
        out += EOT;
        next++;
    }

    for (; next < messages.size(); next++) {
        const Message & message = messages[next];
        if (message.role == "tool" || message.role == "ipython") {
            out += header("ipython");
            out += tool_results == ToolResultStyle::Stock ? json(message.content).dump() : message.content;
            out += EOT;
        } else {
            out += header(message.role) + trim(message.content) + EOT;
        }
    }
    return out + header("assistant");
}

}  // namespace specedge
