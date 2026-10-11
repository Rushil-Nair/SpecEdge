#include "host.h"

#include "grammar.h"
#include "parser.h"
#include "world.h"

#include <iostream>
#include <stdexcept>

namespace specedge {

std::vector<Message> messages_from(const json & list) {
    std::vector<Message> messages;
    for (const auto & message : list) {
        const json & content = message.at("content");
        messages.push_back({message.at("role").get<std::string>(), content.is_string() ? content.get<std::string>() : content.dump()});
    }
    return messages;
}

ToolResultStyle style_from(const std::string & name) {
    if (name == "stock") return ToolResultStyle::Stock;
    if (name == "plain") return ToolResultStyle::Plain;
    throw std::runtime_error("--tool-results must be stock or plain");
}

ToolPlacement placement_from(const std::string & name) {
    if (name == "user") return ToolPlacement::FirstUser;
    if (name == "system") return ToolPlacement::System;
    throw std::runtime_error("--tools-in must be user or system");
}

int run_toolhost(const std::string & tools_path, const std::string & contacts_path) {
    const ToolSet tools = load_tools(tools_path);
    agent::World world(json::parse(read_file(contacts_path)));
    for (std::string line; std::getline(std::cin, line);) {
        json reply;
        try {
            const json request = json::parse(line);
            const std::string op = request.at("op").get<std::string>();
            if (op == "reset") {
                world.reset(request.value("initial", json::object()));
                reply = {{"ok", true}};
            } else if (op == "parse") {
                const ParseResult parsed = parse_reply(request.at("text").get<std::string>(), &tools);
                json calls = json::array();
                for (const auto & call : parsed.calls) calls.push_back(json{{"name", call.name}, {"arguments", call.arguments}});
                reply = {{"calls", calls}, {"text", parsed.text}, {"broken", parsed.broken},
                         {"documented_format", parsed.documented_format}, {"converted", parsed.converted}};
            } else if (op == "call") {
                const json result = world.call(request.at("name").get<std::string>(), request.value("arguments", json::object()));
                reply = {{"result", result}, {"text", dump_spaced(result)}};
            } else if (op == "state") {
                reply = world.state();
            } else if (op == "check") {
                reply = agent::check(request.value("expect", json::object()), world.state(), request.value("final_answer", ""));
            } else if (op == "render") {
                reply = {{"prompt", render_prompt(tools, messages_from(request.at("messages")), request.value("date", today_date()),
                                                  style_from(request.value("tool_results", "stock")),
                                                  placement_from(request.value("tools_in", "user")))}};
            } else if (op == "grammar") {
                reply = {{"grammar", build_tool_grammar(tools)}};
            } else if (op == "tools") {
                reply = {{"tools", tools.tools}};
            } else {
                reply = {{"error", "unknown op: " + op}};
            }
        } catch (const std::exception & problem) {
            reply = {{"error", std::string("toolhost: ") + problem.what()}};
        }
        std::cout << reply.dump() << "\n" << std::flush;
    }
    return 0;
}

}  // namespace specedge
