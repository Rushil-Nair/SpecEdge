// Tool schemas, loaded from JSON files. Each file holds one tool in the OpenAI shape
// {"type": "function", "function": {"name", "description", "parameters"}}, or a list of them.
#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace specedge {

using json = nlohmann::ordered_json;   // ordered: the prompt must list keys as the file does

struct ToolSet {
    std::vector<json> tools;

    // The "function" object of a tool, or nullptr.
    const json * find(const std::string & name) const;
    // The "properties" object of a tool's parameters, or nullptr.
    const json * properties(const std::string & name) const;
};

// Load every *.json file of a directory in file-name order, or a single file.
// Throws std::runtime_error when a file is not a tool.
ToolSet load_tools(const std::string & path);

std::string read_file(const std::string & path);

// JSON on one line with ", " and ": " between items, the way Llama 3.2 writes JSON itself
// and the way its chat template writes a tool result.
std::string dump_spaced(const json & value);

}  // namespace specedge
