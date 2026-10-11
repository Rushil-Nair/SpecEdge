#include "tools.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace specedge {

std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

std::string dump_spaced(const json & value) {
    if (value.is_object()) {
        std::string out = "{";
        for (auto it = value.begin(); it != value.end(); ++it)
            out += (out.size() > 1 ? ", " : "") + json(it.key()).dump() + ": " + dump_spaced(it.value());
        return out + "}";
    }
    if (value.is_array()) {
        std::string out = "[";
        for (const auto & item : value) out += (out.size() > 1 ? ", " : "") + dump_spaced(item);
        return out + "]";
    }
    return value.dump();
}

const json * ToolSet::find(const std::string & name) const {
    for (const auto & tool : tools)
        if (tool["function"]["name"] == name) return &tool["function"];
    return nullptr;
}

const json * ToolSet::properties(const std::string & name) const {
    const json * function = find(name);
    if (!function || !function->contains("parameters")) return nullptr;
    const json & parameters = (*function)["parameters"];
    if (!parameters.is_object() || !parameters.contains("properties") || !parameters["properties"].is_object()) return nullptr;
    return &parameters["properties"];
}

static void add_tool(ToolSet & set, const json & tool, const std::string & path) {
    if (!tool.is_object() || !tool.contains("function") || !tool["function"].is_object() ||
        !tool["function"].contains("name") || !tool["function"]["name"].is_string())
        throw std::runtime_error(path + ": not a tool (expected {\"type\": \"function\", \"function\": {\"name\": ...}})");
    if (set.find(tool["function"]["name"].get<std::string>()))
        throw std::runtime_error(path + ": tool name used twice: " + tool["function"]["name"].get<std::string>());
    set.tools.push_back(tool);
}

ToolSet load_tools(const std::string & path) {
    namespace fs = std::filesystem;
    std::vector<std::string> files;
    if (fs::is_directory(path)) {
        for (const auto & entry : fs::directory_iterator(path))
            if (entry.path().extension() == ".json") files.push_back(entry.path().string());
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(path);
    }
    ToolSet set;
    for (const auto & file : files) {
        const json parsed = json::parse(read_file(file));
        if (parsed.is_array()) {
            for (const auto & tool : parsed) add_tool(set, tool, file);
        } else {
            add_tool(set, parsed, file);
        }
    }
    return set;
}

}  // namespace specedge
