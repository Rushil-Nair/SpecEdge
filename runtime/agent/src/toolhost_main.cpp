// specedge-toolhost: the tool host on its own, for machines without a llama.cpp build.
//     specedge-toolhost [toolhost] --tools DIR --contacts FILE
// The optional first word lets scripts call it the way they call `specedge-agent toolhost`.
#include "host.h"

#include <cstdio>
#include <exception>
#include <string>

int main(int argc, char ** argv) {
    std::string tools, contacts;
    for (int i = 1; i < argc; i++) {
        const std::string word = argv[i];
        if (word == "toolhost") continue;
        if (word == "--tools" && i + 1 < argc) tools = argv[++i];
        else if (word == "--contacts" && i + 1 < argc) contacts = argv[++i];
        else { std::fprintf(stderr, "specedge-toolhost: unexpected argument %s\n", word.c_str()); return 1; }
    }
    if (tools.empty() || contacts.empty()) { std::fprintf(stderr, "usage: specedge-toolhost --tools DIR --contacts FILE\n"); return 1; }
    try {
        return specedge::run_toolhost(tools, contacts);
    } catch (const std::exception & problem) {
        std::fprintf(stderr, "specedge-toolhost: %s\n", problem.what());
        return 1;
    }
}
