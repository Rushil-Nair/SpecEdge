# The part 6 agent runtime. Builds against an existing llama.cpp checkout and its build
# directory (shared libraries), at the pinned commit f498f864:
#   cmake -S runtime/agent -B build -DLLAMA_DIR=/workspace/llama.cpp && cmake --build build -j4
cmake_minimum_required(VERSION 3.16)
project(specedge_agent CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(LLAMA_DIR "/workspace/llama.cpp" CACHE PATH "llama.cpp source checkout")
set(LLAMA_BUILD_DIR "${LLAMA_DIR}/build" CACHE PATH "llama.cpp build directory")
set(AGENT_TOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../agent/tools" CACHE PATH "the in-memory tools")
set(CONTROLLER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../controller" CACHE PATH "the drafting policy and loop")

execute_process(COMMAND git -C ${LLAMA_DIR} rev-parse HEAD
                OUTPUT_VARIABLE LLAMA_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT LLAMA_COMMIT)
    set(LLAMA_COMMIT "unknown")
endif()
message(STATUS "llama.cpp commit: ${LLAMA_COMMIT}")

add_compile_options(-Wall -Wextra)

# Everything that needs no model: schemas, prompt, parser, grammar, tools, checker.
add_library(specedge_core STATIC
    src/tools.cpp src/prompt.cpp src/parser.cpp src/grammar.cpp src/host.cpp src/policy_config.cpp
    ${AGENT_TOOLS_DIR}/world.cpp
    ${CONTROLLER_DIR}/policy.cpp ${CONTROLLER_DIR}/drafting.cpp)
target_include_directories(specedge_core PUBLIC src ${AGENT_TOOLS_DIR} ${CONTROLLER_DIR})
target_include_directories(specedge_core SYSTEM PUBLIC ${LLAMA_DIR}/vendor)

# -DSPECEDGE_WITH_LLAMA=OFF builds only what needs no model (unit checks and the tool host);
# LLAMA_DIR is then used for its vendored JSON header alone and need not be built.
option(SPECEDGE_WITH_LLAMA "build specedge-agent against a llama.cpp build" ON)
if(SPECEDGE_WITH_LLAMA)
    add_executable(specedge-agent src/main.cpp src/engine.cpp src/engine_draft.cpp src/loop.cpp)
    target_compile_definitions(specedge-agent PRIVATE SPECEDGE_LLAMA_COMMIT="${LLAMA_COMMIT}")
    target_include_directories(specedge-agent SYSTEM PRIVATE ${LLAMA_DIR}/include ${LLAMA_DIR}/ggml/include)
    target_link_directories(specedge-agent PRIVATE ${LLAMA_BUILD_DIR}/bin)
    target_link_libraries(specedge-agent PRIVATE specedge_core llama ggml ggml-base)
    set_target_properties(specedge-agent PROPERTIES BUILD_RPATH "${LLAMA_BUILD_DIR}/bin")
endif()

add_executable(specedge-toolhost src/toolhost_main.cpp)
target_link_libraries(specedge-toolhost PRIVATE specedge_core)

add_executable(test_units tests/test_units.cpp)
target_link_libraries(test_units PRIVATE specedge_core)
enable_testing()
add_test(NAME units COMMAND test_units ${AGENT_TOOLS_DIR})
