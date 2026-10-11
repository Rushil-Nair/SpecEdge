# The drafting policy (part 5's policy and part 8's fallback rule). Plain C++17, no llama.cpp:
#   cmake -S runtime/controller -B build_controller && cmake --build build_controller -j2
#   ./build_controller/test_policy runtime/controller/traces
#   ./build_controller/test_drafting
cmake_minimum_required(VERSION 3.16)
project(specedge_controller CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

add_library(specedge_policy STATIC policy.cpp drafting.cpp)
target_include_directories(specedge_policy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(policy_replay policy_replay.cpp)
target_link_libraries(policy_replay PRIVATE specedge_policy)

add_executable(test_policy tests/test_policy.cpp)
target_link_libraries(test_policy PRIVATE specedge_policy)
add_executable(test_drafting tests/test_drafting.cpp)
target_link_libraries(test_drafting PRIVATE specedge_policy)
enable_testing()
add_test(NAME policy COMMAND test_policy ${CMAKE_CURRENT_SOURCE_DIR}/traces)
add_test(NAME drafting COMMAND test_drafting)
