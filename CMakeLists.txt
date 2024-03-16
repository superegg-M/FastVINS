cmake_minimum_required(VERSION 3.16)
project(GraphOptimization)

set(CMAKE_CXX_STANDARD 14)

include_directories("E:\\CLionProjects\\GraphOptimization\\lib\\eigen3")

set(GRAPH_OPTIMIZATION_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "grath optimization location" FORCE)

include_directories("${GRAPH_OPTIMIZATION_SOURCE_DIR}/lib")
include_directories("${GRAPH_OPTIMIZATION_SOURCE_DIR}/thirdparty/Sophus")

add_subdirectory(lib)
add_subdirectory(modules)

add_executable(GraphOptimization main.cpp)
target_link_libraries(GraphOptimization PUBLIC backend fdsi test_ba vins)