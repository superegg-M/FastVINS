cmake_minimum_required(VERSION 3.16)
project(GraphOptimization)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

include_directories("${GRAPH_OPTIMIZATION_SOURCE_DIR}/lib/eigen3")

set(GRAPH_OPTIMIZATION_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "grath optimization location" FORCE)

include_directories("${GRAPH_OPTIMIZATION_SOURCE_DIR}/lib")
include_directories("${GRAPH_OPTIMIZATION_SOURCE_DIR}/thirdparty/Sophus")

add_subdirectory(lib)
add_subdirectory(modules)

find_package (OpenMP REQUIRED)

target_link_libraries(backend so_plus tic_toc OpenMP::OpenMP_CXX)

target_link_libraries(fdsi backend so_plus tic_toc OpenMP::OpenMP_CXX)

target_link_libraries(vo_test backend so_plus tic_toc OpenMP::OpenMP_CXX)

#add_executable(GraphOptimization main.cpp)
#add_executable(GraphOptimization app/system_identification.cpp)
#add_executable(GraphOptimization app/vio_visual_test.cpp)
#add_executable(GraphOptimization app/vio_inertial_init.cpp)
#add_executable(GraphOptimization app/vio_init.cpp)
#add_executable(GraphOptimization app/vio_test.cpp)
add_executable(GraphOptimization app/vio_test2.cpp)
# add_executable(init_test1 app/vio_init_simulator.cpp)
# add_executable(init_test app/vio_init.cpp)

target_link_libraries(GraphOptimization fdsi backend so_plus tic_toc vo_test OpenMP::OpenMP_CXX)
# target_link_libraries(init_test fdsi backend so_plus tic_toc vo_test )
# target_link_libraries(init_test1 fdsi backend so_plus tic_toc vo_test)
