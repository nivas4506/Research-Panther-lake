cmake_minimum_required(VERSION 3.15)
project(panther_lake_simulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(panther_lake_lib src/cores.cpp src/gpu.cpp src/npu.cpp src/memory.cpp)

add_executable(panther_lake_sim main.cpp)
target_link_libraries(panther_lake_sim panther_lake_lib)

add_executable(test_runner tests/test_runner.cpp)
target_link_libraries(test_runner panther_lake_lib)
