cmake_minimum_required(VERSION 3.15)
project(panther_lake_simulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(panther_lake_sim main.cpp)
add_executable(test_runner tests/test_runner.cpp)
