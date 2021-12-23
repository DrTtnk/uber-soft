cmake_minimum_required(VERSION 3.16)
project(cpp_to_python)

set(CMAKE_C_STANDARD 20)

add_subdirectory(pybind11)
add_subdirectory(glm)

pybind11_add_module(cpp_to_python base.cpp)