cmake_minimum_required(VERSION 3.18)
project(cgal_repair LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(CGAL REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cgal_repair src/cgal_repair.cpp)
target_link_libraries(cgal_repair PRIVATE CGAL::CGAL)
install(TARGETS cgal_repair DESTINATION .)