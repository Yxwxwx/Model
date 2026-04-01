cmake_minimum_required(VERSION 3.24)
project(MixedInt)

set(CMAKE_CXX_STANDARD 23)

# executable
add_executable(MixedInt src/main.cpp)
target_sources(MixedInt PRIVATE external/Faddeeva/Faddeeva.cc)
target_include_directories(MixedInt PRIVATE external/Faddeeva)

# library
find_package(pybind11 REQUIRED)
pybind11_add_module(libMixedInt src/pybind.cpp)
target_sources(libMixedInt PRIVATE external/Faddeeva/Faddeeva.cc)
target_include_directories(libMixedInt PRIVATE external/Faddeeva)
target_link_libraries(libMixedInt PRIVATE stdc++)