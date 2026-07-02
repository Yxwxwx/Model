cmake_minimum_required(VERSION 3.20)

project(gto_integral LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add source files
file(GLOB SOURCES "src/*.cpp" "src/*.hpp")

# Add the executable
add_executable(gto_integral ${SOURCES})
