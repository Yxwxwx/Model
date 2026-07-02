cmake_minimum_required(VERSION 3.28)

project(IDMRG LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_definitions(-DEIGEN_USE_THREADS)

# Set the C++ compiler flags
# Enable instruction set optimizations and extra warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -funroll-loops")

# Find OpenMP
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
# --------------------> Eigen3 <--------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/eigen3)
# --------------------> fmt <--------------------
add_subdirectory(external/fmt)
# --------------------> hptt <--------------------
# HPTT is required but not vendored. Install from:
#   https://github.com/springer13/hptt
# After installation, uncomment and set HPTT_ROOT:
#   include_directories(${HPTT_ROOT}/include)
#   target_link_libraries(IDMRG PRIVATE ${HPTT_ROOT}/lib/libhptt.a)

# List all header-only sources
set(src
    src/mps.hpp
    src/mpo.hpp
    # src/dmrg.hpp
    # Add other .hpp files here as needed
)

add_executable(IDMRG main.cpp ${src})
target_link_libraries(IDMRG PRIVATE OpenMP::OpenMP_CXX Threads::Threads fmt::fmt)