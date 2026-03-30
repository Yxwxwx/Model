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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
# --------------------> hptt <--------------------
# add_library(hptt STATIC IMPORTED)
# set_target_properties(hptt PROPERTIES IMPORTED_LOCATION
#     ${CMAKE_CURRENT_SOURCE_DIR}/external/hptt/build/libhptt.a)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/hptt/include)
add_subdirectory(external/hptt)
set_target_properties(hptt PROPERTIES IMPORTED_LOCATION
   ${CMAKE_CURRENT_SOURCE_DIR}/build/hptt/build/libhptt.a)
# Link hptt library
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/hptt/include)

# List all header-only sources
set(src
    src/mps.hpp
    src/mpo.hpp
    # src/dmrg.hpp
    # Add other .hpp files here as needed
)

add_executable(IDMRG main.cpp ${src})
target_link_libraries(IDMRG PRIVATE OpenMP::OpenMP_CXX Threads::Threads fmt::fmt hptt)