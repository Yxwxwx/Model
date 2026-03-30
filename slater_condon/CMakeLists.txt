cmake_minimum_required(VERSION 3.28)

project(Edgerunner LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenMP
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# --------------------> MKL (required) <--------------------
message(STATUS "MKL support enabled.")
set(MKLLIB "-L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")
include_directories(${MKLROOT}/include)
# --------------------> fmt <--------------------
add_subdirectory(external/fmt)
# --------------------> END  <--------------------

file(GLOB_RECURSE src_lianlg "src/linalg/*.[ch]pp")
file(GLOB_RECURSE src_integral "src/integral/*.[ch]pp")
file(GLOB_RECURSE src_ci "src/ci/*.[ch]pp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_executable(Edgerunner ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp 
                ${src_lianlg} 
                ${src_integral} 
                ${src_ci})

target_link_libraries(Edgerunner PRIVATE
                      ${MKLLIB}
                      fmt::fmt
                      OpenMP::OpenMP_CXX 
                      Threads::Threads  )

