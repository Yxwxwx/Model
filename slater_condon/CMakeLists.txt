cmake_minimum_required(VERSION 3.28)

project(Edgerunner LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenMP
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# --------------------> MKL (required) <--------------------
if(NOT DEFINED MKLROOT)
    if(DEFINED ENV{MKLROOT})
        set(MKLROOT $ENV{MKLROOT})
    else()
        message(FATAL_ERROR "MKLROOT is not set. Please source the Intel oneAPI environment.")
    endif()
endif()
message(STATUS "MKL support enabled (MKLROOT=${MKLROOT}).")
set(MKLLIB "-L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")
include_directories(${MKLROOT}/include)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_executable(Edgerunner ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(Edgerunner PRIVATE
                      ${MKLLIB}
                      OpenMP::OpenMP_CXX
                      Threads::Threads  )

# ---- test ----
enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME n2_sto3g_fci
         COMMAND "${Python3_EXECUTABLE}"
         "${CMAKE_CURRENT_SOURCE_DIR}/test/run_test.py"
         WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
