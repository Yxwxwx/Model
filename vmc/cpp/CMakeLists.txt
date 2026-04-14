CMAKE_minimum_required(VERSION 3.28)

set(PROJECT_NAME "VMC")
set(PROJECT_VERSION "0.1.0")

project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
add_compile_definitions(_DEBUG)
endif()

# Set OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(OMP_FLAG ${OpenMP_CXX_FLAGS})
    set(OMP_LIB_NAME ${OpenMP_CXX_LIBRARIES})
else()
    message(FATAL_ERROR "OpenMP not found. Please install OpenMP or set CMAKE_CXX_COMPILER to a compiler that supports OpenMP.")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OPT_FLAG -O0 -g ${OMP_FLAG} -march=native -mtune=native -fno-omit-frame-pointer -fno-optimize-sibling-calls -Werror -Werror=return-type)
else()
    set(OPT_FLAG -O3 -funroll-loops ${OMP_FLAG} -march=native -mtune=native -Werror -Werror=return-type -Wno-unused-command-line-argument -Wno-tautological-constant-compare)
endif()

# TBB
add_compile_definitions(_USE_TBB_)
find_path(TBB_INCLUDE_DIR NAMES tbb/tbb.h HINTS $ENV{TBBROOT}/include)
find_library(TBB_LIB NAMES tbb PATHS $ENV{TBBROOT}/lib )
find_library(TBB_LIB_MAL NAMES tbbmalloc PATHS $ENV{TBBROOT}/lib )
find_library(TBB_LIBS_MALP NAMES tbbmalloc_proxy PATHS $ENV{TBBROOT}/lib )
set(TBB_LIBS ${TBB_LIB} ${TBB_LIB_MAL} ${TBB_LIBS_MALP})
include_directories(${TBB_INCLUDE_DIR})
set(MEMORY_ALLOCATOR_LIBS
${TBB_LIB}
${TBB_LIB_MAL}
${TBB_LIB_MALP}
)

# MKL
add_compile_definitions(_USE_MKL_)
include_directories(${MKLROOT}/include)
set(BLAS_LIBS   -m64  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)

# mdspan
include_directories(${PROJECT_SOURCE_DIR}/external/mdspan/mdspan-stable/include)

add_executable(${PROJECT_NAME}_exe src/main.cpp)
target_compile_options(${PROJECT_NAME}_exe PRIVATE ${OPT_FLAG})
target_link_libraries(${PROJECT_NAME}_exe PRIVATE ${BLAS_LIBS} ${OMP_LIB_NAME} ${MEMORY_ALLOCATOR_LIBS})