cmake_minimum_required(VERSION 3.24)
project(MixedInt)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT mixedint_ipo_supported)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ${mixedint_ipo_supported})

option(MIXEDINT_FAST_MATH "Relax floating-point rules (may reduce integral accuracy)" ON)
option(MIXEDINT_BUILD_EXECUTABLE "Build the MixedInt executable" OFF)

# executable
set(mixedint_targets libMixedInt)
if(MIXEDINT_BUILD_EXECUTABLE)
  add_executable(MixedInt src/main.cpp)
  target_sources(MixedInt PRIVATE external/Faddeeva/Faddeeva.cc)
  target_include_directories(MixedInt PRIVATE external/Faddeeva)
  list(APPEND mixedint_targets MixedInt)
endif()

# library
find_package(pybind11 REQUIRED)
pybind11_add_module(libMixedInt src/pybind.cpp)
target_sources(libMixedInt PRIVATE external/Faddeeva/Faddeeva.cc)
target_include_directories(libMixedInt PRIVATE external/Faddeeva)
target_link_libraries(libMixedInt PRIVATE stdc++)

foreach(target IN LISTS mixedint_targets)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # These binaries are optimized for the CPU used to build them.
    target_compile_options(${target} PRIVATE
      $<$<CONFIG:Release>:-O3;-march=native;-mtune=native;-funroll-loops>)
    if(MIXEDINT_FAST_MATH)
      target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-ffast-math>)
      target_link_options(${target} PRIVATE $<$<CONFIG:Release>:-ffast-math>)
    endif()
  elseif(MSVC)
    target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/O2>)
    if(MIXEDINT_FAST_MATH)
      target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/fp:fast>)
    endif()
  endif()
endforeach()
