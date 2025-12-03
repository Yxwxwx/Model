cmake_minimum_required(VERSION 3.28)
project(CUBLAS_DEMO LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_compile_options(-Wall -Wextra)

find_package(CUDAToolkit REQUIRED)

add_executable(cublas_demo src/cublas.cu)

target_link_libraries(cublas_demo
    PRIVATE
        CUDA::cublas
        CUDA::cudart
)

set_target_properties(cublas_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 89
)