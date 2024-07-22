cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#set(CUDA_ARCHITECTURES native)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "-G")
    set(CUDA_NVCC_FLAGS -G;-g)
    set(CMAKE_CUDA_FLAGS "-m64 -lineinfo")
endif ()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-suppress=177,550")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} -diag-suppress=177,550")

project(conv CUDA CXX C)

set(CUDA_TOOLKIT_PATH "/usr/local/cuda/")

set(SPUTNIK_ROOT "${CMAKE_SOURCE_DIR}/third_party/sputnik")
set(CUSPARSELT_ROOT "${CMAKE_SOURCE_DIR}/third_party/libcusparse_lt")
set(ARGPARSE_ROOT "${CMAKE_SOURCE_DIR}/third_party/argparse")
set(ARGPARSE_INCLUDE_DIR "${ARGPARSE_ROOT}/include")
set(CUTLASS_ROOT "${CMAKE_SOURCE_DIR}/third_party/cutlass")
set(cudnn_frontend_ROOT "${CMAKE_SOURCE_DIR}/third_party/cudnn-frontend")

add_subdirectory(src)