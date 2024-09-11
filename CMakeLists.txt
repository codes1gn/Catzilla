cmake_minimum_required(VERSION 3.19)
project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
# this is set by Makefile automatically
# set(CUDA_COMPUTE_CAPABILITY 86)

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance

# -G and -lineinfo conflicts
# add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;-lineinfo>")
add_compile_options("$<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-src-in-ptx;-lineinfo>")
# add_compile_options("--ptxas-options=-v")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
# Configure the source file path to be compiled
add_subdirectory(${PROJECT_SOURCE_DIR}/references)

# Catzilla impls on various operators
add_subdirectory(${PROJECT_SOURCE_DIR}/recipes)

# ==============================================================
# ==   house-keeping code
# ==============================================================

# config clang-format target
file(GLOB_RECURSE ALL_SOURCE_FILES
 ${CMAKE_SOURCE_DIR}/recipes/*.cpp
 ${CMAKE_SOURCE_DIR}/recipes/*.h)

# Define CMAKE target for invoking clang-format
add_custom_command(
    OUTPUT clang_format_check_only
    COMMAND clang-format
    ARGS --dry-run -Werror ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format"
    VERBATIM
)

add_custom_command(
    OUTPUT clang_format_and_revise
    COMMAND clang-format
    ARGS -i -Werror ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format"
    VERBATIM
)

add_custom_target(
    format-code
    DEPENDS clang_format_and_revise
)

add_custom_target(
    check-code-format
    DEPENDS clang_format_check_only
)
