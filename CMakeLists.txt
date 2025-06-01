cmake_minimum_required(VERSION 3.19)
project(CATZILLA LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# config catch2 for testing
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.3.0
)
FetchContent_MakeAvailable(Catch2)
include_directories(${catch2_SOURCE_DIR}/single_include)

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

# Set CUDA compute capability
set(CUDA_COMPUTE_CAPABILITY "80")  # Set to 80 for Ampere architecture (A100, A6000, etc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)  # need type_traits
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance

# -G and -lineinfo conflicts
# add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;-lineinfo>")
add_compile_options("$<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-src-in-ptx;-lineinfo>")
# add_compile_options("--ptxas-options=-v")

# Add CUDA relaxed constexpr flag
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/catz)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/recipes)

# References and version-3/6 are just for reference, not for compilation
# add_subdirectory(${PROJECT_SOURCE_DIR}/references)

# Catzilla impls on various operators
add_subdirectory(${PROJECT_SOURCE_DIR}/catz)
add_subdirectory(${PROJECT_SOURCE_DIR}/recipes)
add_subdirectory(${PROJECT_SOURCE_DIR}/benchmark)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)

# ==============================================================
# ==   house-keeping code
# ==============================================================

# config clang-format target
# Catz: cuda-enhanced programming model for catzilla
# Recipes: kernel/operator recipes for computing
file(GLOB_RECURSE ALL_SOURCE_FILES
 ${CMAKE_SOURCE_DIR}/catz/*.cpp
 ${CMAKE_SOURCE_DIR}/catz/*.h
 ${CMAKE_SOURCE_DIR}/recipes/*.cpp
 ${CMAKE_SOURCE_DIR}/recipes/*.h
)

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
