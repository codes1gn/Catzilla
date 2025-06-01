cmake_minimum_required(VERSION 3.18)

# Add C++ tests
add_executable(recipe_tests
    matmul/matmul_test.cu
)

# Set CUDA properties for .cu files
set_source_files_properties(matmul/matmul_test.cu PROPERTIES LANGUAGE CUDA)
set_target_properties(recipe_tests PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

target_include_directories(recipe_tests PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(recipe_tests PRIVATE
    test_utils
    catz
    catz-recipes
    Catch2::Catch2WithMain
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)

add_test(NAME recipe_cpp_tests COMMAND recipe_tests)

# Register the test
add_test(
    NAME recipe_tests
    COMMAND recipe_tests
) 
