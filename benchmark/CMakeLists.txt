# generate executable
add_executable(catzilla-matmul
    runner.cu
    benchmark_utils.cu
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(catzilla-matmul PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(catzilla-matmul 
PRIVATE 
${CUDA_LIBRARIES} 
${CUDA_CUBLAS_LIBRARIES}

catz-recipes
)

