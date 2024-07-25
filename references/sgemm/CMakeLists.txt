# generate executable
add_executable(sgemm-ref
    sgemm_main.cu
    runner.cu
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(sgemm-ref PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(sgemm-ref ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
