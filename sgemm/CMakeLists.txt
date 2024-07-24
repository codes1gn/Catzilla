# generate executable
add_executable(sgemm
    sgemm_main.cu
    runner.cu
)
include_directories(${PROJECT_SOURCE_DIR}/sgemm)

set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
