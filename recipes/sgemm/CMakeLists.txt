# generate executable
add_executable(catzilla-sgemm
    runner.cu
    main.cu
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(catzilla-sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(catzilla-sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
