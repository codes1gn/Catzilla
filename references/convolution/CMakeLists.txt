add_executable(convolution-ref
  main.cu
)
target_include_directories(convolution-ref 
PRIVATE 
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/mm_utils"
)
set_target_properties(convolution-ref PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(convolution-ref ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

