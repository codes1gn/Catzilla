add_executable(convolution
  main.cu
)
target_include_directories(convolution 
PRIVATE 
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/mm_utils"
)
set_target_properties(convolution PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(convolution ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

