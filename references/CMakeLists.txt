
add_subdirectory(sgemm)

# CONVOLUTION impls requires cp.async or memcpy_async supported since sm_80.
if(${CUDA_COMPUTE_CAPABILITY} GREATER_EQUAL 80)
  add_subdirectory(convolution)
endif()
