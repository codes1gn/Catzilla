# Catz: cuda-enhanced programming model for catzilla
add_library(catz INTERFACE)

set_target_properties(catz PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_include_directories(catz INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(catz INTERFACE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

