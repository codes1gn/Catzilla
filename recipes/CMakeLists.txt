# catz-recipes: cuda-enhanced programming model for catz-recipesilla
add_library(catz-recipes INTERFACE)

set_target_properties(catz-recipes PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_include_directories(catz-recipes INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(catz-recipes 
INTERFACE 
${CUDA_LIBRARIES} 
${CUDA_CUBLAS_LIBRARIES}
catz
)

