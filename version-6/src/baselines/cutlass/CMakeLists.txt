add_executable(conv_cutlass_float main_float.cu)
target_include_directories(conv_cutlass_float PRIVATE
        ${CUTLASS_ROOT}/include
        ${CUTLASS_ROOT}/tools/util/include
        ${CUTLASS_ROOT}/examples/common)

add_executable(conv_cutlass_half main_half.cu)
target_include_directories(conv_cutlass_half PRIVATE
        ${CUTLASS_ROOT}/include
        ${CUTLASS_ROOT}/tools/util/include
        ${CUTLASS_ROOT}/examples/common)