set(SPCONV_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
message(SPConv_INCLUDE_DIR: ${SPCONV_INCLUDE_DIR})

#add_executable(implgemm_back
#        main_back.cu
#        implgemm_NCHW.h
#        implgemm_NHWC.h
#        implgemm_temp.h)
#target_include_directories(implgemm_back PRIVATE ${ARGPARSE_INCLUDE_DIR} ${SPCONV_INCLUDE_DIR})

add_executable(implgemm
        main.cu
        implgemm_NCHW.h
        implgemm_NHWC.h)
target_include_directories(implgemm PRIVATE ${ARGPARSE_INCLUDE_DIR} ${SPCONV_INCLUDE_DIR})

# test
add_subdirectory(tests)