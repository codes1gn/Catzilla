set(CUDNN_FRONTEND_SOURCES
        main.cu

        utils/helpers.h
        utils/helpers.cpp

        utils/fp16_dev.h
        utils/fp16_emu.h
        utils/fp16_dev.cu
        utils/fp16_emu.cpp
)
set_source_files_properties(${CUDNN_FRONTEND_SOURCES} PROPERTIES LANGUAGE CUDA)
add_executable(conv_cudnn_frontend ${CUDNN_FRONTEND_SOURCES})
target_link_libraries(conv_cudnn_frontend cudnn)
target_include_directories(conv_cudnn_frontend PRIVATE
        ${cudnn_frontend_ROOT}/include
        ${ARGPARSE_INCLUDE_DIR})