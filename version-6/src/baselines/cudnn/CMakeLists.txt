add_executable(conv_cudnn main.cu verify.h)
target_link_libraries(conv_cudnn cudnn)