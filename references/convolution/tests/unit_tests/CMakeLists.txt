add_executable(storage_convert_test storage_convert_test.cpp)
set_source_files_properties(storage_convert_test.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(storage_convert_test PRIVATE ${ARGPARSE_INCLUDE_DIR} ${SPCONV_INCLUDE_DIR})

#add_executable(verify_test verify_test.cu)
#target_include_directories(verify_test PRIVATE ${ARGPARSE_INCLUDE_DIR} ${SPCONV_INCLUDE_DIR})