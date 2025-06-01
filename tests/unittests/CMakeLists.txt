# Add CUDA test files
file(GLOB_RECURSE TEST_SOURCES
    "*.cu"
)

# Add C++ test files
file(GLOB_RECURSE CPP_TEST_SOURCES
    "*.cpp"
)

# Create test executables
foreach(TEST_SOURCE ${TEST_SOURCES} ${CPP_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    # Set CUDA architecture for CUDA files
    if(${TEST_SOURCE} MATCHES ".*\\.cu$")
        set_source_files_properties(${TEST_SOURCE} PROPERTIES LANGUAGE CUDA)
        set_target_properties(${TEST_NAME} PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
    endif()
    
    target_link_libraries(${TEST_NAME} PRIVATE
        test_utils
        catz
        Catch2::Catch2WithMain
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
    )
    
    # Add test to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set(TEST_BINARIES ${TEST_BINARIES} ${TEST_NAME} PARENT_SCOPE)
endforeach()
