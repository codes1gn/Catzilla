# 启用测试
enable_testing()

# 添加测试可执行文件
add_executable(coord_test unittests/coord_test.cu)
add_executable(smoke_test unittests/smoke_test.cpp)

# 链接 Catch2 库（如果需要）
target_link_libraries(coord_test PRIVATE catz Catch2::Catch2WithMain ${CUDA_LIBRARIES})
target_link_libraries(smoke_test PRIVATE catz Catch2::Catch2WithMain ${CUDA_LIBRARIES})

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

# 自动发现测试用例
catch_discover_tests(coord_test)
catch_discover_tests(smoke_test)
