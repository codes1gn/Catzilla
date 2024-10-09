# 启用测试
enable_testing()

set(SOURCES "")

file(GLOB_RECURSE CU_TEST_SOURCES "unittests/*_test.cu")
list(APPEND SOURCES ${CU_TEST_SOURCES})

file(GLOB_RECURSE CPP_TEST_SOURCES "unittests/*_test.cu")
list(APPEND SOURCES ${CPP_TEST_SOURCES})

foreach(src ${SOURCES})
  get_filename_component(binary ${src} NAME_WE)
  add_executable(${binary} ${src})

  target_link_libraries(${binary} PRIVATE catz Catch2::Catch2WithMain ${CUDA_LIBRARIES})

  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(CTest)
  include(Catch)

  catch_discover_tests(${binary})
endforeach()
