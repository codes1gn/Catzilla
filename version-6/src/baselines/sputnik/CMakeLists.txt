add_executable(conv_sputnik main.cu)
target_include_directories(conv_sputnik PUBLIC
        ${SPUTNIK_ROOT}
        ${SPUTNIK_ROOT}/third_party/abseil-cpp/)
message(${SPUTNIK_ROOT})
target_link_libraries(conv_sputnik PUBLIC
        ${SPUTNIK_ROOT}/build/sputnik/libsputnik.so)