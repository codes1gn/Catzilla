#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "matrix.h"
#include "macro.h"


using namespace catz;

// NOTE: how to reverse type info w.o. type decay in function parameters
// this is essential for creating helper functions that wraps compile-time
// runtime mixed programming style
// we do not want to bother users and attempt to omit all details on
// which vars are compile-time or runtime.
template <typename T>
std::string pass_to_func(T&& value) {
    if constexpr(std::is_const_v<std::remove_reference_t<T>>) {
        std::cout << "Is constexpr: " << std::boolalpha 
          << std::is_same_v<T, const int> 
          << ", value = " << value << "\n";
        // auto constV = std::integral_constant<int, value>();
        return "const int";
    } else {
        std::cout << "Is constexpr: " << std::boolalpha 
          << std::is_same_v<T, const int> 
          << ", value = " << value << "\n";
        return "int";
    }
}

TEST_CASE("type decay test", "[type][decay]") {
    const int a = 42;
    std::string ret = pass_to_func(a);

    CHECK(ret == "const int");

    int b = 42;
    std::string ret2 = pass_to_func(b);

    CHECK(ret2 == "int");
}

