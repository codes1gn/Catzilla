
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "matrix.h"
#include "macro.h"
#include "trait.h"
#include "index.h"

using namespace catz;

TEST_CASE("index construction", "[index][construction]") {
  
  // 编译期常量
  constexpr int compileTimeValue = 42;
  constexpr auto indexConst = make_index(compileTimeValue); // 自动使用特化版本
  std::cout << "Compile-time constant: " << indexConst() << std::endl;
  SCHECK(indexConst.value == 42);

  // 运行时变量
  int runtimeValue = 42;
  auto indexDyn = make_index(runtimeValue); // 自动使用基础版本
  std::cout << "Runtime variable: " << indexDyn() << std::endl;
  CHECK(indexDyn.value == 42);

}


TEST_CASE("index sum", "[index][sum]") {
  constexpr auto idx1 = make_index(10);  // 编译期常量
    constexpr auto idx2 = make_index(3);   // 编译期常量

    constexpr auto sum = idx1 + idx2;
  SCHECK(sum.value == 13);

  auto idx1_dyn = make_index(10);  // 编译期常量
    auto idx2_dyn = make_index(3);   // 编译期常量

    auto sum_dyn = idx1 + idx2;
  CHECK(sum_dyn.value == 13);

}
