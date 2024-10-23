
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


// BUG: try to unify helper but not work, runtime value cannot use as template param, this is the killing dot
TEST_CASE("index construction", "[index][construction]") {
  const int compileTimeValue = 42;
  auto indexConst = make_index<compileTimeValue>();
  SCHECK(indexConst.value == 42);
  SCHECK(indexConst.isStatic() == true);

  int runTimeValue = 42;
  auto indexDyn = make_index(runTimeValue); // 自动使用基础版本
  CHECK(indexDyn.value == 42);
  SCHECK(indexDyn.isStatic() == false);
}

TEST_CASE("index sum", "[index][sum]") {
  auto idx1 = make_index<10>();  // 编译期常量
  auto idx2 = make_index<3>();   // 编译期常量
  auto sum = idx1 + idx2;
  SCHECK(sum.value == 13);

  auto idx1_dyn = make_index(10);  // 编译期常量
    auto idx2_dyn = make_index(3);   // 编译期常量
    auto sum_dyn = idx1 + idx2;
  CHECK(sum_dyn.value == 13);
}

TEST_CASE("index sub", "[index][sub]") {
  auto idx1 = make_index<10>();  // 编译期常量
    auto idx2 = make_index<3>();   // 编译期常量
    auto res = idx1 - idx2;
  SCHECK(res.value == 7);

  auto idx1_dyn = make_index(10);  // 编译期常量
    auto idx2_dyn = make_index(3);   // 编译期常量
    auto res_dyn = idx1 - idx2;
  CHECK(res_dyn.value == 7);
}

TEST_CASE("index mul", "[index][mul]") {
  auto idx1 = make_index<10>();  // 编译期常量
    auto idx2 = make_index<3>();   // 编译期常量
    auto res = idx1 * idx2;
  SCHECK(res.value == 30);

  auto idx1_dyn = make_index(10);  // 编译期常量
    auto idx2_dyn = make_index(3);   // 编译期常量
    auto res_dyn = idx1 * idx2;
  CHECK(res_dyn.value == 30);
}

TEST_CASE("index div", "[index][div]") {
  auto idx1 = make_index<10>();  // 编译期常量
    auto idx2 = make_index<3>();   // 编译期常量
    auto res = idx1 / idx2;
  SCHECK(res.value == 3);

  auto idx1_dyn = make_index(10);  // 编译期常量
    auto idx2_dyn = make_index(3);   // 编译期常量
    auto res_dyn = idx1 / idx2;
  CHECK(res_dyn.value == 3);
}

TEST_CASE("index mod", "[index][mod]") {
  auto idx1 = make_index<10>();  // 编译期常量
    auto idx2 = make_index<3>();   // 编译期常量
    auto res = idx1 % idx2;
  SCHECK(res.value == 1);

  auto idx1_dyn = make_index(10);  // 编译期常量
    auto idx2_dyn = make_index(3);   // 编译期常量
    auto res_dyn = idx1 % idx2;
  CHECK(res_dyn.value == 1);
}

TEST_CASE("index self-increment", "[index][self-increment]") {
  auto idx1 = make_index<10>();  // 编译期常量
  auto idx2 = ++idx1;
  SCHECK(idx1.value == 10);
  SCHECK(idx2.value == 11);

  auto idx1_dyn = make_index(10);  // 编译期常量
  ++idx1_dyn;
  CHECK(idx1_dyn.value == 11);
}

