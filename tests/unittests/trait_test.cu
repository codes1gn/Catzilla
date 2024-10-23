
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "macro.h"
#include "trait.h"
#include "index.h"

using namespace catz;

TEST_CASE("isIndexLike trait", "[isIndexLike][trait]") {
  const int compileTimeValue = 42;
  auto indexConst = make_index<compileTimeValue>();
  SCHECK(is_index_like_v<decltype(indexConst)> == true);

  int runTimeValue = 42;
  auto indexDyn = make_index(runTimeValue); // 自动使用基础版本
  SCHECK(is_index_like_v<int> == false);
  SCHECK(is_index_like_v<decltype(indexDyn)> == true);
}
