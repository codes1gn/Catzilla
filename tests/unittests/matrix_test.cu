#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "matrix.h"

using namespace catz;

TEST_CASE("matrix construction", "[matrix][construction]") {
  const int M_TILE = 32;
  const int K_TILE = 32;
  constexpr Coord _shape = Coord(M_TILE, K_TILE);
  float lhs_data[M_TILE*K_TILE] = {0.0};
  float* const ptr = &lhs_data;
  constexpr Matrix lhs_mat = Matrix(ptr, _shape);

  static_assert(lhs_mat.shape.first == 32,
                "matrix.shape.first not available at compile-time");
  // static_assert(lhs_shape_mat.shape.second == 2,
  //               "matrix.shape.first not available at compile-time");
}
