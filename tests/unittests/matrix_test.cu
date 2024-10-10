#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "matrix.h"

using namespace catz;

TEST_CASE("matrix construction full", "[matrix][construction]") {
  const int M_TILE = 32;
  const int K_TILE = 32;
  auto _shape = CoordNightly<M_TILE, K_TILE>();
  auto _stride = CoordNightly<K_TILE, 1>();
  float lhs_data[M_TILE*K_TILE] = {0.0};
  auto lhs_mat = make_matrix(lhs_data, _shape, _stride);

  static_assert(lhs_mat.shape.first == 32,
                "matrix.shape.first not available at compile-time");
  static_assert(lhs_mat.shape.second == 32,
                "matrix.shape.first not available at compile-time");

  static_assert(lhs_mat.stride.first == 32,
                "matrix.stride.first not available at compile-time");
  static_assert(lhs_mat.stride.second == 1,
                "matrix.stride.first not available at compile-time");
}

TEST_CASE("matrix construction contiguous", "[matrix][construction]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  auto _shape = CoordNightly<M_TILE, K_TILE>();
  float lhs_data[M_TILE*K_TILE] = {0.0};
  auto lhs_mat = make_matrix(lhs_data, _shape);

  static_assert(lhs_mat.shape.first == 26,
                "matrix.shape.first not available at compile-time");
  static_assert(lhs_mat.shape.second == 32,
                "matrix.shape.first not available at compile-time");

  static_assert(lhs_mat.stride.first == 32,
                "matrix.stride.first not available at compile-time");
  static_assert(lhs_mat.stride.second == 1,
                "matrix.stride.first not available at compile-time");
}
