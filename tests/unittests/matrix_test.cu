#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "matrix.h"

using namespace catz;

// TEST_CASE("matrix construction full", "[matrix][construction]") {
//   const int M_TILE = 32;
//   const int K_TILE = 32;
//   auto _shape = CoordNightly<M_TILE, K_TILE>();
//   auto _stride = CoordNightly<K_TILE, 1>();
//   float lhs_data[M_TILE*K_TILE] = {0.0};
//   auto lhs_mat = make_matrix(lhs_data, _shape, _stride);
//
//   static_assert(lhs_mat.shape.first == 32,
//                 "matrix.shape.first not available at compile-time");
//   static_assert(lhs_mat.shape.second == 32,
//                 "matrix.shape.first not available at compile-time");
//
//   static_assert(lhs_mat.stride.first == 32,
//                 "matrix.stride.first not available at compile-time");
//   static_assert(lhs_mat.stride.second == 1,
//                 "matrix.stride.first not available at compile-time");
// }
//
// TEST_CASE("matrix construction contiguous", "[matrix][construction]") {
//   const int M_TILE = 26;
//   const int K_TILE = 32;
//   auto _shape = CoordNightly<M_TILE, K_TILE>();
//   float lhs_data[M_TILE*K_TILE] = {0.0};
//   auto lhs_mat = make_matrix(lhs_data, _shape);
//
//   static_assert(lhs_mat.shape.first == 26,
//                 "matrix.shape.first not available at compile-time");
//   static_assert(lhs_mat.shape.second == 32,
//                 "matrix.shape.first not available at compile-time");
//
//   static_assert(lhs_mat.stride.first == 32,
//                 "matrix.stride.first not available at compile-time");
//   static_assert(lhs_mat.stride.second == 1,
//                 "matrix.stride.first not available at compile-time");
// }

// TEST_CASE("matrix tile contiguous legacy", "[matrix][tile]") {
//   const int M_TILE = 26;
//   const int K_TILE = 32;
//   const int M_REG = 13;
//   const int K_REG = 4;
//   auto _shape = CoordNightly<M_TILE, K_TILE>();
//   float lhs_data[M_TILE*K_TILE] = {0.0};
//   auto lhs_mat = make_matrix(lhs_data, _shape);
//   auto _tile_shape = CoordNightly<M_REG, K_REG>();
//   auto rhs_mat = lhs_mat.tile(Coord(0, 0), _tile_shape);
//
//   static_assert(rhs_mat.shape.first == 13,
//                 "matrix.shape.first not available at compile-time");
//   static_assert(rhs_mat.shape.second == 4,
//                 "matrix.shape.first not available at compile-time");
//
//   static_assert(rhs_mat.stride.first == 32,
//                 "matrix.stride.first not available at compile-time");
//   static_assert(rhs_mat.stride.second == 1,
//                 "matrix.stride.first not available at compile-time");
// }

template <int M_TILE, int K_TILE, int M_REG, int K_REG>
__global__ void matrix_tile_test_on_gpu(float* lhs_data) {
  auto _shape = CoordNightly<M_TILE, K_TILE>();

  auto lhs_mat = make_matrix(lhs_data, _shape);
  auto _tile_shape = CoordNightly<M_REG, K_REG>();
  auto rhs_mat = lhs_mat.tile(Coord(0, 0), _tile_shape);

  static_assert(rhs_mat.shape.first == 13,
                "matrix.shape.first not available at compile-time");
  static_assert(rhs_mat.shape.second == 4,
                "matrix.shape.first not available at compile-time");

  static_assert(rhs_mat.stride.first == 32,
                "matrix.stride.first not available at compile-time");
  static_assert(rhs_mat.stride.second == 1,
                "matrix.stride.first not available at compile-time");
}

TEST_CASE("matrix tile contiguous", "[matrix][tile]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  const int M_REG = 13;
  const int K_REG = 4;
  float lhs_data[M_TILE*K_TILE] = {0.0};
  matrix_tile_test_on_gpu<
    M_TILE,
    K_TILE,
    M_REG,
    K_REG><<<1, 1>>>(lhs_data);
}
