#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "macro.h"
#include "matrix.h"

using namespace catz;


TEST_CUDA_CASE(matrix_construction_1, "matrix construction full", "[matrix][construction]") {
  const int M_TILE = 32;
  const int K_TILE = 32;
  auto _shape = Coord<M_TILE, K_TILE>();
  auto _stride = Coord<K_TILE, 1>();
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

TEST_CUDA_CASE(matrix_construction_2, "matrix construction contiguous", "[matrix][construction]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  auto _shape = Coord<M_TILE, K_TILE>();
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

// TODO: allow for TEST_CASE style entry, no need for first identifier
TEST_CASE("matrix tile contiguous legacy", "[matrix][tile]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  const int M_REG = 13;
  const int K_REG = 4;
  auto _shape = CoordDyn(M_TILE, K_TILE);
  float lhs_data[M_TILE*K_TILE] = {0.0};
  auto lhs_mat = MatrixDyn(lhs_data, _shape);
  auto _tile_shape = CoordDyn(M_REG, K_REG);
  auto rhs_mat = lhs_mat.tile(CoordDyn(0, 0), _tile_shape);

  CHECK(rhs_mat.shape.first == 13);
  CHECK(rhs_mat.shape.second == 4);
  CHECK(rhs_mat.stride.first == 32);
  CHECK(rhs_mat.stride.second == 1);

  CHECK_FALSE(rhs_mat.stride.second == 2);
}

TEST_CUDA_CASE(matrix_tile_contiguous, "matrix tile contiguous", "[matrix][tile]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  const int M_REG = 13;
  const int K_REG = 4;
  float lhs_data[M_TILE*K_TILE] = {0.0};
  auto _shape = Coord<M_TILE, K_TILE>();

  auto lhs_mat = make_matrix(lhs_data, _shape);
  auto _tile_shape = Coord<M_REG, K_REG>();
  auto rhs_mat = lhs_mat.tile(CoordDyn(0, 0), _tile_shape);

  SCHECK(rhs_mat.shape.first == 13);
  SCHECK(rhs_mat.shape.second == 4);
  SCHECK(rhs_mat.stride.first == 32);
  SCHECK(rhs_mat.stride.second == 1);
}

TEST_CUDA_CASE(matrix_dist_to_coord, "matrix dist to coord", "[matrix][dist]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  const int M_REG = 13;
  const int K_REG = 4;
  float lhs_data[M_TILE*K_TILE] = {0.0};
  auto _shape = Coord<M_TILE, K_TILE>();

  auto lhs_mat = make_matrix(lhs_data, _shape);
  auto _tile_shape = Coord<M_REG, K_REG>();
  auto rhs_mat = lhs_mat.tile(CoordDyn(0, 0), _tile_shape);
  for (int row = 0; row < M_REG; row++)
    for (int col = 0; col < K_REG; col++)
      rhs_mat.dist_to(CoordDyn(row, col)) = 0.13;

  SCHECK(rhs_mat.shape.first == 13);
  SCHECK(rhs_mat.shape.second == 4);
  SCHECK(rhs_mat.stride.first == 32);
  SCHECK(rhs_mat.stride.second == 1);
  // TODO: assert at device not caught by CATCH2, need fix
  assert(rhs_mat.data[0] == 0.13);
  assert(rhs_mat.data[3] == 0.13);
  assert(rhs_mat.data[4] == 0.13);
}
