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
  auto lhs_mat = make_matrix_nightly(lhs_data, _shape, _stride);

  static_assert(lhs_mat.shape.rows == 32,
                "matrix.shape.rows not available at compile-time");
  static_assert(lhs_mat.shape.cols == 32,
                "matrix.shape.rows not available at compile-time");

  static_assert(lhs_mat.stride.rows == 32,
                "matrix.stride.rows not available at compile-time");
  static_assert(lhs_mat.stride.cols == 1,
                "matrix.stride.rows not available at compile-time");
}

TEST_CUDA_CASE(matrix_construction_2, "matrix construction contiguous", "[matrix][construction]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  auto _shape = Coord<M_TILE, K_TILE>();
  float lhs_data[M_TILE*K_TILE] = {0.0};
  auto lhs_mat = make_matrix_nightly(lhs_data, _shape);

  static_assert(lhs_mat.shape.rows == 26,
                "matrix.shape.rows not available at compile-time");
  static_assert(lhs_mat.shape.cols == 32,
                "matrix.shape.rows not available at compile-time");

  static_assert(lhs_mat.stride.rows == 32,
                "matrix.stride.rows not available at compile-time");
  static_assert(lhs_mat.stride.cols == 1,
                "matrix.stride.rows not available at compile-time");
}

// TODO: allow for TEST_CASE style entry, no need for rows identifier
TEST_CASE("matrix tile contiguous legacy", "[matrix][tile][legacy]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  const int M_REG = 13;
  const int K_REG = 4;
  auto _shape = CoordDyn(M_TILE, K_TILE);
  float lhs_data[M_TILE*K_TILE] = {0.0};
  auto lhs_mat = MatrixDyn(lhs_data, _shape);
  auto _tile_shape = CoordDyn(M_REG, K_REG);
  auto rhs_mat = lhs_mat.tile(CoordDyn(0, 0), _tile_shape);

  CHECK(rhs_mat.shape.rows == 13);
  CHECK(rhs_mat.shape.cols == 4);
  CHECK(rhs_mat.stride.rows == 32);
  CHECK(rhs_mat.stride.cols == 1);
  CHECK_FALSE(rhs_mat.stride.cols == 2);
}

TEST_CUDA_CASE(matrix_tile_contiguous, "matrix tile contiguous", "[matrix][tile]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  const int M_REG = 13;
  const int K_REG = 4;
  float lhs_data[M_TILE*K_TILE] = {0.0};
  auto _shape = Coord<M_TILE, K_TILE>();

  auto lhs_mat = make_matrix_nightly(lhs_data, _shape);
  auto _tile_shape = Coord<M_REG, K_REG>();
  auto rhs_mat = lhs_mat.tile(CoordDyn(0, 0), _tile_shape);

  SCHECK(rhs_mat.shape.rows == 13);
  SCHECK(rhs_mat.shape.cols == 4);
  SCHECK(rhs_mat.stride.rows == 32);
  SCHECK(rhs_mat.stride.cols == 1);
}

TEST_CUDA_CASE(matrix_dist_to_coord, "matrix dist to coord", "[matrix][dist]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  const int M_REG = 13;
  const int K_REG = 4;
  float lhs_data[M_TILE*K_TILE] = {0.0};
  auto _shape = Coord<M_TILE, K_TILE>();

  auto lhs_mat = make_matrix_nightly(lhs_data, _shape);
  auto _tile_shape = Coord<M_REG, K_REG>();
  auto rhs_mat = lhs_mat.tile(CoordDyn(0, 0), _tile_shape);
  for (int row = 0; row < M_REG; row++)
    for (int col = 0; col < K_REG; col++)
      rhs_mat.dist_to(CoordDyn(row, col)) = 0.13;

  SCHECK(rhs_mat.shape.rows == 13);
  SCHECK(rhs_mat.shape.cols == 4);
  SCHECK(rhs_mat.stride.rows == 32);
  SCHECK(rhs_mat.stride.cols == 1);
  // TODO: assert at device not caught by CATCH2, need fix
  assert(rhs_mat.data[0] == 0.13);
  assert(rhs_mat.data[3] == 0.13);
  assert(rhs_mat.data[4] == 0.13);
}

TEST_CUDA_CASE(matrix_create, "matrix create", "[matrix][create]") {
  int data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  auto shape = make_coord(3, 4);
  auto matrix = make_matrix(data, shape);

  SCHECK(shape.rows() == 3);
  SCHECK(matrix.shape.rows() == 3);
  SCHECK(matrix.stride.rows() == 4);
  SCHECK(matrix.shape.isStatic() == true);
  SCHECK(matrix.stride.isStatic() == true);
}

TEST_CASE("matrix tile", "[matrix][tile]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  const int M_REG = 13;
  const int K_REG = 4;
  float lhs_data[M_TILE*K_TILE] = {0.0};
  auto _shape = make_coord(26, 32);
  auto _tile_shape = make_coord(13, 4);
  auto _mat = make_matrix(lhs_data, _shape);
  auto _tiled_mat = _mat.tile(CoordDyn(3, 0), _tile_shape);

  SCHECK(_tiled_mat.shape.rows() == 13);
  SCHECK(_tiled_mat.shape.cols() == 4);
  SCHECK(_tiled_mat.stride.rows() == 32);
  SCHECK(_tiled_mat.stride.cols() == 1);
  SCHECK_FALSE(_tiled_mat.stride.cols() == 2);
}

TEST_CUDA_CASE(matrix_tile, "matrix tile cuda", "[matrix][tile]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  const int M_REG = 13;
  const int K_REG = 4;
  float lhs_data[M_TILE*K_TILE] = {0.0};
  auto _shape = make_coord(26, 32);
  auto _tile_shape = make_coord(13, 4);
  auto _mat = make_matrix(lhs_data, _shape);
  auto _tiled_mat = _mat.tile(CoordDyn(3, 0), _tile_shape);

  SCHECK(_tiled_mat.shape.rows() == 13);
  SCHECK(_tiled_mat.shape.cols() == 4);
  SCHECK(_tiled_mat.stride.rows() == 32);
  SCHECK(_tiled_mat.stride.cols() == 1);
  SCHECK_FALSE(_tiled_mat.stride.cols() == 2);
}
