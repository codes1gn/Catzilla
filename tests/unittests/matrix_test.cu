#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "macro.h"
#include "coord.h"
#include "matrix.h"

using namespace catz;

TEST_CUDA_CASE(matrix_create, "matrix create", "[matrix][create]") {
  float data[12];
  auto shape = make_coord(3, 4);
  auto matrix = make_matrix(data, shape);

  SCHECK(shape.rows() == 3);
  SCHECK(matrix.shape.rows() == 3);
  SCHECK(matrix.stride.rows() == 4);
  SCHECK(matrix.shape.isStatic() == true);
  SCHECK(matrix.stride.isStatic() == true);
}

TEST_CUDA_CASE(matrix_create_dyn, "matrix create dyn", "[matrix][create]") {
  float data[12];
  auto shape = make_coord_dyn(3, 4);
  auto matrix = make_matrix(data, shape);

  assert(shape.rows() == 3);
  assert(matrix.shape.rows() == 3);
  assert(matrix.stride.rows() == 4);
  SCHECK(matrix.shape.isStatic() == false);
  SCHECK(matrix.stride.isStatic() == false);
}

TEST_CASE("matrix dist", "[matrix][dist]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  const int M_REG = 13;
  const int K_REG = 4;
  float lhs_data[M_TILE * K_TILE] = {0.0};
  auto _shape = make_coord(26, 32);
  auto _tile_shape = make_coord(13, 4);
  auto _mat = make_matrix(lhs_data, _shape);
  auto _tiled_mat = _mat.tile(make_coord_dyn(0, 0), _tile_shape);
  for (int i = 0; i < 13; i++)
    for (int j = 0; j < 4; j++)
      _tiled_mat.dist_to(make_coord_dyn(i, j)) = 0.131;

  SCHECK(_tiled_mat.shape.rows() == 13);
  SCHECK(_tiled_mat.shape.cols() == 4);
  SCHECK(_tiled_mat.stride.rows() == 32);
  SCHECK(_tiled_mat.stride.cols() == 1);
  SCHECK_FALSE(_tiled_mat.stride.cols() == 2);

  assert(_tiled_mat.data[0] == 0.131f);
  assert(_tiled_mat.data[3] == 0.131f);
  assert(_tiled_mat.data[32] == 0.131f);
}

TEST_CUDA_CASE(matrix_dist, "matrix dist cuda", "[matrix][dist]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  const int M_REG = 13;
  const int K_REG = 4;
  float lhs_data[M_TILE * K_TILE] = {0.0};
  auto _shape = make_coord(26, 32);
  auto _tile_shape = make_coord(13, 4);
  auto _mat = make_matrix(lhs_data, _shape);
  auto _tiled_mat = _mat.tile(make_coord_dyn(3, 0), _tile_shape);
  for (int i = 0; i < 13; i++)
    for (int j = 0; j < 4; j++)
      _tiled_mat.dist_to(make_coord_dyn(i, j)) = 0.131;

  SCHECK(_tiled_mat.shape.rows() == 13);
  SCHECK(_tiled_mat.shape.cols() == 4);
  SCHECK(_tiled_mat.stride.rows() == 32);
  SCHECK(_tiled_mat.stride.cols() == 1);
  SCHECK_FALSE(_tiled_mat.stride.cols() == 2);
}

TEST_CUDA_CASE_WITH_THREADS(matrix_dist_to_threadx, 4,
                            "matrix dist cuda threadx", "[matrix][dist]") {
  const int M_TILE = 26;
  const int K_TILE = 32;
  const int M_REG = 13;
  const int K_REG = 4;
  float lhs_data[M_TILE * K_TILE] = {0.0};
  auto _shape = make_coord(26, 32);
  auto _tile_shape = make_coord(13, 4);
  auto _mat = make_matrix(lhs_data, _shape);
  auto _tiled_mat = _mat.tile(make_coord_dyn(3, 0), _tile_shape);
  for (int i = 0; i < 13; i++)
    _tiled_mat.dist_to(make_coord_dyn(i, threadIdx.x)) = 0.131;

  SCHECK(_tiled_mat.shape.rows() == 13);
  SCHECK(_tiled_mat.shape.cols() == 4);
  SCHECK(_tiled_mat.stride.rows() == 32);
  SCHECK(_tiled_mat.stride.cols() == 1);
  SCHECK_FALSE(_tiled_mat.stride.cols() == 2);
}
