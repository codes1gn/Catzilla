
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "macro.h"
#include "matrix.h"

using namespace catz;

TEST_CUDA_CASE(kernel_smoke_test, "kernel_smoke_test", "[kernel][smoke]") {

  float lhs[128*128] = {0.0};
  float rhs[128*128] = {0.0};
  float out[128*128];

  int M, K, N = 128;
  constexpr int M_TILE = 32;
  constexpr int K_TILE = 32;
  constexpr int N_TILE = 32;
  constexpr int X_THREAD = 16;
  constexpr int Y_THREAD = 16;

  auto lhs_shape = make_coord_dyn(M, K);
  auto rhs_shape = make_coord_dyn(K, N);
  auto out_shape = make_coord_dyn(M, N);

  assert(lhs_shape.rows() == 128);
  assert(out_shape.cols() == 128);

  auto lhs_sm_tile_shape = make_coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = make_coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = make_coord(M_TILE, N_TILE);

  SCHECK(lhs_sm_tile_shape.rows() == 32);
  SCHECK(out_sm_tile_shape.cols() == 32);

  auto per_block_data_shape = make_coord(Y_THREAD, X_THREAD);
  SCHECK(per_block_data_shape.rows() == 16);

  auto lhs_mat = make_matrix(lhs, lhs_shape);
  auto rhs_mat = make_matrix(rhs, rhs_shape);
  auto out_mat = make_matrix(out, out_shape);
  assert(lhs_mat.shape.rows() == 128);
  assert(out_mat.shape.rows() == 128);

  MAKE_SHARED_MATRIX(lhs_shared_mat, lhs_sm_tile_shape, float);
  MAKE_SHARED_MATRIX(rhs_shared_mat, rhs_sm_tile_shape, float);
  MAKE_LOCAL_MATRIX(partial_sum, out_sm_tile_shape / per_block_data_shape, float);
  SCHECK(lhs_shared_mat.shape.rows() == 32);
  SCHECK(rhs_shared_mat.shape.rows() == 32);
  SCHECK(partial_sum.shape.rows() == 2);

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
      for (int kin = 0; kin < CEIL_DIV(K_TILE, X_THREAD); kin++) {
        lhs_shared_mat.tile(CoordS(IndexDyn(m), IndexDyn(kin)), per_block_data_shape)
          .dist_to(CoordS(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x)))
          = lhs_mat.tile(CoordS(IndexDyn(blockIdx.y), IndexDyn(ko)), lhs_sm_tile_shape)
          .tile(CoordS(IndexDyn(m), IndexDyn(kin)), per_block_data_shape)
          .dist_to(CoordS(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x)));
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, Y_THREAD); kin++) {
      for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
        rhs_shared_mat.tile(CoordS(IndexDyn(kin), IndexDyn(n)), per_block_data_shape)
          .dist_to(CoordS(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x))) =
          rhs_mat.tile(CoordS(IndexDyn(ko), IndexDyn(blockIdx.x)), rhs_sm_tile_shape)
          .tile(CoordS(IndexDyn(kin), IndexDyn(n)), per_block_data_shape)
          .dist_to(CoordS(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x)));
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    // matmul_kernel_coalesced<M_TILE, N_TILE, K_TILE, Y_THREAD, X_THREAD>(
    //     lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
    for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
      out_mat.tile(CoordS(IndexDyn(blockIdx.y), IndexDyn(blockIdx.x)), out_sm_tile_shape)
          .tile(CoordS(IndexDyn(m), IndexDyn(n)), per_block_data_shape)
          .dist_to(CoordS(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x)))
          = partial_sum.dist_to(CoordS(IndexDyn(m), IndexDyn(n)));
    }
  }
}
