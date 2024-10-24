#ifndef CATZILLA_RECIPES_MATMUL_PAD_SWIZZLED_H_
#define CATZILLA_RECIPES_MATMUL_PAD_SWIZZLED_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "coord.h"
#include "coord_legacy.h"
#include "index.h"
#include "macro.h"
#include "matrix.h"
#include "matrix_legacy.h"
#include "ukernels.h"

using namespace catz;
using namespace catz::cuda;
using namespace catz::wmma;
using namespace catz::mma;

namespace catz::recipes {

// hand tuned for HPs
// use templated kernel
// PAD-SWIZZLE
template <const int M_TILE, const int N_TILE, const int K_TILE,
          const int X_THREAD, const int Y_THREAD>
__global__ void _matmul_pad_swizzled(int M, int N, int K, float alpha,
                                     float *lhs, float *rhs, float beta,
                                     float *out) {
  auto lhs_shape = make_coord_dyn(M, K);
  auto rhs_shape = make_coord_dyn(K, N);
  auto out_shape = make_coord_dyn(M, N);

  auto lhs_sm_tile_shape = make_coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = make_coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = make_coord(M_TILE, N_TILE);

  auto lhs_sm_tile_shape_pad = make_coord(M_TILE, K_TILE + 1);
  auto rhs_sm_tile_shape_pad = make_coord(K_TILE, N_TILE + 1);

  auto per_block_data_shape = make_coord(Y_THREAD, X_THREAD);

  auto lhs_mat = make_matrix(lhs, lhs_shape);
  auto rhs_mat = make_matrix(rhs, rhs_shape);
  auto out_mat = make_matrix(out, out_shape);

  MAKE_SHARED_MATRIX(lhs_shared_mat, lhs_sm_tile_shape_pad, float);
  MAKE_SHARED_MATRIX(rhs_shared_mat, rhs_sm_tile_shape_pad, float);

  MAKE_LOCAL_MATRIX(partial_sum, out_sm_tile_shape / per_block_data_shape,
                    float);

  for (auto ko = I(0); ko < make_index(CEIL_DIV(K, K_TILE)); ++ko) {
    for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, Y_THREAD)>(); ++m) {
#pragma unroll
      for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, X_THREAD)>();
           ++kin) {
        lhs_shared_mat.tile(Coord(m, kin), per_block_data_shape)
            .dist_to(Coord(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x))) =
            lhs_mat.tile(Coord(IndexDyn(blockIdx.y), ko), lhs_sm_tile_shape)
                .tile(Coord(m, kin), per_block_data_shape)
                .dist_to(Coord(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x)));
      }
    }
    for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, Y_THREAD)>();
         ++kin) {
#pragma unroll
      for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, X_THREAD)>(); ++n) {
        rhs_shared_mat.tile(Coord(kin, n), per_block_data_shape)
            .dist_to(Coord(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x))) =
            rhs_mat.tile(Coord(ko, IndexDyn(blockIdx.x)), rhs_sm_tile_shape)
                .tile(Coord(kin, n), per_block_data_shape)
                .dist_to(Coord(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x)));
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_pad_swizzled<M_TILE, N_TILE, K_TILE, Y_THREAD, X_THREAD>(
        lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);
  }
  __syncthreads();

  for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, Y_THREAD)>(); ++m) {
#pragma unroll
    for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, X_THREAD)>(); ++n) {
      out_mat
          .tile(Coord(IndexDyn(blockIdx.y), IndexDyn(blockIdx.x)),
                out_sm_tile_shape)
          .tile(Coord(m, n), per_block_data_shape)
          .dist_to(Coord(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x))) =
          partial_sum.dist_to(Coord(m, n));
    }
  }
}

void matmul_pad_swizzled(int M, int N, int K, float alpha, float *A, float *B,
                         float beta, float *C) {
  // for bench
  // sec 1, determine, gridDim
  const int M_TILE = 128;
  const int N_TILE = 128;
  // sec 2, determine blockDim
  const int X_THREAD = 16;
  const int Y_THREAD = 16;
  // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  const int K_TILE = 32;

  // for debug only
  // const int M_TILE = 32;
  // const int N_TILE = 32;
  // // sec 2, determine blockDim
  // const int X_THREAD = 4;
  // const int Y_THREAD = 4;
  // // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // const int K_TILE = 8;
  // K_TILE > Y_THREAD
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
      _matmul_pad_swizzled<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _matmul_pad_swizzled<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

} // namespace catz::recipes

#endif // CATZILLA_RECIPES_MATMUL_PAD_SWIZZLED_H_
