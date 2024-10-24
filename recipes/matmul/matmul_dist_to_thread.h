#ifndef CATZILLA_RECIPES_MATMUL_DIST_TO_THREAD_H_
#define CATZILLA_RECIPES_MATMUL_DIST_TO_THREAD_H_

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "macro.h"
#include "matrix_legacy.h"
#include "ukernels.h"

using namespace catz;
using namespace catz::cuda;
using namespace catz::wmma;
using namespace catz::mma;

namespace catz::recipes {

template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int X_THREAD,
          const int Y_THREAD>
__global__ void _matmul_dist_to_thread(int M, int N, int K, float alpha,
                                       float *lhs, float *rhs, float beta,
                                       float *out) {
  auto lhs_shape = CoordDyn(M, K);
  auto rhs_shape = CoordDyn(K, N);
  auto out_shape = CoordDyn(M, N);

  auto lhs_sm_tile_shape = CoordDyn(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = CoordDyn(K_TILE, N_TILE);
  auto out_sm_tile_shape = CoordDyn(M_TILE, N_TILE);

  auto lhs_reg_tile_shape = CoordDyn(M_REG, K_REG);
  auto rhs_reg_tile_shape = CoordDyn(K_REG, N_REG);
  auto out_reg_tile_shape = CoordDyn(M_REG, N_REG);

  // make sure inner block looks like this, to ensure coalescing

  MatrixDyn<float> lhs_mat = MatrixDyn<float>(lhs, lhs_shape);
  MatrixDyn<float> rhs_mat = MatrixDyn<float>(rhs, rhs_shape);
  MatrixDyn<float> out_mat = MatrixDyn<float>(out, out_shape);

  // __shared__ float lhs_shared[M_TILE* (K_TILE+1)];

  MatrixDyn<float> lhs_shared_mat = make_shared<M_TILE, K_TILE, float>();
  MatrixDyn<float> rhs_shared_mat = make_shared<K_TILE, N_TILE, float>();
  // MatrixDyn<float> out_shared_mat = make_shared<M_TILE_SM, N_TILE_SM>();
  //

  MatrixDyn<float> partial_sum =
      make_local<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(N_TILE, X_THREAD),
                 float>();

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, M_REG); m++) {
#pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, K_REG); kin++) {
        // TODO: span_as(y * 4, x / 4)
        int x = threadIdx.y * X_THREAD + threadIdx.x;
        lhs_shared_mat.tile(CoordDyn(m, kin), lhs_reg_tile_shape)
            .dist_to_thread() =
            lhs_mat.tile(CoordDyn(blockIdx.y, ko), lhs_sm_tile_shape)
                .tile(CoordDyn(m, kin), lhs_reg_tile_shape)
                .dist_to_thread();
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, K_REG); kin++) {
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, N_REG); n++) {
        rhs_shared_mat.tile(CoordDyn(kin, n), rhs_reg_tile_shape)
            .dist_to_thread() =
            rhs_mat.tile(CoordDyn(ko, blockIdx.x), rhs_sm_tile_shape)
                .tile(CoordDyn(kin, n), rhs_reg_tile_shape)
                .dist_to_thread();
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_coalesced<M_TILE, N_TILE, K_TILE, Y_THREAD, X_THREAD>(
        lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);
    // identity<16, 16>(
    //   rhs_shared_mat,
    //   out_mat.tile(CoordDyn(blockIdx.y, blockIdx.x), out_sm_tile_shape));
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
#pragma unroll
    for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
      // out_mat.tile(CoordDyn(blockIdx.y, blockIdx.x), out_sm_tile_shape)
      //   .tile(CoordDyn(m, n), CoordDyn(Y_THREAD, X_THREAD))
      //   <= partial_sum;
      out_mat.tile(CoordDyn(blockIdx.y, blockIdx.x), out_sm_tile_shape)
          .tile(CoordDyn(m, n), CoordDyn(Y_THREAD, X_THREAD))
          .dist_to_thread() = partial_sum.dist_to(CoordDyn(m, n));
    }
  }
}

void matmul_dist_to_thread(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const int M_TILE = 128;
  const int K_TILE = 32;
  const int N_TILE = 128;
  const int M_REG = 32;
  const int K_REG = 8;
  const int N_REG = 32;
  const int X_THREAD = 32;
  const int Y_THREAD = 8;
  assert(M_REG * K_REG > X_THREAD * Y_THREAD);
  assert(N_REG * K_REG > X_THREAD * Y_THREAD);

  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(_matmul_dist_to_thread<M_TILE, N_TILE, K_TILE, M_REG,
                                              N_REG, K_REG, X_THREAD, Y_THREAD>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _matmul_dist_to_thread<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, X_THREAD,
                         Y_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

  // const int M_TILE = 16;
  // const int K_TILE = 8;
  // const int N_TILE = 16;
  // const int M_REG = 8;
  // const int K_REG = 4;
  // const int N_REG = 8;
  // const int X_THREAD = 8;
  // const int Y_THREAD = 4;
  // assert(M_REG * K_REG > X_THREAD * Y_THREAD);
  // assert(N_REG * K_REG > X_THREAD * Y_THREAD);
  //
  // dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  // dim3 blockDim(X_THREAD, Y_THREAD);
  // cudaFuncSetAttribute(
  //   _matmul_dist_to_thread<M_TILE, N_TILE, K_TILE, M_REG, N_REG,
  //   K_REG,
  //                                 X_THREAD, Y_THREAD>,
  //   cudaFuncAttributePreferredSharedMemoryCarveout,
  //   cudaSharedmemCarveoutMaxShared);
  // _matmul_dist_to_thread<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG,
  //                               X_THREAD, Y_THREAD>
  //   <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  return;
}

} // namespace catz::recipes

#endif // CATZILLA_RECIPES_MATMUL_DIST_TO_THREAD_H_
