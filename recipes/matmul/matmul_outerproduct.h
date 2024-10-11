#ifndef CATZILLA_RECIPES_MATMUL_OUTER_PRODUCT_H_
#define CATZILLA_RECIPES_MATMUL_OUTER_PRODUCT_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "macro.h"
#include "matrix.h"
#include "ukernels.h"

using namespace catz;
using namespace catz::cuda;
using namespace catz::wmma;
using namespace catz::mma;

namespace catz::recipes {

/////////////////////////////////////////////////////////////////////////////////
/// version 2:
/// - basic gmem to smem coalescing
/// - use smem for lhs, rhs, out
/// - thread-level blocking, compute a tile in reg-level C, not a scalar
/////////////////////////////////////////////////////////////////////////////////
template <const int M_TILE_SM, const int N_TILE_SM, const int K_TILE_SM,
          const int M_TILE_REG, const int N_TILE_REG>
__global__ void _matmul_outerproduct(int M, int N, int K, float alpha,
                                     float *lhs, float *rhs, float beta,
                                     float *out) {
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  // smem stationary
  // M=64, N=64, K=32, Mreg=2, Nreg=2
  __shared__ float lhs_shared[M_TILE_SM * K_TILE_SM];
  __shared__ float rhs_shared[K_TILE_SM * N_TILE_SM];
  __shared__ float out_shared[M_TILE_SM * N_TILE_SM];
  // __shared__ float out_shared[1];

  // can remove
  for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      out_shared[distribute_(tid_x * N_TILE_REG + nreg,
                             tid_y * M_TILE_REG + mreg, 1, N_TILE_SM)] = 0.;
    }
  }

  float partial_sum[M_TILE_REG * N_TILE_REG] = {0.};

  // QZ: per thread m and m+1
  for (int k = 0; k < CEIL_DIV(K, K_TILE_SM); k++) { // range(0, 128, 1)
    // use distribute_ for in-tile thread calculation
    // use tile_ for out-tile thread calculation
    for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
      lhs_shared[distribute_(tid_x, tid_y * M_TILE_REG + mreg, 1, K_TILE_SM)] =
          lhs[tiling_(bid_y, k, M_TILE_SM, K_TILE_SM, K, 1) +
              distribute_(tid_x, tid_y * M_TILE_REG + mreg, 1, K)];
    }
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      rhs_shared[distribute_(tid_x * N_TILE_REG + nreg, tid_y, 1, N_TILE_SM)] =
          rhs[tiling_(k, bid_x, K_TILE_SM, N_TILE_SM, N, 1) +
              distribute_(tid_x * N_TILE_REG + nreg, tid_y, 1, N)];
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_64x64x16_perthread_4x4(lhs_shared, rhs_shared, partial_sum);
    // more syncs
    // local => gmem 4930 GF
    // local => smem, smem => gmem 5255 GF
    // __syncthreads();
  }
  // local => smem, smem => gmem 5485 GF
  // local => gmem error in results
  __syncthreads();

  for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      // out_shared[distribute_(tid_x * N_TILE_REG, tid_y * M_TILE_REG, 1,
      // N_TILE_SM) + distribute_(mreg, nreg, N_TILE_SM, 1)] = alpha *
      // partial_sum[mreg * 4 + nreg]; out[tiling_(bid_y, bid_x, M_TILE_SM,
      // N_TILE_SM, N, 1) + distribute_(tid_x*N_TILE_REG, tid_y*M_TILE_REG, 1,
      // N) + distribute_(mreg, nreg, N, 1)] =
      // out_shared[distribute_(tid_x*N_TILE_REG, tid_y*M_TILE_REG, 1,
      // N_TILE_SM) + distribute_(mreg, nreg, N_TILE_SM, 1)];
      out[tiling_(bid_y, bid_x, M_TILE_SM, N_TILE_SM, N, 1) +
          distribute_(tid_x * N_TILE_REG, tid_y * M_TILE_REG, 1, N) +
          distribute_(mreg, nreg, N, 1)] =
          alpha * partial_sum[mreg * N_TILE_REG + nreg];
    }
  }
}

void matmul_outerproduct(int M, int N, int K, float alpha, float *A, float *B,
                         float beta, float *C) {
  const int M_TILE = 64;
  const int N_TILE = 64;
  const int K_TILE = 16;
  const int mreg = 4;
  const int nreg = 4;
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(CEIL_DIV(M_TILE, mreg), CEIL_DIV(N_TILE, nreg));
  cudaFuncSetAttribute(_matmul_outerproduct<M_TILE, N_TILE, K_TILE, mreg, nreg>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _matmul_outerproduct<M_TILE, N_TILE, K_TILE, mreg, nreg>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

} // namespace catz::recipes

#endif // CATZILLA_RECIPES_MATMUL_OUTER_PRODUCT_H_
