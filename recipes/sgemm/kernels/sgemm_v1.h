#ifndef CATZILLA_RECIPES_KERNELS_SGEMM_V1_H_
#define CATZILLA_RECIPES_KERNELS_SGEMM_V1_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils/index_utils.h"
#include "utils/macros.h"

/*

Matrix sizes:
MxK * KxN = MxN

*/
// TODO extract common utils
/*
 * gmem access coalescing
 * smem stationary
 *
 * TODO:
 * reduce warp divergence
 * blocking tiling
 * pipeline
 * multi-buffer
 * swizzle
 * unroll
 * tensor.vmm.contracts
 * vector loads
*/
template <const int M_TILE, const int N_TILE, const int K_TILE>
__global__ void catzilla_sgemm_v1(int M, int N, int K, float alpha,
                                       float *lhs, float *rhs,
                                       float beta, float *output) {
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  // smem stationary
  __shared__ float lhs_shared[M_TILE * K_TILE];
  __shared__ float rhs_shared[K_TILE * N_TILE];
  __shared__ float output_shared[M_TILE * N_TILE];

  // coalescing
  output_shared[at_thread_(tid_x, tid_y, 1, N_TILE)] = 0.0;

  float partial_sum = 0.;
  for (int k = 0; k < CEIL_DIV(K, K_TILE); k++) { // range(0, 128, 1)
    // use at_thread_ for in-tile thread calculation
    // use at_tile_ for out-tile thread calculation
    lhs_shared[at_thread_(tid_x, tid_y, 1, K_TILE)] = lhs[at_tile_(bid_y, k, M_TILE, K_TILE, K, 1) + at_thread_(tid_x, tid_y, 1, K)];
    rhs_shared[at_thread_(tid_x, tid_y, 1, N_TILE)] = rhs[at_tile_(k, bid_x, K_TILE, N_TILE, N, 1) + at_thread_(tid_x, tid_y, 1, N)];
    __syncthreads();

    // contract at 32x32x32 micro-kernel
    matmul_kernel_32x32x32(lhs_shared, rhs_shared, partial_sum);
    __syncthreads();
  }
  output_shared[at_thread_(tid_x, tid_y, 1, N_TILE)] = alpha * partial_sum;
  output[at_tile_(bid_y, bid_x, M_TILE, N_TILE, N, 1) + at_thread_(tid_x, tid_y, 1, N)] = output_shared[at_thread_(tid_x, tid_y, 1, N_TILE)];
}


void catzilla_sgemm_v1_host(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  cudaFuncSetAttribute(catzilla_sgemm_v1<32, 32, 32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  catzilla_sgemm_v1<32, 32, 32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

#endif // CATZILLA_RECIPES_KERNELS_SGEMM_V1_H_
