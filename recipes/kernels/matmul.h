#ifndef CATZILLA_RECIPES_KERNELS_MATMUL_V1_H_
#define CATZILLA_RECIPES_KERNELS_MATMUL_V1_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils/index_utils.h"
#include "utils/macros.h"
#include "utils/micro_kernels.h"

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

/////////////////////////////////////////////////////////////////////////////////
/// version 1:
/// - basic gmem to smem coalescing
/// - use smem for lhs, rhs, out
/////////////////////////////////////////////////////////////////////////////////
template <const int M_TILE, const int N_TILE, const int K_TILE>
__global__ void _catzilla_matmul_v1(int M, int N, int K, float alpha,
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
  output_shared[distribute_(tid_x, tid_y, 1, N_TILE)] = 0.0;

  float partial_sum = 0.;
  for (int k = 0; k < CEIL_DIV(K, K_TILE); k++) { // range(0, 128, 1)
    // use distribute_ for in-tile thread calculation
    // use tile_ for out-tile thread calculation
    lhs_shared[distribute_(tid_x, tid_y, 1, K_TILE)] = lhs[tiling_(bid_y, k, M_TILE, K_TILE, K, 1) + distribute_(tid_x, tid_y, 1, K)];
    rhs_shared[distribute_(tid_x, tid_y, 1, N_TILE)] = rhs[tiling_(k, bid_x, K_TILE, N_TILE, N, 1) + distribute_(tid_x, tid_y, 1, N)];
    __syncthreads();

    // contract at 32x32x32 micro-kernel
    matmul_kernel_32x32x32(lhs_shared, rhs_shared, partial_sum);
    __syncthreads();
  }
  output_shared[distribute_(tid_x, tid_y, 1, N_TILE)] = alpha * partial_sum;
  output[tiling_(bid_y, bid_x, M_TILE, N_TILE, N, 1) + distribute_(tid_x, tid_y, 1, N)] = output_shared[distribute_(tid_x, tid_y, 1, N_TILE)];
}


void catzilla_matmul_v1(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  cudaFuncSetAttribute(_catzilla_matmul_v1<32, 32, 32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v1<32, 32, 32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}


/////////////////////////////////////////////////////////////////////////////////
/// version 2:
/// - basic gmem to smem coalescing
/// - use smem for lhs, rhs, out
/// - thread-level blocking, compute a tile in reg-level C, not a scalar
/////////////////////////////////////////////////////////////////////////////////
template <const int M_TILE_SM, const int N_TILE_SM, const int K_TILE_SM, const int M_TILE_REG, const int N_TILE_REG>
__global__ void _catzilla_matmul_v2(int M, int N, int K, float alpha,
                                       float *lhs, float *rhs,
                                       float beta, float *output) {
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  // smem stationary
  // M=128, N=128, K=32, Mreg=4, Nreg=4
  __shared__ float lhs_shared[M_TILE_SM * K_TILE_SM];
  __shared__ float rhs_shared[K_TILE_SM * N_TILE_SM];
  __shared__ float output_shared[M_TILE_SM * N_TILE_SM];
  // __shared__ float output_shared[1];

  // init output
  output_shared[distribute_(tid_x, tid_y, 1, N_TILE_SM)] = 0.0;

  float partial_sum[M_TILE_REG * N_TILE_REG] = {0.};

  // QZ: per thread m and m+1
  for (int k = 0; k < CEIL_DIV(K, K_TILE_SM); k++) { // range(0, 128, 1)
    // use distribute_ for in-tile thread calculation
    // use tile_ for out-tile thread calculation
    for (int m_reg = 0; m_reg < M_TILE_REG; m_reg++) {
      lhs_shared[m_reg*K_TILE_SM + distribute_(tid_x, tid_y, 1, K_TILE_SM)] = lhs[m_reg*K + tiling_(bid_y, k, M_TILE_SM, K_TILE_SM, K, 1) + distribute_(tid_x, tid_y, 1, K)];
    }
    for (int n_reg = 0; n_reg < N_TILE_REG; n_reg++) {
      rhs_shared[n_reg + distribute_(tid_x, tid_y, 1, N_TILE_SM)] = rhs[n_reg + tiling_(k, bid_x, K_TILE_SM, N_TILE_SM, N, 1) + distribute_(tid_x, tid_y, 1, N)];
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_128x128x32_perthread_4x4(lhs_shared, rhs_shared, partial_sum);
    __syncthreads();

    // store to smem
    for (int m_reg = 0; m_reg < M_TILE_REG; m_reg++)
      for (int n_reg = 0; n_reg < N_TILE_REG; n_reg++)
        output_shared[distribute_(tid_x * N_TILE_REG, tid_y * M_TILE_REG, 1, N_TILE_SM) + distribute_(m_reg, n_reg, N_TILE_SM, 1)] = alpha * partial_sum[m_reg * 4 + n_reg];

    // store to gmem
    for (int m_reg = 0; m_reg < M_TILE_REG; m_reg++)
      for (int n_reg = 0; n_reg < N_TILE_REG; n_reg++)
    output[tiling_(bid_y, bid_x, M_TILE_SM, N_TILE_SM, N, 1) + distribute_(tid_x*N_TILE_REG, tid_y*M_TILE_REG, 1, N) + distribute_(m_reg, n_reg, N, 1)] = output_shared[distribute_(tid_x*N_TILE_REG, tid_y*M_TILE_REG, 1, N_TILE_SM) + distribute_(m_reg, n_reg, N_TILE_SM, 1)];
  }
}


void catzilla_matmul_v2(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  const int M_TILE = 64;
  const int N_TILE = 64;
  const int M_REG = 4;
  const int N_REG = 4;
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(CEIL_DIV(M_TILE, M_REG), CEIL_DIV(N_TILE, N_REG));
  cudaFuncSetAttribute(_catzilla_matmul_v2<M_TILE, N_TILE, 32, M_REG, N_REG>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v2<M_TILE, N_TILE, 32, M_REG, N_REG><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

/////////////////////////////////////////////////////////////////////////////////
/// version 3:
/// - basic gmem to smem coalescing
/// - use smem for lhs, rhs, out
/// - thread-level blocking, compute a tile in reg-level C, not a scalar
/// - refactor with tiling and transfer helpers
/// - try different tiling config with hand
/////////////////////////////////////////////////////////////////////////////////
template <const int M_TILE_SM, const int N_TILE_SM, const int K_TILE_SM, const int M_TILE_REG, const int N_TILE_REG>
__global__ void _catzilla_matmul_v3(int M, int N, int K, float alpha,
                                       float *lhs, float *rhs,
                                       float beta, float *output) {
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  // smem stationary
  // M=128, N=128, K=32, Mreg=4, Nreg=4
  __shared__ float lhs_shared[M_TILE_SM * K_TILE_SM];
  __shared__ float rhs_shared[K_TILE_SM * N_TILE_SM];
  __shared__ float output_shared[M_TILE_SM * N_TILE_SM];
  // __shared__ float output_shared[1];

  // init output
  // output_shared[distribute_(tid_x, tid_y, 1, N_TILE_SM)] = 0.0;

  float partial_sum[M_TILE_REG * N_TILE_REG] = {0.};

  // distribute bid_y for lhs

  // QZ: per thread m and m+1
  for (int k = 0; k < CEIL_DIV(K, K_TILE_SM); k++) { // range(0, 128, 1)
    // use distribute_ for in-tile thread calculation
    // use tile_ for out-tile thread calculation
    // newlhs += tiling_(0, 1, M_TILE_SM, K_TILE_SM, K, 1);
    for (int m_reg = 0; m_reg < M_TILE_REG; m_reg++) {
      lhs_shared[m_reg*K_TILE_SM + distribute_(tid_x, tid_y, 1, K_TILE_SM)] = lhs[m_reg*K + tiling_(bid_y, k, M_TILE_SM, K_TILE_SM, K, 1) + distribute_(tid_x, tid_y, 1, K)];
    }
    for (int n_reg = 0; n_reg < N_TILE_REG; n_reg++) {
      rhs_shared[n_reg + distribute_(tid_x, tid_y, 1, N_TILE_SM)] = rhs[n_reg + tiling_(k, bid_x, K_TILE_SM, N_TILE_SM, N, 1) + distribute_(tid_x, tid_y, 1, N)];
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_128x128x32_perthread_4x4(lhs_shared, rhs_shared, partial_sum);
    __syncthreads();

    // store to smem
    for (int m_reg = 0; m_reg < M_TILE_REG; m_reg++)
      for (int n_reg = 0; n_reg < N_TILE_REG; n_reg++)
        output_shared[distribute_(tid_x * N_TILE_REG, tid_y * M_TILE_REG, 1, N_TILE_SM) + distribute_(m_reg, n_reg, N_TILE_SM, 1)] = alpha * partial_sum[m_reg * 4 + n_reg];

    // store to gmem
    for (int m_reg = 0; m_reg < M_TILE_REG; m_reg++)
      for (int n_reg = 0; n_reg < N_TILE_REG; n_reg++)
    output[tiling_(bid_y, bid_x, M_TILE_SM, N_TILE_SM, N, 1) + distribute_(tid_x*N_TILE_REG, tid_y*M_TILE_REG, 1, N) + distribute_(m_reg, n_reg, N, 1)] = output_shared[distribute_(tid_x*N_TILE_REG, tid_y*M_TILE_REG, 1, N_TILE_SM) + distribute_(m_reg, n_reg, N_TILE_SM, 1)];
  }
}


void catzilla_matmul_v3(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  const int M_TILE = 64;
  const int N_TILE = 64;
  const int M_REG = 4;
  const int N_REG = 4;
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(CEIL_DIV(M_TILE, M_REG), CEIL_DIV(N_TILE, N_REG));
  cudaFuncSetAttribute(_catzilla_matmul_v3<M_TILE, N_TILE, 32, M_REG, N_REG>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v3<M_TILE, N_TILE, 32, M_REG, N_REG><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

#endif // CATZILLA_RECIPES_KERNELS_MATMUL_V1_H_
