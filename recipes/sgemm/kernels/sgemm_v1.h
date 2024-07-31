#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

#define CEIL_DIV(dividend, divisor) (((dividend) + (divisor)-1) / (divisor))

template <const int M_TILE, const int N_TILE, const int K_TILE>
__global__ void catzilla_sgemm_v1(int M, int N, int K, float alpha,
                                       float *lhs, float *rhs,
                                       float beta, float *output) {
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  // 申请共享内存空间
  __shared__ float lhs_shared[M_TILE * K_TILE];
  __shared__ float rhs_shared[K_TILE * N_TILE];
  __shared__ float output_shared[M_TILE * N_TILE];

  float tmp = 0.;
  // TODO change to tiled size, not full size
  for (int k = 0; k < K; k += K_TILE) {
      // 缓存A_tile和B_tile
      lhs_shared[tid_y * K_TILE + tid_x] = lhs[(bid_y * M_TILE + tid_y) * K + k + tid_x];
      rhs_shared[tid_y * N_TILE + tid_x] = rhs[bid_x * N_TILE + tid_y * N + k * N + tid_x];
      // 同步所有线程缓存完成
      __syncthreads();
      for (int i = 0; i < K_TILE; i++) {
          tmp += lhs_shared[tid_y * K_TILE + i] * rhs_shared[i * N_TILE + tid_x];
      }
      // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
      __syncthreads();
  }
  output_shared[tid_y * N_TILE + tid_x] = alpha * tmp;
  output[bid_y * M_TILE * N + bid_x * N_TILE + tid_y * N + tid_x] = output_shared[tid_y * N_TILE + tid_x];//alpha * tmp;// + beta * output[bid_y * M_TILE * N + bid_x * N_TILE + tid_y * N + tid_x];
}


inline void catzilla_sgemm_v1_host(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  cudaFuncSetAttribute(catzilla_sgemm_v1<32, 32, 32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  catzilla_sgemm_v1<32, 32, 32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
