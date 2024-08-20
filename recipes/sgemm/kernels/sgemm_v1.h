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

__device__ void matmul_kernel_32x32x32(float* lhs, float* rhs, float& out) {
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int i = 0; i < 32; i++) {
      out += lhs[tid_y * 32 + i] * rhs[i * 32 + tid_x];
  }
  return;
}

__device__ int tile_addr_(int x_offset, int y_offset, int x_stride, int y_stride) {
  int idx = x_offset * x_stride + y_offset * y_stride;
  return idx;
}

__device__ int tile_addr_ex_(int x_tile, int y_tile, int x_offset, int y_offset, int x_tile_size, int y_tile_size, int x_stride, int y_stride) {
  int idx = x_tile * x_tile_size * x_stride + y_tile * y_tile_size * y_stride + x_offset * x_stride + y_offset * y_stride;
  return idx;
}

/*
 * gmem access coalescing
 * smem stationary
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
  output_shared[tile_addr_(tid_x, tid_y, 1, N_TILE)] = 0.0;

  float partial_sum = 0.;
  for (int k = 0; k < CEIL_DIV(K, K_TILE); k++) { // range(0, 128, 1)
      // 缓存A_tile和B_tile
      // lhs_shared[tile_addr_(tid_x, tid_y, 1, K_TILE)] = lhs[bid_y * M_TILE * K + k * K_TILE + tid_y * K + tid_x];
      lhs_shared[tile_addr_(tid_x, tid_y, 1, K_TILE)] = lhs[tile_addr_ex_(bid_y, k, tid_y, tid_x, M_TILE, K_TILE, K, 1)];
      // rhs_shared[tile_addr_(tid_x, tid_y, 1, N_TILE)] = rhs[bid_x * N_TILE + tid_y * N + k * K_TILE * N + tid_x];
      rhs_shared[tile_addr_(tid_x, tid_y, 1, N_TILE)] = rhs[tile_addr_ex_(k, bid_x, tid_y, tid_x, K_TILE, N_TILE, N, 1)];
      // 同步所有线程缓存完成
      // __syncthreads();
      // float partial_sum = 0.;
      // for (int i = 0; i < K_TILE; i++) {
      //     partial_sum += lhs_shared[tid_y * K_TILE + i] * rhs_shared[i * N_TILE + tid_x];
      // }
      matmul_kernel_32x32x32(lhs_shared, rhs_shared, partial_sum);
      // output_shared[tid_y * N_TILE + tid_x] += alpha * partial_sum;
      // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
      __syncthreads();
  }
  output_shared[tile_addr_(tid_x, tid_y, 1, N_TILE)] = alpha * partial_sum;
  output[tile_addr_ex_(bid_y, bid_x, tid_y, tid_x, M_TILE, N_TILE, N, 1)] = output_shared[tile_addr_(tid_x, tid_y, 1, N_TILE)];//alpha * partial_sum;// + beta * output[bid_y * M_TILE * N + bid_x * N_TILE + tid_y * N + tid_x];
  
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
