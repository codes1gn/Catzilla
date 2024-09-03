#ifndef CATZILLA_RECIPES_UTILS_MICRO_KERNELS_H_
#define CATZILLA_RECIPES_UTILS_MICRO_KERNELS_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// TODO: move
inline __device__ void matmul_kernel_32x32x32(float* lhs, float* rhs, float& out) {
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int i = 0; i < 32; i++) {
      out += lhs[tid_y * 32 + i] * rhs[i * 32 + tid_x];
  }
  __syncthreads();
  return;
}

// out[2][2]
inline __device__ void matmul_kernel_128x128x32_perthread_4x4(float* lhs, float* rhs, float* out) {
  int K = 32;
  int M = 128;
  int N = 128;
  int M_REG = 4;
  int N_REG = 4;
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int k = 0; k < K; k++)
    #pragma unroll
    for (int m = 0; m < M_REG; m++)
      for (int n = 0; n < N_REG; n++)
        out[m * N_REG + n] += lhs[tid_y * M_REG * 32 + m * 32 + k] * rhs[k * N + N_REG * tid_x + n];
  return;
}

inline __device__ void matmul_kernel_64x64x32_perthread_2x2(float* lhs, float* rhs, float* out) {
  int K = 32;
  int M = 64;
  int N = 64;
  int M_REG = 2;
  int N_REG = 2;
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int k = 0; k < K; k++)
    #pragma unroll
    for (int m = 0; m < M_REG; m++)
      for (int n = 0; n < N_REG; n++)
        out[m * N_REG + n] += lhs[tid_y * M_REG * K + m * K + k] * rhs[k * N + N_REG * tid_x + n];
  return;
}


#endif // CATZILLA_RECIPES_UTILS_MICRO_KERNELS_H_
