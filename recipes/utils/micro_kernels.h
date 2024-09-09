#ifndef CATZILLA_RECIPES_UTILS_MICRO_KERNELS_H_
#define CATZILLA_RECIPES_UTILS_MICRO_KERNELS_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "index_utils.h"

// TODO: move
inline __device__ void matmul_kernel_32x32x32(float *lhs, float *rhs,
                                              float &out) {
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int i = 0; i < 32; i++) {
    out += lhs[tid_y * 32 + i] * rhs[i * 32 + tid_x];
  }
  __syncthreads();
  return;
}

// out[2][2]
inline __device__ void
matmul_kernel_128x128x32_perthread_4x4(float *lhs, float *rhs, float *out) {
  int K = 32;
  int M = 128;
  int N = 128;
  int M_tessel = 4;
  int N_tessel = 4;
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int k = 0; k < K; k++)
#pragma unroll
    for (int m = 0; m < M_tessel; m++)
      for (int n = 0; n < N_tessel; n++)
        out[m * N_tessel + n] += lhs[tid_y * M_tessel * 32 + m * 32 + k] *
                                 rhs[k * N + N_tessel * tid_x + n];
  __syncthreads();
  return;
}

inline __device__ void
matmul_kernel_64x64x32_perthread_2x2(float *lhs, float *rhs, float *out) {
  int K = 32;
  int M = 64;
  int N = 64;
  int M_tessel = 2;
  int N_tessel = 2;
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int k = 0; k < K; k++)
#pragma unroll
    for (int m = 0; m < M_tessel; m++)
      for (int n = 0; n < N_tessel; n++)
        out[m * N_tessel + n] += lhs[tid_y * M_tessel * K + m * K + k] *
                                 rhs[k * N + N_tessel * tid_x + n];
  __syncthreads();
  return;
}

inline __device__ void
matmul_kernel_64x64x16_perthread_4x4(float *lhs, float *rhs, float *out) {
  int K = 16;
  int M = 64;
  int N = 64;
  int M_tessel = 4;
  int N_tessel = 4;
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int k = 0; k < K; k++)
#pragma unroll
    for (int m = 0; m < M_tessel; m++)
      for (int n = 0; n < N_tessel; n++)
        out[m * N_tessel + n] += lhs[tid_y * M_tessel * K + m * K + k] *
                                 rhs[k * N + N_tessel * tid_x + n];
  __syncthreads();
  return;
}

template <const int M, const int N, const int K, const int M_REG,
          const int N_REG>
inline __device__ void matmul_kernel_scalar(float *lhs, float *rhs,
                                            float *out) {
  for (int k = 0; k < K; k++)
#pragma unroll
    for (int m = 0; m < M_REG; m++)
      for (int n = 0; n < N_REG; n++)
        out[m * N_REG + n] += lhs[threadIdx.y * M_REG * K + m * K + k] *
                              rhs[k * N + N_REG * threadIdx.x + n];
  __syncthreads();
  return;
}

template <const int M, const int N, const int K, const int THD_Y, const int THD_X>
inline __device__ void matmul_kernel_coalesced(float *lhs, float *rhs,
                                               float *out) {
  for (int k = 0; k < K; k++)
    #pragma unroll
    for (int m = 0; m < CEIL_DIV(M, THD_Y); m++)
      #pragma unroll
      for (int n = 0; n < CEIL_DIV(N, THD_X); n++)
        out[m * CEIL_DIV(N, THD_X) + n] +=
            lhs[m * THD_Y * K + threadIdx.y * K + k] *
            rhs[k * N + THD_X * n + threadIdx.x];
  __syncthreads();
  return;
}

template <const int M, const int N, const int K, const int THD_Y, const int THD_X>
inline __device__ void matmul_kernel_xor_swizzled(float *lhs, float *rhs,
                                               float *out) {
  int x_swz = xor_swizzle(threadIdx.x);
  // int x_swz = threadIdx.x;
  for (int k = 0; k < K; k++)
    #pragma unroll
    for (int m = 0; m < CEIL_DIV(M, THD_Y); m++)
      #pragma unroll
      for (int n = 0; n < CEIL_DIV(N, THD_X); n++)
        out[m * CEIL_DIV(N, THD_X) + n] +=
            lhs[m * THD_Y * K + threadIdx.y * K + k] *
            rhs[k * N + THD_X * n + x_swz];
  __syncthreads();
  return;
}


template <const int M, const int N, const int K, const int THD_Y, const int THD_X>
inline __device__ void matmul_kernel_pad_swizzled(float *lhs, float *rhs,
                                               float *out) {
  for (int k = 0; k < K; k++)
    #pragma unroll
    for (int m = 0; m < CEIL_DIV(M, THD_Y); m++)
      #pragma unroll
      for (int n = 0; n < CEIL_DIV(N, THD_X); n++)
        out[m * CEIL_DIV(N, THD_X) + n] +=
            lhs[m * THD_Y * (K+1) + threadIdx.y * (K+1) + k] *
            rhs[k * (N+1) + THD_X * n + threadIdx.x];
  __syncthreads();
  return;
}

#endif // CATZILLA_RECIPES_UTILS_MICRO_KERNELS_H_
