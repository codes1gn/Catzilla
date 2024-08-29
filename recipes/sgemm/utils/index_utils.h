#ifndef CATZILLA_RECIPES_UTILS_INDEX_UTILS_H_
#define CATZILLA_RECIPES_UTILS_INDEX_UTILS_H_

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
  return;
}

// out[2][2]
inline __device__ void matmul_kernel_128x128x32_perthread_4x4(float* lhs, float* rhs, float* out) {
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
        out[m * N_tessel + n] += lhs[tid_y * M_tessel * 32 + m * 32 + k] * rhs[k * N + N_tessel * tid_x + n];
  return;
}


// convert multi-dim index to flatten-index
inline __device__ int at_thread_(int x_offset, int y_offset, int x_stride, int y_stride) {
  int idx = x_offset * x_stride + y_offset * y_stride;
  return idx;
}

// convert multi-dim index of a tile-block to flatten-index
inline __device__ int at_tile_(int x_tile, int y_tile, int x_tile_size, int y_tile_size, int x_stride, int y_stride) {
  int idx = x_tile * x_tile_size * x_stride + y_tile * y_tile_size * y_stride;
  return idx;
}

#endif // CATZILLA_RECIPES_UTILS_INDEX_UTILS_H_
