#ifndef CATZILLA_RECIPES_UTILS_MICRO_KERNELS_H_
#define CATZILLA_RECIPES_UTILS_MICRO_KERNELS_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "index_utils.h"

using namespace nvcuda;
using namespace nvcuda::wmma;

// TODO: move
inline __device__ void matmul_kernel_32x32x32(float *lhs, float *rhs,
                                              float &out)
{
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int i = 0; i < 32; i++) {
    out += lhs[tid_y * 32 + i] * rhs[i * 32 + tid_x];
  }
  __syncthreads();
  return;
}

inline __device__ void matmul_kernel_16x16x16(half *lhs, half *rhs,
                                              float *out_shared)
{
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  float out = out_shared[distribute_(tid_x, tid_y, 1, 16)];
  for (int i = 0; i < 16; i++) {
    out += __half2float(lhs[tid_y * 16 + i] * rhs[i * 16 + tid_x]);
  }
  out_shared[distribute_(tid_x, tid_y, 1, 16)] = out;
  return;
}

__global__ void convertFloatToHalf(float *input, half *output, int numElements)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    output[idx] = __float2half(input[idx]); // 使用CUDA提供的转换函数
  }
}

// C += A * B
// m16.n16.k16
// row-major, col-major
inline __device__ void matmul_kernel_m16n16k16(half *a_half, half *b_half,
                                               float *c)
{
  // Declare the fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> A_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> C_frag;

  // wmma::fill_fragment(C_frag, 0.0f);

  // NOTE: this is wrong, reinterpret only casts ptr type, not contents
  // half *a_half = reinterpret_cast<half *>(a);
  // half *b_half = reinterpret_cast<half *>(b);

  // Assuming leading dimension is 16
  wmma::load_matrix_sync(A_frag, a_half, 16);
  wmma::load_matrix_sync(B_frag, b_half, 16);
  wmma::load_matrix_sync(C_frag, c, 16, wmma::mem_row_major);

  // Perform the matrix multiplication
  wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

  // Store the result
  wmma::store_matrix_sync(c, C_frag, 16, wmma::mem_row_major);
}

// out[2][2]
inline __device__ void
matmul_kernel_128x128x32_perthread_4x4(float *lhs, float *rhs, float *out)
{
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
        out[m * N_tessel + n] += lhs[tid_y * M_tessel * 32 + m * 32 + k]
                                 * rhs[k * N + N_tessel * tid_x + n];
  __syncthreads();
  return;
}

inline __device__ void
matmul_kernel_64x64x32_perthread_2x2(float *lhs, float *rhs, float *out)
{
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
        out[m * N_tessel + n] += lhs[tid_y * M_tessel * K + m * K + k]
                                 * rhs[k * N + N_tessel * tid_x + n];
  __syncthreads();
  return;
}

inline __device__ void
matmul_kernel_64x64x16_perthread_4x4(float *lhs, float *rhs, float *out)
{
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
        out[m * N_tessel + n] += lhs[tid_y * M_tessel * K + m * K + k]
                                 * rhs[k * N + N_tessel * tid_x + n];
  __syncthreads();
  return;
}

template <const int M, const int N, const int K, const int M_REG,
          const int N_REG>
inline __device__ void matmul_kernel_scalar(float *lhs, float *rhs, float *out)
{
  for (int k = 0; k < K; k++)
#pragma unroll
    for (int m = 0; m < M_REG; m++)
      for (int n = 0; n < N_REG; n++)
        out[m * N_REG + n] += lhs[threadIdx.y * M_REG * K + m * K + k]
                              * rhs[k * N + N_REG * threadIdx.x + n];
  __syncthreads();
  return;
}

template <const int M, const int N, const int K, const int THD_Y,
          const int THD_X>
inline __device__ void matmul_kernel_coalesced(float *lhs, float *rhs,
                                               float *out)
{
  for (int k = 0; k < K; k++)
#pragma unroll
    for (int m = 0; m < CEIL_DIV(M, THD_Y); m++)
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N, THD_X); n++)
        out[m * CEIL_DIV(N, THD_X) + n]
          += lhs[m * THD_Y * K + threadIdx.y * K + k]
             * rhs[k * N + THD_X * n + threadIdx.x];
  __syncthreads();
  return;
}

// template <const int M, const int N, const int K, const int THD_Y, const int
// THD_X> inline __device__ void matmul_kernel_coalesced_v(float4 *lhs, float4
// *rhs,
//                                                float4 *out) {
//   auto lhs_data = (float*) lhs; // M, K
//   auto rhs_data = (float*) rhs; // K, N
//   auto out_data = (float*) out; // M, N
//
//   for (int k = 0; k < K; k++)
//     #pragma unroll
//     for (int m = 0; m < CEIL_DIV(M, THD_Y); m++)
//       #pragma unroll
//       for (int n = 0; n < CEIL_DIV(N, THD_X); n++)
//         out[m * CEIL_DIV(N, THD_X) + n] +=
//             lhs[m * THD_Y * K + threadIdx.y * K + k] *
//             rhs[k * N + THD_X * n + threadIdx.x];
//   __syncthreads();
//   return;
// }

template <const int M, const int N, const int K, const int THD_Y,
          const int THD_X>
inline __device__ void matmul_kernel_xor_swizzled(float *lhs, float *rhs,
                                                  float *out)
{
  int x_swz = xor_swizzle(threadIdx.x);
  // int x_swz = threadIdx.x;
  for (int k = 0; k < K; k++)
#pragma unroll
    for (int m = 0; m < CEIL_DIV(M, THD_Y); m++)
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N, THD_X); n++)
        out[m * CEIL_DIV(N, THD_X) + n]
          += lhs[m * THD_Y * K + threadIdx.y * K + k]
             * rhs[k * N + THD_X * n + x_swz];
  __syncthreads();
  return;
}

template <const int M, const int N, const int K, const int THD_Y,
          const int THD_X>
inline __device__ void matmul_kernel_pad_swizzled(float *lhs, float *rhs,
                                                  float *out)
{
  for (int k = 0; k < K; k++)
#pragma unroll
    for (int m = 0; m < CEIL_DIV(M, THD_Y); m++)
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N, THD_X); n++)
        out[m * CEIL_DIV(N, THD_X) + n]
          += lhs[m * THD_Y * (K + 1) + threadIdx.y * (K + 1) + k]
             * rhs[k * (N + 1) + THD_X * n + threadIdx.x];
  __syncthreads();
  return;
}

#endif // CATZILLA_RECIPES_UTILS_MICRO_KERNELS_H_
