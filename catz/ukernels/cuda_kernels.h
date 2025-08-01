#ifndef CATZILLA_RECIPES_UTILS_MICRO_KERNELS_RAW_KERNELS_H_
#define CATZILLA_RECIPES_UTILS_MICRO_KERNELS_RAW_KERNELS_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "matrix.h"
#include "matrix_legacy.h"

using namespace catz;

namespace catz::cuda {

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

inline __device__ void
matmul_kernel_16x16x16_thread_16x16(float *lhs, float *rhs, float *out) {
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int i = 0; i < 16; i++) {
    out[tid_y * 16 + tid_x] += lhs[tid_y * 16 + i] * rhs[i * 16 + tid_x];
  }
  __syncthreads();
  return;
}

inline __device__ void cuda_m16n8k16_f16f32(float *out_shared, const half *lhs,
                                            const half *rhs,
                                            const float *bias) {
  int row_offset = 2 * (threadIdx.x / 4);
  int col_offset = 2 * (threadIdx.x % 4);

  for (int row = row_offset; row < row_offset + 2; row++) {
    for (int col = col_offset; col < col_offset + 2; col++) {
      for (int k = 0; k < 16; k++) {
        out_shared[row * 8 + col] =
            __half2float(lhs[row * 16 + k] * rhs[k * 8 + col]) +
            bias[row * 8 + col];
      }
    }
  }
  return;
}

inline __device__ void cuda_m16n8k16_f16f32(float *out_shared, const float *lhs,
                                            const float *rhs,
                                            const float *bias) {
  int row_offset = threadIdx.x / 4;
  int col_offset = threadIdx.x % 4;

  for (int row = row_offset; row < row_offset + 2; row++) {
    for (int col = col_offset; col < col_offset + 2; col++) {
      for (int k = 0; k < 16; k++) {
        out_shared[row * 8 + col] =
            lhs[row * 16 + k] * rhs[k * 8 + col] + bias[row * 8 + col];
      }
    }
  }
  return;
}

inline __device__ void matmul_kernel_16x16x16_thread_32(MatrixDyn<float> lhs,
                                                        MatrixDyn<float> rhs,
                                                        float *out_shared) {
  int tid_x = threadIdx.x % 16;
  int tid_y = threadIdx.x / 16; // 0-1
  for (int pos = 0; pos < 8; pos++) {
    float out = out_shared[distribute_(tid_x, tid_y + 2 * pos, 1, 16)];
    for (int i = 0; i < 16; i++) {
      out += lhs.data[pos * 32 + tid_y * 16 + i] * rhs.data[i * 16 + tid_x];
    }
    out_shared[distribute_(tid_x, tid_y + 2 * pos, 1, 16)] = out;
  }
  return;
}

inline __device__ void matmul_kernel_16x16x16_thread_32(MatrixDyn<half> lhs,
                                                        MatrixDyn<half> rhs,
                                                        float *out_shared) {
  int tid_x = threadIdx.x % 16;
  int tid_y = threadIdx.x / 16; // 0-1
  for (int pos = 0; pos < 8; pos++) {
    float out = out_shared[distribute_(tid_x, tid_y + 2 * pos, 1, 16)];
    for (int i = 0; i < 16; i++) {
      out += __half2float(lhs.data[pos * 32 + tid_y * 16 + i] *
                          rhs.data[i * 16 + tid_x]);
    }
    out_shared[distribute_(tid_x, tid_y + 2 * pos, 1, 16)] = out;
  }
  return;
}

inline __device__ void matmul_kernel_16x16x16_thread_32(float *lhs, float *rhs,
                                                        float *out_shared) {
  int tid_x = threadIdx.x % 16;
  int tid_y = threadIdx.x / 16; // 0-1
  for (int pos = 0; pos < 8; pos++) {
    float out = out_shared[distribute_(tid_x, tid_y + 2 * pos, 1, 16)];
    for (int i = 0; i < 16; i++) {
      out += lhs[pos * 32 + tid_y * 16 + i] * rhs[i * 16 + tid_x];
    }
    out_shared[distribute_(tid_x, tid_y + 2 * pos, 1, 16)] = out;
  }
  return;
}

inline __device__ void matmul_kernel_16x16x16_thread_32(half *lhs, half *rhs,
                                                        float *out_shared) {
  int tid_x = threadIdx.x % 16;
  int tid_y = threadIdx.x / 16; // 0-1
  for (int pos = 0; pos < 8; pos++) {
    float out = out_shared[distribute_(tid_x, tid_y + 2 * pos, 1, 16)];
    for (int i = 0; i < 16; i++) {
      out += __half2float(lhs[pos * 32 + tid_y * 16 + i] * rhs[i * 16 + tid_x]);
    }
    out_shared[distribute_(tid_x, tid_y + 2 * pos, 1, 16)] = out;
  }
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
  __syncthreads();
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

template <const int M, const int N, const int K, const int THD_Y,
          const int THD_X>
inline __device__ void matmul_kernel_coalesced(float *lhs, float *rhs,
                                               float *out) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int tid_y = tid / THD_X;
  int tid_x = tid % THD_X;
  for (int k = 0; k < K; k++) {
    __syncthreads();
    // if (tid == 0) printf("k = %d\n", k);
    for (int m = 0; m < CEIL_DIV(M, THD_Y); m+=1) {
      // if (tid == 0) printf("  m = %d\n", m);
      for (int n = 0; n < CEIL_DIV(N, THD_X); n+=1) {
        // if (tid == 0) printf("    n = %d\n", n);
        out[(m) * CEIL_DIV(N, THD_X) + (n)] +=
            lhs[(m * THD_X + tid_y) * K + k] * rhs[k * N + (n * THD_X + tid_x)];
        // if (tid == 0) printf("    output offset = %d\n", (m) * CEIL_DIV(N, THD_X) + (n));
      }
    }
  }
  __syncthreads();
  return;
}

// SPECILISED KERNEL FOR CHOREO, INPS/OUTS are all SM buffer
template <const int M, const int N, const int K, const int THD_Y,
          const int THD_X>
inline __device__ void matmul_kernel_sm_stationary(float *lhs, float *rhs,
                                                   float *out) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  // TODO: fix it
  int tid_y = tid / THD_X;
  int tid_x = tid % THD_X;
  for (int m = 0; m < M; m+=THD_Y)
    for (int n = 0; n < N; n+=THD_X)
      for (int k = 0; k < K; k++)
        out[(m + tid_y) * N + (n + tid_x)] +=
            lhs[(m + tid_y) * K + k] * rhs[k * N + (n + tid_x)];
  __syncthreads();
  return;
}

template <const int M, const int N, const int K, const int THD_Y,
          const int THD_X>
inline __device__ void matmul_kernel_sm_stationary(half *lhs, half *rhs,
                                                   float *out) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int tid_y = tid / 32;
  int tid_x = tid % 32;
  for (int m = 0; m < CEIL_DIV(M, THD_Y); m++)
    for (int n = 0; n < CEIL_DIV(N, THD_X); n++)
      for (int k = 0; k < K; k++)
        out[m * THD_Y * N + tid_y * N + n * THD_X + tid_x] +=
            __half2float(lhs[m * THD_Y * K + tid_y * K + k] *
                         rhs[k * N + THD_X * n + tid_x]);
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

template <const int M, const int N, const int K, const int THD_Y,
          const int THD_X>
inline __device__ void matmul_kernel_pad_swizzled(float *lhs, float *rhs,
                                                  float *out) {
  __syncthreads();
  for (int k = 0; k < K; k++) {
    __syncthreads();
#pragma unroll
    for (int m = 0; m < CEIL_DIV(M, THD_Y); m++)
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N, THD_X); n++)
        out[m * CEIL_DIV(N, THD_X) + n] +=
            lhs[m * THD_Y * (K + 1) + threadIdx.y * (K + 1) + k] *
            rhs[k * (N + 1) + THD_X * n + threadIdx.x];

  }
  __syncthreads();
  return;
}

// first kernel use MatrixDyn<float> as type
// TODO: merge generics
template <const int M, const int N>
inline __device__ void identity(MatrixDyn<float> inp, MatrixDyn<float> out) {
  int THREADS = blockDim.x * blockDim.y;
  int ELEMENTS = M * N;
  int CHUNKS = CEIL_DIV(ELEMENTS, THREADS);
  int ROWPERCK = CEIL_DIV(THREADS, N);
  for (int ck = 0; ck < CHUNKS; ck++)
    out.tile(CoordDyn(ck, 0), CoordDyn(ROWPERCK, N)).dist_to_thread() =
        inp.tile(CoordDyn(ck, 0), CoordDyn(ROWPERCK, N)).dist_to_thread();
  __syncthreads();
  return;
}

template <const int M, const int N>
inline __device__ void identity(MatrixDyn<half> inp, MatrixDyn<float> out) {
  int THREADS = blockDim.x * blockDim.y;
  int ELEMENTS = M * N;
  int CHUNKS = CEIL_DIV(ELEMENTS, THREADS);
  int ROWPERCK = CEIL_DIV(THREADS, N);
  for (int ck = 0; ck < CHUNKS; ck++)
    out.tile(CoordDyn(ck, 0), CoordDyn(ROWPERCK, N)).dist_to_thread() =
        inp.tile(CoordDyn(ck, 0), CoordDyn(ROWPERCK, N)).dist_to_thread();
  __syncthreads();
  return;
}

} // namespace catz::cuda

#endif // CATZILLA_RECIPES_UTILS_MICRO_KERNELS_RAW_KERNELS_H_
