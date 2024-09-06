#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// BM=128, BN=128, BK=8, TM=8, TN=8
// GD=128, 128
// TD=16, 16 = 256
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm2DBlocktiling(int M, int N, int K, float alpha, float *A,
                                   float *B, float beta, float *C) {
  int bx = blockIdx.x; // 128
  int by = blockIdx.y; // 128

  int block_row_thread = BN / TN; // 16
  int block_col_thread = BM / TM; // 16
  int thread_num = block_row_thread *
                   block_col_thread; // 一个线程负责计算block中TM*TN个元素 256

  int tx = (threadIdx.x % block_row_thread) * TN; // 0 16 32 64 ... 128
  int ty = (threadIdx.x / block_row_thread) *
           TM; // 0 0 0 0 16 16 16 16 ... 128 128 128 128

  __shared__ float As[BM * BK]; // 128 * 8 = 1024
  __shared__ float Bs[BK * BN]; // 1024

  // 移动到当前block, tile to SM
  A = &A[by * BM * K];
  B = &B[bx * BN];
  C = &C[by * BM * N + bx * BN];

  /*
  当前线程负责搬运全局内存中第a_tile_row行，第a_tile_col列元素至共享内存第a_tile_row行，第a_tile_col列
  a_tile_stride表示block中线程可搬运a_tile_stride行至共享内存；

  若BM=64,BK=8,thread_num=512,则a_tile_stride=64,a_tile_stride=BM，表示每个线程搬运一轮即可完成所需元素的搬运;
  若BM=128,BK=8,thread_num=512,则a_tile_stride=64,表示每个线程搬运两轮即可完成所需元素的搬运;
  */
  int a_tile_row = threadIdx.x / BK;
  int a_tile_col = threadIdx.x % BK;
  int a_tile_stride = thread_num / BK;

  int b_tile_row = threadIdx.x / BN;
  int b_tile_col = threadIdx.x % BN;
  int b_tile_stride = thread_num / BN;

  float tmp[TM][TN] = {
      0.}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；

#pragma unroll
  for (int k = 0; k < K; k += BK) {
#pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride) {
      As[(a_tile_row + i) * BK + a_tile_col] =
          A[(a_tile_row + i) * K + a_tile_col];
    }
#pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
      Bs[(b_tile_row + i) * BN + b_tile_col] =
          B[(b_tile_row + i) * N + b_tile_col];
    }
    __syncthreads();
    A += BK;
    B += BK * N;
#pragma unroll
    for (int i = 0; i < BK; i++) {
// #pragma unroll
//       for (int j = 0; j < TM; j++) {
//         a_frag[j] = As[(ty + j) * BK + i];
//       }
// #pragma unroll
//       for (int l = 0; l < TN; l++) {
//         b_frag[l] = Bs[tx + l + i * BN];
//       }
#pragma unroll
      for (int j = 0; j < TM; j++) {
#pragma unroll
        for (int l = 0; l < TN; l++)
          tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];
        // tmp[j][l] += a_frag[j] * b_frag[l];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < TM; j++) {
    for (int l = 0; l < TN; l++)
      C[(ty + j) * N + tx + l] =
          alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
  }
}

void runSgemm2DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BK = 16;
  const uint TM = 8;
  const uint TN = 8;
  printf("hello");
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    const uint M_THREAD = CEIL_DIV(BM, TM); // 16
    const uint N_THREAD = CEIL_DIV(BN, TN); // 16
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim(M_THREAD, N_THREAD);
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    const uint M_THREAD = CEIL_DIV(BM, TM); // 16
    const uint N_THREAD = CEIL_DIV(BN, TN); // 16
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim(M_THREAD, N_THREAD);
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}
