#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  const int BM = BLOCKSIZE;
  const int BN = BLOCKSIZE;
  const int BK = BLOCKSIZE;

  int tx = threadIdx.x % BN;
  int ty = threadIdx.x / BN;

  // 申请共享内存空间
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 移动到当前block
  A = &A[by * BM * K];
  B = &B[bx * BN];
  C = &C[by * BM * N + bx * BN];

  float tmp = 0.;
  for (int k = 0; k < K; k += BK) {
      // 缓存A_tile和B_tile
      As[ty * BK + tx] = A[ty * K + tx];
      Bs[ty * BN + tx] = B[ty * N + tx];
      // 同步所有线程缓存完成
      __syncthreads();
      A += BK;
      B += BK * N;
      for (int i = 0; i < BK; i++) {
          tmp += As[ty * BK + i] * Bs[i * BN + tx];
      }
      // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
      __syncthreads();
  }
  C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}
