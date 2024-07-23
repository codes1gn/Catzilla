#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template<const int BM,
         const int BN,
         const int BK,
         const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  int thread_num = BM * BN / TM;

  int tx = threadIdx.x % BN;
  int ty = threadIdx.x / BN * TM;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A = &A[by * BM * K];
  B = &B[bx * BN];
  C = &C[by * BM * N + bx * BN];

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  int a_tile_row = threadIdx.x / BK;
  int a_tile_col = threadIdx.x % BK;
  int b_tile_row = threadIdx.x / BN;
  int b_tile_col = threadIdx.x % BN;

  int a_tile_stride = thread_num / BK;
  int b_tile_stride = thread_num / BN;

  // allocate thread-local cache for results in registerfile
  float tmp[TM] = {0.};

  #pragma unroll
  for (int k = 0; k < K; k += BK) {
      #pragma unroll
      for (int i = 0; i < BM; i += a_tile_stride) {
          As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
      }
      #pragma unroll
      for (int i = 0; i < BK; i += b_tile_stride) {
          Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
      }
      __syncthreads();
      A += BK;
      B += BK * N;
      #pragma unroll
      for (int i = 0; i < BK; i++) {
          tmp[TM] = Bs[tx + i * BN];
          #pragma unroll
          for (int j = 0; j < TM; j++) {
              tmp[j] += As[(ty + j) * BK + i] * tmp[TM];
          }
      }
      __syncthreads();
  }
  #pragma unroll
  for (int j = 0; j < TM; j++) {
      C[(ty + j) * N + tx] = alpha * tmp[j] + beta * C[(ty + j) * N + tx];
  }
}
