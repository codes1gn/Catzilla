#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template<const int M_TILE,
         const int N_TILE,
         const int K_TILE,
         const int M_THREAD,
         const int N_THREAD>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const int TM = CEIL_DIV(M_TILE*N_TILE, M_THREAD);

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  int tx = threadIdx.x % N_TILE;
  int ty = threadIdx.x * M_TILE / M_THREAD;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[M_TILE * K_TILE];
  __shared__ float Bs[K_TILE * N_TILE];

  // Move blocktile to beginning of A's row and B's column
  A = &A[blockIdx.y * M_TILE * K];
  B = &B[blockIdx.x * N_TILE];
  C = &C[blockIdx.y * M_TILE * N + blockIdx.x * N_TILE];

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  int a_tile_row = threadIdx.x / K_TILE;
  int a_tile_col = threadIdx.x % K_TILE;
  int b_tile_row = threadIdx.x / N_TILE;
  int b_tile_col = threadIdx.x % N_TILE;

  int a_tile_stride = M_THREAD / K_TILE;
  int b_tile_stride = M_THREAD / N_TILE;

  // allocate thread-local cache for results in registerfile
  float tmp[TM] = {0.};

  #pragma unroll
  for (int k = 0; k < K; k += K_TILE) {
    #pragma unroll
    for (int i = 0; i < M_TILE; i += a_tile_stride) {
      As[(a_tile_row + i) * K_TILE + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
    }
    #pragma unroll
    for (int i = 0; i < K_TILE; i += b_tile_stride) {
      Bs[(b_tile_row + i) * N_TILE + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
    }
    __syncthreads();
    A += K_TILE;
    B += K_TILE * N;
    #pragma unroll
    for (int i = 0; i < K_TILE; i++) {
      tmp[TM] = Bs[tx + i * N_TILE];
      #pragma unroll
      for (int j = 0; j < TM; j++) {
        tmp[j] += As[(ty + j) * K_TILE + i] * tmp[TM];
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int j = 0; j < TM; j++) {
    C[(ty + j) * N + tx] = alpha * tmp[j] + beta * C[(ty + j) * N + tx];
  }
}

void runSgemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint M_TILE = 64;
  const uint N_TILE = 64;
  const uint K_TILE = 8;
  const uint M_THREAD = 512;
  const uint N_THREAD = 1;
  dim3 gridDim(CEIL_DIV(N, N_TILE), CEIL_DIV(M, M_TILE));
  dim3 blockDim(M_THREAD);
  sgemm1DBlocktiling<M_TILE, N_TILE, K_TILE, M_THREAD, N_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

