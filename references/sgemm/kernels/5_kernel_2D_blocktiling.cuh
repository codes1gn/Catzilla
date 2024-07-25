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
        const int TM,
        const int TN>
__global__ void sgemm2DBlocktiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread; // 一个线程负责计算block中TM*TN个元素

    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
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

    float tmp[TM][TN] = {0.}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；
    float a_frag[TM] = {0.};
    float b_frag[TN] = {0.};

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
            #pragma unroll
            for (int j = 0; j < TM; j++) {
                a_frag[j] = As[(ty + j) * BK + i];
            }
            #pragma unroll
            for (int l = 0; l < TN; l++) {
                b_frag[l] = Bs[tx + l + i * BN];
            }
            #pragma unroll
            for (int j = 0; j < TM; j++) {
                #pragma unroll
                for (int l = 0; l < TN; l++)
                    tmp[j][l] += a_frag[j] * b_frag[l];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++)
            C[(ty + j) * N + tx + l] = alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
    }
}
// template <const int BM, const int BN, const int BK, const int TM, const int TN>
// __global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
//     sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
//                        const float *B, float beta, float *C) {
//   const uint cRow = blockIdx.y;
//   const uint cCol = blockIdx.x;
//
//   const uint totalResultsBlocktile = BM * BN;
//   // A thread is responsible for calculating TM*TN elements in the blocktile
//   const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
//
//   // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
//   assert(numThreadsBlocktile == blockDim.x);
//
//   // BN/TN are the number of threads to span a column
//   const int threadCol = threadIdx.x % (BN / TN);
//   const int threadRow = threadIdx.x / (BN / TN);
//
//   // allocate space for the current blocktile in smem
//   __shared__ float As[BM * BK];
//   __shared__ float Bs[BK * BN];
//
//   // Move blocktile to beginning of A's row and B's column
//   A += cRow * BM * K;
//   B += cCol * BN;
//   C += cRow * BM * N + cCol * BN;
//
//   // calculating the indices that this thread will load into SMEM
//   const uint innerRowA = threadIdx.x / BK;
//   const uint innerColA = threadIdx.x % BK;
//   // calculates the number of rows of As that are being loaded in a single step
//   // by a single block
//   const uint strideA = numThreadsBlocktile / BK;
//   const uint innerRowB = threadIdx.x / BN;
//   const uint innerColB = threadIdx.x % BN;
//   // for both As and Bs we want each load to span the full column-width, for
//   // better GMEM coalescing (as opposed to spanning full row-width and iterating
//   // across columns)
//   const uint strideB = numThreadsBlocktile / BN;
//
//   // allocate thread-local cache for results in registerfile
//   float threadResults[TM * TN] = {0.0};
//   // register caches for As and Bs
//   float regM[TM] = {0.0};
//   float regN[TN] = {0.0};
//
//   // outer-most loop over block tiles
//   for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
//     // populate the SMEM caches
//     for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
//       As[(innerRowA + loadOffset) * BK + innerColA] =
//           A[(innerRowA + loadOffset) * K + innerColA];
//     }
//     for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
//       Bs[(innerRowB + loadOffset) * BN + innerColB] =
//           B[(innerRowB + loadOffset) * N + innerColB];
//     }
//     __syncthreads();
//
//     // advance blocktile
//     A += BK;     // move BK columns to right
//     B += BK * N; // move BK rows down
//
//     // calculate per-thread results
//     for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
//       // block into registers
//       for (uint i = 0; i < TM; ++i) {
//         regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
//       }
//       for (uint i = 0; i < TN; ++i) {
//         regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
//       }
//       for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
//         for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
//           threadResults[resIdxM * TN + resIdxN] +=
//               regM[resIdxM] * regN[resIdxN];
//         }
//       }
//     }
//     __syncthreads();
//   }
//
//   // write out the results
//   #pragma unroll
//   for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
//     for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
//       C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
//           alpha * threadResults[resIdxM * TN + resIdxN] +
//           beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
//     }
//   }
// }
