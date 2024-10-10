#ifndef CATZILLA_RECIPES_MATMUL_PAD_SWIZZLED_H_
#define CATZILLA_RECIPES_MATMUL_PAD_SWIZZLED_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "macro.h"
#include "matrix.h"
#include "ukernels.h"

using namespace catz;
using namespace catz::cuda;
using namespace catz::wmma;
using namespace catz::mma;

namespace catz::recipes
{

// hand tuned for HPs
// use templated kernel
// PAD-SWIZZLE
template <const int M_TILE, const int N_TILE, const int K_TILE,
          const int X_THREAD, const int Y_THREAD>
__global__ void _matmul_pad_swizzled(int M, int N, int K, float alpha,
                                     float *lhs, float *rhs, float beta,
                                     float *out)
{
  auto lhs_shape = make_coord(M, K);
  auto rhs_shape = make_coord(K, N);
  auto out_shape = make_coord(M, N);

  auto lhs_sm_tile_shape = Coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = Coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = Coord(M_TILE, N_TILE);

  // make sure inner block looks like this, to ensure coalescing
  auto per_block_data_shape = Coord(Y_THREAD, X_THREAD);

  Matrix<float> lhs_mat = Matrix<float>(lhs, lhs_shape);
  Matrix<float> rhs_mat = Matrix<float>(rhs, rhs_shape);
  Matrix<float> out_mat = Matrix<float>(out, out_shape);

  // __shared__ float lhs_shared[M_TILE* (K_TILE+1)];

  Matrix<float> lhs_shared_mat = make_shared<M_TILE, K_TILE + 1, float>();
  Matrix<float> rhs_shared_mat = make_shared<K_TILE, N_TILE + 1, float>();
  // Matrix<float> out_shared_mat = make_shared<M_TILE_SM, N_TILE_SM>();
  //

  Matrix<float> partial_sum = make_local<CEIL_DIV(M_TILE, Y_THREAD),
                                         CEIL_DIV(N_TILE, X_THREAD), float>();

  // int threadId = threadIdx.y * X_THREAD + threadIdx.x;

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
#pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, X_THREAD); kin++) {
        lhs_shared_mat.tile(Coord(m, kin), per_block_data_shape)
          .dist_to(Coord(threadIdx.y, threadIdx.x))
          // .dist_to(Coord(threadId / K_TILE, threadId % K_TILE))
          = lhs_mat.tile(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
              .tile(Coord(m, kin), per_block_data_shape)
              // .dist_to(Coord(threadId / K_TILE, threadId % K_TILE));
              .dist_to(Coord(threadIdx.y, threadIdx.x));
        // lhs_shared[m * Y_THREAD * (K_TILE+1) + kin * X_THREAD + threadIdx.y *
        // (K_TILE+1) + threadIdx.x]
        //   = lhs_mat
        //   .tile(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
        //   .tile(Coord(m, kin), per_block_data_shape)
        //   // .dist_to(Coord(threadId / K_TILE, threadId % K_TILE));
        //   .dist_to(Coord(threadIdx.y, threadIdx.x)).data[0];
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, Y_THREAD); kin++) {
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
        rhs_shared_mat.tile(Coord(kin, n), per_block_data_shape)
          .dist_to(Coord(threadIdx.y, threadIdx.x))
          = rhs_mat.tile(Coord(ko, blockIdx.x), rhs_sm_tile_shape)
              .tile(Coord(kin, n), per_block_data_shape)
              .dist_to(Coord(threadIdx.y, threadIdx.x));
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_pad_swizzled<M_TILE, N_TILE, K_TILE, Y_THREAD, X_THREAD>(
      lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
#pragma unroll
    for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
      out_mat.tile(Coord(blockIdx.y, blockIdx.x), out_sm_tile_shape)
        .tile(Coord(m, n), per_block_data_shape)
        .dist_to(Coord(threadIdx.y, threadIdx.x))
        = partial_sum.dist_to(Coord(m, n));
    }
  }
}

void matmul_pad_swizzled(int M, int N, int K, float alpha, float *A, float *B,
                         float beta, float *C)
{
  // for bench
  // sec 1, determine, gridDim
  const int M_TILE = 128;
  const int N_TILE = 128;
  // sec 2, determine blockDim
  const int X_THREAD = 16;
  const int Y_THREAD = 16;
  // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  const int K_TILE = 32;

  // for debug only
  // const int M_TILE = 32;
  // const int N_TILE = 32;
  // // sec 2, determine blockDim
  // const int X_THREAD = 4;
  // const int Y_THREAD = 4;
  // // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // const int K_TILE = 8;
  // K_TILE > Y_THREAD
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
    _matmul_pad_swizzled<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared);
  _matmul_pad_swizzled<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>
    <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

} // namespace catz::recipes

#endif // CATZILLA_RECIPES_MATMUL_PAD_SWIZZLED_H_
