#ifndef CATZILLA_RECIPES_KERNELS_MATMUL_FLATTEN_COPY_H_
#define CATZILLA_RECIPES_KERNELS_MATMUL_FLATTEN_COPY_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils/index_utils.h"
#include "utils/macros.h"
#include "utils/micro_kernels.h"

// flatten copy
// TODO: xor-swizzle
template <const int M_TILE, const int N_TILE, const int K_TILE,
          const int X_THREAD, const int Y_THREAD>
__global__ void _catzilla_matmul_flatten_copy(int M, int N, int K, float alpha,
                                              float *lhs, float *rhs,
                                              float beta, float *out)
{
  auto lhs_shape = make_coord(M, K);
  auto rhs_shape = make_coord(K, N);
  auto out_shape = make_coord(M, N);

  auto lhs_sm_tile_shape = Coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = Coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = Coord(M_TILE, N_TILE);

  // make sure inner block looks like this, to ensure coalescing
  auto per_block_data_shape = Coord(Y_THREAD, X_THREAD);

  Matrix lhs_mat = Matrix(lhs, lhs_shape);
  Matrix rhs_mat = Matrix(rhs, rhs_shape);
  Matrix out_mat = Matrix(out, out_shape);

  // __shared__ float lhs_shared[M_TILE* (K_TILE+1)];

  Matrix lhs_shared_mat = make_shared<M_TILE, K_TILE>();
  Matrix rhs_shared_mat = make_shared<K_TILE, N_TILE>();
  // Matrix out_shared_mat = make_shared<M_TILE_SM, N_TILE_SM>();
  //

  Matrix partial_sum
    = make_local<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(N_TILE, X_THREAD)>();

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, 8); m++) {
#pragma unroll
      // for (int kin = 0; kin < CEIL_DIV(K_TILE, X_THREAD); kin++) {
      for (int kin = 0; kin < CEIL_DIV(K_TILE, 2); kin++) {
        // TODO: span_as(y * 4, x / 4)
        int x = threadIdx.y * X_THREAD + threadIdx.x;
        lhs_shared_mat.tile_ex(Coord(m, kin), Coord(8, 2)).dist_to_thread()
          = lhs_mat.tile_ex(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
              .tile_ex(Coord(m, kin), Coord(8, 2))
              .dist_to_thread();
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, Y_THREAD); kin++) {
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
        rhs_shared_mat.tile_ex(Coord(kin, n), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x))
          = rhs_mat.tile_ex(Coord(ko, blockIdx.x), rhs_sm_tile_shape)
              .tile_ex(Coord(kin, n), per_block_data_shape)
              .dist_ex(Coord(threadIdx.y, threadIdx.x));
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_coalesced<M_TILE, N_TILE, K_TILE, Y_THREAD, X_THREAD>(
      lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
#pragma unroll
    for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
      out_mat.tile_ex(Coord(blockIdx.y, blockIdx.x), out_sm_tile_shape)
        .tile_ex(Coord(m, n), per_block_data_shape)
        .dist_ex(Coord(threadIdx.y, threadIdx.x))
        = partial_sum.dist_ex(Coord(m, n));
    }
  }
}

// TODO: what if K_TILE < X_THREAD OR Y_THREAD, a thread-var cannot bind to
// K_TILE dim solely XOR-SWIZZLE
void catzilla_matmul_flatten_copy(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C)
{
  // for bench
  // sec 1, determine, gridDim
  // const int M_TILE = 128;
  // const int N_TILE = 128;
  // // sec 2, determine blockDim
  // const int X_THREAD = 16;
  // const int Y_THREAD = 16;
  // // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // const int K_TILE = 32;

  // for debug only
  const int M_TILE = 16;
  const int N_TILE = 16;
  // sec 2, determine blockDim
  const int X_THREAD = 4;
  const int Y_THREAD = 4;
  // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  const int K_TILE = 2;

  // K_TILE > Y_THREAD
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
    _catzilla_matmul_flatten_copy<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_flatten_copy<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>
    <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

#endif // CATZILLA_RECIPES_KERNELS_MATMUL_FLATTEN_COPY_H_
