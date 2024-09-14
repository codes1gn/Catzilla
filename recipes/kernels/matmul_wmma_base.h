#ifndef CATZILLA_RECIPES_KERNELS_MATMUL_WMMA_BASE_H_
#define CATZILLA_RECIPES_KERNELS_MATMUL_WMMA_BASE_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "index_utils.h"
#include "macros.h"
#include "micro_kernels.h"

template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int X_THREAD,
          const int Y_THREAD>
__global__ void
_catzilla_matmul_wmma_basic_test(int M, int N, int K, float alpha, float *lhs,
                                 float *rhs, float beta, float *out)
{
  auto lhs_shape = make_coord(M, K);
  auto rhs_shape = make_coord(K, N);
  auto out_shape = make_coord(M, N);

  auto lhs_sm_tile_shape = Coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = Coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = Coord(M_TILE, N_TILE);

  // make sure inner block looks like this, to ensure coalescing

  Matrix lhs_mat = Matrix(lhs, lhs_shape);
  Matrix rhs_mat = Matrix(rhs, rhs_shape);
  Matrix out_mat = Matrix(out, out_shape);

  __shared__ half lhs_shared[M_TILE * K_TILE];
  __shared__ half rhs_shared[K_TILE * N_TILE];
  __shared__ float out_shared[M_TILE * N_TILE];

  MatrixH lhs_shared_mat = MatrixH(lhs_shared, lhs_sm_tile_shape);
  MatrixH rhs_shared_mat = MatrixH(rhs_shared, rhs_sm_tile_shape);
  Matrix out_shared_mat = Matrix(out_shared, out_sm_tile_shape);
  out_shared_mat.fill(0.);

  // TODO: wrong, since this will release shared after make_shared call
  // impl a Macro based make_shared and make_local, ensure it won't release
  //
  // Matrix lhs_shared_mat = make_shared<M_TILE, K_TILE>();
  // Matrix rhs_shared_mat = make_shared<K_TILE, N_TILE>();
  // Matrix out_shared_mat = make_shared<M_TILE, N_TILE>(); // 16*16*4 = 1024
  // Bytes
  //

  // Matrix partial_sum
  //   = make_local<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(N_TILE, X_THREAD)>();

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, 2); m++) {
#pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, 16); kin++) {
        lhs_shared_mat.tile_ex(Coord(m, kin), Coord(2, 16)).dist_to_thread()
          = lhs_mat.tile_ex(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
              .tile_ex(Coord(m, kin), Coord(2, 16))
              .dist_to_thread();
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, 2); kin++) {
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, 16); n++) {
        rhs_shared_mat.tile_ex(Coord(kin, n), Coord(2, 16)).dist_to_thread()
          = rhs_mat.tile_ex(Coord(ko, blockIdx.x), rhs_sm_tile_shape)
              .tile_ex(Coord(kin, n), Coord(2, 16))
              .dist_to_thread();
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    // matmul_kernel_16x16x16_thread_32(lhs_shared_mat, rhs_shared_mat,
    // out_shared_mat.data); identity<M_TILE, N_TILE>(lhs_shared_mat,
    // out_mat.tile_ex(Coord(blockIdx.y, blockIdx.x), out_sm_tile_shape));
    // identity<M_TILE, N_TILE>(lhs_shared_mat,
    // out_mat.tile_ex(Coord(blockIdx.y, blockIdx.x), out_sm_tile_shape));
    matmul_kernel_m16n16k16(lhs_shared_mat.data, rhs_shared_mat.data,
                            out_shared_mat.data);
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, 2); m++) {
#pragma unroll
    out_mat.tile_ex(Coord(blockIdx.y, blockIdx.x), out_sm_tile_shape)
      .tile_ex(Coord(m, 0), Coord(2, 16))
      .dist_to_thread()
      = out_shared_mat.tile_ex(Coord(m, 0), Coord(2, 16)).dist_to_thread();
  }
}

void catzilla_matmul_wmma_basic_test(int M, int N, int K, float alpha, float *A,
                                     float *B, float beta, float *C)
{
  const int M_TILE = 16;
  const int K_TILE = 16;
  const int N_TILE = 16;
  const int M_REG = 2;
  const int K_REG = 2;
  const int N_REG = 16;
  const int X_THREAD = 32;
  const int Y_THREAD = 1;
  assert(M_REG * K_REG > X_THREAD * Y_THREAD);
  assert(N_REG * K_REG > X_THREAD * Y_THREAD);

  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
    _catzilla_matmul_wmma_basic_test<M_TILE, N_TILE, K_TILE, M_REG, N_REG,
                                     K_REG, X_THREAD, Y_THREAD>,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_wmma_basic_test<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG,
                                   X_THREAD, Y_THREAD>
    <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  return;
}

#endif // CATZILLA_RECIPES_KERNELS_MATMUL_WMMA_BASE_H_
