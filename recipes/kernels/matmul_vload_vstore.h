#ifndef CATZILLA_RECIPES_KERNELS_MATMUL_VLOAD_VSTORE_H_
#define CATZILLA_RECIPES_KERNELS_MATMUL_VLOAD_VSTORE_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils/index_utils.h"
#include "utils/macros.h"
#include "utils/micro_kernels.h"



// + vload vstore 
// TODO: let lhs also vectorise, currently only apply on rhs and out
template <const int M_TILE, const int N_TILE, const int K_TILE,
          const int X_THREAD, const int Y_THREAD>
__global__ void _catzilla_matmul_vload_vstore(int M, int N, int K, float alpha,
                                    float *lhs, float *rhs, float beta,
                                    float *out) {
  auto lhs_shape = make_coord(M, K);
  auto rhs_shape_v = make_coord(K, CEIL_DIV(N, 4));
  auto out_shape_v = make_coord(M, CEIL_DIV(N, 4)); // in float4 type, overall shape is M, N/4

  auto lhs_sm_tile_shape = Coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape_v = Coord(K_TILE, CEIL_DIV(N_TILE, 4));
  auto out_sm_tile_shape_v = Coord(M_TILE, CEIL_DIV(N_TILE, 4)); // in float4 type, sm shape is M_tile, N_tile/4 as well

  auto per_block_data_shape = Coord(Y_THREAD, X_THREAD); // same threading

  Matrix lhs_mat = Matrix(lhs, lhs_shape);
  MatrixV rhs_mat = MatrixV((float4*)rhs, rhs_shape_v);
  MatrixV out_mat = MatrixV((float4*)out, out_shape_v); // use MatrixV


  Matrix lhs_shared_mat = make_shared<M_TILE, K_TILE>();
  MatrixV rhs_shared_mat = make_shared_v<K_TILE, CEIL_DIV(N_TILE, 4)>();
  MatrixV partial_sum =
      make_local_v<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(CEIL_DIV(N_TILE, 4), X_THREAD)>();


  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
      #pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, X_THREAD); kin++) {
        // TODO: span_as(y * 4, x / 4)
        lhs_shared_mat
          .tile_ex(Coord(m, kin), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x))
          = lhs_mat
          .tile_ex(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
          .tile_ex(Coord(m, kin), per_block_data_shape)
          // .dist_ex(Coord(threadId / K_TILE, threadId % K_TILE));
          .dist_ex(Coord(threadIdx.y, threadIdx.x));
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, Y_THREAD); kin++) {
      #pragma unroll
      for (int n = 0; n < CEIL_DIV(CEIL_DIV(N_TILE, 4), X_THREAD); n++) {
        rhs_shared_mat
          .tile_ex(Coord(kin, n), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x))
          = rhs_mat
          .tile_ex(Coord(ko, blockIdx.x), rhs_sm_tile_shape_v)
          .tile_ex(Coord(kin, n), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x));
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    // matmul_kernel_coalesced_v<M_TILE, N_TILE, K_TILE, Y_THREAD, X_THREAD>(
    //     lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);
    matmul_kernel_coalesced<M_TILE, N_TILE, K_TILE, Y_THREAD, X_THREAD>(
        (float*)lhs_shared_mat.data, (float*)rhs_shared_mat.data, (float*)partial_sum.data);
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
    #pragma unroll
    for (int n = 0; n < CEIL_DIV(CEIL_DIV(N_TILE, 4), X_THREAD); n++) {
      out_mat
        .tile_ex(Coord(blockIdx.y, blockIdx.x), out_sm_tile_shape_v)
        .tile_ex(Coord(m, n), per_block_data_shape)
        .dist_ex(Coord(threadIdx.y, threadIdx.x)) 
        = partial_sum.dist_ex(Coord(m, n));
    }
  }
}

void catzilla_matmul_vload_vstore(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // const int M_TILE = 128;
  // const int N_TILE = 128;
  // const int X_THREAD = 16;
  // const int Y_THREAD = 16;
  // const int K_TILE = 32;

  const int M_TILE = 32;
  const int N_TILE = 32;
  const int X_THREAD = 2;
  const int Y_THREAD = 2;
  const int K_TILE = 8;

  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
      _catzilla_matmul_vload_vstore<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_vload_vstore<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

#endif // CATZILLA_RECIPES_KERNELS_MATMUL_VLOAD_VSTORE_H_
