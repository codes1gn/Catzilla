#ifndef CATZILLA_RECIPES_KERNELS_MATMUL_VLOAD_VSTORE_H_
#define CATZILLA_RECIPES_KERNELS_MATMUL_VLOAD_VSTORE_H_

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils/index_utils.h"
#include "utils/macros.h"
#include "utils/micro_kernels.h"

// flatten copy
// TODO: xor-swizzle
template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int X_THREAD,
          const int Y_THREAD>
__global__ void _catzilla_matmul_vload_vstore(int M, int N, int K, float alpha,
                                              float *lhs, float *rhs,
                                              float beta, float *out)
{
  auto lhs_shape = make_coord(M, K); // 16x4
  auto rhs_shape = make_coord(K, N);
  auto lhs_shape_v = make_coord(M, K / 4); // 16x4
  auto rhs_shape_v = make_coord(K, N / 4);
  auto out_shape = make_coord(M, N);

  auto lhs_sm_tile_shape = Coord(M_TILE, K_TILE); // 16x2
  auto rhs_sm_tile_shape = Coord(K_TILE, N_TILE);
  auto lhs_sm_tile_shape_v = Coord(M_TILE, K_TILE / 4); // 16x2
  auto rhs_sm_tile_shape_v = Coord(K_TILE, N_TILE / 4);
  auto out_sm_tile_shape = Coord(M_TILE, N_TILE);

  auto lhs_reg_tile_shape = Coord(M_REG, K_REG); // 8x1
  auto rhs_reg_tile_shape = Coord(K_REG, N_REG);
  auto lhs_reg_tile_shape_v = Coord(M_REG, K_REG / 4); // 8x1
  auto rhs_reg_tile_shape_v = Coord(K_REG, N_REG / 4);
  auto out_reg_tile_shape = Coord(M_REG, N_REG);

  // make sure inner block looks like this, to ensure coalescing

  Matrix lhs_mat = Matrix(lhs, lhs_shape);
  // MatrixV lhs_mat = MatrixV((float4*)lhs, lhs_shape);
  MatrixV lhs_mat_v = MatrixV((float4 *)lhs, lhs_shape_v); // 16x4
  //
  Matrix rhs_mat = Matrix(rhs, rhs_shape);
  MatrixV rhs_mat_v = MatrixV((float4 *)rhs, rhs_shape_v);
  // MatrixV rhs_mat = MatrixV((float4*)rhs, rhs_shape);
  Matrix out_mat = Matrix(out, out_shape);
  // MatrixV out_mat_v = MatrixV((float4*)out, out_shape_v); // 16x4

  // __shared__ float lhs_shared[M_TILE* (K_TILE+1)];

  // Matrix lhs_shared_mat = make_shared<M_TILE, K_TILE>();
  MatrixV lhs_shared_mat_v = make_shared_v<M_TILE, K_TILE / 4>();
  // Matrix rhs_shared_mat = make_shared<K_TILE, N_TILE>();
  MatrixV rhs_shared_mat_v = make_shared_v<K_TILE, N_TILE / 4>();
  //

  Matrix partial_sum
    = make_local<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(N_TILE, X_THREAD)>();

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, M_REG); m++) {
#pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, K_REG); kin++) {
        // lhs_shared_mat
        //   .tile_ex(Coord(m, kin), lhs_reg_tile_shape)
        //   .dist_to_thread()
        //   = lhs_mat
        //   .tile_ex(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
        //     .tile_ex(Coord(m, kin), lhs_reg_tile_shape)
        //     .dist_to_thread();
        lhs_shared_mat_v.tile_ex(Coord(m, kin), lhs_reg_tile_shape_v)
          .dist_to_thread()
          = lhs_mat_v.tile_ex(Coord(blockIdx.y, ko), lhs_sm_tile_shape_v)
              .tile_ex(Coord(m, kin), lhs_reg_tile_shape_v)
              .dist_to_thread();
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, K_REG); kin++) {
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, N_REG); n++) {
        // rhs_shared_mat
        //   .tile_ex(Coord(kin, n), rhs_reg_tile_shape)
        //   .dist_to_thread()
        //   = rhs_mat
        //   .tile_ex(Coord(ko, blockIdx.x), rhs_sm_tile_shape)
        //   .tile_ex(Coord(kin, n), rhs_reg_tile_shape)
        //   .dist_to_thread();
        rhs_shared_mat_v.tile_ex(Coord(kin, n), rhs_reg_tile_shape_v)
          .dist_to_thread()
          = rhs_mat_v.tile_ex(Coord(ko, blockIdx.x), rhs_sm_tile_shape_v)
              .tile_ex(Coord(kin, n), rhs_reg_tile_shape_v)
              .dist_to_thread();
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_coalesced<M_TILE, N_TILE, K_TILE, Y_THREAD, X_THREAD>(
      (float *)lhs_shared_mat_v.data, (float *)rhs_shared_mat_v.data,
      (float *)partial_sum.data);
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
#pragma unroll
    for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
      out_mat.tile_ex(Coord(blockIdx.y, blockIdx.x), out_sm_tile_shape)
        .tile_ex(Coord(m, n), Coord(Y_THREAD, X_THREAD))
        .dist_to_thread()
        = partial_sum.dist_ex(Coord(m, n));
    }
  }
}

template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int X_THREAD,
          const int Y_THREAD>
__global__ void _catzilla_matmul_vload_vstore_test_lhs(int M, int N, int K,
                                                       float alpha, float *lhs,
                                                       float *rhs, float beta,
                                                       float *out)
{
  auto lhs_shape = make_coord(M, K);       // 16x16
  auto lhs_shape_v = make_coord(M, K / 4); // 16x16
  auto rhs_shape = make_coord(K, N);
  auto out_shape = make_coord(M, N);
  auto out_shape_v = make_coord(M, K / 4);

  auto lhs_sm_tile_shape = Coord(M_TILE, K_TILE);       // 16x8
  auto lhs_sm_tile_shape_v = Coord(M_TILE, K_TILE / 4); // 16x8
  auto rhs_sm_tile_shape = Coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = Coord(M_TILE, N_TILE);

  auto lhs_reg_tile_shape = Coord(M_REG, K_REG);       // 8x4
  auto lhs_reg_tile_shape_v = Coord(M_REG, K_REG / 4); // 8x4
  auto rhs_reg_tile_shape = Coord(K_REG, N_REG);
  auto out_reg_tile_shape = Coord(M_REG, N_REG);

  // make sure inner block looks like this, to ensure coalescing

  Matrix lhs_mat = Matrix(lhs, lhs_shape);
  MatrixV lhs_mat_v = MatrixV((float4 *)lhs, lhs_shape_v); // 16x4
  Matrix rhs_mat = Matrix(rhs, rhs_shape);
  // MatrixV rhs_mat = MatrixV((float4*)rhs, rhs_shape);
  MatrixV out_mat_v = MatrixV((float4 *)out, out_shape_v); // 16x4
  // Matrix out_mat = Matrix(out, out_shape);

  // __shared__ float lhs_shared[M_TILE* (K_TILE+1)];

  // Matrix lhs_shared_mat = make_shared<M_TILE, K_TILE>();
  MatrixV lhs_shared_mat_v = make_shared_v<M_TILE, K_TILE / 4>();
  Matrix rhs_shared_mat = make_shared<K_TILE, N_TILE>();
  // MatrixV rhs_shared_mat = make_shared_v<K_TILE, N_TILE>();
  // Matrix out_shared_mat = make_shared<M_TILE_SM, N_TILE_SM>();
  //

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    // for (int m = 0; m < CEIL_DIV(M_TILE, M_REG); m++) { // if use float* to
    // transfer data with float4 space, will results in empty data
    for (int m = 0; m < CEIL_DIV(M_TILE, M_REG); m++) {
      for (int kin = 0; kin < CEIL_DIV(K_TILE, K_REG); kin++) {
        lhs_shared_mat_v.tile_ex(Coord(m, kin), lhs_reg_tile_shape_v)
          .dist_to_thread()
          = lhs_mat_v.tile_ex(Coord(blockIdx.y, ko), lhs_sm_tile_shape_v)
              .tile_ex(Coord(m, kin), lhs_reg_tile_shape_v)
              .dist_to_thread();
      }
    }
    __syncthreads();

    // Matrix lhs_shared_mat = Matrix((float*)lhs_shared_mat_v.data,
    // Coord(M_TILE, K_TILE));
    for (int m = 0; m < CEIL_DIV(M_TILE, M_REG); m++) {
      for (int kin = 0; kin < CEIL_DIV(K_TILE, K_REG); kin++) {
        out_mat_v.tile_ex(Coord(blockIdx.y, ko), lhs_sm_tile_shape_v)
          .tile_ex(Coord(m, kin), lhs_reg_tile_shape_v)
          .dist_to_thread()
          = lhs_shared_mat_v.tile_ex(Coord(m, kin), lhs_reg_tile_shape_v)
              .dist_to_thread();
      }
    }
  }
  __syncthreads();
}

// move to identity
template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int X_THREAD,
          const int Y_THREAD>
__global__ void _catzilla_matmul_vload_vstore_test_rhs(int M, int N, int K,
                                                       float alpha, float *lhs,
                                                       float *rhs, float beta,
                                                       float *out)
{
  auto rhs_shape = make_coord(K, N);
  auto rhs_shape_v = make_coord(K, N / 4);

  auto rhs_sm_tile_shape = Coord(K_TILE, N_TILE);
  auto rhs_sm_tile_shape_v = Coord(K_TILE, N_TILE / 4);

  auto rhs_reg_tile_shape = Coord(K_REG, N_REG);
  auto rhs_reg_tile_shape_v = Coord(K_REG, N_REG / 4);

  // make sure inner block looks like this, to ensure coalescing

  // Matrix rhs_mat = Matrix(rhs, rhs_shape);
  MatrixV rhs_mat_v = MatrixV((float4 *)rhs, rhs_shape_v);
  MatrixV out_mat_v = MatrixV((float4 *)out, rhs_shape_v); // 16x4
  // Matrix out_mat = Matrix(out, out_shape);

  // __shared__ float lhs_shared[M_TILE* (K_TILE+1)];

  // Matrix lhs_shared_mat = make_shared<M_TILE, K_TILE>();
  // Matrix rhs_shared_mat = make_shared<K_TILE, N_TILE/4>();
  MatrixV rhs_shared_mat_v = make_shared_v<K_TILE, N_TILE / 4>();
  //

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int kin = 0; kin < CEIL_DIV(K_TILE, K_REG); kin++) {
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, N_REG); n++) {
        rhs_shared_mat_v.tile_ex(Coord(kin, n), rhs_reg_tile_shape_v)
          .dist_to_thread()
          = rhs_mat_v.tile_ex(Coord(ko, blockIdx.x), rhs_sm_tile_shape_v)
              .tile_ex(Coord(kin, n), rhs_reg_tile_shape_v)
              .dist_to_thread();
      }
    }
    __syncthreads();

    // Matrix lhs_shared_mat = Matrix((float*)lhs_shared_mat_v.data,
    // Coord(M_TILE, K_TILE));
    for (int kin = 0; kin < CEIL_DIV(K_TILE, K_REG); kin++) {
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, N_REG); n++) {
        out_mat_v.tile_ex(Coord(ko, blockIdx.x), rhs_sm_tile_shape_v)
          .tile_ex(Coord(kin, n), rhs_reg_tile_shape_v)
          .dist_to_thread()
          = rhs_shared_mat_v.tile_ex(Coord(kin, n), rhs_reg_tile_shape_v)
              .dist_to_thread();
      }
    }
  }
  __syncthreads();
}

// TODO: what if K_TILE < X_THREAD OR Y_THREAD, a thread-var cannot bind to
// K_TILE dim solely XOR-SWIZZLE
void catzilla_matmul_vload_vstore(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C)
{
  const int M_TILE = 128;
  const int K_TILE = 32;
  const int N_TILE = 128;
  const int M_REG = 32;
  const int K_REG = 32;
  const int N_REG = 32;
  const int X_THREAD = 32;
  const int Y_THREAD = 8;
  assert(M_REG * K_REG > X_THREAD * Y_THREAD);
  assert(N_REG * K_REG > X_THREAD * Y_THREAD);

  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
    _catzilla_matmul_vload_vstore<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG,
                                  X_THREAD, Y_THREAD>,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_vload_vstore<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG,
                                X_THREAD, Y_THREAD>
    <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  //
  // const int M_TILE = 16;
  // const int K_TILE = 8;
  // const int N_TILE = 16;
  // const int M_REG = 8;
  // const int K_REG = 8;
  // const int N_REG = 8;
  // const int X_THREAD = 2; // if only lhs use float4, threads volume can be
  // 1/4 of M/K-REG, but other place cannot match; thus we have to solve
  // elements = K*threads where K = 2, 3, 4 const int Y_THREAD = 8; assert(M_REG
  // * K_REG > X_THREAD * Y_THREAD); assert(N_REG * K_REG > X_THREAD *
  // Y_THREAD);
  //
  // dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  // dim3 blockDim(X_THREAD, Y_THREAD);
  // // cudaFuncSetAttribute(
  // //   _catzilla_matmul_vload_vstore_test_lhs<M_TILE, N_TILE, K_TILE, M_REG,
  // N_REG, K_REG,
  // //                                 X_THREAD, Y_THREAD>,
  // //   cudaFuncAttributePreferredSharedMemoryCarveout,
  // //   cudaSharedmemCarveoutMaxShared);
  // // _catzilla_matmul_vload_vstore_test_lhs<M_TILE, N_TILE, K_TILE, M_REG,
  // N_REG, K_REG,
  // //                               X_THREAD, Y_THREAD>
  // //   <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  //
  // // _catzilla_matmul_vload_vstore_test_rhs<M_TILE, N_TILE, K_TILE, M_REG,
  // N_REG, K_REG,
  // //                               X_THREAD, Y_THREAD>
  // //   <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  //
  // _catzilla_matmul_vload_vstore<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG,
  //                               X_THREAD, Y_THREAD>
  //   <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  return;
}

#endif // CATZILLA_RECIPES_KERNELS_MATMUL_VLOAD_VSTORE_H_
