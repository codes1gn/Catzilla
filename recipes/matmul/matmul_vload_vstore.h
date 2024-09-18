#ifndef CATZILLA_RECIPES_MATMUL_VLOAD_VSTORE_H_
#define CATZILLA_RECIPES_MATMUL_VLOAD_VSTORE_H_

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "index_utils.h"
#include "macros.h"
#include "micro_kernels.h"

using namespace catz;
using namespace catz::cuda;
using namespace catz::wmma;
using namespace catz::mma;

namespace catz::recipes
{

// TODO: about code
// move lhs and rhs tests to identity
// move reg tile into dist-to-thread, make it performant and opaque to user
//
// NOTE: current advantages
// no handle threads
// no handle tiles within sm-block
// reg-level tiling size and binding rule to threads are auto'd
// coalescing and swizzle are auto'd
//
// TODO: new features
// vmm instuctions in micro-kernels
// xor-swizzle not impl'd
//
template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int X_THREAD,
          const int Y_THREAD>
__global__ void _matmul_vload_vstore(int M, int N, int K, float alpha,
                                     float *lhs, float *rhs, float beta,
                                     float *out)
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

  Matrix<float> lhs_mat = Matrix<float>(lhs, lhs_shape);
  // MatriV lhs_mat = Matrix<float4>((float4*)lhs, lhs_shape);
  Matrix<float4> lhs_mat_v = Matrix<float4>((float4 *)lhs, lhs_shape_v); // 16x4
  //
  Matrix<float> rhs_mat = Matrix<float>(rhs, rhs_shape);
  Matrix<float4> rhs_mat_v = Matrix<float4>((float4 *)rhs, rhs_shape_v);
  // Matrix<float4> rhs_mat = Matrix<float4>((float4*)rhs, rhs_shape);
  Matrix<float> out_mat = Matrix<float>(out, out_shape);
  // Matrix<float4> out_mat_v = Matrix<float4>((float4*)out, out_shape_v); //
  // 16x4

  // __shared__ float lhs_shared[M_TILE* (K_TILE+1)];

  // Matrix<float> lhs_shared_mat = make_shared<M_TILE, K_TILE>();
  Matrix<float4> lhs_shared_mat_v = make_shared<M_TILE, K_TILE / 4, float4>();
  // MAKE_SHARED(lhs_shared_mat_v, M_TILE, K_TILE / 4, float4);
  // Matrix<float> rhs_shared_mat = make_shared<K_TILE, N_TILE>();
  Matrix<float4> rhs_shared_mat_v = make_shared<K_TILE, N_TILE / 4, float4>();
  // MAKE_SHARED(rhs_shared_mat_v, K_TILE, N_TILE / 4, float4);
  //

  Matrix<float> partial_sum = make_local<CEIL_DIV(M_TILE, Y_THREAD),
                                         CEIL_DIV(N_TILE, X_THREAD), float>();

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

// TODO: what if K_TILE < X_THREAD OR Y_THREAD, a thread-var cannot bind to
// K_TILE dim solely XOR-SWIZZLE
void matmul_vload_vstore(int M, int N, int K, float alpha, float *A, float *B,
                         float beta, float *C)
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
  cudaFuncSetAttribute(_matmul_vload_vstore<M_TILE, N_TILE, K_TILE, M_REG,
                                            N_REG, K_REG, X_THREAD, Y_THREAD>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _matmul_vload_vstore<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, X_THREAD,
                       Y_THREAD>
    <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  return;
}

} // namespace catz::recipes

#endif // CATZILLA_RECIPES_KERNELS_MATMUL_VLOAD_VSTORE_H_
