#ifndef CATZILLA_RECIPES_MATMUL_TENSOR_CORES_TUNED_H_
#define CATZILLA_RECIPES_MATMUL_TENSOR_CORES_TUNED_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "macro.h"
#include "ukernels.h"

using namespace catz;
using namespace catz::cuda;
using namespace catz::wmma;
using namespace catz::mma;

namespace catz::recipes {

template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int WRAP_SIZE,
          const int WRAP_AMOUNT>
__global__ void
_matmul_tensor_cores_mma_m16n8k8_f16f32_tuned(int M, int N, int K, float alpha,
                                              float *lhs, float *rhs,
                                              float beta, float *out) {
  auto lhs_shape = CoordDyn(M, K);
  auto rhs_shape = CoordDyn(K, N);
  auto out_shape = CoordDyn(M, N);

  auto lhs_sm_tile_shape = CoordDyn(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = CoordDyn(K_TILE, N_TILE);
  auto out_sm_tile_shape = CoordDyn(M_TILE, N_TILE);

  auto lhs_reg_tile_shape = CoordDyn(M_REG, K_REG);
  auto rhs_reg_tile_shape = CoordDyn(K_REG, N_REG);
  auto out_reg_tile_shape = CoordDyn(M_REG, N_REG);

  // make sure inner block looks like this, to ensure coalescing

  MatrixDyn<float> lhs_mat = MatrixDyn<float>(lhs, lhs_shape);
  MatrixDyn<float> rhs_mat = MatrixDyn<float>(rhs, rhs_shape);
  MatrixDyn<float> out_mat = MatrixDyn<float>(out, out_shape);

  MAKE_SHARED(lhs_shared_mat, M_TILE, K_TILE, half);
  MAKE_SHARED(rhs_shared_mat, K_TILE, N_TILE, half);
  MAKE_SHARED(out_shared_mat, M_TILE, N_TILE, float);

  out_shared_mat.fill(0.);

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int kin = 0; kin < CEIL_DIV(K_TILE, K_REG); kin++) {
      lhs_shared_mat.tile(CoordDyn(0, kin), CoordDyn(M_TILE, K_REG)) <=
          lhs_mat.tile(CoordDyn(blockIdx.x, ko), lhs_sm_tile_shape)
              .tile(CoordDyn(0, kin), CoordDyn(M_TILE, K_REG));
      rhs_shared_mat.tile(CoordDyn(kin, 0), CoordDyn(K_REG, N_TILE)) <=
          rhs_mat.tile(CoordDyn(ko, blockIdx.y), rhs_sm_tile_shape)
              .tile(CoordDyn(kin, 0), CoordDyn(K_REG, N_TILE));
      __syncthreads();

      for (int n = 0; n < CEIL_DIV(N_TILE, N_REG); n++) {
        mma_m16n8k8_f16f32_neo(
            out_shared_mat.tile(CoordDyn(threadIdx.y, n),
                                CoordDyn(M_REG, N_REG)),
            lhs_shared_mat.tile(CoordDyn(threadIdx.y, kin),
                                CoordDyn(M_REG, K_REG)),
            rhs_shared_mat.tile(CoordDyn(kin, n), CoordDyn(K_REG, N_REG)),
            out_shared_mat.tile(CoordDyn(threadIdx.y, n),
                                CoordDyn(M_REG, N_REG)));
      }
    }
  }
  __syncthreads();

  out_mat.tile(CoordDyn(blockIdx.x, blockIdx.y), out_sm_tile_shape) <=
      out_shared_mat;
}

void matmul_tensor_cores_mma_m16n8k8_f16f32_tuned(int M, int N, int K,
                                                  float alpha, float *A,
                                                  float *B, float beta,
                                                  float *C) {
  const int M_TILE = 128;
  const int N_TILE = 64;
  const int K_TILE = 32; // slice K_TILE TO K_REG with Y_thread
  const int M_REG = 16;
  const int N_REG = 8;
  const int K_REG = 8;
  const int WRAP_SIZE = 32;
  const int WRAP_AMOUNT = 8;
  assert(M_REG * K_REG > WRAP_SIZE * WRAP_AMOUNT);
  assert(N_REG * K_REG > WRAP_SIZE * WRAP_AMOUNT);

  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(WRAP_SIZE, WRAP_AMOUNT);
  cudaFuncSetAttribute(
      _matmul_tensor_cores_mma_m16n8k8_f16f32_tuned<
          M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, WRAP_SIZE, WRAP_AMOUNT>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _matmul_tensor_cores_mma_m16n8k8_f16f32_tuned<
      M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, WRAP_SIZE, WRAP_AMOUNT>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  return;
}

template <const int M_TILE, const int N_TILE, const int K_TILE,
          const int X_THREAD, const int Y_THREAD>
__global__ void _matmul_tuned_with_mma_kernel(int M, int N, int K, float alpha,
                                              float *lhs, float *rhs,
                                              float beta, float *out) {
  const int M_REG = 16;
  const int K_REG = 8;
  const int N_REG = 8;

  auto lhs_shape = make_coord_dyn(M, K);
  auto rhs_shape = make_coord_dyn(K, N);
  auto out_shape = make_coord_dyn(M, N);

  auto lhs_sm_tile_shape = make_coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = make_coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = make_coord(M_TILE, N_TILE);

  auto lhs_reg_tile_shape = make_coord(M_REG, K_REG);
  auto rhs_reg_tile_shape = make_coord(K_REG, N_REG);
  auto out_reg_tile_shape = make_coord(M_REG, N_REG);

  auto per_block_data_shape = make_coord(Y_THREAD, X_THREAD);

  auto lhs_mat = make_matrix(lhs, lhs_shape);
  auto rhs_mat = make_matrix(rhs, rhs_shape);
  auto out_mat = make_matrix(out, out_shape);

  MAKE_SHARED_MATRIX(lhs_shared_mat, lhs_sm_tile_shape, float);
  MAKE_SHARED_MATRIX(rhs_shared_mat, rhs_sm_tile_shape, float);

  MAKE_LOCAL_MATRIX(partial_sum, out_sm_tile_shape / per_block_data_shape,
                    float);

  for (auto ko = I(0); ko < make_index(CEIL_DIV(K, K_TILE)); ++ko) {
    __syncthreads();
    for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, Y_THREAD)>(); ++m) {
      for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, X_THREAD)>();
           ++kin) {
        lhs_shared_mat.tile(Coord(m, kin), per_block_data_shape)
            .dist_to(Coord(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x))) =
            lhs_mat.tile(Coord(IndexDyn(blockIdx.y), ko), lhs_sm_tile_shape)
                .tile(Coord(m, kin), per_block_data_shape)
                .dist_to(Coord(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x)));
      }
    }
    // lhs_shared_mat <= lhs_mat.tile(Coord(IndexDyn(blockIdx.y), ko),
    // lhs_sm_tile_shape);

    for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, Y_THREAD)>();
         ++kin) {
      for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, X_THREAD)>(); ++n) {
        rhs_shared_mat.tile(Coord(kin, n), per_block_data_shape)
            .dist_to(Coord(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x))) =
            rhs_mat.tile(Coord(ko, IndexDyn(blockIdx.x)), rhs_sm_tile_shape)
                .tile(Coord(kin, n), per_block_data_shape)
                .dist_to(Coord(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x)));
      }
    }

    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_coalesced<M_TILE, N_TILE, K_TILE, Y_THREAD, X_THREAD>(
        lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);
  }
  __syncthreads();

  for (auto m = I(I(0)); m < make_index<CEIL_DIV(M_TILE, Y_THREAD)>(); ++m) {
    for (auto n = I(I(0)); n < make_index<CEIL_DIV(N_TILE, X_THREAD)>(); ++n) {
      out_mat
          .tile(Coord(IndexDyn(blockIdx.y), IndexDyn(blockIdx.x)),
                out_sm_tile_shape)
          .tile(Coord(m, n), per_block_data_shape)
          .dist_to(Coord(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x))) =
          partial_sum.dist_to(Coord(m, n));
    }
  }
}

void matmul_tuned_with_mma_kernel(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  // // sec 1, determine, gridDim
  // const int M_TILE = 128;
  // const int N_TILE = 128;
  // // sec 2, determine blockDim
  // const int X_THREAD = 16;
  // const int Y_THREAD = 16;
  // // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // const int K_TILE = 32;
  const int M_TILE = 128;
  const int N_TILE = 128;
  const int K_TILE = 32;
  // sec 2, determine blockDim
  const int X_THREAD = 16;
  const int Y_THREAD = 16;
  // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // K_TILE > Y_THREAD
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
      _matmul_tuned_with_mma_kernel<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _matmul_tuned_with_mma_kernel<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

} // namespace catz::recipes

#endif // CATZILLA_RECIPES_MATMUL_TENSOR_CORES_TUNED_H_
