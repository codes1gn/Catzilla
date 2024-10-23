#ifndef CATZILLA_RECIPES_MATMUL_REFACTORED_H_
#define CATZILLA_RECIPES_MATMUL_REFACTORED_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "coord.h"
#include "index.h"
#include "macro.h"
#include "matrix.h"
#include "ukernels.h"

using namespace catz;
using namespace catz::cuda;
using namespace catz::wmma;
using namespace catz::mma;

namespace catz::recipes {

/////////////////////////////////////////////////////////////////////////////////
/// version 3:
/// - basic gmem to smem coalescing
/// - use smem for lhs, rhs, out
/// - thread-level blocking, compute a tile in reg-level C, not a scalar
/// - refactor with tiling and transfer helpers
/// - try different tiling config with hand
/////////////////////////////////////////////////////////////////////////////////
template <const int M_TILE_SM, const int N_TILE_SM, const int K_TILE_SM,
          const int M_TILE_REG, const int N_TILE_REG>
__global__ void _matmul_affine_api(int M, int N, int K, float alpha, float *lhs,
                                   float *rhs, float beta, float *out) {
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  // smem stationary
  // M=128, N=128, K=32, Mreg=4, Nreg=4
  __shared__ float lhs_shared[M_TILE_SM * K_TILE_SM];
  __shared__ float rhs_shared[K_TILE_SM * N_TILE_SM];
  __shared__ float out_shared[M_TILE_SM * N_TILE_SM];
  // __shared__ float out_shared[1];

  // init out
  // out_shared[distribute_(tid_x, tid_y, 1, N_TILE_SM)] = 0.0;

  float partial_sum[M_TILE_REG * N_TILE_REG] = {0.};

  // distribute bid_y for lhs

  // QZ: per thread m and m+1
  for (int k = 0; k < CEIL_DIV(K, K_TILE_SM); k++) { // range(0, 128, 1)
    // use distribute_ for in-tile thread calculation
    // use tile_ for out-tile thread calculation
    float *lhs_cursor = lhs + tiling_(bid_y, k, M_TILE_SM, K_TILE_SM, K, 1);
    float *rhs_cursor = rhs + tiling_(k, bid_x, K_TILE_SM, N_TILE_SM, N, 1);
    for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
      lhs_shared[distribute_(tid_x, tid_y * M_TILE_REG + mreg, 1, K_TILE_SM)] =
          lhs_cursor[distribute_(tid_x, tid_y * M_TILE_REG + mreg, 1, K)];
    }
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      rhs_shared[distribute_(tid_x * N_TILE_REG + nreg, tid_y, 1, N_TILE_SM)] =
          rhs_cursor[distribute_(tid_x * N_TILE_REG + nreg, tid_y, 1, N)];
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_64x64x16_perthread_4x4(lhs_shared, rhs_shared, partial_sum);
  }
  __syncthreads();

  float *out_cursor = out + tiling_(bid_y, bid_x, M_TILE_SM, N_TILE_SM, N, 1);
  for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      // out_shared[distribute_(tid_x * N_TILE_REG + nreg, tid_y * M_TILE_REG +
      // mreg, 1, N_TILE_SM)] = alpha * partial_sum[mreg * N_TILE_REG + nreg];
      // out_cursor[distribute_(tid_x * N_TILE_REG + nreg, tid_y * M_TILE_REG +
      // mreg, 1, N)] = out_shared[distribute_(tid_x*N_TILE_REG + nreg,
      // tid_y*M_TILE_REG + mreg, 1, N_TILE_SM)];
      out_cursor[distribute_(tid_x * N_TILE_REG + nreg,
                             tid_y * M_TILE_REG + mreg, 1, N)] =
          alpha * partial_sum[mreg * N_TILE_REG + nreg];
    }
  }
}

void matmul_affine_api(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  const int M_TILE = 64;
  const int N_TILE = 64;
  const int K_TILE = 16;
  const int mreg = 4;
  const int nreg = 4;
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(CEIL_DIV(M_TILE, mreg), CEIL_DIV(N_TILE, nreg));
  cudaFuncSetAttribute(_matmul_affine_api<M_TILE, N_TILE, K_TILE, mreg, nreg>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _matmul_affine_api<M_TILE, N_TILE, K_TILE, mreg, nreg>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// simplify with stream-style
template <const int M_TILE_SM, const int N_TILE_SM, const int K_TILE_SM,
          const int M_TILE_REG, const int N_TILE_REG>
__global__ void _matmul_stream_api(int M, int N, int K, float alpha, float *lhs,
                                   float *rhs, float beta, float *out) {
  auto lhs_shape = CoordDyn(M, K);
  auto rhs_shape = CoordDyn(K, N);
  auto out_shape = CoordDyn(M, N);

  auto lhs_sm_tile_shape = CoordDyn(M_TILE_SM, K_TILE_SM);
  auto rhs_sm_tile_shape = CoordDyn(K_TILE_SM, N_TILE_SM);
  auto out_sm_tile_shape = CoordDyn(M_TILE_SM, N_TILE_SM);

  auto lhs_reg_tile_shape = CoordDyn(M_TILE_REG, K_TILE_SM);
  auto rhs_reg_tile_shape = CoordDyn(K_TILE_SM, N_TILE_REG);
  auto out_reg_tile_shape = CoordDyn(M_TILE_REG, N_TILE_REG);

  MatrixDyn<float> lhs_mat = MatrixDyn<float>(lhs, lhs_shape);
  MatrixDyn<float> rhs_mat = MatrixDyn<float>(rhs, rhs_shape);
  MatrixDyn<float> out_mat = MatrixDyn<float>(out, out_shape);

  MatrixDyn<float> lhs_shared_mat = make_shared<M_TILE_SM, K_TILE_SM, float>();
  MatrixDyn<float> rhs_shared_mat = make_shared<K_TILE_SM, N_TILE_SM, float>();
  // MatrixDyn<float> out_shared_mat = make_shared<M_TILE_SM, N_TILE_SM>();

  MatrixDyn<float> partial_sum = make_local<M_TILE_REG, N_TILE_REG, float>();

  // iter vars we got, block.x, block.y, thread.x, thread.y, k, mreg.
  // because we distribute out into blocks, we select one of blk.x or blk.y
  // for row: blk.y, thd.y, mreg
  // for col: k, thd.x
  for (int k = 0; k < CEIL_DIV(K, K_TILE_SM); k++) {
#pragma unroll 4
    for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
      lhs_shared_mat // (M_TILE_SM, K_TILE_SM)
          .tile(CoordDyn(threadIdx.y, 0),
                lhs_reg_tile_shape) // (M_TILE_SM, K_TILE_SM) => (M_TILE_REG,
                                    // K_TILE_SM)
          .dist_to(CoordDyn(
              mreg,
              threadIdx.x)) // (M_TILE_REG, K_TILE_SM) distribute to threads
          = lhs_mat         // (M, K)
                .tile(CoordDyn(blockIdx.y, k),
                      lhs_sm_tile_shape) // (M, K) => (M_TILE_SM, K_TILE_SM)
                .tile(CoordDyn(threadIdx.y, 0),
                      lhs_reg_tile_shape) // (M_TILE_SM, K_TILE_SM) =>
                                          // (M_TILE_REG, K_TILE_SM)
                .dist_to(
                    CoordDyn(mreg, threadIdx.x)); // (M_TILE_REG, K_TILE_SM)
                                                  // distribute to threads
    }
#pragma unroll 4
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      rhs_shared_mat
          .tile(CoordDyn(0, threadIdx.x),
                rhs_reg_tile_shape) // (M_TILE_SM, K_TILE_SM) => (M_TILE_REG,
                                    // K_TILE_SM)
          .dist_to(CoordDyn(threadIdx.y, nreg)) =
          rhs_mat.tile(CoordDyn(k, blockIdx.x), rhs_sm_tile_shape)
              .tile(CoordDyn(0, threadIdx.x),
                    rhs_reg_tile_shape) // (M_TILE_SM, K_TILE_SM) =>
                                        // (M_TILE_REG, K_TILE_SM)
              .dist_to(CoordDyn(threadIdx.y, nreg));
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_64x64x16_perthread_4x4(lhs_shared_mat.data,
                                         rhs_shared_mat.data, partial_sum.data);
  }
  __syncthreads();

  for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
#pragma unroll 4
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      // out_shared_mat
      //   .tile(CoordDyn(threadIdx.y, threadIdx.x), out_reg_tile_shape)
      //   .dist_to(CoordDyn(mreg, nreg))
      //   = partial_sum
      //   .dist_to(CoordDyn(mreg, nreg));
      //
      // out_mat
      //   .tile(CoordDyn(blockIdx.y, blockIdx.x), out_sm_tile_shape)
      //   .tile(CoordDyn(threadIdx.y, threadIdx.x), out_reg_tile_shape)
      //   .dist_to(CoordDyn(mreg, nreg))
      //   = out_shared_mat
      //   .tile(CoordDyn(threadIdx.y, threadIdx.x), out_reg_tile_shape)
      //   .dist_to(CoordDyn(mreg, nreg));
      out_mat.tile(CoordDyn(blockIdx.y, blockIdx.x), out_sm_tile_shape)
          .tile(CoordDyn(threadIdx.y, threadIdx.x), out_reg_tile_shape)
          .dist_to(CoordDyn(mreg, nreg)) =
          partial_sum.dist_to(CoordDyn(mreg, nreg));
    }
  }
}

void matmul_stream_api(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  const int M_TILE = 64;
  const int N_TILE = 64;
  const int K_TILE = 16;
  const int mreg = 4;
  const int nreg = 4;
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(CEIL_DIV(M_TILE, mreg), CEIL_DIV(N_TILE, nreg));
  cudaFuncSetAttribute(_matmul_stream_api<M_TILE, N_TILE, K_TILE, mreg, nreg>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _matmul_stream_api<M_TILE, N_TILE, K_TILE, mreg, nreg>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// hand tuned for HPs
// use templated kernel
template <const int M_TILE, const int N_TILE, const int K_TILE,
          const int X_THREAD, const int Y_THREAD>
__global__ void _matmul_stream_api_tuned(int M, int N, int K, float alpha,
                                         float *lhs, float *rhs, float beta,
                                         float *out) {
  auto lhs_shape = make_coord_dyn(M, K);
  auto rhs_shape = make_coord_dyn(K, N);
  auto out_shape = make_coord_dyn(M, N);

  auto lhs_sm_tile_shape = make_coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = make_coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = make_coord(M_TILE, N_TILE);

  // make sure inner block looks like this, to ensure coalescing
  auto per_block_data_shape = make_coord(Y_THREAD, X_THREAD);

  auto lhs_mat = make_matrix(lhs, lhs_shape);
  auto rhs_mat = make_matrix(rhs, rhs_shape);
  auto out_mat = make_matrix(out, out_shape);

  // auto lhs_shared_mat = make_shared_test(lhs_sm_tile_shape);
  MAKE_SHARED_MATRIX(lhs_shared_mat, lhs_sm_tile_shape, float);
  MAKE_SHARED_MATRIX(rhs_shared_mat, rhs_sm_tile_shape, float);

  // auto lhs_shape = CoordDyn(M, K);
  // auto rhs_shape = CoordDyn(K, N);
  // auto out_shape = CoordDyn(M, N);

  // auto lhs_sm_tile_shape = CoordDyn(M_TILE, K_TILE);
  // auto rhs_sm_tile_shape = CoordDyn(K_TILE, N_TILE);
  auto out_sm_tile_shape_old = CoordDyn(M_TILE, N_TILE);

  // make sure inner block looks like this, to ensure coalescing
  // auto per_block_data_shape_old = CoordDyn(Y_THREAD, X_THREAD);

  // MatrixDyn<float> lhs_mat = MatrixDyn<float>(lhs, lhs_shape);
  // MatrixDyn<float> rhs_mat = MatrixDyn<float>(rhs, rhs_shape);
  // MatrixDyn<float> out_mat = MatrixDyn<float>(out, out_shape);

  // MatrixDyn<float> lhs_shared_mat = make_shared<M_TILE, K_TILE, float>();
  // MatrixDyn<float> rhs_shared_mat = make_shared<K_TILE, N_TILE, float>();
  // MatrixDyn<float> partial_sum =
  //     make_local<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(N_TILE, X_THREAD),
  //                float>();

  auto partial_sum_shape =
      make_coord(CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(N_TILE, X_THREAD));
  MAKE_LOCAL_MATRIX(partial_sum, partial_sum_shape, float);

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
#pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, X_THREAD); kin++) {
        lhs_shared_mat
            .tile(CoordS(IndexDyn(m), IndexDyn(kin)), per_block_data_shape)
            .dist_to(CoordS(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x))) =
            lhs_mat
                .tile(CoordS(IndexDyn(blockIdx.y), IndexDyn(ko)),
                      lhs_sm_tile_shape)
                .tile(CoordS(IndexDyn(m), IndexDyn(kin)), per_block_data_shape)
                .dist_to(CoordS(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x)));
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, Y_THREAD); kin++) {
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
        rhs_shared_mat
            .tile(CoordS(IndexDyn(kin), IndexDyn(n)), per_block_data_shape)
            .dist_to(CoordS(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x))) =
            rhs_mat
                .tile(CoordS(IndexDyn(ko), IndexDyn(blockIdx.x)),
                      rhs_sm_tile_shape)
                .tile(CoordS(IndexDyn(kin), IndexDyn(n)), per_block_data_shape)
                .dist_to(CoordS(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x)));
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_coalesced<M_TILE, N_TILE, K_TILE, Y_THREAD, X_THREAD>(
        lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);
  }
  __syncthreads();

  for (auto m = IndexDyn(0); m < make_index<CEIL_DIV(M_TILE, Y_THREAD)>();
       ++m) {
#pragma unroll
    for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
      out_mat
          .tile(CoordS(IndexDyn(blockIdx.y), IndexDyn(blockIdx.x)),
                out_sm_tile_shape)
          .tile(CoordS(m, IndexDyn(n)), per_block_data_shape)
          .dist_to(CoordS(IndexDyn(threadIdx.y), IndexDyn(threadIdx.x)))
          //     partial_sum.dist_to(CoordS(IndexDyn(m), IndexDyn(n)));
          // out_mat.tile(CoordDyn(blockIdx.y, blockIdx.x),
          // out_sm_tile_shape_old)
          //     .tile(CoordDyn(m, n), per_block_data_shape_old)
          //     .dist_to(CoordDyn(threadIdx.y, threadIdx.x))
          = partial_sum.data[m.value * CEIL_DIV(M_TILE, Y_THREAD) + n];
      // = partial_sum.dist_to(CoordDyn(m, n));
    }
  }
}

void matmul_stream_api_tuned(int M, int N, int K, float alpha, float *A,
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
  // sec 2, determine blockDim
  const int X_THREAD = 16;
  const int Y_THREAD = 16;
  // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  const int K_TILE = 32;
  // K_TILE > Y_THREAD
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
      _matmul_stream_api_tuned<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _matmul_stream_api_tuned<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

} // namespace catz::recipes

#endif // CATZILLA_RECIPES_MATMUL_REFACTORED_H_
