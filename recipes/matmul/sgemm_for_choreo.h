#ifndef CATZILLA_RECIPES_SGEMM_FOR_CHOREO_H_
#define CATZILLA_RECIPES_SGEMM_FOR_CHOREO_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "macro.h"
#include "matrix_legacy.h"
#include "ukernels.h"

using namespace catz;
using namespace catz::cuda;
using namespace catz::wmma;
using namespace catz::mma;

namespace catz::recipes {

__global__ void _sgemm_for_choreo(float *lhs, float *rhs, float *out) {
  const int M = 4096;
  const int N = 4096;
  const int K = 4096;
  auto bid_x = IndexDyn(blockIdx.x);
  auto bid_y = IndexDyn(blockIdx.y);
  auto bid_z = IndexDyn(blockIdx.z);
  auto tid_x = IndexDyn(threadIdx.x);
  auto tid_y = IndexDyn(threadIdx.x);
  auto tid_z = IndexDyn(threadIdx.z);

  auto lhs_mat = make_matrix(lhs, make_coord_dyn(M, K));
  auto rhs_mat = make_matrix(rhs, make_coord_dyn(K, N));
  auto out_mat = make_matrix(out, make_coord_dyn(M, N));

  // MAKE_LOCAL_MATRIX(partial_sum, make_coord(64, 64) / make_coord(16, 16),
  //                   float);
  // TODO: this become super slow
  MAKE_SHARED_MATRIX(partial_sum, make_coord(64, 64),
                    float);

  for (auto ko = I(0); ko < make_index(CEIL_DIV(K, 32)); ++ko) {
    MAKE_SHARED_MATRIX(lhs_shared_mat, make_coord(64, 32), float);
    MAKE_SHARED_MATRIX(rhs_shared_mat, make_coord(32, 64), float);

    for (auto m = I(0); m < make_index<CEIL_DIV(64, 16)>(); ++m) {
      for (auto kin = I(0); kin < make_index<CEIL_DIV(32, 16)>(); ++kin) {
        lhs_shared_mat
            .tile(Coord(m, kin), make_coord(16, 16))
            .dist_to(Coord(tid_y, tid_x)) =
            lhs_mat.tile(Coord(bid_y, ko), make_coord(64, 32))
            .tile(Coord(m, kin), make_coord(16, 16))
            .dist_to(Coord(tid_y, tid_x));
      }
    }

    for (auto kin = I(0); kin < make_index<CEIL_DIV(32, 16)>(); ++kin) {
      for (auto n = I(0); n < make_index<CEIL_DIV(64, 16)>(); ++n) {
        rhs_shared_mat.tile(Coord(kin, n), make_coord(16, 16))
            .dist_to(Coord(tid_y, tid_x)) =
            rhs_mat.tile(Coord(ko, bid_x), make_coord(32, 64))
                .tile(Coord(kin, n), make_coord(16, 16))
                .dist_to(Coord(tid_y, tid_x));
      }
    }

    __syncthreads();

    // contract at 64x64x32 micro-kernel
    matmul_kernel_at_shared<64, 64, 32, 16, 16>(
    // matmul_kernel_coalesced<64, 64, 32, 16, 16>(
        lhs_shared_mat.data, rhs_shared_mat.data,
        partial_sum.data);
  }
  __syncthreads();

  for (auto m = I(0); m < make_index<CEIL_DIV(64, 16)>(); ++m) {
    for (auto n = I(0); n < make_index<CEIL_DIV(64, 16)>(); ++n) {
      out_mat.tile(Coord(bid_y, bid_x), make_coord(64, 64))
          .tile(Coord(m, n), make_coord(16, 16))
          .dist_to(Coord(tid_y, tid_x)) =
          // partial_sum.dist_to(Coord(m, n));
          partial_sum
          .tile(Coord(m, n), make_coord(16, 16))
          .dist_to(Coord(tid_y, tid_x));
    }
  }
}

// parallel {p, q} by [32, 32]
void sgemm_for_choreo(float *A, float *B, float *C) {
  dim3 gridDim(64, 64);
  // TODO: HC
  dim3 blockDim(16, 16);
  cudaFuncSetAttribute(_sgemm_for_choreo,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _sgemm_for_choreo<<<gridDim, blockDim>>>(A, B, C);
}

template <const int M_TILE, const int N_TILE, const int K_TILE,
          const int M_REG, const int N_REG, const int K_REG,
          const int THREADS>
__global__ void _matmul_dataflow_plus_mma(int M, int N, int K, float alpha,
                                              float *lhs, float *rhs,
                                              float beta, float *out) {
  auto lhs_shape = make_coord_dyn(M, K);
  auto rhs_shape = make_coord_dyn(K, N);
  auto out_shape = make_coord_dyn(M, N);

  auto lhs_sm_tile_shape = make_coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = make_coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = make_coord(M_TILE, N_TILE);

  auto lhs_reg_tile_shape = make_coord(M_REG, K_REG);
  auto rhs_reg_tile_shape = make_coord(K_REG, N_REG);
  auto out_reg_tile_shape = make_coord(M_REG, N_REG);

  auto tb_shape = make_coord(THREADS/32, 32);

  auto lhs_mat = make_matrix(lhs, lhs_shape);
  auto rhs_mat = make_matrix(rhs, rhs_shape);
  auto out_mat = make_matrix(out, out_shape);

  MAKE_SHARED_MATRIX(lhs_shared_mat, lhs_sm_tile_shape, float);
  MAKE_SHARED_MATRIX(rhs_shared_mat, rhs_sm_tile_shape, float);

  MAKE_LOCAL_MATRIX(partial_sum, out_sm_tile_shape / tb_shape,
                    float);

  for (auto ko = I(0); ko < make_index(CEIL_DIV(K, K_TILE)); ++ko) {
    for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, tb_shape.rows())>(); ++m) {
      for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, tb_shape.cols())>();
           ++kin) {
        lhs_shared_mat
            .tile(Coord(m, kin), tb_shape)
            .dist_to(Coord(IndexDyn(threadIdx.x/32), IndexDyn(threadIdx.x%32))) 
          = lhs_mat
            .tile(Coord(IndexDyn(blockIdx.y), ko), lhs_sm_tile_shape)
            .tile(Coord(m, kin), tb_shape)
            .dist_to(Coord(IndexDyn(threadIdx.x/32), IndexDyn(threadIdx.x%32)));
      }
    }

    // for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, tb_shape.rows())>();
    //      ++kin) {
    //   for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, tb_shape.cols())>(); ++n) {
    //     rhs_shared_mat.tile(Coord(kin, n), tb_shape)
    //         .dist_to(Coord(IndexDyn(threadIdx.x/32), IndexDyn(threadIdx.x%32))) =
    //         rhs_mat.tile(Coord(ko, IndexDyn(blockIdx.x)), rhs_sm_tile_shape)
    //             .tile(Coord(kin, n), tb_shape)
    //             .dist_to(Coord(IndexDyn(threadIdx.x/32), IndexDyn(threadIdx.x%32)));
    //   }
    // }

    __syncthreads();
    out_mat
        .tile(Coord(IndexDyn(blockIdx.y), IndexDyn(blockIdx.x)),
              out_sm_tile_shape)
    <= lhs_shared_mat;

    // contract at 128x128x32 micro-kernel
    // matmul_kernel_coalesced<M_TILE, N_TILE, K_TILE, tb_shape.rows(), tb_shape.cols()>(
    //     lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);
  }
  __syncthreads();


  // for (auto m = I(I(0)); m < make_index<CEIL_DIV(M_TILE, tb_shape.rows())>(); ++m) {
  //   for (auto n = I(I(0)); n < make_index<CEIL_DIV(N_TILE, tb_shape.cols())>(); ++n) {
  //     out_mat
  //         .tile(Coord(IndexDyn(blockIdx.y), IndexDyn(blockIdx.x)),
  //               out_sm_tile_shape)
  //         .tile(Coord(m, n), tb_shape)
  //         .dist_to(Coord(IndexDyn(threadIdx.x/32), IndexDyn(threadIdx.x%32))) =
  //         partial_sum.dist_to(Coord(m, n));
  //   }
  // }
}

void matmul_dataflow_plus_mma(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  // // sec 1, determine, gridDim
  // const int M_TILE = 128;
  // const int N_TILE = 128;
  // // sec 2, determine blockDim
  // const int tb_shape.cols() = 16;
  // const int tb_shape.rows() = 16;
  // // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // const int K_TILE = 32;
  const int M_TILE = 64;
  const int N_TILE = 64;
  const int K_TILE = 64;
  const int M_REG = 16;
  const int K_REG = 8;
  const int N_REG = 8;
  const int THREADS = 256;
  // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // K_TILE > tb_shape.rows()
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(THREADS);
  cudaFuncSetAttribute(
      _matmul_dataflow_plus_mma<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, THREADS>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _matmul_dataflow_plus_mma<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

} // namespace catz::recipes

#endif // CATZILLA_RECIPES_SGEMM_FOR_CHOREO_H_
