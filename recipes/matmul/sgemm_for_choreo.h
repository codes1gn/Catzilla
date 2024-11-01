#ifndef CATZILLA_RECIPES_matmul_choreo_v6_gm_H_
#define CATZILLA_RECIPES_matmul_choreo_v6_gm_H_

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

template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int THREADS>
__global__ void _matmul_choreo_v6_gm(int M, int N, int K, float *lhs,
                                     float *rhs, float *out) {
  auto lhs_shape = make_coord_dyn(M, K);
  auto rhs_shape = make_coord_dyn(K, N);
  auto out_shape = make_coord_dyn(M, N);

  auto lhs_sm_tile_shape = make_coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = make_coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = make_coord(M_TILE, N_TILE);

  auto lhs_reg_tile_shape = make_coord(M_REG, K_REG);
  auto rhs_reg_tile_shape = make_coord(K_REG, N_REG);
  auto out_reg_tile_shape = make_coord(M_REG, N_REG);

  auto tb_shape = make_coord(THREADS / 32, 32);

  auto lhs_mat = make_matrix(lhs, lhs_shape);
  auto rhs_mat = make_matrix(rhs, rhs_shape);
  auto out_mat = make_matrix(out, out_shape);

  MAKE_SHARED_MATRIX(lhs_shared_mat, lhs_sm_tile_shape, half);
  MAKE_SHARED_MATRIX(rhs_shared_mat, rhs_sm_tile_shape, half);

  for (auto ko = I(0); ko < make_index(CEIL_DIV(K, K_TILE)); ++ko) {
    for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, tb_shape.rows())>();
         ++m) {
      for (auto kin = I(0);
           kin < make_index<CEIL_DIV(K_TILE, tb_shape.cols())>(); ++kin) {
        lhs_shared_mat.tile(Coord(m, kin), tb_shape)
            .dist_to(
                Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32))) =
            lhs_mat.tile(Coord(IndexDyn(blockIdx.y), ko), lhs_sm_tile_shape)
                .tile(Coord(m, kin), tb_shape)
                .dist_to(Coord(IndexDyn(threadIdx.x / 32),
                               IndexDyn(threadIdx.x % 32)));
      }
    }
    for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, tb_shape.rows())>();
         ++kin) {
      for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, tb_shape.cols())>();
           ++n) {
        rhs_shared_mat.tile(Coord(kin, n), tb_shape)
            .dist_to(
                Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32))) =
            rhs_mat.tile(Coord(ko, IndexDyn(blockIdx.x)), rhs_sm_tile_shape)
                .tile(Coord(kin, n), tb_shape)
                .dist_to(Coord(IndexDyn(threadIdx.x / 32),
                               IndexDyn(threadIdx.x % 32)));
      }
    }

    __syncthreads();

    // contract at 128x128x32 micro-kernel
    for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, M_REG)>(); ++m) {
      // for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, N_REG)>(); ++n) {
      // }
      float partial_sum[4] = {0.};
      out_mat
          .tile(Coord(IndexDyn(blockIdx.y), IndexDyn(blockIdx.x)),
                out_sm_tile_shape)
          .tile(make_coord_dyn(m, threadIdx.x / 32), Coord(M_REG, N_REG))
          .load_fragments_c(partial_sum);

      for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, K_REG)>();
           ++kin) {
        // TODO: make signs into template arguments
        mma_m16n8k8_f16f32_choreo(
            (float *)partial_sum,
            lhs_shared_mat.tile(make_coord_dyn(m, kin), Coord(M_REG, K_REG)),
            rhs_shared_mat.tile(make_coord_dyn(kin, threadIdx.x / 32),
                                Coord(K_REG, N_REG)),
            (float *)partial_sum);
      }
      out_mat
          .tile(Coord(IndexDyn(blockIdx.y), IndexDyn(blockIdx.x)),
                out_sm_tile_shape)
          .tile(make_coord_dyn(m, threadIdx.x / 32), Coord(M_REG, N_REG))
          .store_fragments_c(partial_sum);
    }
  }
  __syncthreads();
}

// parallel {p, q} by [32, 32]
void matmul_choreo_v6_gm(const int M, const int N, const int K, float *A,
                         float *B, float *C) {
  const int M_TILE = 128;
  const int N_TILE = 128;
  const int K_TILE = 32;
  const int M_REG = 16;
  const int K_REG = 8;
  const int N_REG = 8;
  const int THREADS = 512; // 16 * 32
  // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // K_TILE > tb_shape.rows()
  dim3 gridDim(CEIL_DIV(N, N_TILE), CEIL_DIV(M, M_TILE));
  dim3 blockDim(THREADS);
  cudaFuncSetAttribute(_matmul_choreo_v6_gm<M_TILE, N_TILE, K_TILE, M_REG,
                                            N_REG, K_REG, THREADS>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _matmul_choreo_v6_gm<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, THREADS>
      <<<gridDim, blockDim>>>(M, N, K, A, B, C);
}

template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int THREADS>
__global__ void _matmul_choreo_v2(int M, int N, int K, float alpha, float *lhs,
                                  float *rhs, float beta, float *out) {
  auto lhs_shape = make_coord_dyn(M, K);
  auto rhs_shape = make_coord_dyn(K, N);
  auto out_shape = make_coord_dyn(M, N);

  auto lhs_sm_tile_shape = make_coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = make_coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = make_coord(M_TILE, N_TILE);

  auto lhs_reg_tile_shape = make_coord(M_REG, K_REG);
  auto rhs_reg_tile_shape = make_coord(K_REG, N_REG);
  auto out_reg_tile_shape = make_coord(M_REG, N_REG);

  auto tb_shape = make_coord(THREADS / 32, 32);

  auto lhs_mat = make_matrix(lhs, lhs_shape);
  auto rhs_mat = make_matrix(rhs, rhs_shape);
  auto out_mat = make_matrix(out, out_shape);

  MAKE_SHARED_MATRIX(lhs_shared_mat, lhs_sm_tile_shape, float);
  MAKE_SHARED_MATRIX(rhs_shared_mat, rhs_sm_tile_shape, float);

  MAKE_LOCAL_MATRIX(partial_sum, out_sm_tile_shape / tb_shape, float);

  for (auto ko = I(0); ko < make_index(CEIL_DIV(K, K_TILE)); ++ko) {
    for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, tb_shape.rows())>();
         ++m) {
      for (auto kin = I(0);
           kin < make_index<CEIL_DIV(K_TILE, tb_shape.cols())>(); ++kin) {
        lhs_shared_mat.tile(Coord(m, kin), tb_shape)
            .dist_to(
                Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32))) =
            lhs_mat.tile(Coord(IndexDyn(blockIdx.y), ko), lhs_sm_tile_shape)
                .tile(Coord(m, kin), tb_shape)
                .dist_to(Coord(IndexDyn(threadIdx.x / 32),
                               IndexDyn(threadIdx.x % 32)));
      }
    }

    for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, tb_shape.rows())>();
         ++kin) {
      for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, tb_shape.cols())>();
           ++n) {
        rhs_shared_mat.tile(Coord(kin, n), tb_shape)
            .dist_to(
                Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32))) =
            rhs_mat.tile(Coord(ko, IndexDyn(blockIdx.x)), rhs_sm_tile_shape)
                .tile(Coord(kin, n), tb_shape)
                .dist_to(Coord(IndexDyn(threadIdx.x / 32),
                               IndexDyn(threadIdx.x % 32)));
      }
    }

    __syncthreads();
    matmul_kernel_coalesced<M_TILE, N_TILE, K_TILE, tb_shape.rows(),
                            tb_shape.cols()>(
        lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);
  }
  __syncthreads();

  for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, tb_shape.rows())>();
       ++m) {
    for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, tb_shape.cols())>();
         ++n) {
      out_mat
          .tile(Coord(IndexDyn(blockIdx.y), IndexDyn(blockIdx.x)),
                out_sm_tile_shape)
          .tile(Coord(m, n), tb_shape)
          .dist_to(
              Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32))) =
          partial_sum.dist_to(Coord(m, n));
    }
  }
}

void matmul_choreo_v2(int M, int N, int K, float alpha, float *A, float *B,
                      float beta, float *C) {
  // // sec 1, determine, gridDim
  // const int M_TILE = 128;
  // const int N_TILE = 128;
  // // sec 2, determine blockDim
  // const int tb_shape.cols() = 16;
  // const int tb_shape.rows() = 16;
  // // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // const int K_TILE = 32;
  const int M_TILE = 128;
  const int N_TILE = 128;
  const int K_TILE = 32;
  const int M_REG = 16;
  const int K_REG = 8;
  const int N_REG = 8;
  const int THREADS = 512; // 16 * 32
  // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // K_TILE > tb_shape.rows()
  dim3 gridDim(CEIL_DIV(N, N_TILE), CEIL_DIV(M, M_TILE));
  dim3 blockDim(THREADS);
  cudaFuncSetAttribute(
      _matmul_choreo_v2<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, THREADS>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _matmul_choreo_v2<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// float*float = float
template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int THREADS>
__global__ void _matmul_choreo_v3(int M, int N, int K, float alpha, float *lhs,
                                  float *rhs, float beta, float *out) {
  auto lhs_shape = make_coord_dyn(M, K);
  auto rhs_shape = make_coord_dyn(K, N);
  auto out_shape = make_coord_dyn(M, N);

  auto lhs_sm_tile_shape = make_coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = make_coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = make_coord(M_TILE, N_TILE);

  auto lhs_reg_tile_shape = make_coord(M_REG, K_REG);
  auto rhs_reg_tile_shape = make_coord(K_REG, N_REG);
  auto out_reg_tile_shape = make_coord(M_REG, N_REG);

  auto tb_shape = make_coord(THREADS / 32, 32);

  auto lhs_mat = make_matrix(lhs, lhs_shape);
  auto rhs_mat = make_matrix(rhs, rhs_shape);
  auto out_mat = make_matrix(out, out_shape);

  MAKE_SHARED_MATRIX(lhs_shared_mat, lhs_sm_tile_shape, float);
  MAKE_SHARED_MATRIX(rhs_shared_mat, rhs_sm_tile_shape, float);

  MAKE_SHARED_MATRIX(partial_sum, out_sm_tile_shape, float);

  for (auto ko = I(0); ko < make_index(CEIL_DIV(K, K_TILE)); ++ko) {
    for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, tb_shape.rows())>();
         ++m) {
      for (auto kin = I(0);
           kin < make_index<CEIL_DIV(K_TILE, tb_shape.cols())>(); ++kin) {
        lhs_shared_mat.tile(Coord(m, kin), tb_shape)
            .dist_to(
                Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32))) =
            lhs_mat.tile(Coord(IndexDyn(blockIdx.y), ko), lhs_sm_tile_shape)
                .tile(Coord(m, kin), tb_shape)
                .dist_to(Coord(IndexDyn(threadIdx.x / 32),
                               IndexDyn(threadIdx.x % 32)));
      }
    }
    for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, tb_shape.rows())>();
         ++kin) {
      for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, tb_shape.cols())>();
           ++n) {
        rhs_shared_mat.tile(Coord(kin, n), tb_shape)
            .dist_to(
                Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32))) =
            rhs_mat.tile(Coord(ko, IndexDyn(blockIdx.x)), rhs_sm_tile_shape)
                .tile(Coord(kin, n), tb_shape)
                .dist_to(Coord(IndexDyn(threadIdx.x / 32),
                               IndexDyn(threadIdx.x % 32)));
      }
    }

    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_sm_stationary<M_TILE, N_TILE, K_TILE, tb_shape.rows(),
                                tb_shape.cols()>(
        lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);

    // for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, M_REG)>(); ++m) {
    //   // for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, N_REG)>(); ++n)
    //   {
    //     for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, K_REG)>();
    //     ++kin) {
    //       // TODO: make signs into template arguments
    //       mma_m16n8k8_f16f32_choreo(
    //           partial_sum.tile(make_coord_dyn(m, threadIdx.x / 32),
    //                               Coord(M_REG, N_REG)),
    //           lhs_shared_mat.tile(make_coord_dyn(m, kin),
    //                               Coord(M_REG, K_REG)),
    //           rhs_shared_mat.tile(make_coord_dyn(kin, threadIdx.x / 32),
    //           Coord(K_REG, N_REG)), partial_sum.tile(make_coord_dyn(m,
    //           threadIdx.x / 32),
    //                               Coord(M_REG, N_REG)));
    //     }
    //   // }
    // }
  }
  __syncthreads();

  out_mat.tile(Coord(IndexDyn(blockIdx.y), IndexDyn(blockIdx.x)),
               out_sm_tile_shape) <= partial_sum;
  // for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, tb_shape.rows())>();
  //      ++m) {
  //   for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, tb_shape.cols())>();
  //        ++n) {
  //     out_mat
  //         .tile(Coord(IndexDyn(blockIdx.y), IndexDyn(blockIdx.x)),
  //               out_sm_tile_shape)
  //         .tile(Coord(m, n), tb_shape)
  //         .dist_to(
  //             Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32)))
  //             =
  //         partial_sum.tile(Coord(m, n), tb_shape)
  //         .dist_to(
  //             Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32)));
  //   }
  // }
}

void matmul_choreo_v3(int M, int N, int K, float alpha, float *A, float *B,
                      float beta, float *C) {
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
  const int K_TILE = 32;
  const int M_REG = 16;
  const int K_REG = 8;
  const int N_REG = 8;
  const int THREADS = 256; // 16 * 32
  // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // K_TILE > tb_shape.rows()
  dim3 gridDim(CEIL_DIV(N, N_TILE), CEIL_DIV(M, M_TILE));
  dim3 blockDim(THREADS);
  cudaFuncSetAttribute(
      _matmul_choreo_v3<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, THREADS>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _matmul_choreo_v3<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// half * half = float
template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int THREADS>
__global__ void _matmul_choreo_v4(int M, int N, int K, float alpha, float *lhs,
                                  float *rhs, float beta, float *out) {
  auto lhs_shape = make_coord_dyn(M, K);
  auto rhs_shape = make_coord_dyn(K, N);
  auto out_shape = make_coord_dyn(M, N);

  auto lhs_sm_tile_shape = make_coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = make_coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = make_coord(M_TILE, N_TILE);

  auto lhs_reg_tile_shape = make_coord(M_REG, K_REG);
  auto rhs_reg_tile_shape = make_coord(K_REG, N_REG);
  auto out_reg_tile_shape = make_coord(M_REG, N_REG);

  auto tb_shape = make_coord(THREADS / 32, 32);

  auto lhs_mat = make_matrix(lhs, lhs_shape);
  auto rhs_mat = make_matrix(rhs, rhs_shape);
  auto out_mat = make_matrix(out, out_shape);

  MAKE_SHARED_MATRIX(lhs_shared_mat, lhs_sm_tile_shape, half);
  MAKE_SHARED_MATRIX(rhs_shared_mat, rhs_sm_tile_shape, half);

  MAKE_SHARED_MATRIX(partial_sum, out_sm_tile_shape, float);

  for (auto ko = I(0); ko < make_index(CEIL_DIV(K, K_TILE)); ++ko) {
    for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, tb_shape.rows())>();
         ++m) {
      for (auto kin = I(0);
           kin < make_index<CEIL_DIV(K_TILE, tb_shape.cols())>(); ++kin) {
        lhs_shared_mat.tile(Coord(m, kin), tb_shape)
            .dist_to(
                Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32))) =
            lhs_mat.tile(Coord(IndexDyn(blockIdx.y), ko), lhs_sm_tile_shape)
                .tile(Coord(m, kin), tb_shape)
                .dist_to(Coord(IndexDyn(threadIdx.x / 32),
                               IndexDyn(threadIdx.x % 32)));
      }
    }
    for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, tb_shape.rows())>();
         ++kin) {
      for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, tb_shape.cols())>();
           ++n) {
        rhs_shared_mat.tile(Coord(kin, n), tb_shape)
            .dist_to(
                Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32))) =
            rhs_mat.tile(Coord(ko, IndexDyn(blockIdx.x)), rhs_sm_tile_shape)
                .tile(Coord(kin, n), tb_shape)
                .dist_to(Coord(IndexDyn(threadIdx.x / 32),
                               IndexDyn(threadIdx.x % 32)));
      }
    }

    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_sm_stationary<M_TILE, N_TILE, K_TILE, tb_shape.rows(),
                                tb_shape.cols()>(
        lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);

    // for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, M_REG)>(); ++m) {
    //   // for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, N_REG)>(); ++n)
    //   {
    //     for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, K_REG)>();
    //     ++kin) {
    //       // TODO: make signs into template arguments
    //       mma_m16n8k8_f16f32_choreo(
    //           partial_sum.tile(make_coord_dyn(m, threadIdx.x / 32),
    //                               Coord(M_REG, N_REG)),
    //           lhs_shared_mat.tile(make_coord_dyn(m, kin),
    //                               Coord(M_REG, K_REG)),
    //           rhs_shared_mat.tile(make_coord_dyn(kin, threadIdx.x / 32),
    //           Coord(K_REG, N_REG)), partial_sum.tile(make_coord_dyn(m,
    //           threadIdx.x / 32),
    //                               Coord(M_REG, N_REG)));
    //     }
    //   // }
    // }
  }
  __syncthreads();

  out_mat.tile(Coord(IndexDyn(blockIdx.y), IndexDyn(blockIdx.x)),
               out_sm_tile_shape) <= partial_sum;
  // for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, tb_shape.rows())>();
  //      ++m) {
  //   for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, tb_shape.cols())>();
  //        ++n) {
  //     out_mat
  //         .tile(Coord(IndexDyn(blockIdx.y), IndexDyn(blockIdx.x)),
  //               out_sm_tile_shape)
  //         .tile(Coord(m, n), tb_shape)
  //         .dist_to(
  //             Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32)))
  //             =
  //         partial_sum.tile(Coord(m, n), tb_shape)
  //         .dist_to(
  //             Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32)));
  //   }
  // }
}

void matmul_choreo_v4(int M, int N, int K, float alpha, float *A, float *B,
                      float beta, float *C) {
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
  const int K_TILE = 32;
  const int M_REG = 16;
  const int K_REG = 8;
  const int N_REG = 8;
  const int THREADS = 256; // 16 * 32
  // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // K_TILE > tb_shape.rows()
  dim3 gridDim(CEIL_DIV(N, N_TILE), CEIL_DIV(M, M_TILE));
  dim3 blockDim(THREADS);
  cudaFuncSetAttribute(
      _matmul_choreo_v4<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, THREADS>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _matmul_choreo_v4<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// half * half = float with mma kenrel
template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int THREADS>
__global__ void _matmul_choreo_v5(int M, int N, int K, float alpha, float *lhs,
                                  float *rhs, float beta, float *out) {
  auto lhs_shape = make_coord_dyn(M, K);
  auto rhs_shape = make_coord_dyn(K, N);
  auto out_shape = make_coord_dyn(M, N);

  auto lhs_sm_tile_shape = make_coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = make_coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = make_coord(M_TILE, N_TILE);

  auto lhs_reg_tile_shape = make_coord(M_REG, K_REG);
  auto rhs_reg_tile_shape = make_coord(K_REG, N_REG);
  auto out_reg_tile_shape = make_coord(M_REG, N_REG);

  auto tb_shape = make_coord(THREADS / 32, 32);

  auto lhs_mat = make_matrix(lhs, lhs_shape);
  auto rhs_mat = make_matrix(rhs, rhs_shape);
  auto out_mat = make_matrix(out, out_shape);

  MAKE_SHARED_MATRIX(lhs_shared_mat, lhs_sm_tile_shape, half);
  MAKE_SHARED_MATRIX(rhs_shared_mat, rhs_sm_tile_shape, half);

  MAKE_SHARED_MATRIX(partial_sum, out_sm_tile_shape, float);

  for (auto ko = I(0); ko < make_index(CEIL_DIV(K, K_TILE)); ++ko) {
    for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, tb_shape.rows())>();
         ++m) {
      for (auto kin = I(0);
           kin < make_index<CEIL_DIV(K_TILE, tb_shape.cols())>(); ++kin) {
        lhs_shared_mat.tile(Coord(m, kin), tb_shape)
            .dist_to(
                Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32))) =
            lhs_mat.tile(Coord(IndexDyn(blockIdx.y), ko), lhs_sm_tile_shape)
                .tile(Coord(m, kin), tb_shape)
                .dist_to(Coord(IndexDyn(threadIdx.x / 32),
                               IndexDyn(threadIdx.x % 32)));
      }
    }
    for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, tb_shape.rows())>();
         ++kin) {
      for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, tb_shape.cols())>();
           ++n) {
        rhs_shared_mat.tile(Coord(kin, n), tb_shape)
            .dist_to(
                Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32))) =
            rhs_mat.tile(Coord(ko, IndexDyn(blockIdx.x)), rhs_sm_tile_shape)
                .tile(Coord(kin, n), tb_shape)
                .dist_to(Coord(IndexDyn(threadIdx.x / 32),
                               IndexDyn(threadIdx.x % 32)));
      }
    }

    __syncthreads();

    // contract at 128x128x32 micro-kernel

    for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, M_REG)>(); ++m) {
      // for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, N_REG)>(); ++n) {
      for (auto kin = I(0); kin < make_index<CEIL_DIV(K_TILE, K_REG)>();
           ++kin) {
        // TODO: make signs into template arguments
        mma_m16n8k8_f16f32_choreo(
            partial_sum.tile(make_coord_dyn(m, threadIdx.x / 32),
                             Coord(M_REG, N_REG)),
            lhs_shared_mat.tile(make_coord_dyn(m, kin), Coord(M_REG, K_REG)),
            rhs_shared_mat.tile(make_coord_dyn(kin, threadIdx.x / 32),
                                Coord(K_REG, N_REG)),
            partial_sum.tile(make_coord_dyn(m, threadIdx.x / 32),
                             Coord(M_REG, N_REG)));
      }
      // }
    }
  }
  __syncthreads();

  for (auto m = I(0); m < make_index<CEIL_DIV(M_TILE, tb_shape.rows())>();
       ++m) {
    for (auto n = I(0); n < make_index<CEIL_DIV(N_TILE, tb_shape.cols())>();
         ++n) {
      out_mat
          .tile(Coord(IndexDyn(blockIdx.y), IndexDyn(blockIdx.x)),
                out_sm_tile_shape)
          .tile(Coord(m, n), tb_shape)
          .dist_to(
              Coord(IndexDyn(threadIdx.x / 32), IndexDyn(threadIdx.x % 32))) =
          partial_sum.tile(Coord(m, n), tb_shape)
              .dist_to(Coord(IndexDyn(threadIdx.x / 32),
                             IndexDyn(threadIdx.x % 32)));
    }
  }
}

void matmul_choreo_v5(int M, int N, int K, float alpha, float *A, float *B,
                      float beta, float *C) {
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
  const int K_TILE = 32;
  const int M_REG = 16;
  const int K_REG = 8;
  const int N_REG = 8;
  const int THREADS = 256; // 16 * 32
  // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // K_TILE > tb_shape.rows()
  dim3 gridDim(CEIL_DIV(N, N_TILE), CEIL_DIV(M, M_TILE));
  dim3 blockDim(THREADS);
  cudaFuncSetAttribute(
      _matmul_choreo_v5<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, THREADS>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _matmul_choreo_v5<M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

} // namespace catz::recipes

#endif // CATZILLA_RECIPES_matmul_choreo_v6_gm_H_
