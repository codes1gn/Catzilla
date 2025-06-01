#pragma once

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

__global__ void  __choreo_gpu_matmul_parallel(int M, int N, int K, float* lhs, float* rhs, float* output);

void __choreo_gpu_matmul_host(int M, int N, int K, float* lhs, float* rhs, float* output) {
    dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64));
    dim3 blockDim(256);
    cudaFuncSetAttribute(
        __choreo_gpu_matmul_parallel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    __choreo_gpu_matmul_parallel<<<gridDim, blockDim>>>(M, N, K, lhs, rhs, output);
} // end of choreo-cuda dataflow program

// TODO:
// 1. verbose print for debug
// 2. logic support for index arith and logic
// 3. make it correct
// 4. fit with choreo
__global__ void  __choreo_gpu_matmul_parallel(int M, int N, int K, float* lhs, float* rhs, float* output) {

  auto p = IndexDyn(blockIdx.x);
  auto q = IndexDyn(blockIdx.y);
  auto tid_x = IndexDyn(threadIdx.x);
  auto tid_y = IndexDyn(threadIdx.y);
  auto tid_z = IndexDyn(threadIdx.z);
  // auto lhs_mat = make_matrix(lhs, make_coord(128, 128));
  auto lhs_mat = make_matrix(lhs, make_coord_dyn(M, K));
  // TODO: support 4096 and dyn shape
  // auto rhs_mat = make_matrix(rhs, make_coord(128, 128));
  auto rhs_mat = make_matrix(rhs, make_coord_dyn(K, N));
  // auto output_mat = make_matrix(output, make_coord(128, 128));
  auto output_mat = make_matrix(output, make_coord_dyn(M, N));
  MAKE_SHARED_MATRIX(l2_out, make_coord(64, 64), float);
  for (auto k_tile = I(0); k_tile < make_index(K) / make_index<32>(); ++k_tile) {
    MAKE_SHARED_MATRIX(lhs_load__buf__, make_coord(64, 32), float);
    // 8 = 64x32 / nth (256)
    for (auto iv_row_lhs_load__buf__ = I(0); iv_row_lhs_load__buf__ < make_index<8>(); ++iv_row_lhs_load__buf__) {
      for (auto iv_col_lhs_load__buf__ = I(0); iv_col_lhs_load__buf__ < make_index<1>(); ++iv_col_lhs_load__buf__) {
        lhs_load__buf__
          .tile(Coord(iv_row_lhs_load__buf__, iv_col_lhs_load__buf__), make_coord(8, 32))
          .dist_to_thread()
          // .dist_to(Coord(tid_y, tid_x))
        = lhs_mat.tile(Coord(p, k_tile), make_coord(64, 32))
          .tile(Coord(iv_row_lhs_load__buf__, iv_col_lhs_load__buf__), make_coord(8, 32))
          .dist_to_thread();
          // .dist_to(Coord(tid_y, tid_x));
      }
    }
    // PRINT_MATRIX(lhs_load__buf__);
    __syncthreads();
    MAKE_SHARED_MATRIX(rhs_load__buf__, make_coord(32, 64), float);
    // 8 = 32X64 / NUM THERAD
    for (auto iv_row_rhs_load__buf__ = I(0); iv_row_rhs_load__buf__ < make_index<8>(); ++iv_row_rhs_load__buf__) {
      for (auto iv_col_rhs_load__buf__ = I(0); iv_col_rhs_load__buf__ < make_index<1>(); ++iv_col_rhs_load__buf__) {
        rhs_load__buf__
          .tile(Coord(iv_row_rhs_load__buf__, iv_col_rhs_load__buf__), make_coord(4, 64))
          .dist_to_thread()
          // .dist_to(Coord(tid_y, tid_x))
        = rhs_mat.tile(Coord(k_tile, q), make_coord(32, 64))
          .tile(Coord(iv_row_rhs_load__buf__, iv_col_rhs_load__buf__), make_coord(4, 64))
          .dist_to_thread();
          // .dist_to(Coord(tid_y, tid_x));
      }
    }
    // PRINT_MATRIX(rhs_load__buf__);
    // TODO: above correct, but seams here is wrong
    __syncthreads();
    matmul_kernel_sm_stationary<64, 64, 32, 16, 16>(lhs_load__buf__.data,rhs_load__buf__.data,l2_out.data);
    __syncthreads();

  } // end of choreo-foreach block on 'k_tile'.
  for (auto iv_row_output = I(0); iv_row_output < make_index<16>(); ++iv_row_output) {
    for (auto iv_col_output = I(0); iv_col_output < make_index<1>(); ++iv_col_output) {
      output_mat.tile(Coord(p, q), make_coord(64, 64))
        .tile(Coord(iv_row_output, iv_col_output), make_coord(4, 64))
        .dist_to_thread()
        // .dist_to(Coord(tid_y, tid_x))
      = l2_out
        .tile(Coord(iv_row_output, iv_col_output), make_coord(4, 64))
        .dist_to_thread();
        // .dist_to(Coord(tid_y, tid_x));
    }
  }
  __syncthreads();
  // PRINT_MATRIX(l2_out);
  // ASSERT_EQUAL_MATRIX(l2_out, output_mat.tile(Coord(p, q), make_coord(64, 64)));
  
} // end of choreo-cuda kernel function

}
