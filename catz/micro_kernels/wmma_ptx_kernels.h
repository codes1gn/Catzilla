#ifndef CATZILLA_RECIPES_UTILS_MICRO_KERNELS_WMMA_PTX_KERNELS_H_
#define CATZILLA_RECIPES_UTILS_MICRO_KERNELS_WMMA_PTX_KERNELS_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "index_utils.h"

using namespace nvcuda;
using namespace nvcuda::wmma;

// C += A * B
// m16.n16.k16
// f16.f16.f32
// row-major, col-major, row-major
inline __device__ void matmul_kernel_m16n16k16_ptx(half *a_half, half *b_half,
                                                   float *c)
{
  // Declare the fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> A_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> C_frag;

  // wmma::fill_fragment(C_frag, 0.0f);

  // NOTE: this is wrong, reinterpret only casts ptr type, not contents
  // half *a_half = reinterpret_cast<half *>(a);
  // half *b_half = reinterpret_cast<half *>(b);

  // Assuming leading dimension is 16
  wmma::load_matrix_sync(A_frag, a_half, 16);
  wmma::load_matrix_sync(B_frag, b_half, 16);
  wmma::load_matrix_sync(C_frag, c, 16, wmma::mem_row_major);

  // Perform the matrix multiplication
  wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

  // Store the result
  wmma::store_matrix_sync(c, C_frag, 16, wmma::mem_row_major);
}

#endif // CATZILLA_RECIPES_UTILS_MICRO_KERNELS_WMMA_PTX_KERNELS_H_
