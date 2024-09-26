#ifndef CATZILLA_RECIPES_UTILS_MICRO_KERNELS_WMMA_PTX_KERNELS_H_
#define CATZILLA_RECIPES_UTILS_MICRO_KERNELS_WMMA_PTX_KERNELS_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "matrix_utils.h"

using namespace catz;

namespace catz::wmma
{

// // C += A * B
// // m16.n16.k16
// // f16.f16.f32
// // row-major, col-major, row-major
// inline __device__ void mma_m16n16k8_tf32_f32_ptx(float *a, float *b,
//                                                    float *c)
// {
//   // Declare the fragments
//   // TODO: incomplete error, does not know why yet?
//   nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, float,
//   nvcuda::wmma::row_major> a_frag;
//   nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, float,
//   nvcuda::wmma::row_major> b_frag;
//   nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> c_frag;
//
//   // nvcuda::wmma::fill_fragment(C_frag, 0.0f);
//
//   // NOTE: this is wrong, reinterpret only casts ptr type, not contents
//   // half *a_half = reinterpret_cast<half *>(a);
//   // half *b_half = reinterpret_cast<half *>(b);
//
//   // Assuming leading dimension is 16
//   nvcuda::wmma::load_matrix_sync(a_frag, a, 16);
//   nvcuda::wmma::load_matrix_sync(b_frag, b, 16);
//   nvcuda::wmma::load_matrix_sync(c_frag, c, 16, nvcuda::wmma::mem_row_major);
//
//   // NOTE: strangely, nvcuda::wmma::fragment has half as basetype, but the
//   datastruct
//   // f16x2 requires "r" constraints, thus the asm inline triggers implicit
//   // conversion from half to "r" allowed types, which has many candidates and
//   // confused at last.
//   // uint32_t const *A_frag = reinterpret_cast<uint32_t const *>(a_frag.x);
//   // uint32_t const *B_frag = reinterpret_cast<uint32_t const *>(b_frag.x);
//
//   // Perform the matrix multiplication
//   nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
//   // asm volatile("wmma.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32
//   \t"
//   //              "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
//   //              "{%8, %9, %10, %11}, {%12, %13, %14, %15}, \t"
//   //              "{%16, %17, %18, %19, %20, %21, %22, %23};"
//   //              : "=f"(c_frag.x[0]), "=f"(c_frag.x[1]), "=f"(c_frag.x[2]),
//   //                "=f"(c_frag.x[3]), "=f"(c_frag.x[4]), "=f"(c_frag.x[5]),
//   //                "=f"(c_frag.x[6]), "=f"(c_frag.x[7])
//   //              : "f"(a_frag.x[0]), "f"(a_frag.x[1]), "f"(a_frag.x[2]),
//   //              "f"(a_frag.x[3]),
//   //                "f"(b_frag.x[0]), "f"(b_frag.x[1]), "f"(b_frag.x[2]),
//   //                "f"(b_frag.x[3]), "f"(c_frag.x[0]), "f"(c_frag.x[1]),
//   //                "f"(c_frag.x[2]), "f"(c_frag.x[3]), "f"(c_frag.x[4]),
//   //                "f"(c_frag.x[5]), "f"(c_frag.x[6]), "f"(c_frag.x[7]));
//
//   // Store the result
//   nvcuda::wmma::store_matrix_sync(c, c_frag, 16,
//   nvcuda::wmma::mem_row_major);
// }

inline __device__ void mma_m16n16k16_f16_f32_ptx(half *a_half, half *b_half,
                                                 float *c)
{
  // Declare the fragments
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
    a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
    b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;

  // nvcuda::wmma::fill_fragment(C_frag, 0.0f);

  // NOTE: this is wrong, reinterpret only casts ptr type, not contents
  // half *a_half = reinterpret_cast<half *>(a);
  // half *b_half = reinterpret_cast<half *>(b);

  // Assuming leading dimension is 16
  nvcuda::wmma::load_matrix_sync(a_frag, a_half, 16);
  // asm volatile("wmma.load.a.sync.aligned.m16n16k16.global.row.f16 \t"
  //              "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8]\t"
  //              : "=r"(A_frag[0]), "=r"(A_frag[1]), "=r"(A_frag[2]),
  //                "=r"(A_frag[3]), "=r"(A_frag[4]), "=r"(A_frag[5]),
  //                "=r"(A_frag[6]), "=r"(A_frag[7])
  //              : "h"(a_half[0]));
  nvcuda::wmma::load_matrix_sync(b_frag, b_half, 16);
  nvcuda::wmma::load_matrix_sync(c_frag, c, 16, nvcuda::wmma::mem_row_major);

  // NOTE: strangely, nvcuda::wmma::fragment has half as basetype, but the
  // datastruct f16x2 requires "r" constraints, thus the asm inline triggers
  // implicit conversion from half to "r" allowed types, which has many
  // candidates and confused at last.
  uint32_t const *A_frag = reinterpret_cast<uint32_t const *>(a_frag.x);
  uint32_t const *B_frag = reinterpret_cast<uint32_t const *>(b_frag.x);

  // Perform the matrix multiplication
  // nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  asm volatile("wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32 \t"
               "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
               "{%8, %9, %10, %11, %12, %13, %14, %15}, \t"
               "{%16, %17, %18, %19, %20, %21, %22, %23}, \t"
               "{%24, %25, %26, %27, %28, %29, %30, %31};"
               : "=f"(c_frag.x[0]), "=f"(c_frag.x[1]), "=f"(c_frag.x[2]),
                 "=f"(c_frag.x[3]), "=f"(c_frag.x[4]), "=f"(c_frag.x[5]),
                 "=f"(c_frag.x[6]), "=f"(c_frag.x[7])
               : "r"(A_frag[0]), "r"(A_frag[1]), "r"(A_frag[2]), "r"(A_frag[3]),
                 "r"(A_frag[4]), "r"(A_frag[5]), "r"(A_frag[6]), "r"(A_frag[7]),
                 "r"(B_frag[0]), "r"(B_frag[1]), "r"(B_frag[2]), "r"(B_frag[3]),
                 "r"(B_frag[4]), "r"(B_frag[5]), "r"(B_frag[6]), "r"(B_frag[7]),
                 "f"(c_frag.x[0]), "f"(c_frag.x[1]), "f"(c_frag.x[2]),
                 "f"(c_frag.x[3]), "f"(c_frag.x[4]), "f"(c_frag.x[5]),
                 "f"(c_frag.x[6]), "f"(c_frag.x[7]));

  // Store the result
  nvcuda::wmma::store_matrix_sync(c, c_frag, 16, nvcuda::wmma::mem_row_major);
}

} // namespace catz::wmma

#endif // CATZILLA_RECIPES_UTILS_MICRO_KERNELS_WMMA_PTX_KERNELS_H_
