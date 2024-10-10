#ifndef CATZILLA_RECIPES_UTILS_MICRO_KERNELS_WMMA_KERNELS_H_
#define CATZILLA_RECIPES_UTILS_MICRO_KERNELS_WMMA_KERNELS_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "matrix.h"

using namespace catz;

namespace catz::wmma
{

// C += A * B
// m16.n16.k16
// f16.f16.f32
// row-major, col-major, row-major
inline __device__ void matmul_kernel_m16n16k16(half *a_half, half *b_half,
                                               float *c)
{
  // Declare the fragments
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
    A_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
    B_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> C_frag;

  // nvcuda::wmma::fill_fragment(C_frag, 0.0f);

  // NOTE: this is wrong, reinterpret only casts ptr type, not contents
  // half *a_half = reinterpret_cast<half *>(a);
  // half *b_half = reinterpret_cast<half *>(b);

  // Assuming leading dimension is 16
  nvcuda::wmma::load_matrix_sync(A_frag, a_half, 16);
  nvcuda::wmma::load_matrix_sync(B_frag, b_half, 16);
  nvcuda::wmma::load_matrix_sync(C_frag, c, 16, nvcuda::wmma::mem_row_major);

  // Perform the matrix multiplication
  nvcuda::wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

  // Store the result
  nvcuda::wmma::store_matrix_sync(c, C_frag, 16, nvcuda::wmma::mem_row_major);
}

inline __device__ void matmul_kernel_m16n16k16(__nv_bfloat16 *a_half,
                                               __nv_bfloat16 *b_half, float *c)
{
  // Declare the fragments
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                         nvcuda::wmma::row_major>
    A_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                         nvcuda::wmma::row_major>
    B_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> C_frag;

  // nvcuda::wmma::fill_fragment(C_frag, 0.0f);

  // NOTE: this is wrong, reinterpret only casts ptr type, not contents
  // half *a_half = reinterpret_cast<half *>(a);
  // half *b_half = reinterpret_cast<half *>(b);

  // Assuming leading dimension is 16
  nvcuda::wmma::load_matrix_sync(A_frag, a_half, 16);
  nvcuda::wmma::load_matrix_sync(B_frag, b_half, 16);
  nvcuda::wmma::load_matrix_sync(C_frag, c, 16, nvcuda::wmma::mem_row_major);

  // Perform the matrix multiplication
  nvcuda::wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

  // Store the result
  nvcuda::wmma::store_matrix_sync(c, C_frag, 16, nvcuda::wmma::mem_row_major);
}

// // assume we always use blockDim.x == 32, AS CONFIG
// // and only binds threadIdx.y outer from this kernel
// // this setting ensures this kernels uses same threadIdx.y, AND DIFF X (0-31)
// inline __device__ void mma_m16n16k8(const float *A, const float *B, float *C)
// {
//   // 线程索引和块索引
//   int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / 32; // 计算 warp
//   的 ID int lane_id = threadIdx.x % 32; // 当前线程在 warp 中的索引
//
//   // 声明用于存储 A, B, C 块的寄存器
//   float a_frag[8];          // 存储 A 的一部分
//   float b_frag[8];          // 存储 B 的一部分
//   float c_frag[8] = {0.0f}; // 初始化 C 的累加寄存器为 0
//
//   // 将 A 和 B 的子矩阵（16x16 的块）通过 ldmatrix 指令加载到寄存器
//   asm volatile("ldmatrix.sync.aligned.m8n8.x4.row.b16 {%0, %1, %2, %3},
//   [%4];\n"
//                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]),
//                  "=r"(a_frag[3])
//                : "l"(A + warp_id * 16));
//   asm volatile("ldmatrix.sync.aligned.m8n8.x4.row.b16 {%0, %1, %2, %3},
//   [%4];\n"
//                : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]),
//                  "=r"(b_frag[3])
//                : "l"(B + lane_id * 16));
//
//   // 使用内联 PTX 执行矩阵乘法，并累加结果
//   asm volatile(
//     "mma.sync.aligned.m16n16k8.row.row.f32.tf32.tf32.f32 "
//     "{%0, %1, %2, %3, %4, %5, %6, %7}, "
//     "{%8, %9, %10, %11, %12, %13, %14, %15}, "
//     "{%16, %17, %18, %19, %20, %21, %22, %23}, "
//     "{%24, %25, %26, %27, %28, %29, %30, %31};\n"
//     : "=f"(c_frag[0]), "=f"(c_frag[1]), "=f"(c_frag[2]), "=f"(c_frag[3]),
//       "=f"(c_frag[4]), "=f"(c_frag[5]), "=f"(c_frag[6]), "=f"(c_frag[7])
//     : "f"(a_frag[0]), "f"(a_frag[1]), "f"(a_frag[2]), "f"(a_frag[3]),
//       "f"(b_frag[0]), "f"(b_frag[1]), "f"(b_frag[2]), "f"(b_frag[3]),
//       "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]),
//       "f"(c_frag[4]), "f"(c_frag[5]), "f"(c_frag[6]), "f"(c_frag[7]));
//
//   // 将累加结果存储回全局内存 C
//   for (int i = 0; i < 8; ++i) {
//     C[warp_id * 16 + i] = c_frag[i];
//   }
// }

} // namespace catz::wmma

#endif // CATZILLA_RECIPES_UTILS_MICRO_KERNELS_WMMA_KERNELS_H_
