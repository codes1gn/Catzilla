#ifndef CATZILLA_RECIPES_UTILS_MICRO_KERNELS_MMA_KERNELS_H_
#define CATZILLA_RECIPES_UTILS_MICRO_KERNELS_MMA_KERNELS_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "index_utils.h"

using namespace catz;

namespace catz::mma
{

inline __device__ void initialize_unsigned_half(unsigned *A, int elems,
                                                half value)
{
  // half 类型 1.0 的二进制表示为 0x3C00
  unsigned short half_value
    = __half_as_short(value); // 取得 half 类型值的二进制表示
  unsigned packed_value
    = half_value | (half_value << 16); // 将两个 half 值打包成一个 unsigned 值

  // 初始化 unsigned 数组
  for (int i = 0; i < elems; i++) {
    A[i] = packed_value;
  }
}

__device__ __forceinline__ uint get_smem_ptr(const void *ptr)
{
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

// assume we always use blockDim.x == 32, AS CONFIG
// and only binds threadIdx.y outer from this kernel
// this setting ensures this kernels uses same threadIdx.y, AND DIFF X (0-31)
// TODO: align naming with ptx manual
inline __device__ void mma_m16n8k16_f16f32(float *d, const half *a,
                                           const half *b, const float *c)
{
  int lane_id = threadIdx.x % 32;
  int group_id = lane_id / 4; // id of quad group where 32 threads are arranged
  int thread_in_group = lane_id % 4;

  int row_a = group_id;        // 0, 1, 4, 5
  int row_a_ex = group_id + 8; // 2, 3, 6, 7

  int col_a = thread_in_group;        // 0, 1, 2, 3
  int col_a_ex = thread_in_group + 4; // 4, 5, 6, 7
  //
  int row_b = thread_in_group;        // i < 2
  int row_b_ex = thread_in_group + 4; // i >= 2
  //
  int col_b = group_id;

  unsigned A[4];
  unsigned B[2];
  float C[4];
  float D[4];

  initialize_unsigned_half(A, 4, __float2half(1.0f));
  initialize_unsigned_half(B, 2, __float2half(0.0f));

  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(B[0]), "=r"(B[1])
               : "r"(get_smem_ptr(&b)));

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               " { %0, %1, %2, %3 }, "
               " { %4, %5, %6, %7 }, "
               " { %8, %9 }, "
               " { %10, %11, %12, %13 };"
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]),
                 "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

  int row_c = group_id; // i < 2
  int row_c_ex = group_id + 8;
  int col_c = thread_in_group * 2; // each thread takes f32x2 contiguous mem

  // TODO: convert to float2 transfer, may be faster
  d[row_c * 8 + col_c] = D[0];
  d[row_c * 8 + col_c + 1] = D[1];
  d[row_c_ex * 8 + col_c] = D[2];
  d[row_c_ex * 8 + col_c + 1] = D[3];

  // asm volatile("stmatrix.sync.aligned.m8n8.x4.b16 "
  //              "[%0], {%1, %2, %3, %4};"
  //              : "=l"(d)
  //              : "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

  // uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  // uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  // float const *C = reinterpret_cast<float const *>(&c);
  // float *D = reinterpret_cast<float *>(&d);

  // asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
  //     " { %0, %1, %2, %3 }, "
  //     " { %4, %5, %6, %7 }, "
  //     " { %8, %9 }, "
  //     " { %10, %11, %12, %13 };"
  //     : "=f"(D[row_c*8+col_c]),
  //       "=f"(D[row_c*8+col_c + 1]),
  //       "=f"(D[row_c_ex*8+col_c]),
  //       "=f"(D[row_c_ex*8+col_c + 1])
  //     : "r"(A[row_a*8+col_a]),
  //       "r"(A[row_a_ex*8+col_a]),
  //       "r"(A[row_a*8+col_a_ex]),
  //       "r"(A[row_a_ex*8+col_a_ex]),
  //       "r"(B[row_b*8+col_b]),
  //       "r"(B[row_b_ex*8+col_b]),
  //       "f"(C[row_c*8+col_c]),
  //       "f"(C[row_c*8+col_c + 1]),
  //       "f"(C[row_c_ex*8+col_c]),
  //       "f"(C[row_c_ex*8+col_c + 1]));
}

inline __device__ void mma_m16n8k4_tf32f32(float *d, const float *a,
                                           const float *b, const float *c)
{
  int lane_id = threadIdx.x % 32;
  int outer = lane_id / 4;
  int inner = lane_id % 4;

  int c_outer = lane_id / 4;
  int c_inner = 2 * (lane_id % 4);

  int ab_idx = outer * 4 + inner;
  int cd_idx = c_outer * 8 + c_inner;

  int cd_stride = 64;
  int ab_stride_inner = 4;
  int ab_stride_outer = 64;

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  asm("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5 }, "
      " { %6 }, "
      " { %7, %8, %9, %10 };"
      : "=f"(D[cd_idx]), "=f"(D[cd_idx + 1]), "=f"(D[cd_stride + cd_idx]),
        "=f"(D[cd_stride + cd_idx + 1])
      : "r"(A[ab_idx]), "r"(A[ab_stride_outer + ab_idx]), "r"(B[ab_idx]),
        "f"(C[cd_idx]), "f"(C[cd_idx + 1]), "f"(C[cd_stride + cd_idx]),
        "f"(C[cd_stride + cd_idx + 1]));
}

inline __device__ void mma_m16n8k8_tf32f32(float *d, const float *a,
                                           const float *b, const float *c)
{
  int lane_id = threadIdx.x % 32;
  int outer = lane_id / 4;
  int inner = lane_id % 4;

  int c_outer = lane_id / 4;
  int c_inner = 2 * (lane_id % 4);

  int ab_idx = outer * 4 + inner;
  int cd_idx = c_outer * 8 + c_inner;

  int cd_stride = 64;
  int ab_stride_inner = 4;
  int ab_stride_outer = 64;

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5, %6, %7 }, "
      " { %8, %9 }, "
      " { %10, %11, %12, %13 };"
      : "=f"(D[cd_idx]), "=f"(D[cd_idx + 1]), "=f"(D[cd_stride + cd_idx]),
        "=f"(D[cd_stride + cd_idx + 1])
      : "r"(A[ab_idx]), "r"(A[ab_idx + ab_stride_inner]),
        "r"(A[ab_stride_outer + ab_idx]),
        "r"(A[ab_stride_outer + ab_idx + ab_stride_inner]), "r"(B[ab_idx]),
        "r"(B[ab_stride_inner + ab_idx]), "f"(C[cd_idx]), "f"(C[cd_idx + 1]),
        "f"(C[cd_stride + cd_idx]), "f"(C[cd_stride + cd_idx + 1]));
}

} // namespace catz::mma

#endif // CATZILLA_RECIPES_UTILS_MICRO_KERNELS_MMA_KERNELS_H_
