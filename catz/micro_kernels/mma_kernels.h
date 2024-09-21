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

inline __device__ void initialize_unsigned_tf32(unsigned *A, int elems,
                                                float value)
{
  for (int i = 0; i < elems; i++) {
    uint32_t b;
    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(b) : "f"(value));
    A[i] = b;
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
// TODO: need housekeeping
inline __device__ void mma_m16n8k16_f16f32(float *d, const half *a,
                                           const half *b, const float *c)
{
  int lane_id = threadIdx.x % 32;
  int group_id = lane_id / 4; // id of quad group where 32 threads are arranged
  int thread_in_group = lane_id % 4;

  unsigned A[4];
  unsigned B[2];
  float C[4];
  float D[4];

  // initialize_unsigned_half(A, 4, __float2half(0.0f));
  // initialize_unsigned_half(B, 2, __float2half(1.0f));

  int lorr = lane_id / 16;
  int lorr_id = lane_id % 16;
  const half *a_new = a + lorr_id * 16 + lorr * 8;

  // TODO: pack all these abstractions back to Matrix
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
    : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3])
    : "r"(get_smem_ptr(a_new)));

  int uord = lane_id / 8;
  int uord_id = lane_id % 8;
  const half *b_new = b + uord * 8 + uord_id * 16;

  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
               : "=r"(B[0]), "=r"(B[1])
               : "r"(get_smem_ptr(b_new)));

  // TODO: make it ldmatrix
  int row_c = group_id; // i < 2
  int row_c_ex = group_id + 8;
  int col_c = thread_in_group * 2; // each thread takes f32x2 contiguous mem

  C[0] = c[row_c * 8 + col_c];
  C[1] = c[row_c * 8 + col_c + 1];
  C[2] = c[row_c_ex * 8 + col_c];
  C[3] = c[row_c_ex * 8 + col_c + 1];

  // const float *c_new = c + lane_id * 8;
  //
  // asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1, %2, %3},
  // [%2];"
  //              : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
  //              : "r"(get_smem_ptr(c_new)));

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               " { %0, %1, %2, %3 }, "
               " { %4, %5, %6, %7 }, "
               " { %8, %9 }, "
               " { %10, %11, %12, %13 };"
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]),
                 "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

  // TODO: convert to float2 transfer, may be faster
  d[row_c * 8 + col_c] = D[0];
  d[row_c * 8 + col_c + 1] = D[1];
  d[row_c_ex * 8 + col_c] = D[2];
  d[row_c_ex * 8 + col_c + 1] = D[3];
}

inline __device__ void mma_m16n8k8_f16f32(float *d, const half *a,
                                          const half *b, const float *c)
{
  int lane_id = threadIdx.x % 32;

  unsigned A[2];
  unsigned B[1];
  float C[4];
  float D[4];

  // initialize_unsigned_half(A, 4, __float2half(0.0f));
  // initialize_unsigned_half(B, 2, __float2half(1.0f));

  int lorr_id = lane_id % 16;
  const half *a_new = a + lorr_id * 8;

  // TODO: pack all these abstractions back to Matrix
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(A[0]), "=r"(A[1])
               : "r"(get_smem_ptr(a_new)));

  int uord_id = lane_id % 8;
  const half *b_new = b + uord_id * 8;

  asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
               : "=r"(B[0])
               : "r"(get_smem_ptr(b_new)));

  C[0] = c[(lane_id / 4) * 8 + (lane_id % 4) * 2];
  C[1] = c[(lane_id / 4) * 8 + (lane_id % 4) * 2 + 1];
  C[2] = c[(lane_id / 4) * 8 + (lane_id % 4) * 2 + 64];
  C[3] = c[(lane_id / 4) * 8 + (lane_id % 4) * 2 + 65];

  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               " { %0, %1, %2, %3 }, "
               " { %4, %5 }, "
               " { %6 }, "
               " { %7, %8, %9, %10 };"
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
               : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(C[0]), "f"(C[1]),
                 "f"(C[2]), "f"(C[3]));

  // TODO: convert to float2 transfer, may be faster
  d[(lane_id / 4) * 8 + (lane_id % 4) * 2] = D[0];
  d[(lane_id / 4) * 8 + (lane_id % 4) * 2 + 1] = D[1];
  d[(lane_id / 4) * 8 + (lane_id % 4) * 2 + 64] = D[2];
  d[(lane_id / 4) * 8 + (lane_id % 4) * 2 + 65] = D[3];
}

// TODO: enable this kernel
// inline __device__ void mma_m16n8k4_tf32f32(float *d, const float *a,
//                                            const float *b, const float *c)
// {
//   int lane_id = threadIdx.x % 32;
//   int outer = lane_id / 4;
//   int inner = lane_id % 4;
//
//   int c_outer = lane_id / 4;
//   int c_inner = 2 * (lane_id % 4);
//
//   int ab_idx = outer * 4 + inner;
//   int cd_idx = c_outer * 8 + c_inner;
//
//   int cd_stride = 64;
//   int ab_stride_inner = 4;
//   int ab_stride_outer = 64;
//
//   uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
//   uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
//   float const *C = reinterpret_cast<float const *>(&c);
//   float *D = reinterpret_cast<float *>(&d);
//
//   asm("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
//       " { %0, %1, %2, %3 }, "
//       " { %4, %5 }, "
//       " { %6 }, "
//       " { %7, %8, %9, %10 };"
//       : "=f"(D[cd_idx]), "=f"(D[cd_idx + 1]), "=f"(D[cd_stride + cd_idx]),
//         "=f"(D[cd_stride + cd_idx + 1])
//       : "r"(A[ab_idx]), "r"(A[ab_stride_outer + ab_idx]), "r"(B[ab_idx]),
//         "f"(C[cd_idx]), "f"(C[cd_idx + 1]), "f"(C[cd_stride + cd_idx]),
//         "f"(C[cd_stride + cd_idx + 1]));
// }

inline __device__ void mma_m16n8k8_tf32f32(float *d, const float *a,
                                           const float *b, const float *c)
{
  int lane_id = threadIdx.x % 32;

  unsigned A[4];
  unsigned B[2];
  float C[4];
  float D[4] = {1.0, 1.0, 1.0, 1.0};

  // initialize_unsigned_tf32(A, 4, 1.0f);
  // initialize_unsigned_tf32(B, 2, 1.0f);

  int lorr = lane_id / 16;
  int lorr_id = lane_id % 16;
  // NOTE: in fp16 case, was 8 and 16 as x, y strides
  const float *a_new = a + lorr_id * 8 + lorr * 4;

  // TODO: pack all these abstractions back to Matrix
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
    : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3])
    : "r"(get_smem_ptr(a_new)));

  int uord = lane_id / 16;
  int uord_id = (lane_id % 16);
  const float *b_up = b + uord_id * 8;
  const float *b_down = b + uord_id * 8 + 4;

  asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
               : "=r"(B[0])
               : "r"(get_smem_ptr(b_up)));

  asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
               : "=r"(B[1])
               : "r"(get_smem_ptr(b_down)));

  // TODO: make it ldmatrix
  int group_id = lane_id / 4; // id of quad group where 32 threads are
  int thread_in_group = lane_id % 4;
  int row_c = group_id; // i < 2
  int row_c_ex = group_id + 8;
  int col_c = thread_in_group * 2; // each thread takes f32x2 contiguous mem

  C[0] = c[row_c * 8 + col_c];
  C[1] = c[row_c * 8 + col_c + 1];
  C[2] = c[row_c_ex * 8 + col_c];
  C[3] = c[row_c_ex * 8 + col_c + 1];

  asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5, %6, %7 }, "
      " { %8, %9 }, "
      " { %10, %11, %12, %13 };"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

  d[row_c * 8 + col_c] = D[0];
  d[row_c * 8 + col_c + 1] = D[1];
  d[row_c_ex * 8 + col_c] = D[2];
  d[row_c_ex * 8 + col_c + 1] = D[3];
}

inline __device__ void mma_m16n8k4_tf32f32(float *d, const float *a,
                                           const float *b, const float *c)
{
  int lane_id = threadIdx.x % 32;

  unsigned A[2];
  unsigned B[1];
  float C[4];
  float D[4];

  // initialize_unsigned_half(A, 4, __float2half(0.0f));
  // initialize_unsigned_tf32(B, 2, 1.0f);

  int lorr_id = lane_id % 16;
  const float *a_new = a + lorr_id * 4;

  // TODO: pack all these abstractions back to Matrix
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(A[0]), "=r"(A[1])
               : "r"(get_smem_ptr(a_new)));

  int uord_id = lane_id % 8;
  const float *b_new = b + uord_id * 4;

  asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
               : "=r"(B[0])
               : "r"(get_smem_ptr(b_new)));

  C[0] = c[(lane_id / 4) * 8 + (lane_id % 4) * 2];
  C[1] = c[(lane_id / 4) * 8 + (lane_id % 4) * 2 + 1];
  C[2] = c[(lane_id / 4) * 8 + (lane_id % 4) * 2 + 64];
  C[3] = c[(lane_id / 4) * 8 + (lane_id % 4) * 2 + 65];

  asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
               " { %0, %1, %2, %3 }, "
               " { %4, %5 }, "
               " { %6 }, "
               " { %7, %8, %9, %10 };"
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
               : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(C[0]), "f"(C[1]),
                 "f"(C[2]), "f"(C[3]));

  // TODO: convert to float2 transfer, may be faster
  d[(lane_id / 4) * 8 + (lane_id % 4) * 2] = D[0];
  d[(lane_id / 4) * 8 + (lane_id % 4) * 2 + 1] = D[1];
  d[(lane_id / 4) * 8 + (lane_id % 4) * 2 + 64] = D[2];
  d[(lane_id / 4) * 8 + (lane_id % 4) * 2 + 65] = D[3];
}

} // namespace catz::mma

#endif // CATZILLA_RECIPES_UTILS_MICRO_KERNELS_MMA_KERNELS_H_
