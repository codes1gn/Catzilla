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

// assume we always use blockDim.x == 32, AS CONFIG
// and only binds threadIdx.y outer from this kernel
// this setting ensures this kernels uses same threadIdx.y, AND DIFF X (0-31)
inline __device__ void mma_m16n16k8_f16f32(float *d, const half *a,
                                           const half *b, const float *c)
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

  asm("mma.sync.aligned.m16n8k16.row.row.f32.f16.f16.f32 "
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
