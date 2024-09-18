#ifndef CATZILLA_RECIPES_MATMUL_VANILLA_H_
#define CATZILLA_RECIPES_MATMUL_VANILLA_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "index_utils.h"
#include "macros.h"
#include "micro_kernels.h"

using namespace catz;
using namespace catz::cuda;
using namespace catz::wmma;
using namespace catz::mma;

namespace catz::recipes
{

/////////////////////////////////////////////////////////////////////////////////
/// version 1:
/// - basic gmem to smem coalescing
/// - use smem for lhs, rhs, out
/////////////////////////////////////////////////////////////////////////////////
template <const int M_TILE, const int N_TILE, const int K_TILE>
__global__ void _matmul_vanilla(int M, int N, int K, float alpha, float *lhs,
                                float *rhs, float beta, float *out)
{
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  // smem stationary
  // __shared__ float lhs_shared[M_TILE * K_TILE];
  // __shared__ float rhs_shared[K_TILE * N_TILE];
  __shared__ __nv_bfloat16 lhs_shared[M_TILE * K_TILE];
  __shared__ __nv_bfloat16 rhs_shared[K_TILE * N_TILE];
  __shared__ float out_shared[M_TILE * N_TILE];

  // coalescing
  out_shared[distribute_(tid_x, tid_y, 1, N_TILE)] = 0.0;

  for (int k = 0; k < CEIL_DIV(K, K_TILE); k++) { // range(0, 128, 1)
    // use distribute_ for in-tile thread calculation
    // use tile_ for out-tile thread calculation
    // lhs_shared[distribute_(tid_x, tid_y, 1, K_TILE)]
    //   = lhs[tiling_(bid_y, k, M_TILE, K_TILE, K, 1)
    //         + distribute_(tid_x, tid_y, 1, K)];
    lhs_shared[distribute_(tid_x, tid_y, 1, K_TILE)]
      = __float2bfloat16(lhs[tiling_(bid_y, k, M_TILE, K_TILE, K, 1)
                             + distribute_(tid_x, tid_y, 1, K)]);

    // rhs_shared[distribute_(tid_x, tid_y, 1, N_TILE)]
    //   = rhs[tiling_(k, bid_x, K_TILE, N_TILE, N, 1)
    //         + distribute_(tid_x, tid_y, 1, N)];
    rhs_shared[distribute_(tid_x, tid_y, 1, N_TILE)]
      = __float2bfloat16(rhs[tiling_(k, bid_x, K_TILE, N_TILE, N, 1)
                             + distribute_(tid_x, tid_y, 1, N)]);
    __syncthreads();

    // contract at 32x32x32 micro-kernel
    // matmul_kernel_16x16x16(lhs_shared, rhs_shared, out_shared);
    matmul_kernel_m16n16k16(lhs_shared, rhs_shared, out_shared);
    __syncthreads();
  }
  out[tiling_(bid_y, bid_x, M_TILE, N_TILE, N, 1)
      + distribute_(tid_x, tid_y, 1, N)]
    = out_shared[distribute_(tid_x, tid_y, 1, N_TILE)];
  // out[tiling_(bid_y, bid_x, M_TILE, N_TILE, N, 1) + distribute_(tid_x, tid_y,
  // 1, N)] = alpha * partial_sum;
}

void matmul_f16f32(int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C)
{
  dim3 gridDim(CEIL_DIV(M, 16), CEIL_DIV(N, 16));
  dim3 blockDim(16, 16);
  cudaFuncSetAttribute(_matmul_vanilla<16, 16, 16>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _matmul_vanilla<16, 16, 16>
    <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

} // namespace catz::recipes

#endif // CATZILLA_RECIPES_MATMUL_VANILLA_H_
