#ifndef CATZILLA_RECIPES_KERNELS_MATMUL_WMMA_BASE_H_
#define CATZILLA_RECIPES_KERNELS_MATMUL_WMMA_BASE_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "index_utils.h"
#include "macros.h"
#include "micro_kernels.h"

template <const int M_TILE, const int N_TILE, const int K_TILE>
__global__ void _catzilla_matmul_v1_t32(int M, int N, int K, float alpha,
                                        float *lhs, float *rhs, float beta,
                                        float *out)
{
  auto lhs_shape = make_coord(M, K);
  auto rhs_shape = make_coord(K, N);
  auto out_shape = make_coord(M, N);

  auto lhs_sm_tile_shape = Coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = Coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = Coord(M_TILE, N_TILE);

  // make sure inner block looks like this, to ensure coalescing

  Matrix lhs_mat = Matrix(lhs, lhs_shape);
  Matrix rhs_mat = Matrix(rhs, rhs_shape);
  Matrix out_mat = Matrix(out, out_shape);

  // __shared__ float lhs_shared[M_TILE* (K_TILE+1)];

  MatrixH lhs_shared_mat = make_shared_half<M_TILE, K_TILE>();
  MatrixH rhs_shared_mat = make_shared_half<K_TILE, N_TILE>();
  Matrix out_shared_mat = make_shared<M_TILE, N_TILE>();

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, 2); m++) { // range(0, 128, 1)
      out_shared_mat.tile_ex(Coord(m, 0), Coord(2, 16)).dist_to_thread() = 0.0;
    }
  }

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, 2); m++) { // range(0, 128, 1)
      lhs_shared_mat.tile_ex(Coord(m, 0), Coord(2, 16)).dist_to_thread()
        = lhs_mat.tile_ex(Coord(blockIdx.y, ko), Coord(16, 16))
            .tile_ex(Coord(m, 0), Coord(2, 16))
            .dist_to_thread();
      rhs_shared_mat.tile_ex(Coord(m, 0), Coord(2, 16)).dist_to_thread()
        = rhs_mat.tile_ex(Coord(ko, blockIdx.x), Coord(16, 16))
            .tile_ex(Coord(m, 0), Coord(2, 16))
            .dist_to_thread();
    }
    __syncthreads();

    // matmul_kernel_16x16x16_thread_32(lhs_shared_mat.data,
    // rhs_shared_mat.data,
    //                                  out_shared_mat.data);
    matmul_kernel_m16n16k16_ptx(lhs_shared_mat.data, rhs_shared_mat.data,
                                out_shared_mat.data);
    __syncthreads();
  }
  for (int m = 0; m < CEIL_DIV(M_TILE, 2); m++) { // range(0, 128, 1)
    out_mat.tile_ex(Coord(blockIdx.y, blockIdx.x), Coord(16, 16))
      .tile_ex(Coord(m, 0), Coord(2, 16))
      .dist_to_thread()
      = out_shared_mat.tile_ex(Coord(m, 0), Coord(2, 16)).dist_to_thread();
  }
}

void catzilla_matmul_v1_t32(int M, int N, int K, float alpha, float *A,
                            float *B, float beta, float *C)
{
  const int M_TILE = 16;
  const int N_TILE = 16;
  const int K_TILE = 16;
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(32, 1);
  cudaFuncSetAttribute(_catzilla_matmul_v1_t32<M_TILE, N_TILE, K_TILE>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v1_t32<M_TILE, N_TILE, K_TILE>
    <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

#endif // CATZILLA_RECIPES_KERNELS_MATMUL_WMMA_BASE_H_
