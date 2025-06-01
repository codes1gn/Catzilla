#ifndef CATZILLA_RECIPES_MATMUL_TENSOR_CORES_H_
#define CATZILLA_RECIPES_MATMUL_TENSOR_CORES_H_

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

template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int X_THREAD,
          const int Y_THREAD>
__global__ void
_matmul_tensor_cores_wmma_m16n16k16_f16f32(int M, int N, int K, float alpha,
                                           float *lhs, float *rhs, float beta,
                                           float *out) {
  auto lhs_shape = CoordDyn(M, K);
  auto rhs_shape = CoordDyn(K, N);
  auto out_shape = CoordDyn(M, N);

  auto lhs_sm_tile_shape = CoordDyn(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = CoordDyn(K_TILE, N_TILE);
  auto out_sm_tile_shape = CoordDyn(M_TILE, N_TILE);

  // make sure inner block looks like this, to ensure coalescing

  MatrixDyn<float> lhs_mat = MatrixDyn<float>(lhs, lhs_shape);
  MatrixDyn<float> rhs_mat = MatrixDyn<float>(rhs, rhs_shape);
  MatrixDyn<float> out_mat = MatrixDyn<float>(out, out_shape);

  __shared__ half lhs_shared[M_TILE * K_TILE];
  __shared__ half rhs_shared[K_TILE * N_TILE];
  // __shared__ float out_shared[M_TILE * N_TILE];

  MatrixDyn<half> lhs_shared_mat =
      MatrixDyn<half>(lhs_shared, lhs_sm_tile_shape);
  MatrixDyn<half> rhs_shared_mat =
      MatrixDyn<half>(rhs_shared, rhs_sm_tile_shape);
  // MAKE_SHARED(rhs_shared_mat, K_TILE, N_TILE, half);
  // MatrixDyn<float> out_shared_mat = MatrixDyn<float>(out_shared,
  // out_sm_tile_shape);
  MAKE_SHARED(out_shared_mat, M_TILE, N_TILE, float);

  out_shared_mat.fill(0.);

  // TODO: wrong, since this will release shared after make_shared call
  // impl a Macro based make_shared and make_local, ensure it won't release
  //
  // MatrixDyn<half> lhs_shared_mat = make_shared<M_TILE, K_TILE, half>();
  // MatrixDyn<half> rhs_shared_mat = make_shared<K_TILE, N_TILE, half>();
  // MatrixDyn<float> out_shared_mat = make_shared<M_TILE, N_TILE>(); // 16*16*4
  // = 1024 Bytes
  //

  // MatrixDyn<float> partial_sum
  //   = make_local<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(N_TILE, X_THREAD)>();

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int kin = 0; kin < CEIL_DIV(K_TILE, 16); kin++) {
      lhs_shared_mat.tile(CoordDyn(0, kin), CoordDyn(M_TILE, 16)) <=
          lhs_mat.tile(CoordDyn(blockIdx.y, ko), lhs_sm_tile_shape)
              .tile(CoordDyn(0, kin), CoordDyn(M_TILE, 16));
    }
    rhs_shared_mat <= rhs_mat.tile(CoordDyn(ko, blockIdx.x), rhs_sm_tile_shape);
    __syncthreads();

    mma_m16n16k16_f16_f32_ptx(lhs_shared_mat.data, rhs_shared_mat.data,
                              out_shared_mat.data);
  }
  __syncthreads();

  out_mat.tile(CoordDyn(blockIdx.y, blockIdx.x), out_sm_tile_shape) <=
      out_shared_mat;
}

void matmul_tensor_cores_wmma_m16n16k16_f16f32(int M, int N, int K, float alpha,
                                               float *A, float *B, float beta,
                                               float *C) {
  const int M_TILE = 16;
  const int K_TILE = 16;
  const int N_TILE = 16;
  const int M_REG = 2;
  const int K_REG = 2;
  const int N_REG = 16;
  const int X_THREAD = 32;
  const int Y_THREAD = 1;
  assert(M_REG * K_REG > X_THREAD * Y_THREAD);
  assert(N_REG * K_REG > X_THREAD * Y_THREAD);

  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
      _matmul_tensor_cores_wmma_m16n16k16_f16f32<
          M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, X_THREAD, Y_THREAD>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _matmul_tensor_cores_wmma_m16n16k16_f16f32<M_TILE, N_TILE, K_TILE, M_REG,
                                             N_REG, K_REG, X_THREAD, Y_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  return;
}

template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int X_THREAD,
          const int Y_THREAD>
__global__ void _matmul_tensor_cores_mma_m16n8k8_f16f32(int M, int N, int K,
                                                        float alpha, float *lhs,
                                                        float *rhs, float beta,
                                                        float *out) {
  auto lhs_shape = CoordDyn(M, K);
  auto rhs_shape = CoordDyn(K, N);
  auto out_shape = CoordDyn(M, N);

  auto lhs_sm_tile_shape = CoordDyn(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = CoordDyn(K_TILE, N_TILE);
  auto out_sm_tile_shape = CoordDyn(M_TILE, N_TILE);

  // make sure inner block looks like this, to ensure coalescing

  MatrixDyn<float> lhs_mat = MatrixDyn<float>(lhs, lhs_shape);
  MatrixDyn<float> rhs_mat = MatrixDyn<float>(rhs, rhs_shape);
  MatrixDyn<float> out_mat = MatrixDyn<float>(out, out_shape);

  // __shared__ float lhs_shared[M_TILE * K_TILE];
  // __shared__ float rhs_shared[K_TILE * N_TILE];
  // __shared__ float out_shared[M_TILE * N_TILE];

  // MatrixDyn<float> lhs_shared_mat = MatrixDyn<float>(lhs_shared,
  // lhs_sm_tile_shape); MatrixDyn<float> rhs_shared_mat =
  // MatrixDyn<float>(rhs_shared, rhs_sm_tile_shape);
  // MAKE_SHARED(rhs_shared_mat, K_TILE, N_TILE, half); MatrixDyn<float>
  // out_shared_mat = MatrixDyn<float>(out_shared, out_sm_tile_shape);
  MAKE_SHARED(lhs_shared_mat, M_TILE, K_TILE, half);
  MAKE_SHARED(rhs_shared_mat, K_TILE, N_TILE, half);
  MAKE_SHARED(out_shared_mat, M_TILE, N_TILE, float);

  out_shared_mat.fill(0.);

  // TODO: wrong, since this will release shared after make_shared call
  // impl a Macro based make_shared and make_local, ensure it won't release
  //
  // MatrixDyn<half> lhs_shared_mat = make_shared<M_TILE, K_TILE, half>();
  // MatrixDyn<half> rhs_shared_mat = make_shared<K_TILE, N_TILE, half>();
  // MatrixDyn<float> out_shared_mat = make_shared<M_TILE, N_TILE>(); // 16*16*4
  // = 1024 Bytes
  //

  // MatrixDyn<float> partial_sum
  //   = make_local<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(N_TILE, X_THREAD)>();

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, CEIL_DIV(32, K_TILE)); m++) {
#pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, K_TILE); kin++) {
        lhs_shared_mat.tile(CoordDyn(m, kin),
                            CoordDyn(CEIL_DIV(32, K_TILE), K_TILE)) <=
            lhs_mat.tile(CoordDyn(blockIdx.x, ko), lhs_sm_tile_shape)
                .tile(CoordDyn(m, kin), CoordDyn(CEIL_DIV(32, K_TILE), K_TILE));
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, CEIL_DIV(32, N_TILE)); kin++) {
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, N_TILE); n++) {
        rhs_shared_mat.tile(CoordDyn(kin, n),
                            CoordDyn(CEIL_DIV(32, N_TILE), N_TILE)) <=
            rhs_mat.tile(CoordDyn(ko, blockIdx.y), rhs_sm_tile_shape)
                .tile(CoordDyn(kin, n), CoordDyn(CEIL_DIV(32, N_TILE), N_TILE));
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    // cuda_m16n8k16_f16f32(out_shared_mat.data, lhs_shared_mat.data,
    //                      rhs_shared_mat.data, out_shared_mat.data);
    mma_m16n8k8_f16f32_neo(out_shared_mat, lhs_shared_mat, rhs_shared_mat,
                           out_shared_mat);
    // mma_m16n8k8_f16f32(out_shared_mat.data, lhs_shared_mat.data,
    //                    rhs_shared_mat.data, out_shared_mat.data);
    // identity<16, 8>(lhs_shared_mat, out_shared_mat.tile(CoordDyn(0, ko),
    // CoordDyn(16, 8))); identity<8, 16>(rhs_shared_mat,
    // out_shared_mat.tile(CoordDyn(ko, 0), CoordDyn(8, 16)));
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, CEIL_DIV(32, N_TILE)); m++) {
    out_mat.tile(CoordDyn(blockIdx.x, blockIdx.y), out_sm_tile_shape)
            .tile(CoordDyn(m, 0), CoordDyn(CEIL_DIV(32, N_TILE), N_TILE)) <=
        out_shared_mat.tile(CoordDyn(m, 0),
                            CoordDyn(CEIL_DIV(32, N_TILE), N_TILE));
  }
  // out_mat.tile(CoordDyn(blockIdx.x, blockIdx.y), out_sm_tile_shape) <=
  // out_shared_mat;
}

void matmul_tensor_cores_mma_m16n8k8_f16f32(int M, int N, int K, float alpha,
                                            float *A, float *B, float beta,
                                            float *C) {
  const int M_TILE = 16;
  const int K_TILE = 8;
  const int N_TILE = 8;
  const int M_REG = 2;
  const int K_REG = 2;
  const int N_REG = 16;
  const int X_THREAD = 32;
  const int Y_THREAD = 1;
  assert(M_REG * K_REG > X_THREAD * Y_THREAD);
  assert(N_REG * K_REG > X_THREAD * Y_THREAD);

  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
      _matmul_tensor_cores_mma_m16n8k8_f16f32<M_TILE, N_TILE, K_TILE, M_REG,
                                              N_REG, K_REG, X_THREAD, Y_THREAD>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _matmul_tensor_cores_mma_m16n8k8_f16f32<M_TILE, N_TILE, K_TILE, M_REG, N_REG,
                                          K_REG, X_THREAD, Y_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  return;
}

template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int X_THREAD,
          const int Y_THREAD>
__global__ void
_matmul_tensor_cores_mma_m16n8k16_f16f32(int M, int N, int K, float alpha,
                                         float *lhs, float *rhs, float beta,
                                         float *out) {
  auto lhs_shape = CoordDyn(M, K);
  auto rhs_shape = CoordDyn(K, N);
  auto out_shape = CoordDyn(M, N);

  auto lhs_sm_tile_shape = CoordDyn(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = CoordDyn(K_TILE, N_TILE);
  auto out_sm_tile_shape = CoordDyn(M_TILE, N_TILE);

  // make sure inner block looks like this, to ensure coalescing

  MatrixDyn<float> lhs_mat = MatrixDyn<float>(lhs, lhs_shape);
  MatrixDyn<float> rhs_mat = MatrixDyn<float>(rhs, rhs_shape);
  MatrixDyn<float> out_mat = MatrixDyn<float>(out, out_shape);

  // __shared__ float lhs_shared[M_TILE * K_TILE];
  // __shared__ float rhs_shared[K_TILE * N_TILE];
  // __shared__ float out_shared[M_TILE * N_TILE];

  // MatrixDyn<float> lhs_shared_mat = MatrixDyn<float>(lhs_shared,
  // lhs_sm_tile_shape); MatrixDyn<float> rhs_shared_mat =
  // MatrixDyn<float>(rhs_shared, rhs_sm_tile_shape);
  // MAKE_SHARED(rhs_shared_mat, K_TILE, N_TILE, half); MatrixDyn<float>
  // out_shared_mat = MatrixDyn<float>(out_shared, out_sm_tile_shape);
  MAKE_SHARED(lhs_shared_mat, M_TILE, K_TILE, half);
  MAKE_SHARED(rhs_shared_mat, K_TILE, N_TILE, half);
  MAKE_SHARED(out_shared_mat, M_TILE, N_TILE, float);

  out_shared_mat.fill(0.);

  // TODO: wrong, since this will release shared after make_shared call
  // impl a Macro based make_shared and make_local, ensure it won't release
  //
  // MatrixDyn<half> lhs_shared_mat = make_shared<M_TILE, K_TILE, half>();
  // MatrixDyn<half> rhs_shared_mat = make_shared<K_TILE, N_TILE, half>();
  // MatrixDyn<float> out_shared_mat = make_shared<M_TILE, N_TILE>(); // 16*16*4
  // = 1024 Bytes
  //

  // MatrixDyn<float> partial_sum
  //   = make_local<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(N_TILE, X_THREAD)>();

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, CEIL_DIV(32, K_TILE)); m++) {
#pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, K_TILE); kin++) {
        lhs_shared_mat
            .tile(CoordDyn(m, kin), CoordDyn(CEIL_DIV(32, K_TILE), K_TILE))
            .dist_to_thread() =
            lhs_mat.tile(CoordDyn(blockIdx.x, ko), lhs_sm_tile_shape)
                .tile(CoordDyn(m, kin), CoordDyn(CEIL_DIV(32, K_TILE), K_TILE))
                .dist_to_thread();
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, CEIL_DIV(32, N_TILE)); kin++) {
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, N_TILE); n++) {
        rhs_shared_mat
            .tile(CoordDyn(kin, n), CoordDyn(CEIL_DIV(32, N_TILE), N_TILE))
            .dist_to_thread() =
            rhs_mat.tile(CoordDyn(ko, blockIdx.y), rhs_sm_tile_shape)
                .tile(CoordDyn(kin, n), CoordDyn(CEIL_DIV(32, N_TILE), N_TILE))
                .dist_to_thread();
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    // cuda_m16n8k16_f16f32(out_shared_mat.data, lhs_shared_mat.data,
    //                      rhs_shared_mat.data, out_shared_mat.data);
    mma_m16n8k16_f16f32(out_shared_mat.data, lhs_shared_mat.data,
                        rhs_shared_mat.data, out_shared_mat.data);
    // identity<16, 8>(lhs_shared_mat, out_shared_mat.tile(CoordDyn(0, ko),
    // CoordDyn(16, 8))); identity<8, 16>(rhs_shared_mat,
    // out_shared_mat.tile(CoordDyn(ko, 0), CoordDyn(8, 16)));
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, CEIL_DIV(32, N_TILE)); m++) {
#pragma unroll
    out_mat.tile(CoordDyn(blockIdx.x, blockIdx.y), out_sm_tile_shape)
        .tile(CoordDyn(m, 0), CoordDyn(CEIL_DIV(32, N_TILE), N_TILE))
        .dist_to_thread() =
        out_shared_mat
            .tile(CoordDyn(m, 0), CoordDyn(CEIL_DIV(32, N_TILE), N_TILE))
            .dist_to_thread();
  }
}

void matmul_tensor_cores_mma_m16n8k16_f16f32(int M, int N, int K, float alpha,
                                             float *A, float *B, float beta,
                                             float *C) {
  const int M_TILE = 16;
  const int K_TILE = 16;
  const int N_TILE = 8;
  const int M_REG = 2;
  const int K_REG = 2;
  const int N_REG = 16;
  const int X_THREAD = 32;
  const int Y_THREAD = 1;
  assert(M_REG * K_REG > X_THREAD * Y_THREAD);
  assert(N_REG * K_REG > X_THREAD * Y_THREAD);

  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
      _matmul_tensor_cores_mma_m16n8k16_f16f32<
          M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, X_THREAD, Y_THREAD>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _matmul_tensor_cores_mma_m16n8k16_f16f32<M_TILE, N_TILE, K_TILE, M_REG, N_REG,
                                           K_REG, X_THREAD, Y_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  return;
}

template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int X_THREAD,
          const int Y_THREAD>
__global__ void
_matmul_tensor_cores_mma_m16n8k8_tf32f32(int M, int N, int K, float alpha,
                                         float *lhs, float *rhs, float beta,
                                         float *out) {
  auto lhs_shape = CoordDyn(M, K);
  auto rhs_shape = CoordDyn(K, N);
  auto out_shape = CoordDyn(M, N);

  auto lhs_sm_tile_shape = CoordDyn(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = CoordDyn(K_TILE, N_TILE);
  auto out_sm_tile_shape = CoordDyn(M_TILE, N_TILE);

  // make sure inner block looks like this, to ensure coalescing

  MatrixDyn<float> lhs_mat = MatrixDyn<float>(lhs, lhs_shape);
  MatrixDyn<float> rhs_mat = MatrixDyn<float>(rhs, rhs_shape);
  MatrixDyn<float> out_mat = MatrixDyn<float>(out, out_shape);

  // __shared__ float lhs_shared[M_TILE * K_TILE];
  // __shared__ float rhs_shared[K_TILE * N_TILE];
  // __shared__ float out_shared[M_TILE * N_TILE];

  // MatrixDyn<float> lhs_shared_mat = MatrixDyn<float>(lhs_shared,
  // lhs_sm_tile_shape); MatrixDyn<float> rhs_shared_mat =
  // MatrixDyn<float>(rhs_shared, rhs_sm_tile_shape);
  // MAKE_SHARED(rhs_shared_mat, K_TILE, N_TILE, half); MatrixDyn<float>
  // out_shared_mat = MatrixDyn<float>(out_shared, out_sm_tile_shape);
  MAKE_SHARED(lhs_shared_mat, M_TILE, K_TILE, float);
  MAKE_SHARED(rhs_shared_mat, K_TILE, N_TILE, float);
  MAKE_SHARED(out_shared_mat, M_TILE, N_TILE, float);

  out_shared_mat.fill(0.);

  // TODO: wrong, since this will release shared after make_shared call
  // impl a Macro based make_shared and make_local, ensure it won't release
  //
  // MatrixDyn<half> lhs_shared_mat = make_shared<M_TILE, K_TILE, half>();
  // MatrixDyn<half> rhs_shared_mat = make_shared<K_TILE, N_TILE, half>();
  // MatrixDyn<float> out_shared_mat = make_shared<M_TILE, N_TILE>(); // 16*16*4
  // = 1024 Bytes
  //

  // MatrixDyn<float> partial_sum
  //   = make_local<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(N_TILE, X_THREAD)>();

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, CEIL_DIV(32, K_TILE)); m++) {
#pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, K_TILE); kin++) {
        lhs_shared_mat
            .tile(CoordDyn(m, kin), CoordDyn(CEIL_DIV(32, K_TILE), K_TILE))
            .dist_to_thread() =
            lhs_mat.tile(CoordDyn(blockIdx.x, ko), lhs_sm_tile_shape)
                .tile(CoordDyn(m, kin), CoordDyn(CEIL_DIV(32, K_TILE), K_TILE))
                .dist_to_thread();
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, CEIL_DIV(32, N_TILE)); kin++) {
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, N_TILE); n++) {
        rhs_shared_mat
            .tile(CoordDyn(kin, n), CoordDyn(CEIL_DIV(32, N_TILE), N_TILE))
            .dist_to_thread() =
            rhs_mat.tile(CoordDyn(ko, blockIdx.y), rhs_sm_tile_shape)
                .tile(CoordDyn(kin, n), CoordDyn(CEIL_DIV(32, N_TILE), N_TILE))
                .dist_to_thread();
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    // cuda_m16n8k16_f16f32(out_shared_mat.data, lhs_shared_mat.data,
    //                      rhs_shared_mat.data, out_shared_mat.data);
    mma_m16n8k8_tf32f32(out_shared_mat.data, lhs_shared_mat.data,
                        rhs_shared_mat.data, out_shared_mat.data);
    // identity<16, 8>(lhs_shared_mat, out_shared_mat.tile(CoordDyn(0, ko),
    // CoordDyn(16, 8))); identity<8, 16>(rhs_shared_mat,
    // out_shared_mat.tile(CoordDyn(ko, 0), CoordDyn(8, 16)));
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, CEIL_DIV(32, N_TILE)); m++) {
    out_mat.tile(CoordDyn(blockIdx.x, blockIdx.y), out_sm_tile_shape)
        .tile(CoordDyn(m, 0), CoordDyn(CEIL_DIV(32, N_TILE), N_TILE))
        .dist_to_thread() =
        out_shared_mat
            .tile(CoordDyn(m, 0), CoordDyn(CEIL_DIV(32, N_TILE), N_TILE))
            .dist_to_thread();
  }
}

void matmul_tensor_cores_mma_m16n8k8_tf32f32(int M, int N, int K, float alpha,
                                             float *A, float *B, float beta,
                                             float *C) {
  const int M_TILE = 16;
  const int K_TILE = 8;
  const int N_TILE = 8;
  const int M_REG = 16;
  const int K_REG = 8;
  const int N_REG = 8;
  const int X_THREAD = 32;
  const int Y_THREAD = 1;
  assert(M_REG * K_REG > X_THREAD * Y_THREAD);
  assert(N_REG * K_REG > X_THREAD * Y_THREAD);

  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
      _matmul_tensor_cores_mma_m16n8k8_tf32f32<
          M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, X_THREAD, Y_THREAD>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _matmul_tensor_cores_mma_m16n8k8_tf32f32<M_TILE, N_TILE, K_TILE, M_REG, N_REG,
                                           K_REG, X_THREAD, Y_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  return;
}

template <const int M_TILE, const int N_TILE, const int K_TILE, const int M_REG,
          const int N_REG, const int K_REG, const int X_THREAD,
          const int Y_THREAD>
__global__ void
_matmul_tensor_cores_mma_m16n8k4_tf32f32(int M, int N, int K, float alpha,
                                         float *lhs, float *rhs, float beta,
                                         float *out) {
  auto lhs_shape = CoordDyn(M, K);
  auto rhs_shape = CoordDyn(K, N);
  auto out_shape = CoordDyn(M, N);

  auto lhs_sm_tile_shape = CoordDyn(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = CoordDyn(K_TILE, N_TILE);
  auto out_sm_tile_shape = CoordDyn(M_TILE, N_TILE);

  // make sure inner block looks like this, to ensure coalescing

  MatrixDyn<float> lhs_mat = MatrixDyn<float>(lhs, lhs_shape);
  MatrixDyn<float> rhs_mat = MatrixDyn<float>(rhs, rhs_shape);
  MatrixDyn<float> out_mat = MatrixDyn<float>(out, out_shape);

  // __shared__ float lhs_shared[M_TILE * K_TILE];
  // __shared__ float rhs_shared[K_TILE * N_TILE];
  // __shared__ float out_shared[M_TILE * N_TILE];

  // MatrixDyn<float> lhs_shared_mat = MatrixDyn<float>(lhs_shared,
  // lhs_sm_tile_shape); MatrixDyn<float> rhs_shared_mat =
  // MatrixDyn<float>(rhs_shared, rhs_sm_tile_shape);
  // MAKE_SHARED(rhs_shared_mat, K_TILE, N_TILE, half); MatrixDyn<float>
  // out_shared_mat = MatrixDyn<float>(out_shared, out_sm_tile_shape);
  MAKE_SHARED(lhs_shared_mat, M_TILE, K_TILE, float);
  MAKE_SHARED(rhs_shared_mat, K_TILE, N_TILE, float);
  MAKE_SHARED(out_shared_mat, M_TILE, N_TILE, float);

  out_shared_mat.fill(0.);

  // TODO: wrong, since this will release shared after make_shared call
  // impl a Macro based make_shared and make_local, ensure it won't release
  //
  // MatrixDyn<half> lhs_shared_mat = make_shared<M_TILE, K_TILE, half>();
  // MatrixDyn<half> rhs_shared_mat = make_shared<K_TILE, N_TILE, half>();
  // MatrixDyn<float> out_shared_mat = make_shared<M_TILE, N_TILE>(); // 16*16*4
  // = 1024 Bytes
  //

  // MatrixDyn<float> partial_sum
  //   = make_local<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(N_TILE, X_THREAD)>();

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, CEIL_DIV(32, K_TILE)); m++) {
#pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, K_TILE); kin++) {
        lhs_shared_mat
            .tile(CoordDyn(m, kin), CoordDyn(CEIL_DIV(32, K_TILE), K_TILE))
            .dist_to_thread() =
            lhs_mat.tile(CoordDyn(blockIdx.x, ko), lhs_sm_tile_shape)
                .tile(CoordDyn(m, kin), CoordDyn(CEIL_DIV(32, K_TILE), K_TILE))
                .dist_to_thread();
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, CEIL_DIV(32, N_TILE)); kin++) {
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, N_TILE); n++) {
        rhs_shared_mat
            .tile(CoordDyn(kin, n), CoordDyn(CEIL_DIV(32, N_TILE), N_TILE))
            .dist_to_thread() =
            rhs_mat.tile(CoordDyn(ko, blockIdx.y), rhs_sm_tile_shape)
                .tile(CoordDyn(kin, n), CoordDyn(CEIL_DIV(32, N_TILE), N_TILE))
                .dist_to_thread();
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    // cuda_m16n8k16_f16f32(out_shared_mat.data, lhs_shared_mat.data,
    //                      rhs_shared_mat.data, out_shared_mat.data);
    mma_m16n8k4_tf32f32(out_shared_mat.data, lhs_shared_mat.data,
                        rhs_shared_mat.data, out_shared_mat.data);
    // identity<16, 8>(lhs_shared_mat, out_shared_mat.tile(CoordDyn(0, ko),
    // CoordDyn(16, 8))); identity<8, 16>(rhs_shared_mat,
    // out_shared_mat.tile(CoordDyn(ko, 0), CoordDyn(8, 16)));
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, CEIL_DIV(32, N_TILE)); m++) {
#pragma unroll
    out_mat.tile(CoordDyn(blockIdx.x, blockIdx.y), out_sm_tile_shape)
        .tile(CoordDyn(m, 0), CoordDyn(CEIL_DIV(32, N_TILE), N_TILE))
        .dist_to_thread() =
        out_shared_mat
            .tile(CoordDyn(m, 0), CoordDyn(CEIL_DIV(32, N_TILE), N_TILE))
            .dist_to_thread();
  }
}

void matmul_tensor_cores_mma_m16n8k4_tf32f32(int M, int N, int K, float alpha,
                                             float *A, float *B, float beta,
                                             float *C) {
  const int M_TILE = 16;
  const int K_TILE = 4;
  const int N_TILE = 8;
  const int M_REG = 16;
  const int K_REG = 8;
  const int N_REG = 8;
  const int X_THREAD = 32;
  const int Y_THREAD = 1;
  assert(M_REG * K_REG > X_THREAD * Y_THREAD);
  assert(N_REG * K_REG > X_THREAD * Y_THREAD);

  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
      _matmul_tensor_cores_mma_m16n8k4_tf32f32<
          M_TILE, N_TILE, K_TILE, M_REG, N_REG, K_REG, X_THREAD, Y_THREAD>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _matmul_tensor_cores_mma_m16n8k4_tf32f32<M_TILE, N_TILE, K_TILE, M_REG, N_REG,
                                           K_REG, X_THREAD, Y_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  return;
}

} // namespace catz::recipes

#endif // CATZILLA_RECIPES_MATMUL_TENSOR_CORES_H_
