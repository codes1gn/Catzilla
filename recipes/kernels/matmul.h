#ifndef CATZILLA_RECIPES_KERNELS_MATMUL_V1_H_
#define CATZILLA_RECIPES_KERNELS_MATMUL_V1_H_

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils/index_utils.h"
#include "utils/macros.h"
#include "utils/micro_kernels.h"


/*
 * TODO:
 * reduce warp divergence
 * blocking tiling
 * pipeline
 * multi-buffer
 * xor_swizzle
 * unroll
 * tensor.vmm.contracts
 * vector loads
 */

/////////////////////////////////////////////////////////////////////////////////
/// version 1:
/// - basic gmem to smem coalescing
/// - use smem for lhs, rhs, out
/////////////////////////////////////////////////////////////////////////////////
template <const int M_TILE, const int N_TILE, const int K_TILE>
__global__ void _catzilla_matmul_v1(int M, int N, int K, float alpha,
                                    float *lhs, float *rhs, float beta,
                                    float *out) {
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  // smem stationary
  __shared__ float lhs_shared[M_TILE * K_TILE];
  __shared__ float rhs_shared[K_TILE * N_TILE];
  __shared__ float out_shared[M_TILE * N_TILE];

  // coalescing
  out_shared[distribute_(tid_x, tid_y, 1, N_TILE)] = 0.0;

  float partial_sum = 0.;
  for (int k = 0; k < CEIL_DIV(K, K_TILE); k++) { // range(0, 128, 1)
    // use distribute_ for in-tile thread calculation
    // use tile_ for out-tile thread calculation
    lhs_shared[distribute_(tid_x, tid_y, 1, K_TILE)] =
        lhs[tiling_(bid_y, k, M_TILE, K_TILE, K, 1) +
            distribute_(tid_x, tid_y, 1, K)];
    rhs_shared[distribute_(tid_x, tid_y, 1, N_TILE)] =
        rhs[tiling_(k, bid_x, K_TILE, N_TILE, N, 1) +
            distribute_(tid_x, tid_y, 1, N)];
    __syncthreads();

    // contract at 32x32x32 micro-kernel
    matmul_kernel_32x32x32(lhs_shared, rhs_shared, partial_sum);
    __syncthreads();
  }
  out_shared[distribute_(tid_x, tid_y, 1, N_TILE)] = alpha * partial_sum;
  out[tiling_(bid_y, bid_x, M_TILE, N_TILE, N, 1) +
      distribute_(tid_x, tid_y, 1, N)] =
      out_shared[distribute_(tid_x, tid_y, 1, N_TILE)];
  // out[tiling_(bid_y, bid_x, M_TILE, N_TILE, N, 1) + distribute_(tid_x, tid_y,
  // 1, N)] = alpha * partial_sum;
}

void catzilla_matmul_v1(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  cudaFuncSetAttribute(_catzilla_matmul_v1<32, 32, 32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v1<32, 32, 32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

/////////////////////////////////////////////////////////////////////////////////
/// version 2:
/// - basic gmem to smem coalescing
/// - use smem for lhs, rhs, out
/// - thread-level blocking, compute a tile in reg-level C, not a scalar
/////////////////////////////////////////////////////////////////////////////////
template <const int M_TILE_SM, const int N_TILE_SM, const int K_TILE_SM,
          const int M_TILE_REG, const int N_TILE_REG>
__global__ void _catzilla_matmul_v2(int M, int N, int K, float alpha,
                                    float *lhs, float *rhs, float beta,
                                    float *out) {
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  // smem stationary
  // M=64, N=64, K=32, Mreg=2, Nreg=2
  __shared__ float lhs_shared[M_TILE_SM * K_TILE_SM];
  __shared__ float rhs_shared[K_TILE_SM * N_TILE_SM];
  __shared__ float out_shared[M_TILE_SM * N_TILE_SM];
  // __shared__ float out_shared[1];

  // can remove
  for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      out_shared[distribute_(tid_x * N_TILE_REG + nreg,
                             tid_y * M_TILE_REG + mreg, 1, N_TILE_SM)] = 0.;
    }
  }

  float partial_sum[M_TILE_REG * N_TILE_REG] = {0.};

  // QZ: per thread m and m+1
  for (int k = 0; k < CEIL_DIV(K, K_TILE_SM); k++) { // range(0, 128, 1)
    // use distribute_ for in-tile thread calculation
    // use tile_ for out-tile thread calculation
    for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
      lhs_shared[distribute_(tid_x, tid_y * M_TILE_REG + mreg, 1, K_TILE_SM)] =
          lhs[tiling_(bid_y, k, M_TILE_SM, K_TILE_SM, K, 1) +
              distribute_(tid_x, tid_y * M_TILE_REG + mreg, 1, K)];
    }
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      rhs_shared[distribute_(tid_x * N_TILE_REG + nreg, tid_y, 1, N_TILE_SM)] =
          rhs[tiling_(k, bid_x, K_TILE_SM, N_TILE_SM, N, 1) +
              distribute_(tid_x * N_TILE_REG + nreg, tid_y, 1, N)];
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_64x64x16_perthread_4x4(lhs_shared, rhs_shared, partial_sum);
    // more syncs
    // local => gmem 4930 GF
    // local => smem, smem => gmem 5255 GF
    // __syncthreads();
  }
  // local => smem, smem => gmem 5485 GF
  // local => gmem error in results
  __syncthreads();

  for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      // out_shared[distribute_(tid_x * N_TILE_REG, tid_y * M_TILE_REG, 1,
      // N_TILE_SM) + distribute_(mreg, nreg, N_TILE_SM, 1)] = alpha *
      // partial_sum[mreg * 4 + nreg]; out[tiling_(bid_y, bid_x, M_TILE_SM,
      // N_TILE_SM, N, 1) + distribute_(tid_x*N_TILE_REG, tid_y*M_TILE_REG, 1,
      // N) + distribute_(mreg, nreg, N, 1)] =
      // out_shared[distribute_(tid_x*N_TILE_REG, tid_y*M_TILE_REG, 1,
      // N_TILE_SM) + distribute_(mreg, nreg, N_TILE_SM, 1)];
      out[tiling_(bid_y, bid_x, M_TILE_SM, N_TILE_SM, N, 1) +
          distribute_(tid_x * N_TILE_REG, tid_y * M_TILE_REG, 1, N) +
          distribute_(mreg, nreg, N, 1)] =
          alpha * partial_sum[mreg * N_TILE_REG + nreg];
    }
  }
}

void catzilla_matmul_v2(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  const int M_TILE = 64;
  const int N_TILE = 64;
  const int K_TILE = 16;
  const int mreg = 4;
  const int nreg = 4;
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(CEIL_DIV(M_TILE, mreg), CEIL_DIV(N_TILE, nreg));
  cudaFuncSetAttribute(_catzilla_matmul_v2<M_TILE, N_TILE, K_TILE, mreg, nreg>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v2<M_TILE, N_TILE, K_TILE, mreg, nreg>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

/////////////////////////////////////////////////////////////////////////////////
/// version 3:
/// - basic gmem to smem coalescing
/// - use smem for lhs, rhs, out
/// - thread-level blocking, compute a tile in reg-level C, not a scalar
/// - refactor with tiling and transfer helpers
/// - try different tiling config with hand
/////////////////////////////////////////////////////////////////////////////////
template <const int M_TILE_SM, const int N_TILE_SM, const int K_TILE_SM,
          const int M_TILE_REG, const int N_TILE_REG>
__global__ void _catzilla_matmul_v3(int M, int N, int K, float alpha,
                                    float *lhs, float *rhs, float beta,
                                    float *out) {
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  // smem stationary
  // M=128, N=128, K=32, Mreg=4, Nreg=4
  __shared__ float lhs_shared[M_TILE_SM * K_TILE_SM];
  __shared__ float rhs_shared[K_TILE_SM * N_TILE_SM];
  __shared__ float out_shared[M_TILE_SM * N_TILE_SM];
  // __shared__ float out_shared[1];

  // init out
  // out_shared[distribute_(tid_x, tid_y, 1, N_TILE_SM)] = 0.0;

  float partial_sum[M_TILE_REG * N_TILE_REG] = {0.};

  // distribute bid_y for lhs

  // QZ: per thread m and m+1
  for (int k = 0; k < CEIL_DIV(K, K_TILE_SM); k++) { // range(0, 128, 1)
    // use distribute_ for in-tile thread calculation
    // use tile_ for out-tile thread calculation
    float *lhs_cursor = lhs + tiling_(bid_y, k, M_TILE_SM, K_TILE_SM, K, 1);
    float *rhs_cursor = rhs + tiling_(k, bid_x, K_TILE_SM, N_TILE_SM, N, 1);
    for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
      lhs_shared[distribute_(tid_x, tid_y * M_TILE_REG + mreg, 1, K_TILE_SM)] =
          lhs_cursor[distribute_(tid_x, tid_y * M_TILE_REG + mreg, 1, K)];
    }
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      rhs_shared[distribute_(tid_x * N_TILE_REG + nreg, tid_y, 1, N_TILE_SM)] =
          rhs_cursor[distribute_(tid_x * N_TILE_REG + nreg, tid_y, 1, N)];
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_64x64x16_perthread_4x4(lhs_shared, rhs_shared, partial_sum);
  }
  __syncthreads();

  float *out_cursor = out + tiling_(bid_y, bid_x, M_TILE_SM, N_TILE_SM, N, 1);
  for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      // out_shared[distribute_(tid_x * N_TILE_REG + nreg, tid_y * M_TILE_REG +
      // mreg, 1, N_TILE_SM)] = alpha * partial_sum[mreg * N_TILE_REG + nreg];
      // out_cursor[distribute_(tid_x * N_TILE_REG + nreg, tid_y * M_TILE_REG +
      // mreg, 1, N)] = out_shared[distribute_(tid_x*N_TILE_REG + nreg,
      // tid_y*M_TILE_REG + mreg, 1, N_TILE_SM)];
      out_cursor[distribute_(tid_x * N_TILE_REG + nreg,
                             tid_y * M_TILE_REG + mreg, 1, N)] =
          alpha * partial_sum[mreg * N_TILE_REG + nreg];
    }
  }
}

void catzilla_matmul_v3(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  const int M_TILE = 64;
  const int N_TILE = 64;
  const int K_TILE = 16;
  const int mreg = 4;
  const int nreg = 4;
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(CEIL_DIV(M_TILE, mreg), CEIL_DIV(N_TILE, nreg));
  cudaFuncSetAttribute(_catzilla_matmul_v3<M_TILE, N_TILE, K_TILE, mreg, nreg>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v3<M_TILE, N_TILE, K_TILE, mreg, nreg>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// simplify with stream-style
template <const int M_TILE_SM, const int N_TILE_SM, const int K_TILE_SM,
          const int M_TILE_REG, const int N_TILE_REG>
__global__ void _catzilla_matmul_v4(int M, int N, int K, float alpha,
                                    float *lhs, float *rhs, float beta,
                                    float *out) {
  auto lhs_shape = make_coord(M, K);
  auto rhs_shape = make_coord(K, N);
  auto out_shape = make_coord(M, N);

  auto lhs_sm_tile_shape = Coord(M_TILE_SM, K_TILE_SM);
  auto rhs_sm_tile_shape = Coord(K_TILE_SM, N_TILE_SM);
  auto out_sm_tile_shape = Coord(M_TILE_SM, N_TILE_SM);

  auto lhs_reg_tile_shape = Coord(M_TILE_REG, K_TILE_SM);
  auto rhs_reg_tile_shape = Coord(K_TILE_SM, N_TILE_REG);
  auto out_reg_tile_shape = Coord(M_TILE_REG, N_TILE_REG);

  Matrix lhs_mat = Matrix(lhs, lhs_shape);
  Matrix rhs_mat = Matrix(rhs, rhs_shape);
  Matrix out_mat = Matrix(out, out_shape);

  Matrix lhs_shared_mat = make_shared<M_TILE_SM, K_TILE_SM>();
  Matrix rhs_shared_mat = make_shared<K_TILE_SM, N_TILE_SM>();
  // Matrix out_shared_mat = make_shared<M_TILE_SM, N_TILE_SM>();

  Matrix partial_sum = make_local<M_TILE_REG, N_TILE_REG>();

  // iter vars we got, block.x, block.y, thread.x, thread.y, k, mreg.
  // because we distribute out into blocks, we select one of blk.x or blk.y
  // for row: blk.y, thd.y, mreg
  // for col: k, thd.x
  for (int k = 0; k < CEIL_DIV(K, K_TILE_SM); k++) {
#pragma unroll 4
    for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
      lhs_shared_mat // (M_TILE_SM, K_TILE_SM)
          .tile_ex(Coord(threadIdx.y, 0),
                   lhs_reg_tile_shape) // (M_TILE_SM, K_TILE_SM) => (M_TILE_REG,
                                       // K_TILE_SM)
          .dist_ex(Coord(
              mreg,
              threadIdx.x)) // (M_TILE_REG, K_TILE_SM) distribute to threads
          = lhs_mat         // (M, K)
                .tile_ex(Coord(blockIdx.y, k),
                         lhs_sm_tile_shape) // (M, K) => (M_TILE_SM, K_TILE_SM)
                .tile_ex(Coord(threadIdx.y, 0),
                         lhs_reg_tile_shape)        // (M_TILE_SM, K_TILE_SM) =>
                                                    // (M_TILE_REG, K_TILE_SM)
                .dist_ex(Coord(mreg, threadIdx.x)); // (M_TILE_REG, K_TILE_SM)
                                                    // distribute to threads
    }
#pragma unroll 4
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      rhs_shared_mat
          .tile_ex(Coord(0, threadIdx.x),
                   rhs_reg_tile_shape) // (M_TILE_SM, K_TILE_SM) => (M_TILE_REG,
                                       // K_TILE_SM)
          .dist_ex(Coord(threadIdx.y, nreg)) =
          rhs_mat.tile_ex(Coord(k, blockIdx.x), rhs_sm_tile_shape)
              .tile_ex(Coord(0, threadIdx.x),
                       rhs_reg_tile_shape) // (M_TILE_SM, K_TILE_SM) =>
                                           // (M_TILE_REG, K_TILE_SM)
              .dist_ex(Coord(threadIdx.y, nreg));
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_64x64x16_perthread_4x4(lhs_shared_mat.data,
                                         rhs_shared_mat.data, partial_sum.data);
  }
  __syncthreads();

  for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
#pragma unroll 4
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      // out_shared_mat
      //   .tile_ex(Coord(threadIdx.y, threadIdx.x), out_reg_tile_shape)
      //   .dist_ex(Coord(mreg, nreg))
      //   = partial_sum
      //   .dist_ex(Coord(mreg, nreg));
      //
      // out_mat
      //   .tile_ex(Coord(blockIdx.y, blockIdx.x), out_sm_tile_shape)
      //   .tile_ex(Coord(threadIdx.y, threadIdx.x), out_reg_tile_shape)
      //   .dist_ex(Coord(mreg, nreg))
      //   = out_shared_mat
      //   .tile_ex(Coord(threadIdx.y, threadIdx.x), out_reg_tile_shape)
      //   .dist_ex(Coord(mreg, nreg));
      out_mat.tile_ex(Coord(blockIdx.y, blockIdx.x), out_sm_tile_shape)
          .tile_ex(Coord(threadIdx.y, threadIdx.x), out_reg_tile_shape)
          .dist_ex(Coord(mreg, nreg)) = partial_sum.dist_ex(Coord(mreg, nreg));
    }
  }
}

void catzilla_matmul_v4(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  const int M_TILE = 64;
  const int N_TILE = 64;
  const int K_TILE = 16;
  const int mreg = 4;
  const int nreg = 4;
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(CEIL_DIV(M_TILE, mreg), CEIL_DIV(N_TILE, nreg));
  cudaFuncSetAttribute(_catzilla_matmul_v4<M_TILE, N_TILE, K_TILE, mreg, nreg>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v4<M_TILE, N_TILE, K_TILE, mreg, nreg>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// hand tuned for HPs
// use templated kernel
template <const int M_TILE, const int N_TILE, const int K_TILE,
          const int X_THREAD, const int Y_THREAD>
__global__ void _catzilla_matmul_v5(int M, int N, int K, float alpha,
                                    float *lhs, float *rhs, float beta,
                                    float *out) {
  auto lhs_shape = make_coord(M, K);
  auto rhs_shape = make_coord(K, N);
  auto out_shape = make_coord(M, N);

  auto lhs_sm_tile_shape = Coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = Coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = Coord(M_TILE, N_TILE);

  // make sure inner block looks like this, to ensure coalescing
  auto per_block_data_shape = Coord(Y_THREAD, X_THREAD);

  Matrix lhs_mat = Matrix(lhs, lhs_shape);
  Matrix rhs_mat = Matrix(rhs, rhs_shape);
  Matrix out_mat = Matrix(out, out_shape);

  Matrix lhs_shared_mat = make_shared<M_TILE, K_TILE>();
  Matrix rhs_shared_mat = make_shared<K_TILE, N_TILE>();
  // Matrix out_shared_mat = make_shared<M_TILE_SM, N_TILE_SM>();
  //

  Matrix partial_sum =
      make_local<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(N_TILE, X_THREAD)>();

  // int threadId = threadIdx.y * X_THREAD + threadIdx.x;

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
      #pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, X_THREAD); kin++) {
        lhs_shared_mat
          .tile_ex(Coord(m, kin), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x))
          // .dist_ex(Coord(threadId / K_TILE, threadId % K_TILE))
          = lhs_mat
          .tile_ex(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
          .tile_ex(Coord(m, kin), per_block_data_shape)
          // .dist_ex(Coord(threadId / K_TILE, threadId % K_TILE));
          .dist_ex(Coord(threadIdx.y, threadIdx.x));
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, Y_THREAD); kin++) {
      #pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
        rhs_shared_mat
          .tile_ex(Coord(kin, n), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x))
          = rhs_mat
          .tile_ex(Coord(ko, blockIdx.x), rhs_sm_tile_shape)
          .tile_ex(Coord(kin, n), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x));
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_coalesced<M_TILE, N_TILE, K_TILE, Y_THREAD, X_THREAD>(
        lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
    #pragma unroll
    for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
      out_mat
        .tile_ex(Coord(blockIdx.y, blockIdx.x), out_sm_tile_shape)
        .tile_ex(Coord(m, n), per_block_data_shape)
        .dist_ex(Coord(threadIdx.y, threadIdx.x)) 
        = partial_sum
        .dist_ex(Coord(m, n));
    }
  }
}

void catzilla_matmul_v5(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // // sec 1, determine, gridDim
  // const int M_TILE = 128;
  // const int N_TILE = 128;
  // // sec 2, determine blockDim
  // const int X_THREAD = 16;
  // const int Y_THREAD = 16;
  // // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // const int K_TILE = 32;
  const int M_TILE = 128;
  const int N_TILE = 128;
  // sec 2, determine blockDim
  const int X_THREAD = 16;
  const int Y_THREAD = 16;
  // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  const int K_TILE = 32;
  // K_TILE > Y_THREAD
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
      _catzilla_matmul_v5<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v5<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// hand tuned for HPs
// use templated kernel
// PAD-SWIZZLE
template <const int M_TILE, const int N_TILE, const int K_TILE,
          const int X_THREAD, const int Y_THREAD>
__global__ void _catzilla_matmul_v6(int M, int N, int K, float alpha,
                                    float *lhs, float *rhs, float beta,
                                    float *out) {
  auto lhs_shape = make_coord(M, K);
  auto rhs_shape = make_coord(K, N);
  auto out_shape = make_coord(M, N);

  auto lhs_sm_tile_shape = Coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = Coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = Coord(M_TILE, N_TILE);

  // make sure inner block looks like this, to ensure coalescing
  auto per_block_data_shape = Coord(Y_THREAD, X_THREAD);

  Matrix lhs_mat = Matrix(lhs, lhs_shape);
  Matrix rhs_mat = Matrix(rhs, rhs_shape);
  Matrix out_mat = Matrix(out, out_shape);

  // __shared__ float lhs_shared[M_TILE* (K_TILE+1)];

  Matrix lhs_shared_mat = make_shared<M_TILE, K_TILE+1>();
  Matrix rhs_shared_mat = make_shared<K_TILE, N_TILE+1>();
  // Matrix out_shared_mat = make_shared<M_TILE_SM, N_TILE_SM>();
  //

  Matrix partial_sum =
      make_local<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(N_TILE, X_THREAD)>();

  // int threadId = threadIdx.y * X_THREAD + threadIdx.x;

  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
      #pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, X_THREAD); kin++) {
        lhs_shared_mat
          .tile_ex(Coord(m, kin), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x))
          // .dist_ex(Coord(threadId / K_TILE, threadId % K_TILE))
          = lhs_mat
          .tile_ex(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
          .tile_ex(Coord(m, kin), per_block_data_shape)
          // .dist_ex(Coord(threadId / K_TILE, threadId % K_TILE));
          .dist_ex(Coord(threadIdx.y, threadIdx.x));
        // lhs_shared[m * Y_THREAD * (K_TILE+1) + kin * X_THREAD + threadIdx.y * (K_TILE+1) + threadIdx.x]
        //   = lhs_mat
        //   .tile_ex(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
        //   .tile_ex(Coord(m, kin), per_block_data_shape)
        //   // .dist_ex(Coord(threadId / K_TILE, threadId % K_TILE));
        //   .dist_ex(Coord(threadIdx.y, threadIdx.x)).data[0];
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, Y_THREAD); kin++) {
      #pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
        rhs_shared_mat
          .tile_ex(Coord(kin, n), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x))
          = rhs_mat
          .tile_ex(Coord(ko, blockIdx.x), rhs_sm_tile_shape)
          .tile_ex(Coord(kin, n), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x));
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_pad_swizzled<M_TILE, N_TILE, K_TILE, Y_THREAD, X_THREAD>(
        lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
    #pragma unroll
    for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
      out_mat
        .tile_ex(Coord(blockIdx.y, blockIdx.x), out_sm_tile_shape)
        .tile_ex(Coord(m, n), per_block_data_shape)
        .dist_ex(Coord(threadIdx.y, threadIdx.x)) 
        = partial_sum
        .dist_ex(Coord(m, n));
    }
  }
}

void catzilla_matmul_v6(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // for bench
  // sec 1, determine, gridDim
  const int M_TILE = 128;
  const int N_TILE = 128;
  // sec 2, determine blockDim
  const int X_THREAD = 16;
  const int Y_THREAD = 16;
  // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  const int K_TILE = 32;

  // for debug only
  // const int M_TILE = 32;
  // const int N_TILE = 32;
  // // sec 2, determine blockDim
  // const int X_THREAD = 4;
  // const int Y_THREAD = 4;
  // // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // const int K_TILE = 8;
  // K_TILE > Y_THREAD
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
      _catzilla_matmul_v6<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v6<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// flatten copy
// TODO: xor-swizzle
template <const int M_TILE, const int N_TILE, const int K_TILE,
          const int X_THREAD, const int Y_THREAD>
__global__ void _catzilla_matmul_v7(int M, int N, int K, float alpha,
                                    float *lhs, float *rhs, float beta,
                                    float *out) {
  auto lhs_shape = make_coord(M, K);
  auto rhs_shape = make_coord(K, N);
  auto out_shape = make_coord(M, N);

  auto lhs_sm_tile_shape = Coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = Coord(K_TILE, N_TILE);
  auto out_sm_tile_shape = Coord(M_TILE, N_TILE);

  // make sure inner block looks like this, to ensure coalescing
  auto per_block_data_shape = Coord(Y_THREAD, X_THREAD);

  Matrix lhs_mat = Matrix(lhs, lhs_shape);
  Matrix rhs_mat = Matrix(rhs, rhs_shape);
  Matrix out_mat = Matrix(out, out_shape);

  // __shared__ float lhs_shared[M_TILE* (K_TILE+1)];

  Matrix lhs_shared_mat = make_shared<M_TILE, K_TILE>();
  Matrix rhs_shared_mat = make_shared<K_TILE, N_TILE>();
  // Matrix out_shared_mat = make_shared<M_TILE_SM, N_TILE_SM>();
  //

  Matrix partial_sum =
      make_local<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(N_TILE, X_THREAD)>();


  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
      #pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, X_THREAD); kin++) {
        // TODO: span_as(y * 4, x / 4)
        lhs_shared_mat
          .tile_ex(Coord(m, kin), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x))
          = lhs_mat
          .tile_ex(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
          .tile_ex(Coord(m, kin), per_block_data_shape)
          // .dist_ex(Coord(threadId / K_TILE, threadId % K_TILE));
          .dist_ex(Coord(threadIdx.y, threadIdx.x));
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, Y_THREAD); kin++) {
      #pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
        rhs_shared_mat
          .tile_ex(Coord(kin, n), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x))
          = rhs_mat
          .tile_ex(Coord(ko, blockIdx.x), rhs_sm_tile_shape)
          .tile_ex(Coord(kin, n), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x));
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_coalesced<M_TILE, N_TILE, K_TILE, Y_THREAD, X_THREAD>(
        lhs_shared_mat.data, rhs_shared_mat.data, partial_sum.data);
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
    #pragma unroll
    for (int n = 0; n < CEIL_DIV(N_TILE, X_THREAD); n++) {
      out_mat
        .tile_ex(Coord(blockIdx.y, blockIdx.x), out_sm_tile_shape)
        .tile_ex(Coord(m, n), per_block_data_shape)
        .dist_ex(Coord(threadIdx.y, threadIdx.x)) 
        = partial_sum
        .dist_ex(Coord(m, n));
    }
  }
}

// TODO: what if K_TILE < X_THREAD OR Y_THREAD, a thread-var cannot bind to K_TILE dim solely
// XOR-SWIZZLE
void catzilla_matmul_v7(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // for bench
  // sec 1, determine, gridDim
  const int M_TILE = 128;
  const int N_TILE = 128;
  // sec 2, determine blockDim
  const int X_THREAD = 16;
  const int Y_THREAD = 16;
  // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  const int K_TILE = 32;

  // for debug only
  // const int M_TILE = 32;
  // const int N_TILE = 32;
  // // sec 2, determine blockDim
  // const int X_THREAD = 32;
  // const int Y_THREAD = 32;
  // // sec 3, K_TILE >= N_THREAD, AND M_THREAD
  // const int K_TILE = 32;

  // K_TILE > Y_THREAD
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
      _catzilla_matmul_v7<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v7<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// + vload vstore 
// TODO: let lhs also vectorise, currently only apply on rhs and out
template <const int M_TILE, const int N_TILE, const int K_TILE,
          const int X_THREAD, const int Y_THREAD>
__global__ void _catzilla_matmul_v8(int M, int N, int K, float alpha,
                                    float *lhs, float *rhs, float beta,
                                    float *out) {
  auto lhs_shape = make_coord(M, K);
  auto rhs_shape = make_coord(K, N);
  auto rhs_shape_v = make_coord(K, CEIL_DIV(N, 4));
  auto out_shape = make_coord(M, N);
  auto out_shape_v = make_coord(M, CEIL_DIV(N, 4)); // in float4 type, overall shape is M, N/4

  auto lhs_sm_tile_shape = Coord(M_TILE, K_TILE);
  auto rhs_sm_tile_shape = Coord(K_TILE, N_TILE);
  auto rhs_sm_tile_shape_v = Coord(K_TILE, CEIL_DIV(N_TILE, 4));
  auto out_sm_tile_shape = Coord(M_TILE, N_TILE); // in float4 type, sm shape is M_tile, N_tile/4 as well
  auto out_sm_tile_shape_v = Coord(M_TILE, CEIL_DIV(N_TILE, 4)); // in float4 type, sm shape is M_tile, N_tile/4 as well

  auto per_block_data_shape = Coord(Y_THREAD, X_THREAD); // same threading

  Matrix lhs_mat = Matrix(lhs, lhs_shape);
  MatrixV rhs_mat = MatrixV((float4*)rhs, rhs_shape_v);
  MatrixV out_mat = MatrixV((float4*)out, out_shape_v); // use MatrixV


  Matrix lhs_shared_mat = make_shared<M_TILE, K_TILE>();
  MatrixV rhs_shared_mat = make_shared_v<K_TILE, CEIL_DIV(N_TILE, 4)>();
  MatrixV partial_sum =
      make_local_v<CEIL_DIV(M_TILE, Y_THREAD), CEIL_DIV(CEIL_DIV(N_TILE, 4), X_THREAD)>();


  for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
      #pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, X_THREAD); kin++) {
        // TODO: span_as(y * 4, x / 4)
        lhs_shared_mat
          .tile_ex(Coord(m, kin), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x))
          = lhs_mat
          .tile_ex(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
          .tile_ex(Coord(m, kin), per_block_data_shape)
          // .dist_ex(Coord(threadId / K_TILE, threadId % K_TILE));
          .dist_ex(Coord(threadIdx.y, threadIdx.x));
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, Y_THREAD); kin++) {
      #pragma unroll
      for (int n = 0; n < CEIL_DIV(CEIL_DIV(N_TILE, 4), X_THREAD); n++) {
        rhs_shared_mat
          .tile_ex(Coord(kin, n), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x))
          = rhs_mat
          .tile_ex(Coord(ko, blockIdx.x), rhs_sm_tile_shape_v)
          .tile_ex(Coord(kin, n), per_block_data_shape)
          .dist_ex(Coord(threadIdx.y, threadIdx.x));
      }
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_coalesced<M_TILE, N_TILE, K_TILE, Y_THREAD, X_THREAD>(
        (float*)lhs_shared_mat.data, (float*)rhs_shared_mat.data, (float*)partial_sum.data);
  }
  __syncthreads();

  for (int m = 0; m < CEIL_DIV(M_TILE, Y_THREAD); m++) {
    #pragma unroll
    for (int n = 0; n < CEIL_DIV(CEIL_DIV(N_TILE, 4), X_THREAD); n++) {
      out_mat
        .tile_ex(Coord(blockIdx.y, blockIdx.x), out_sm_tile_shape_v)
        .tile_ex(Coord(m, n), per_block_data_shape)
        .dist_ex(Coord(threadIdx.y, threadIdx.x)) 
        = partial_sum.dist_ex(Coord(m, n));
    }
  }
}

void catzilla_matmul_v8(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  const int M_TILE = 128;
  const int N_TILE = 128;
  const int X_THREAD = 16;
  const int Y_THREAD = 16;
  const int K_TILE = 32;

  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(X_THREAD, Y_THREAD);
  cudaFuncSetAttribute(
      _catzilla_matmul_v8<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v8<M_TILE, N_TILE, K_TILE, X_THREAD, Y_THREAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

#endif // CATZILLA_RECIPES_KERNELS_MATMUL_V1_H_
