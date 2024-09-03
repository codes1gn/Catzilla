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

Matrix sizes:
MxK * KxN = MxN

*/
// TODO extract common utils
/*
 * gmem access coalescing
 * smem stationary
 *
 * TODO:
 * reduce warp divergence
 * blocking tiling
 * pipeline
 * multi-buffer
 * swizzle
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
                                       float *lhs, float *rhs,
                                       float beta, float *out) {
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
    lhs_shared[distribute_(tid_x, tid_y, 1, K_TILE)] = lhs[tiling_(bid_y, k, M_TILE, K_TILE, K, 1) + distribute_(tid_x, tid_y, 1, K)];
    rhs_shared[distribute_(tid_x, tid_y, 1, N_TILE)] = rhs[tiling_(k, bid_x, K_TILE, N_TILE, N, 1) + distribute_(tid_x, tid_y, 1, N)];
    __syncthreads();

    // contract at 32x32x32 micro-kernel
    matmul_kernel_32x32x32(lhs_shared, rhs_shared, partial_sum);
    __syncthreads();
  }
  out_shared[distribute_(tid_x, tid_y, 1, N_TILE)] = alpha * partial_sum;
  out[tiling_(bid_y, bid_x, M_TILE, N_TILE, N, 1) + distribute_(tid_x, tid_y, 1, N)] = out_shared[distribute_(tid_x, tid_y, 1, N_TILE)];
  // out[tiling_(bid_y, bid_x, M_TILE, N_TILE, N, 1) + distribute_(tid_x, tid_y, 1, N)] = alpha * partial_sum;
}


void catzilla_matmul_v1(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  cudaFuncSetAttribute(_catzilla_matmul_v1<32, 32, 32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v1<32, 32, 32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}


/////////////////////////////////////////////////////////////////////////////////
/// version 2:
/// - basic gmem to smem coalescing
/// - use smem for lhs, rhs, out
/// - thread-level blocking, compute a tile in reg-level C, not a scalar
/////////////////////////////////////////////////////////////////////////////////
template <const int M_TILE_SM, const int N_TILE_SM, const int K_TILE_SM, const int M_TILE_REG, const int N_TILE_REG>
__global__ void _catzilla_matmul_v2(int M, int N, int K, float alpha,
                                       float *lhs, float *rhs,
                                       float beta, float *out) {
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
      out_shared[distribute_(tid_x * N_TILE_REG + nreg, tid_y * M_TILE_REG + mreg, 1, N_TILE_SM)] = 0.;
    }
  }

  float partial_sum[M_TILE_REG * N_TILE_REG] = {0.};

  // QZ: per thread m and m+1
  for (int k = 0; k < CEIL_DIV(K, K_TILE_SM); k++) { // range(0, 128, 1)
    // use distribute_ for in-tile thread calculation
    // use tile_ for out-tile thread calculation
    for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
      lhs_shared[distribute_(tid_x, tid_y*M_TILE_REG + mreg, 1, K_TILE_SM)] = lhs[tiling_(bid_y, k, M_TILE_SM, K_TILE_SM, K, 1) + distribute_(tid_x, tid_y*M_TILE_REG + mreg, 1, K)];
    }
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      rhs_shared[distribute_(tid_x*N_TILE_REG + nreg, tid_y, 1, N_TILE_SM)] = rhs[tiling_(k, bid_x, K_TILE_SM, N_TILE_SM, N, 1) + distribute_(tid_x*N_TILE_REG + nreg, tid_y, 1, N)];
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_64x64x32_perthread_2x2(lhs_shared, rhs_shared, partial_sum);
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
      out_shared[distribute_(tid_x * N_TILE_REG, tid_y * M_TILE_REG, 1, N_TILE_SM) + distribute_(mreg, nreg, N_TILE_SM, 1)] = alpha * partial_sum[mreg * 4 + nreg];
      out[tiling_(bid_y, bid_x, M_TILE_SM, N_TILE_SM, N, 1) + distribute_(tid_x*N_TILE_REG, tid_y*M_TILE_REG, 1, N) + distribute_(mreg, nreg, N, 1)] = out_shared[distribute_(tid_x*N_TILE_REG, tid_y*M_TILE_REG, 1, N_TILE_SM) + distribute_(mreg, nreg, N_TILE_SM, 1)];
      // out[tiling_(bid_y, bid_x, M_TILE_SM, N_TILE_SM, N, 1) + distribute_(tid_x*N_TILE_REG, tid_y*M_TILE_REG, 1, N) + distribute_(mreg, nreg, N, 1)] = alpha * partial_sum[mreg * N_TILE_REG + nreg]; 
    }
  }
}


void catzilla_matmul_v2(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  const int M_TILE = 64;
  const int N_TILE = 64;
  const int mreg = 2;
  const int nreg = 2;
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(CEIL_DIV(M_TILE, mreg), CEIL_DIV(N_TILE, nreg));
  cudaFuncSetAttribute(_catzilla_matmul_v2<M_TILE, N_TILE, 32, mreg, nreg>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v2<M_TILE, N_TILE, 32, mreg, nreg><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

/////////////////////////////////////////////////////////////////////////////////
/// version 3:
/// - basic gmem to smem coalescing
/// - use smem for lhs, rhs, out
/// - thread-level blocking, compute a tile in reg-level C, not a scalar
/// - refactor with tiling and transfer helpers
/// - try different tiling config with hand
/////////////////////////////////////////////////////////////////////////////////
template <const int M_TILE_SM, const int N_TILE_SM, const int K_TILE_SM, const int M_TILE_REG, const int N_TILE_REG>
__global__ void _catzilla_matmul_v3(int M, int N, int K, float alpha,
                                       float *lhs, float *rhs,
                                       float beta, float *out) {
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
    float* lhs_cursor = lhs + tiling_(bid_y, k, M_TILE_SM, K_TILE_SM, K, 1);
    float* rhs_cursor = rhs + tiling_(k, bid_x, K_TILE_SM, N_TILE_SM, N, 1);
    for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
      lhs_shared[distribute_(tid_x, tid_y*M_TILE_REG + mreg, 1, K_TILE_SM)] = lhs_cursor[distribute_(tid_x, tid_y*M_TILE_REG + mreg, 1, K)];
    }
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      rhs_shared[distribute_(tid_x*N_TILE_REG + nreg, tid_y, 1, N_TILE_SM)] = rhs_cursor[distribute_(tid_x*N_TILE_REG + nreg, tid_y, 1, N)];
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_64x64x32_perthread_2x2(lhs_shared, rhs_shared, partial_sum);
  }
  __syncthreads();
  
  float* out_cursor = out + tiling_(bid_y, bid_x, M_TILE_SM, N_TILE_SM, N, 1);
  for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      out_shared[distribute_(tid_x * N_TILE_REG + nreg, tid_y * M_TILE_REG + mreg, 1, N_TILE_SM)] = alpha * partial_sum[mreg * 4 + nreg];
      out_cursor[distribute_(tid_x * N_TILE_REG + nreg, tid_y * M_TILE_REG + mreg, 1, N)] = out_shared[distribute_(tid_x*N_TILE_REG + nreg, tid_y*M_TILE_REG + mreg, 1, N_TILE_SM)];
    }
  }
}


void catzilla_matmul_v3(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  const int M_TILE = 64;
  const int N_TILE = 64;
  const int mreg = 2;
  const int nreg = 2;
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(CEIL_DIV(M_TILE, mreg), CEIL_DIV(N_TILE, nreg));
  cudaFuncSetAttribute(_catzilla_matmul_v3<M_TILE, N_TILE, 32, mreg, nreg>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v3<M_TILE, N_TILE, 32, mreg, nreg><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// simplify with stream-style
template <const int M_TILE_SM, const int N_TILE_SM, const int K_TILE_SM, const int M_TILE_REG, const int N_TILE_REG>
__global__ void _catzilla_matmul_v4(int M, int N, int K, float alpha,
                                       float *lhs, float *rhs,
                                       float beta, float *out) {
  // TODO: hide it with RAII
  int bid_x = blockIdx.x;
  int bid_y = blockIdx.y;

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  // smem stationary
  // M=128, N=128, K=32, Mreg=4, Nreg=4
  __shared__ float lhs_shared[M_TILE_SM * K_TILE_SM];
  __shared__ float rhs_shared[K_TILE_SM * N_TILE_SM];
  __shared__ float out_shared[M_TILE_SM * N_TILE_SM];

  Matrix lhs_mat = Matrix(lhs, Coord(M, K));
  Matrix rhs_mat = Matrix(rhs, Coord(K, N));
  Matrix out_mat = Matrix(out, Coord(M, N));

  Matrix lhs_shared_mat = Matrix(lhs_shared, Coord(M_TILE_SM, K_TILE_SM));
  Matrix rhs_shared_mat = Matrix(rhs_shared, Coord(K_TILE_SM, N_TILE_SM));
  Matrix out_shared_mat = Matrix(out_shared, Coord(M_TILE_SM, N_TILE_SM));
  // __shared__ float out_shared[1];

  // init out
  // out_shared[distribute_(tid_x, tid_y, 1, N_TILE_SM)] = 0.0;

  float partial_sum[M_TILE_REG * N_TILE_REG] = {0.};

  // distribute bid_y for lhs

  // QZ: per thread m and m+1
  for (int k = 0; k < CEIL_DIV(K, K_TILE_SM); k++) { // range(0, 128, 1)
    // use distribute_ for in-tile thread calculation
    // use tile_ for out-tile thread calculation
    for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
      lhs_shared_mat
        .distribute(Coord(tid_x, tid_y*M_TILE_REG + mreg), Coord(1, K_TILE_SM)) 
        = lhs_mat
        .tile(Coord(bid_y, k), Coord(M_TILE_SM, K_TILE_SM), Coord(K, 1))
        .distribute(Coord(tid_x, tid_y*M_TILE_REG + mreg), Coord(1, K));
    }
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      rhs_shared_mat
        .distribute(Coord(tid_x*N_TILE_REG + nreg, tid_y), Coord(1, N_TILE_SM)) 
        = rhs_mat
        .tile(Coord(k, bid_x), Coord(K_TILE_SM, N_TILE_SM), Coord(N, 1))
        .distribute(Coord(tid_x*N_TILE_REG + nreg, tid_y), Coord(1, N));
    }
    __syncthreads();

    // contract at 128x128x32 micro-kernel
    matmul_kernel_64x64x32_perthread_2x2(lhs_shared, rhs_shared, partial_sum);
  }
  __syncthreads();
  
  // float* out_cursor = out + tiling_(bid_y, bid_x, M_TILE_SM, N_TILE_SM, N, 1);
  for (int mreg = 0; mreg < M_TILE_REG; mreg++) {
    for (int nreg = 0; nreg < N_TILE_REG; nreg++) {
      // out_cursor[distribute_(tid_x * N_TILE_REG + nreg, tid_y * M_TILE_REG + mreg, 1, N)] = out_shared[distribute_(tid_x*N_TILE_REG + nreg, tid_y*M_TILE_REG + mreg, 1, N_TILE_SM)];
      // out_cursor.distribute(Coord(tid_x * N_TILE_REG + nreg, tid_y * M_TILE_REG + mreg), Coord(1, N)).data[0] = out_shared[distribute_(tid_x*N_TILE_REG + nreg, tid_y*M_TILE_REG + mreg, 1, N_TILE_SM)];
      out_shared_mat
        .tile(Coord(tid_x, tid_y), Coord(N_TILE_REG, M_TILE_REG), Coord(1, N_TILE_SM))
        .distribute(Coord(nreg, mreg), Coord(1, N_TILE_SM)) 
        = alpha * partial_sum[mreg * 4 + nreg];
      out_mat
        .tile(Coord(bid_y, bid_x), Coord(M_TILE_SM, N_TILE_SM), Coord(N, 1))
        .distribute(Coord(tid_x * N_TILE_REG + nreg, tid_y * M_TILE_REG + mreg), Coord(1, N)) 
        = out_shared_mat
        .distribute(Coord(tid_x * N_TILE_REG + nreg, tid_y * M_TILE_REG + mreg), Coord(1, N_TILE_SM));
    }
  }
}


void catzilla_matmul_v4(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  const int M_TILE = 64;
  const int N_TILE = 64;
  const int mreg = 2;
  const int nreg = 2;
  dim3 gridDim(CEIL_DIV(M, M_TILE), CEIL_DIV(N, N_TILE));
  dim3 blockDim(CEIL_DIV(M_TILE, mreg), CEIL_DIV(N_TILE, nreg));
  cudaFuncSetAttribute(_catzilla_matmul_v4<M_TILE, N_TILE, 32, mreg, nreg>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _catzilla_matmul_v3<M_TILE, N_TILE, 32, mreg, nreg><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

#endif // CATZILLA_RECIPES_KERNELS_MATMUL_V1_H_
