#ifndef CATZILLA_RECIPES_KERNELS_IMPORTER_H_
#define CATZILLA_RECIPES_KERNELS_IMPORTER_H_

#include "matmul/matmul_dist_to_thread.h"
#include "matmul/matmul_outerproduct.h"
#include "matmul/matmul_padswizzled.h"
#include "matmul/matmul_refactored.h"
#include "matmul/matmul_tensor_cores.h"
#include "matmul/matmul_tensor_cores_tuned.h"
#include "matmul/matmul_vanilla.h"
#include "matmul/matmul_vload_vstore.h"
#include "matmul/sgemm_for_choreo.h"

namespace catz::recipes {

// TODO: use catz::recipes namespace
inline void matmul_exec(int impl_idx, int M, int N, int K, float alpha,
                        float *A, float *B, float beta, float *C) {
  if (impl_idx == 1) {
    matmul_f16f32(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 2) {
    matmul_outerproduct(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 3) {
    matmul_affine_api(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 4) {
    matmul_stream_api(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 5) {
    matmul_stream_api_tuned(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 6) {
    matmul_pad_swizzled(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 7) {
    matmul_dist_to_thread(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 8) {
    matmul_vload_vstore(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 9) {
    // wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32
    matmul_tensor_cores_wmma_m16n16k16_f16f32(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 10) {
    // mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    matmul_tensor_cores_mma_m16n8k16_f16f32(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 11) {
    // mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
    matmul_tensor_cores_mma_m16n8k8_f16f32(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 12) {
    // mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    // TODO: RHS MATRIX has bug in ldmatrix inst
    matmul_tensor_cores_mma_m16n8k8_tf32f32(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 13) {
    // mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32
    // TODO: RHS MATRIX has bug in ldmatrix inst
    matmul_tensor_cores_mma_m16n8k4_tf32f32(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 14) {
    // improved from k11
    matmul_tensor_cores_mma_m16n8k8_f16f32_tuned(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 15) {
    // improved from k11
    matmul_tuned_with_mma_kernel(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 16) {
    // improved from k11
    matmul_dataflow_plus_mma(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 20) {
    // choreo version
    sgemm_for_choreo(A, B, C);
  } else {
    printf("[ERROR] kernel id not exists\n");
    exit(EXIT_FAILURE);
  }
}

} // namespace catz::recipes

#endif // CATZILLA_RECIPES_KERNELS_IMPORTER_H_
