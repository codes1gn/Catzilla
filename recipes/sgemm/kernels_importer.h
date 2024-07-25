#ifndef CATZILLA_RECIPES_SGEMM_KERNELS_IMPORTER_H_ 
#define CATZILLA_RECIPES_SGEMM_KERNELS_IMPORTER_H_


#include "kernels/sgemm_v1.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

inline void catzilla_sgemm_exec(int impl_idx, int M, int N, int K, float alpha, const float *A,
                                  const float *B, float beta, float *C) {
  if (impl_idx == 1) {
    printf(">> Exec catzilla_sgemm_v1");
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    catzilla_sgemm_v1<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}


#endif // CATZILLA_RECIPES_SGEMM_KERNELS_IMPORTER_H_
