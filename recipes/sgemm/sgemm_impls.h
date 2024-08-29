#ifndef CATZILLA_RECIPES_SGEMM_KERNELS_IMPORTER_H_ 
#define CATZILLA_RECIPES_SGEMM_KERNELS_IMPORTER_H_


#include "kernels/sgemm_v1.h"
#include "kernels/sgemm_v2.h"


inline void catzilla_sgemm_exec(int impl_idx, int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  if (impl_idx == 1) {
    catzilla_sgemm_v1_host(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 2) {
    catzilla_sgemm_v2_host(M, N, K, alpha, A, B, beta, C);
  }
}

#endif // CATZILLA_RECIPES_SGEMM_KERNELS_IMPORTER_H_
