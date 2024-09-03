#ifndef CATZILLA_RECIPES_matmul_KERNELS_IMPORTER_H_ 
#define CATZILLA_RECIPES_matmul_KERNELS_IMPORTER_H_


#include "kernels/matmul.h"


inline void catzilla_matmul_exec(int impl_idx, int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  if (impl_idx == 1) {
    catzilla_matmul_v1(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 2) {
    catzilla_matmul_v2(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 3) {
    // with tiling and distribute seperately
    catzilla_matmul_v3(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 4) {
    // stream-style tiling -> distribute
    catzilla_matmul_v4(M, N, K, alpha, A, B, beta, C);
  }
}

#endif // CATZILLA_RECIPES_matmul_KERNELS_IMPORTER_H_
