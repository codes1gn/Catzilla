#ifndef CATZILLA_RECIPES_matmul_KERNELS_IMPORTER_H_ 
#define CATZILLA_RECIPES_matmul_KERNELS_IMPORTER_H_


#include "kernels/matmul.h"
#include "kernels/matmul_flatten_copy.h"
#include "kernels/matmul_vload_vstore.h"


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
  } else if (impl_idx == 5) {
    // stream-style tiling -> distribute
    catzilla_matmul_v5(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 6) {
    // stream-style tiling -> distribute
    catzilla_matmul_v6(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 7) {
    // stream-style tiling -> distribute
    catzilla_matmul_flatten_copy(M, N, K, alpha, A, B, beta, C);
  } else if (impl_idx == 8) {
    // stream-style tiling -> distribute
    catzilla_matmul_vload_vstore(M, N, K, alpha, A, B, beta, C);
  }
}

#endif // CATZILLA_RECIPES_matmul_KERNELS_IMPORTER_H_
