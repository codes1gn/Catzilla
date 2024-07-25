//
// Created by kiki on 24-7-2.
//

#ifndef CONV_MMA_HELPER_H
#define CONV_MMA_HELPER_H

#include "data_structure_helper.h"

__device__ __forceinline__
void mma_sync_dense(
        C_fragment<Shape_16_8_8, float> &d,
        const A_fragment<Shape_16_8_8> &a,
        const B_fragment<Shape_16_8_8> &b,
        const C_fragment<Shape_16_8_8, float> &c) {
    uint32_t const *A = reinterpret_cast<uint32_t const *>(a.x);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(b.x);
    // float const *C = reinterpret_cast<float const *>(&c);
    // float *D = reinterpret_cast<float *>(&d);

    asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
    "r"(B[0]), "r"(B[1]),
    "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3])
            );
}

#endif //CONV_MMA_HELPER_H
