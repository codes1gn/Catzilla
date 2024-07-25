//
// Created by kiki on 24-7-1.
//

#ifndef CONV_COPY_HELPER_H
#define CONV_COPY_HELPER_H

#include "data_structure_helper.h"

__device__ __forceinline__
uint get_smem_ptr(const void *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__
void load_meta_gm_to_frag_sync_uint(uint *__restrict__ dst,
                                    const uint *base, const int offset) {
    *((uint *) dst) = *((uint *) (base + offset));
}

template<int SizeInBytes>
__device__ __forceinline__
void cp_gm_to_sm_async_zfill(void *smem_ptr, void const *gmem_ptr,
                             const bool zfill = false) {
    unsigned smem_int_ptr = get_smem_ptr(smem_ptr);
    int src_in_bytes = (zfill ? 0 : SizeInBytes);

    asm volatile (
            "{\n"
            "  cp.async.cg.shared.global [%0], [%1], %2, %3;\n"
            "}\n"::"r"(smem_int_ptr), "l"(gmem_ptr), "n"(SizeInBytes), "r"(src_in_bytes)
            );
}

template<int SizeInBytes>
__device__ __forceinline__
void cp_gm_to_sm_async_zfill_ca(void *smem_ptr, void const *gmem_ptr,
                             const bool zfill = false) {
    unsigned smem_int_ptr = get_smem_ptr(smem_ptr);
    int src_in_bytes = (zfill ? 0 : SizeInBytes);

    asm volatile (
            "{\n"
            "  cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
            "}\n"::"r"(smem_int_ptr), "l"(gmem_ptr), "n"(SizeInBytes), "r"(src_in_bytes)
            );
}

template<typename SWIZZLE>
__device__ __forceinline__
void load_matrix_A_sm_to_frag_sync_no_trans(A_fragment<Shape_16_8_8> &dst,
                                          const float *base, const int offset, const int ldm, const int lane_id) {
    SWIZZLE swizzle;
    const float *src = base + swizzle(offset + lane_id % 16 * ldm + lane_id / 16 * 4);
    unsigned smem_ptr = get_smem_ptr(src);
    uint32_t *x = reinterpret_cast<uint32_t *>(dst.x);
    // float *x = dst.x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
            : "r"(smem_ptr));
}

template<typename SWIZZLE>
__device__ __forceinline__
void load_matrix_B_sm_to_frag_sync_no_trans(B_fragment<Shape_16_8_8> &dst1, B_fragment<Shape_16_8_8> &dst2,
                                          const float *base, const int offset, const int ldm, const int lane_id) {
    SWIZZLE swizzle;
    const float *src = base + swizzle(offset + lane_id % 8 * ldm + lane_id / 8 * 4);
    unsigned smem_ptr = get_smem_ptr(src);
    uint32_t *x1 = reinterpret_cast<uint32_t *>(dst1.x);
    uint32_t *x2 = reinterpret_cast<uint32_t *>(dst2.x);
    // float *x = dst.x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(x1[0]), "=r"(x1[1]), "=r"(x2[0]), "=r"(x2[1])
            : "r"(smem_ptr));
}

template<typename SWIZZLE>
__device__ __forceinline__
void load_matrix_B_sm_to_frag_sync_no_trans(B_fragment<Shape_16_8_8> &dst1,
                                            const float *base, const int offset, const int ldm, const int lane_id) {
    SWIZZLE swizzle;
    const float *src = base + swizzle(offset + lane_id % 8 * ldm + lane_id / 8 * 4);
    unsigned smem_ptr = get_smem_ptr(src);
    uint32_t *x1 = reinterpret_cast<uint32_t *>(dst1.x);
    // uint32_t *x2 = reinterpret_cast<uint32_t *>(dst2.x);
    // float *x = dst.x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(x1[0]), "=r"(x1[1])
            : "r"(smem_ptr));
}


#endif //CONV_COPY_HELPER_H
