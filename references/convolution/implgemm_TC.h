//
// Created by kiki on 24-6-28.
//

#ifndef SPCONV_IMPLGEMM_TC_H
#define SPCONV_IMPLGEMM_TC_H


#include <cstdint>
#include <cuda_runtime.h>
#include "include/storage_structure.h"
#include "include/utils.h"

#include "mm_utils/memcpy_pipeline.h"
#include "mm_utils/copy_helper.h"
#include "mm_utils/data_structure_helper.h"
#include "mm_utils/mma_helper.h"

// Block_M * Block_K >= thread_num * 4(16 byte)
// Block_K是4(16 byte)的倍数
#define FILTER_GM_TO_SM_CP_SIZE 16
// Block_K * Block_N >= thread_num * 4(16 byte)
// Block_K是4(16 byte)的倍数
#define INPUT_GM_TO_SM_CP_SIZE 16

// Block_M * Block_N >= thread_num * 4(16 byte)
// Block_N是4(16 byte)的倍数
#define OUTPUT_SM_TO_GM_CP_SIZE 16

#define FILTER_TRANSFER_SIZE_M 16
#define FILTER_TRANSFER_SIZE_K 32

#define INPUT_TRANSFER_SIZE_K 32
#define INPUT_TRANSFER_SIZE_N 16

template<typename T, typename BlockShape, typename WarpShape, typename MmaShape, typename Swizzle, typename CSwizzle, int Stage>
struct implgemm_TC {
    static constexpr int Block_M = BlockShape::M;
    static constexpr int Block_N = BlockShape::N;
    static constexpr int Block_K = BlockShape::K;
    static constexpr int Warp_M = WarpShape::M;
    static constexpr int Warp_N = WarpShape::N;
    static constexpr int Warp_K = WarpShape::K;
    static constexpr int Mma_M = MmaShape::M;
    static constexpr int Mma_N = MmaShape::N;
    static constexpr int Mma_K = MmaShape::K;
    static constexpr int NStage = Stage;

    typedef A_fragment<MmaShape> A_Fragment;
    typedef B_fragment<MmaShape> B_Fragment;
    typedef C_fragment<MmaShape, T> C_Fragment;

    __device__ __forceinline__
    void mainloop(const param_t<T> param, float *shared_mem);

    __device__ __forceinline__
    void epilogue(const param_t<T> param, float *shared_mem);
};

template<typename T, typename BlockShape, typename WarpShape, typename MmaShape, typename Swizzle, typename CSwizzle,int Stage>
__device__ __forceinline__
void implgemm_TC<T, BlockShape, WarpShape, MmaShape, Swizzle, CSwizzle, Stage>::mainloop
        (const param_t<T> param, float *shared_mem) {
    const int block_idx_M = blockIdx.x;
    const int block_idx_N = blockIdx.y;

    const int warp_idx_M = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int filter_tile_size = Block_M * Block_K;
    const int input_tile_size = Block_K * Block_N;

    T *filter_ptr = shared_mem;
    T *input_ptr = filter_ptr + filter_tile_size * NStage;

    Pipeline<NStage, true> pipeline;

    Swizzle swizzle;

    const int num_tiles = CEIL_DIV(param.r * param.s * param.c, Block_K);
    const int num_warp_M = Block_M / Warp_M;
    int fetch = 0;

    const int k_offset = param.r * param.s * param.c;
    const int n_input_offset = param.h * param.w * param.c;

    const int iter_mma_M = Warp_M / Mma_M;
    const int iter_mma_N = Warp_N / Mma_N;
    const int iter_mma_K = Warp_K / Mma_K;

    A_Fragment aFragment[iter_mma_M][iter_mma_K];
    B_Fragment bFragment[iter_mma_N][iter_mma_K];
    C_Fragment cFragment[iter_mma_M][iter_mma_N];

    T *input_gm_ptr = param.input_d + blockIdx.z * n_input_offset;

#pragma unroll
    for (int compute = 0; compute < num_tiles; compute++) {
#pragma unroll
        for (; fetch < compute + NStage; fetch++) {
            pipeline.acquire_writer();

            if (fetch < num_tiles) {
                T *shared_tile_filter = filter_ptr + filter_tile_size * (fetch % NStage);
                T *shared_tile_input = input_ptr + input_tile_size * (fetch % NStage);

                // 加载filter到SM (A_swizzle_old)
                // int iter_filter = filter_tile_size * sizeof(T) / blockDim.x / FILTER_GM_TO_SM_CP_SIZE;
                // for (int i = 0; i < iter_filter; i++) {
                //     int filter_h = block_idx_M * Block_M +
                //                    ((threadIdx.x + blockDim.x * i) * (FILTER_GM_TO_SM_CP_SIZE / sizeof(T))) / Block_K;
                //     int filter_w = fetch * Block_K + ((threadIdx.x + blockDim.x * i) * (FILTER_GM_TO_SM_CP_SIZE / sizeof(T))) % Block_K;
                //
                //     const T *src = param.weight_d + filter_h * k_offset + filter_w;
                //     T *dst = shared_tile_filter +
                //             swizzle(((threadIdx.x + blockDim.x * i) * (FILTER_GM_TO_SM_CP_SIZE / sizeof(T))));
                //     // 拷贝 filter_tile_size * sizeof(T) / blockDim.x 个T元素
                //     // 一次拷贝 FILTER_GM_TO_SM_CP_SIZE 个byte
                //     cp_gm_to_sm_async_zfill<FILTER_GM_TO_SM_CP_SIZE>((void *) (dst), (void *) (src));
                // }

                // 加载filter到SM (A_swizzle_new)
                for (int i = 0; i < Block_M / FILTER_TRANSFER_SIZE_M; ++i) {
                    for (int j = 0; j < Block_K / FILTER_TRANSFER_SIZE_K; ++j) {
                        int filter_h = block_idx_M * Block_M + i * FILTER_TRANSFER_SIZE_M + threadIdx.x / 8;
                        int filter_w = fetch * Block_K + j * FILTER_TRANSFER_SIZE_K + threadIdx.x % 8 * 4;
                        const T *src = param.weight_d + filter_h * k_offset + filter_w;
                        T* dst = shared_tile_filter + (i * Block_K / FILTER_TRANSFER_SIZE_K + j) * FILTER_TRANSFER_SIZE_M * FILTER_TRANSFER_SIZE_K
                                + swizzle(FILTER_GM_TO_SM_CP_SIZE / sizeof(T) * threadIdx.x);
                        cp_gm_to_sm_async_zfill<FILTER_GM_TO_SM_CP_SIZE>((void *) (dst), (void *) (src));
                    }
                }

                // =======================================================

                // 加载input到SM (B_swizzle_old)
                // int iter_input = input_tile_size * sizeof(T) / blockDim.x / INPUT_GM_TO_SM_CP_SIZE;
                // for (int i = 0; i < iter_input; i++) {
                //     // input_matrix 是转置的im2col矩阵
                //     int input_matrix_h = block_idx_N * Block_N +
                //                          ((threadIdx.x + blockDim.x * i) * (INPUT_GM_TO_SM_CP_SIZE / sizeof(T))) /
                //                          Block_K;
                //     int input_matrix_w = fetch * Block_K +
                //                          threadIdx.x % (Block_K / (INPUT_GM_TO_SM_CP_SIZE / sizeof(T))) *
                //                          (INPUT_GM_TO_SM_CP_SIZE / sizeof(T));
                //     // input是原始输入矩阵(N * H * (WC))
                //     int input_h =
                //             (input_matrix_h / param.Ow) * param.u - param.p + input_matrix_w / (param.s * param.c);
                //     int input_w = ((input_matrix_h % param.Ow) * param.v - param.q) * param.c +
                //                   input_matrix_w % (param.s * param.c);
                //
                //     const T *src = input_gm_ptr + input_h * param.w * param.c + input_w;
                //     T *dst = shared_tile_input + swizzle(input_tile_size / iter_input * i +
                //              threadIdx.x * (INPUT_GM_TO_SM_CP_SIZE / sizeof(T)));
                //
                //     // 拷贝 input_tile_size * sizeof(T) / blockDim.x 个T元素
                //     // 一次拷贝 INPUT_GM_TO_SM_CP_SIZE 个byte
                //     bool invalid = input_w < 0 || input_w >= param.w * param.c || input_h < 0 || input_h >= param.h;
                //     cp_gm_to_sm_async_zfill_ca<INPUT_GM_TO_SM_CP_SIZE>((void *) (dst), (void *) (src), invalid);
                // }

                // 加载input到SM (B_swizzle_new)
                for (int i = 0; i < Block_K / INPUT_TRANSFER_SIZE_K; ++i) {
                    for (int j = 0; j < Block_N / INPUT_TRANSFER_SIZE_N; ++j) {
                        int input_matrix_h = block_idx_N * Block_N + j * INPUT_TRANSFER_SIZE_N + threadIdx.x / 8;
                        int input_matrix_w = fetch * Block_K + i * INPUT_TRANSFER_SIZE_K + threadIdx.x % 8 * 4;
                        int input_h = (input_matrix_h / param.Ow) * param.u - param.p + input_matrix_w / (param.s * param.c);
                        int input_w = ((input_matrix_h % param.Ow) * param.v - param.q) * param.c + input_matrix_w % (param.s * param.c);
                        const T *src = input_gm_ptr + input_h * param.w * param.c + input_w;

                        T* dst = shared_tile_input + (i * Block_K / INPUT_TRANSFER_SIZE_K + j) * INPUT_TRANSFER_SIZE_K * INPUT_TRANSFER_SIZE_N
                                + swizzle(INPUT_GM_TO_SM_CP_SIZE / sizeof(T) * threadIdx.x);
                        bool invalid = input_w < 0 || input_w >= param.w * param.c || input_h < 0 || input_h >= param.h;
                        cp_gm_to_sm_async_zfill_ca<INPUT_GM_TO_SM_CP_SIZE>((void *) (dst), (void *) (src), invalid);
                    }
                }

            }
            pipeline.commit_stage();
        }

        pipeline.acquire_reader();

        T *shared_tile_filter = filter_ptr + (compute % NStage) * filter_tile_size;
        T *shared_tile_input = input_ptr + (compute % NStage) * input_tile_size;

        // 加载filter到寄存器 (A_swizzle_old)
// #pragma unroll
//         for (int i = 0; i < iter_mma_M; i++) {
//             for (int j = 0; j < iter_mma_K; j++) {
//                 load_matrix_A_sm_to_frag_sync_no_trans<Swizzle>(aFragment[i][j], shared_tile_filter, warp_idx_M * Mma_M * Block_K + i * num_warp_M * Mma_M * Block_K + j * Mma_K, Block_K, lane_id);
//             }
//         }

        // 加载filter到寄存器 (A_swizzle_new)
#pragma unroll
        for (int i = 0; i < iter_mma_M; i++) {
            for (int j = 0; j < iter_mma_K; j++) {
                int offset = ((i * num_warp_M + warp_idx_M) * (Block_K / FILTER_TRANSFER_SIZE_K) + j / (FILTER_TRANSFER_SIZE_K / Mma_K)) * FILTER_TRANSFER_SIZE_M * FILTER_TRANSFER_SIZE_K + j % 4 * 8;
                        // + (threadIdx.x % FILTER_TRANSFER_SIZE_M) * FILTER_TRANSFER_SIZE_K + threadIdx.x / FILTER_TRANSFER_SIZE_M * 4 ;
                load_matrix_A_sm_to_frag_sync_no_trans<Swizzle>(aFragment[i][j], shared_tile_filter, offset, FILTER_TRANSFER_SIZE_K, lane_id);
            }
        }

//         // 加载input到寄存器 (B_swizzle_old)
// #pragma unroll
//         for (int i = 0; i < iter_mma_N; i++) {
//             for (int j = 0; j < iter_mma_K; j+=2) {
//                 load_matrix_B_sm_to_frag_sync_no_trans<Swizzle>(bFragment[i][j], bFragment[i][j+1], shared_tile_input, i * Mma_N * Block_K + j * Mma_K, Block_K, lane_id);
//             // for (int j = 0; j < iter_mma_K; j+=1) {
//                 // load_matrix_B_sm_to_frag_sync_no_trans<Swizzle>(bFragment[i][j], shared_tile_input, i * Mma_N * Block_K + j * Mma_K, Block_K, lane_id);
//             }
//         }

        // 加载input到寄存器 (B_swizzle_new)
#pragma unroll
        for (int i = 0; i < iter_mma_N; i++) {
            for (int j = 0; j < iter_mma_K; j+=2) {
                int offset = (i / (INPUT_TRANSFER_SIZE_N / Mma_N) * (Block_K / INPUT_TRANSFER_SIZE_K) + j / (INPUT_TRANSFER_SIZE_K / Mma_K)) * INPUT_TRANSFER_SIZE_K * INPUT_TRANSFER_SIZE_N
                        + (i % (INPUT_TRANSFER_SIZE_N / Mma_N)) * Mma_N * INPUT_TRANSFER_SIZE_K + j % 4 * 8;
                load_matrix_B_sm_to_frag_sync_no_trans<Swizzle>(bFragment[i][j], bFragment[i][j + 1], shared_tile_input, offset, INPUT_TRANSFER_SIZE_K, lane_id);
            }
        }

        // 计算
        for (int j = 0; j < iter_mma_K; j++) {
#pragma unroll
            for (int i = 0; i < iter_mma_M; i++) {
#pragma unroll
                for (int k = 0; k < iter_mma_N; k++) {
                    mma_sync_dense(cFragment[i][k], aFragment[i][j], bFragment[k][j], cFragment[i][k]);
                }
            }
        }

        pipeline.release_reader();
    }

    __syncthreads();

    // 输出结果到SM
    T *shared_C = shared_mem;
    int mma_m_offset = 0;

    CSwizzle cSwizzle;

#pragma unroll
    for (int i = 0; i < iter_mma_M; i++) {
#pragma unroll
        for (int j = 0; j < iter_mma_N; j++) {
            int offset_C = lane_id % 4 * 2 + lane_id / 4 * Block_N + j * Mma_N + warp_idx_M * Mma_M * Block_N;

            *((int2 *) (shared_C + cSwizzle(mma_m_offset + offset_C))) = *((int2 *) (cFragment[i][j].x));
            *((int2 *) (shared_C + cSwizzle(mma_m_offset + offset_C + 8 * Block_N))) = *((int2 *) (cFragment[i][j].x + 2));
        }
        mma_m_offset += Mma_M * num_warp_M * Block_N;
    }

    __syncthreads();
}

template<typename T, typename BlockShape, typename WarpShape, typename MmaShape, typename Swizzle, typename CSwizzle, int Stage>
__device__ __forceinline__
void implgemm_TC<T, BlockShape, WarpShape, MmaShape, Swizzle, CSwizzle, Stage>::epilogue
        (const param_t<T> param, float *shared_mem) {
    Swizzle swizzle;
    CSwizzle cSwizzle;

    int iteration = Block_M * Block_N * sizeof(T) / blockDim.x / OUTPUT_SM_TO_GM_CP_SIZE;
    int num_element_per_iter = OUTPUT_SM_TO_GM_CP_SIZE / sizeof(T);
    int grid_offset = blockIdx.z * param.k * param.Oh * param.Ow + blockIdx.x * Block_M * param.Oh * param.Ow + blockIdx.y * Block_N;

    for(int i = 0; i < iteration; i++){
        int h = (threadIdx.x + i * blockDim.x) * num_element_per_iter / Block_N;
        int w = (threadIdx.x + i * blockDim.x) * num_element_per_iter % Block_N;

        float *shared_C = shared_mem + cSwizzle(h * Block_N + w);
        float *global_C = param.output_d + grid_offset + h * param.Oh * param.Ow + w;

        *((float4 *) global_C) = *((float4 *) shared_C);
    }

    // 每个线程拷贝8个float
    // float *shared_C = shared_mem + threadIdx.x % 2 * 8 + threadIdx.x / 2 * Block_N;
    // float *global_C =
    //         param.output_d + blockIdx.z * param.k * param.Oh * param.Ow + blockIdx.x * Block_M * param.Oh * param.Ow +
    //         blockIdx.y * Block_N + threadIdx.x % 2 * 8 + threadIdx.x / 2 * param.Oh * param.Ow;
    //
    // *((float4 *) global_C) = *((float4 *) shared_C);
    // *((float4 *) (global_C + 4)) = *((float4 *) (shared_C + 4));
}

template<typename T, typename BlockShape, typename WarpShape, typename MmaShape, typename Swizzle, typename CSwizzle, int Stage>
__global__ void implgemm_TC_kernel(param_t<T> param) {
    extern __shared__ float shared_mem[];
    implgemm_TC<T, BlockShape, WarpShape, MmaShape, Swizzle, CSwizzle, Stage> kernel;
    kernel.mainloop(param, shared_mem);
    kernel.epilogue(param, shared_mem);
}

struct Swizzle8BWiseXor {
    __device__ __forceinline__
    int operator()(int o) {
        return (o ^ ((o & (7 << 5)) >> 3));
    }
};

struct SwizzleIdentity {
    __device__ __forceinline__
    int operator()(int o) {
        return o;
    }
};

template<typename BlockShape>
struct SwizzlePadding {
    // __device__ __forceinline__
    // int operator()(int x_hat, int y_hat) {
    //     return x_hat / 8 * 2 * 32 + x_hat / 8 * 8 + y_hat * 8;
    // }

    __device__ __forceinline__
    int operator()(int offset) {
        int x_hat = offset % BlockShape::N % 32, y_hat = offset / BlockShape::N % 8;
        int padded_offset = x_hat / 8 * 2 * 32 + x_hat / 8 * 8 + y_hat * 8 + x_hat % 8;
        int x_ = padded_offset % 32, y_ = padded_offset / 32;
        // printf("x_hat: %d, y_hat: %d, -> x_: %d, y_: %d\n", x_hat, y_hat, x_, y_);
        return (offset / BlockShape::N / 8 * 9 + y_) * BlockShape::N + offset % BlockShape::N / 32 * 32 + x_;
    }
};

template<typename T, typename BlockShape, typename WarpShape, typename MmaShape, int Stage>
void implgemm_TC_exec(param_t<T> param) {
    cudaStream_t stream = NULL;

    int blockx = CEIL_DIV(param.k, BlockShape::M);
    int blocky = CEIL_DIV(param.Oh * param.Ow, BlockShape::N);
    int blockz = param.n;
    int threadx = 128;
    int thready = 1;
    int threadz = 1;
    dim3 block_dim(blockx, blocky, blockz);
    dim3 thread_dim(threadx, thready, threadz);

    using Swizzle = Swizzle8BWiseXor;
    using CSwizzle = SwizzlePadding<BlockShape>;

    implgemm_TC_kernel<T, BlockShape, WarpShape, MmaShape, Swizzle, CSwizzle, Stage>
    <<<block_dim, thread_dim, max((BlockShape::M * BlockShape::K + BlockShape::K * BlockShape::N) * Stage *
                              sizeof(T), (BlockShape::M + CEIL_DIV(BlockShape::M, 8)) * BlockShape::N * sizeof(T)), stream>>>(param);
}

#endif //SPCONV_IMPLGEMM_TC_H
