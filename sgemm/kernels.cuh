#pragma once

#include "kernels/1_naive.cuh"
#include "kernels/2_kernel_global_mem_coalesce.cuh"
#include "kernels/3_kernel_shared_mem_blocking.cuh"
#include "kernels/4_kernel_1D_blocktiling.cuh"
#include "kernels/5_kernel_2D_blocktiling.cuh"
#include "kernels/6_kernel_vectorize.cuh"
#include "kernels/7_kernel_autotuned.cuh"
#include "kernels/8_kernel_warptiling.cuh"
#include "kernels/9_kernel_doublebuffer.cuh"
#include "kernels/10_kernel_hardcoded.cuh"
#include "kernels/11_sgemm_manual.cuh"
#include "kernels/12_sgemm_manual.cuh"
