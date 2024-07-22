#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// CUDA错误检查宏
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << error << " \""                      \
                      << cudaGetErrorString(error) << "\"" << std::endl; \
            exit(1);                                                     \
        }                                                                \
    } while (0)

void reference_matmul(float* A, float* B, float* C, int M, int N, int K,
                    float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum * alpha + beta * C[i * N + j];
        }
    }
}

void OptsGemm(int m, int n, int k, float* d_A, float* d_B, float* d_C,
              float alpha, float beta);

int main() {
    // 分配和初始化主机内存
    int problem[5] = {512, 1024, 2048, 4096, 8192};
    int benchmark_times = 10;
    float alpha = 1.0f;
    float beta = 0.0f;
    for (auto matmul_size : problem) {
        float *h_A, *h_B, *h_C, *ref_C, *cublas_h_C;
        int N = matmul_size;
        int M = matmul_size;
        int K = 1024;
        size_t bytes_a = M * K * sizeof(float);
        size_t bytes_b = K * N * sizeof(float);
        size_t bytes_c = M * N * sizeof(float);
        h_A = (float*)malloc(bytes_a);
        h_B = (float*)malloc(bytes_b);
        h_C = (float*)malloc(bytes_c);
        cublas_h_C = (float*)malloc(bytes_c);

        ref_C = (float*)malloc(bytes_c);

        // 初始化矩阵A和B的值
        for (int i = 0; i < M * K; ++i) {
            //! 随机生成 0-1 之间的浮点数
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        for (int i = 0; i < N * K; ++i) {
            h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        memset(h_C, 0, bytes_c);
        memset(cublas_h_C, 0, bytes_c);

        // 分配并初始化设备内存
        float *d_A, *d_B, *d_C, *cublas_C;
        CHECK_CUDA(cudaMalloc((void**)&d_A, bytes_a));
        CHECK_CUDA(cudaMalloc((void**)&d_B, bytes_b));
        CHECK_CUDA(cudaMalloc((void**)&d_C, bytes_c));
        CHECK_CUDA(cudaMalloc((void**)&cublas_C, bytes_c));

        // 将输入矩阵从主机复制到设备
        CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_a, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_b, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_C, h_C, bytes_c, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(cublas_C, cublas_h_C, bytes_c,
                              cudaMemcpyHostToDevice));

        // 定义CUDA内核的执行配置
        constexpr int block_len = 128;
        dim3 threadsPerBlock(256);
        dim3 blocksPerGrid((M + 128 - 1) / 128, (N + 128 - 1) / 128);

        /* Time the "optimized" implementation */
        cudaEvent_t start, stop;
        // Allocate CUDA events that we'll use for timing
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start, NULL));
        for (int i = 0; i < benchmark_times; ++i) {
            OptsGemm(M, N, K, d_A, d_B, d_C, 1.0f, 0.0f);
        }
        CHECK_CUDA(cudaEventRecord(stop, NULL));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

        // Compute and print the performance
        float msecPerMatrixMul = milliseconds / benchmark_times;
        double flopsPerMatrixMul = 2.0 * M * K * N;
        double gflops =
                (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

        std::cout << "Opt Gemm Problem size: " << M << "x" << N << "x" << K;
        std::cout << " Used time Time: " << msecPerMatrixMul << "ms";
        std::cout << " Performance: " << gflops << " GFLOPS" << std::endl;

        CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_c, cudaMemcpyDeviceToHost));
        // cublas compute
        cublasHandle_t handle;
        cublasCreate(&handle);

        CHECK_CUDA(cudaEventRecord(start, NULL));
        for (int i = 0; i < benchmark_times; ++i) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B,
                        N, d_A, K, &beta, cublas_C, N);
        }
        CHECK_CUDA(cudaEventRecord(stop, NULL));
        CHECK_CUDA(cudaEventSynchronize(stop));
        milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        msecPerMatrixMul = milliseconds / benchmark_times;
        gflops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

        std::cout << "Cublas Gemm Problem size: " << M << "x" << N << "x" << K;
        std::cout << " Used time Time: " << msecPerMatrixMul << "ms";
        std::cout << " Performance: " << gflops << " GFLOPS" << std::endl;

        CHECK_CUDA(cudaMemcpy(cublas_h_C, cublas_C, bytes_c,
                              cudaMemcpyDeviceToHost));

        //! reference compute
        // reference_matmul(h_A, h_B, ref_C, M, N, K, 1.0f, 0.0f);
        //  验证结果
        std::cout << "Validating the result of problem M: " << M << ",N: " << N
                  << ",K: " << K << std::endl;
        bool error = false;
        for (int i = 0; i < M * N; ++i) {
            if (abs(h_C[i] - cublas_h_C[i]) > 0.001) {
                std::cerr << "Mismatch at index " << i << std::endl;
                std::cerr << "Opt: " << h_C[i] << " cublas: " << ref_C[i]
                          << std::endl;
                error = true;
                break;
            } 
        }
        if (!error) {
            std::cout << "Verfication passed!!!" << std::endl;
        } else {
            std::cout << "Verfication failed!!!" << std::endl;
        }

        // 释放设备和主机内存
        free(h_A);
        free(h_B);
        free(h_C);
        free(ref_C);
        free(cublas_h_C);
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUDA(cudaFree(cublas_C));
    }

    return 0;
}