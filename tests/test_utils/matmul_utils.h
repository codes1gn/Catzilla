#ifndef CATZILLA_TESTS_TEST_UTILS_MATMUL_UTILS_H_
#define CATZILLA_TESTS_TEST_UTILS_MATMUL_UTILS_H_

#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "catz/cuda_utils.h"

namespace catz {
namespace test {

// Matrix initialization utilities
class MatrixTestUtils {
public:
    // Initialize a matrix with random values
    static void initializeRandomMatrix(float* matrix, int rows, int cols) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (int i = 0; i < rows * cols; ++i) {
            matrix[i] = dis(gen);
        }
    }
    
    // Initialize a matrix with zeros
    static void initializeZeroMatrix(float* matrix, int rows, int cols) {
        std::fill(matrix, matrix + rows * cols, 0.0f);
    }
    
    // Allocate device memory and optionally copy data from host
    static void prepareDeviceMemory(float** d_ptr, const float* h_ptr, size_t size) {
        cudaMalloc(d_ptr, size * sizeof(float));
        if (h_ptr != nullptr) {
            cudaMemcpy(*d_ptr, h_ptr, size * sizeof(float), cudaMemcpyHostToDevice);
        }
    }
    
    // Copy data from device to host and free device memory
    static void cleanupDeviceMemory(float* d_ptr, float* h_ptr, size_t size) {
        if (h_ptr != nullptr) {
            cudaMemcpy(h_ptr, d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
        }
        cudaFree(d_ptr);
    }
    
    // Compute reference result using cuBLAS
    static void computeCublasReference(int M, int N, int K,
                                     float alpha, const float* d_A, const float* d_B,
                                     float beta, float* d_C) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   N, M, K, &alpha,
                   d_B, N,
                   d_A, K,
                   &beta, d_C, N);
        cublasDestroy(handle);
    }
    
    // Verify results against reference
    static bool verifyResults(const float* result, const float* reference,
                            int rows, int cols, float tolerance = 1e-3,
                            bool printDiff = false) {
        bool passed = true;
        for (int i = 0; i < rows * cols; ++i) {
            float diff = std::abs(result[i] - reference[i]);
            if (diff > tolerance) {
                passed = false;
                if (printDiff) {
                    printf("Mismatch at index %d: result = %f, reference = %f, diff = %f\n",
                           i, result[i], reference[i], diff);
                }
            }
        }
        return passed;
    }
};

} // namespace test
} // namespace catz

#endif // CATZILLA_TESTS_TEST_UTILS_MATMUL_UTILS_H_ 