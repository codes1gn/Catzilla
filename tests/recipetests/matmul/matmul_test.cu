#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "recipes/kernel_impls.h"
#include "catz/cuda_utils.h"
#include "test_utils/matmul_utils.h"

using namespace catz;
using namespace catz::test;

namespace {

// Test configurations
struct TestConfig {
    int M, N, K;
    float alpha, beta;
    float tolerance;
    
    TestConfig(int m, int n, int k, float a = 1.0f, float b = 0.0f, float tol = 1e-3)
        : M(m), N(n), K(k), alpha(a), beta(b), tolerance(tol) {}
};

// Helper function to run a single kernel test
void testKernel(int kernel_version, const TestConfig& config, const char* kernel_name) {
    SECTION(kernel_name) {
        // Allocate host memory
        std::vector<float> h_A(config.M * config.K);
        std::vector<float> h_B(config.K * config.N);
        std::vector<float> h_C(config.M * config.N);
        std::vector<float> h_C_ref(config.M * config.N);
        
        // Initialize matrices
        MatrixTestUtils::initializeRandomMatrix(h_A.data(), config.M, config.K);
        MatrixTestUtils::initializeRandomMatrix(h_B.data(), config.K, config.N);
        MatrixTestUtils::initializeZeroMatrix(h_C.data(), config.M, config.N);
        MatrixTestUtils::initializeZeroMatrix(h_C_ref.data(), config.M, config.N);
        
        // Allocate and prepare device memory
        float *d_A, *d_B, *d_C, *d_C_ref;
        MatrixTestUtils::prepareDeviceMemory(&d_A, h_A.data(), config.M * config.K);
        MatrixTestUtils::prepareDeviceMemory(&d_B, h_B.data(), config.K * config.N);
        MatrixTestUtils::prepareDeviceMemory(&d_C, nullptr, config.M * config.N);
        MatrixTestUtils::prepareDeviceMemory(&d_C_ref, nullptr, config.M * config.N);
        
        // Get cuBLAS reference result
        MatrixTestUtils::computeCublasReference(config.M, config.N, config.K,
                                              config.alpha, d_A, d_B,
                                              config.beta, d_C_ref);
        
        // Run the kernel being tested
        recipes::matmul_exec(kernel_version, config.M, config.N, config.K,
                           config.alpha, d_A, d_B, config.beta, d_C);
        
        // Copy results back to host
        MatrixTestUtils::cleanupDeviceMemory(d_C, h_C.data(), config.M * config.N);
        MatrixTestUtils::cleanupDeviceMemory(d_C_ref, h_C_ref.data(), config.M * config.N);
        
        // Verify results
        bool passed = MatrixTestUtils::verifyResults(h_C.data(), h_C_ref.data(),
                                                   config.M, config.N,
                                                   config.tolerance, true);
        REQUIRE(passed);
        
        // Cleanup remaining device memory
        MatrixTestUtils::cleanupDeviceMemory(d_A, nullptr, 0);
        MatrixTestUtils::cleanupDeviceMemory(d_B, nullptr, 0);
    }
}

} // namespace


TEST_CASE("MatMul Kernels - Medium Matrix (128x128)", "[matmul][medium]") {
    TestConfig config(128, 128, 128);
    
    testKernel(1, config, "Kernel 1 - matmul_f16f32");
    testKernel(2, config, "Kernel 2 - matmul_outerproduct");
    testKernel(3, config, "Kernel 3 - matmul_affine_api");
    testKernel(4, config, "Kernel 4 - matmul_stream_api");
    testKernel(5, config, "Kernel 5 - matmul_stream_api_tuned");
    testKernel(6, config, "Kernel 6 - matmul_pad_swizzled");
    testKernel(15, config, "Kernel 15 - matmul_tuned_with_mma_kernel");
    testKernel(23, config, "Kernel 23 - choreo_gpu_matmul");
}

TEST_CASE("MatMul Kernels - Large Matrix (2048x2048)", "[matmul][large]") {
    TestConfig config(2048, 2048, 2048);
    
    testKernel(1, config, "Kernel 1 - matmul_f16f32");
    testKernel(2, config, "Kernel 2 - matmul_outerproduct");
    testKernel(3, config, "Kernel 3 - matmul_affine_api");
    testKernel(4, config, "Kernel 4 - matmul_stream_api");
    testKernel(5, config, "Kernel 5 - matmul_stream_api_tuned");
    testKernel(6, config, "Kernel 6 - matmul_pad_swizzled");
    testKernel(15, config, "Kernel 15 - matmul_tuned_with_mma_kernel");
    testKernel(23, config, "Kernel 23 - choreo_gpu_matmul");
}

// TEST_CASE("MatMul Kernels - Non-square Matrix", "[matmul][non-square]") {
//     TestConfig config(64, 128, 32);
//     
//     testKernel(1, config, "Kernel 1 - matmul_f16f32");
//     testKernel(2, config, "Kernel 2 - matmul_outerproduct");
//     testKernel(3, config, "Kernel 3 - matmul_affine_api");
//     testKernel(4, config, "Kernel 4 - matmul_stream_api");
//     testKernel(5, config, "Kernel 5 - matmul_stream_api_tuned");
//     testKernel(6, config, "Kernel 6 - matmul_pad_swizzled");
//     testKernel(15, config, "Kernel 15 - matmul_tuned_with_mma_kernel");
//     testKernel(23, config, "Kernel 23 - choreo_gpu_matmul");
// } 
