#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.h>
#include <sstream>
#include <string>
#include <vector>

#include "kernel_impls.h"
// TODO, promote this main and baselines/utils to top-level

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";
const std::string devLogFile = "dev.txt";

struct Arguments {
  int SIZE_M = 4096;
  int SIZE_K = 4096;
  int SIZE_N = 4096;
  int WARMUP = 5;
  int REPEAT = 1000;
  int VERSION = 0;
  int DEVICE = 0;
  int IS_PROF = 0;
};

void parse_arguments(int argc, char *argv[], Arguments &args) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-size-m" && i + 1 < argc) {
      std::istringstream(argv[++i]) >> args.SIZE_M;
    } else if (arg == "-size-k" && i + 1 < argc) {
      std::istringstream(argv[++i]) >> args.SIZE_K;
    } else if (arg == "-size-n" && i + 1 < argc) {
      std::istringstream(argv[++i]) >> args.SIZE_N;
    } else if (arg == "-warmup" && i + 1 < argc) {
      std::istringstream(argv[++i]) >> args.WARMUP;
    } else if (arg == "-repeat" && i + 1 < argc) {
      std::istringstream(argv[++i]) >> args.REPEAT;
    } else if (arg == "-version" && i + 1 < argc) {
      std::istringstream(argv[++i]) >> args.VERSION;
    } else if (arg == "-device" && i + 1 < argc) {
      std::istringstream(argv[++i]) >> args.DEVICE;
    } else if (arg == "-profile" && i + 1 < argc) {
      std::istringstream(argv[++i]) >> args.IS_PROF;
    }
  }
}

int main(int argc, char **argv) {

  Arguments args;
  parse_arguments(argc, argv, args);

  const int M = args.SIZE_M;
  const int K = args.SIZE_K;
  const int N = args.SIZE_N;
  const int warmup = args.WARMUP;
  const int repeat = args.REPEAT;
  const int kernel_id = args.VERSION;
  const int deviceIdx = args.DEVICE;
  const int isProf = args.IS_PROF;
  cudaCheck(cudaSetDevice(deviceIdx));
  printf("Running kernel %d on device %d.\n", kernel_id, deviceIdx);

  // print some device info
  // CudaDeviceInfo();

  // Declare the handle, create the handle, cublasCreate will return a value of
  // type cublasStatus_t to determine whether the handle was created
  // successfully (the value is 0)
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  float alpha = 1.0, beta = 0.0; // GEMM input parameters, C=α*AB+β*C

  float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr; // host matrices
  float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices

  A = (float *)malloc(sizeof(float) * M * K);
  B = (float *)malloc(sizeof(float) * K * N);
  C = (float *)malloc(sizeof(float) * M * N);
  C_ref = (float *)malloc(sizeof(float) * M * N);

  // randomise(A, M * K);
  // randomise(B, K * N);

  // ranges(A, M * K);
  // ranges(B, K * N);

  ones(A, M * K);
  ones(B, K * N);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * M * K));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * K * N));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * M * N));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * M * N));

  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * K * N, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * M * N, cudaMemcpyHostToDevice));
  cudaCheck(
      cudaMemcpy(dC_ref, C, sizeof(float) * M * N, cudaMemcpyHostToDevice));

  std::cout << "dimensions(M=" << M << ", K=" << K << ", N=" << N
            << ") , alpha: " << alpha << ", beta: " << beta
            << "is prof: " << isProf << std::endl;
  // Verify the correctness of the calculation, and execute it once before the
  // kernel function timing to avoid cold start errors
  if (kernel_id != 0 && isProf < 1) {
    std::cout << "Verify Results\n";
    run_kernel(0, M, N, K, alpha, dA, dB, beta, dC_ref,
               handle); // cuBLAS
    catzilla_matmul_exec(kernel_id, M, N, K, alpha, dA, dB, beta,
                         dC); // Executes the kernel, modifies the result matrix
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
    cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_ref, dC_ref, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    if (M <= 128) {
      std::cout << " Verbose output into " << errLogFile << "\n";
      std::ofstream fs;
      fs.open(errLogFile);
      fs << "A:\n";
      print_matrix(A, M, K, fs);
      fs << "B:\n";
      print_matrix(B, K, N, fs);
      fs << "C:\n";
      print_matrix(C, M, N, fs);
      fs << "Should:\n";
      print_matrix(C_ref, M, N, fs);
    }
    if (!verify_matrix(C_ref, C, M * N)) {
      std::cout << "Failed to pass the correctness verification against NVIDIA "
                   "cuBLAS."
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  cudaEventRecord(beg);
  for (int j = 0; j < repeat; j++) {
    // We don't reset dC between runs to save time
    catzilla_matmul_exec(kernel_id, M, N, K, alpha, dA, dB, beta,
                         dC); // Executes the kernel, modifies the result matrix
  }
  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, beg, end);
  elapsed_time /= 1000.; // Convert to seconds

  long flops = 2 * (long)M * (long)N * (long)K;
  printf("Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
         "(%ld).\n",
         elapsed_time / repeat, (repeat * flops * 1e-9) / elapsed_time, M);
  fflush(stdout);
  // make dC and dC_ref equal again (we modified dC while calling our kernel
  // for benchmarking)
  cudaCheck(
      cudaMemcpy(dC, dC_ref, sizeof(float) * M * N, cudaMemcpyDeviceToDevice));

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cublasDestroy(handle);

  return 0;
};
