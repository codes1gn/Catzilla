#ifndef CATZILLA_RECIPES_SGEMM_RUNNER_H_
#define CATZILLA_RECIPES_SGEMM_RUNNER_H_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

void cudaCheck(cudaError_t error, const char *file,
               int line); // CUDA error check
void CudaDeviceInfo();    // print CUDA information

void ranges(float *mat, int N);
void ranges(float *mat, int N, int mod);
void randomise(float *mat, int N);
void zeros(float *mat, int N);
void ones(float *mat, int N);
void copy_matrix(const float *src, float *dest, int N);
void print_matrix(const float *A, int M, int N, std::ofstream &fs);
bool verify_matrix(float *mat1, float *mat2, int N);

float get_current_sec();                        // Get the current moment
float cpu_elapsed_time(float &beg, float &end); // Calculate time difference

void run_reference(int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C, cublasHandle_t handle);

#endif // CATZILLA_RECIPES_SGEMM_RUNNER_H_
