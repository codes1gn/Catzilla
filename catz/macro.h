#ifndef CATZILLA_CATZ_MACRO_H_
#define CATZILLA_CATZ_MACRO_H_

namespace catz {

// KERNEL UTILS
#define CEIL_DIV(dividend, divisor) (((dividend) + (divisor)-1) / (divisor))

#define MAKE_SHARED(matrixVar, size, type)                                     \
  __shared__ type matrixVar##_data[size];                                      \
  MatrixDyn<type> matrixVar =                                                  \
      MatrixDyn<type>(matrixVar##_data, CoordDyn(1, size))

#define MAKE_SHARED(matrixVar, size_x, size_y, type)                           \
  __shared__ type matrixVar##_data[size_x * size_y];                           \
  MatrixDyn<type> matrixVar =                                                  \
      MatrixDyn<type>(matrixVar##_data, CoordDyn(size_x, size_y))

// #define MAKE_SHARED(matrixVar, size, type)                                     \
//   __shared__ type matrixVar##_data[size];                                      \
//   Matrix<type> matrixVar = Matrix<type>(matrixVar##_data, CoordDyn(1, size))
//
// #define MAKE_SHARED(matrixVar, size_x, size_y, type) \
//   __shared__ type matrixVar##_data[size_x * size_y]; \ Matrix<type> matrixVar
//   = Matrix<type>(matrixVar##_data, CoordDyn(size_x, size_y))

// TEST UTILS
// static check at compile time
#define SCHECK(cond) static_assert(cond, "compile-time check failed\n")
#define SCHECK_FALSE(cond) static_assert(!(cond), "compile-time check failed\n")

#define CONCATENATE(x, y) x##y

#define CUDA_KERNEL_NAME(name) CONCATENATE(name, _cuda)

// 检查 CUDA 错误的宏
#define CUDA_CHECK_ERROR()                                                     \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    CHECK(err == cudaSuccess);                                                 \
    cudaDeviceSynchronize();                                                   \
  } while (0)

#define TEST_CUDA_CASE(kernel_name, case_name, tags)                           \
  __global__ void CUDA_KERNEL_NAME(kernel_name)();                             \
  TEST_CASE(case_name, tags) {                                                 \
    CUDA_KERNEL_NAME(kernel_name)<<<1, 1>>>();                                 \
    CUDA_CHECK_ERROR();                                                        \
  }                                                                            \
  __global__ void CUDA_KERNEL_NAME(kernel_name)()

} // namespace catz

#endif // CATZILLA_CATZ_MACRO_H_
