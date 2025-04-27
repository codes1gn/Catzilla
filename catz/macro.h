#ifndef CATZILLA_CATZ_MACRO_H_
#define CATZILLA_CATZ_MACRO_H_

namespace catz {

#define PRINT_MATRIX(data) \
    printMatrixFloat(data, #data)

#define PRINT_MATRIX_F(data) \
    printMatrixFloat(data, #data)

#define PRINT_MATRIX_I(data) \
    printMatrixInt(data, #data)

template <typename T>
__device__ void printMatrixFloat(const T& data, const char* label) {
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    printf("===================== %s ===================\n", label);
    for (int i = 0; i < data.shape.rows; i++) {
      for (int j = 0; j < data.shape.cols; j++) {
        printf("%6.1f ", static_cast<float>(data.data[i * data.stride.rows() + j * data.stride.cols()]));
      }
      printf("\n");
    }
  }
}

template <typename T>
__device__ void printMatrixInt(const T& data, const char* label) {
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    printf("===================== %s ===================\n", label);
    for (int i = 0; i < data.shape.rows; i++) {
      for (int j = 0; j < data.shape.cols; j++) {
        printf("%d ", static_cast<float>(data.data[i * data.stride.rows() + j * data.stride.cols()]));
      }
      printf("\n");
    }
  }
}

#define ASSERT_EQUAL_MATRIX(lhs, rhs) \
    assertEqualMatrix(lhs, rhs, #lhs, #rhs)

template <typename T1, typename T2>
__device__ void assertEqualMatrix(const T1& lhs, const T2& rhs, const char* lhs_label, const char* rhs_label) {
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    printf("===================== Check if << %s == %s >> ===================\n", lhs_label, rhs_label);
    assert(lhs.shape.rows == rhs.shape.rows);
    assert(lhs.shape.cols == rhs.shape.cols);
    for (int i = 0; i < lhs.shape.rows; i++) {
      for (int j = 0; j < lhs.shape.cols; j++) {
        assert(lhs.data[i * data.stride.rows() + j * data.stride.cols()] 
            == rhs.data[i * data.stride.rows() + j * data.stride.cols()]);
      }
    }
    printf("===================== Check Passed ===================\n");
  }
}

// KERNEL UTILS
#define CEIL_DIV(dividend, divisor) (((dividend) + (divisor)-1) / (divisor))

#define MAKE_SHARED(matrixVar, size, type)                                     \
  __shared__ type matrixVar##_data[size];                                      \
  MatrixDyn<type> matrixVar =                                                  \
      MatrixDyn<type>(matrixVar##_data, CoordDyn(1, size))

// TODO: impl volume
// TODO: Matrix own.
// #define MAKE_SHARED_MATRIX(matrixVar, shape, type) \
//   __shared__ type matrixVar##_data[(shape).volume()]; \ auto matrixVar =
//   make_matrix(matrixVar##_data, shape)

#define MAKE_SHARED_MATRIX(matrixVar, shape, type)                             \
  __shared__ type matrixVar##_data[(shape).volume()];                          \
  for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < (shape).volume();   \
       i += blockDim.x * blockDim.y) {                                         \
    matrixVar##_data[i] = 0.0;                                                 \
  }                                                                            \
  auto matrixVar = make_matrix(matrixVar##_data, shape)

#define MAKE_LOCAL_MATRIX(matrixVar, shape, type)                              \
  type matrixVar##_data[(shape).volume()];                                     \
  for (int i = 0; i < (shape).volume(); ++i) {                                 \
    matrixVar##_data[i] = 0.0;                                                 \
  }                                                                            \
  auto matrixVar = make_matrix(matrixVar##_data, shape)

#define MAKE_SHARED(matrixVar, size_x, size_y, type)                           \
  __shared__ type matrixVar##_data[size_x * size_y];                           \
  MatrixDyn<type> matrixVar =                                                  \
      MatrixDyn<type>(matrixVar##_data, CoordDyn(size_x, size_y))

// TODO: consider to mute
#define make_shared_matrix(matrixVar, size_x, size_y, type)                    \
  __shared__ type matrixVar##_data[size_x * size_y];                           \
  auto matrixVar = make_matrix(matrixVar##_data, make_coord_dyn(size_x, size_y))

// #define MAKE_SHARED(matrixVar, size, type)                                     \
//   __shared__ type matrixVar##_data[size];                                      \
//   Matrix<type> matrixVar = Matrix<type>(matrixVar##_data, CoordDyn(1, size))
//
// #define MAKE_SHARED(matrixVar, size_x, size_y, type) \
//   __shared__ type matrixVar##_data[size_x * size_y]; \ Matrix<type> matrixVar
//   = Matrix<type>(matrixVar##_data, CoordDyn(size_x, size_y))

// TEST UTILS
// static check at compile time
#define SCHECK(cond)                                                           \
  static_assert(cond, "compile-time assertion check failed\n")
#define SCHECK_FALSE(cond) static_assert(!(cond), "compile-time check failed\n")

#define CONCATENATE(x, y) x##y

#define CUDA_KERNEL_NAME(name) CONCATENATE(name, _cuda)

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

#define TEST_CUDA_CASE_WITH_THREADS(kernel_name, num_of_threads, case_name,    \
                                    tags)                                      \
  __global__ void CUDA_KERNEL_NAME(kernel_name)();                             \
  TEST_CASE(case_name, tags) {                                                 \
    CUDA_KERNEL_NAME(kernel_name)<<<1, num_of_threads>>>();                    \
    CUDA_CHECK_ERROR();                                                        \
  }                                                                            \
  __global__ void CUDA_KERNEL_NAME(kernel_name)()

} // namespace catz

#endif // CATZILLA_CATZ_MACRO_H_
