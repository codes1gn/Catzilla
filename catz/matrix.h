#ifndef CATZILLA_CATZ_MATRIX_H_
#define CATZILLA_CATZ_MATRIX_H_

#include <algorithm>
#include <cassert>
// #include <concepts>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <tuple>
#include <type_traits>

#include "cuda_utils.h"
#include "macro.h"

// TODO: rename to matrix.h
namespace catz {

template <typename T>
struct is_allowed_type
    : std::disjunction<std::is_same<T, float>, std::is_same<T, half>,
                       std::is_same<T, __nv_bfloat16>,
                       std::is_same<T, float4>> {};

// concept DataType = std::is_same_v<T, float> || std::is_same_v<T, half> ||
// std::is_same_v<T, float4>;

// convert multi-dim index to flatten-index
// make it stream-style coding
//
// legacy:
// output[tiling_(bid_y, bid_x, M_TILE_SM, N_TILE_SM, N, 1) +
// distribute_(tid_x*N_TILE_REG, tid_y*M_TILE_REG, 1, N) + distribute_(m_reg,
// n_reg, N, 1)]
//
// stream-like:
// lhs.tiling_(xxx).distribute_(xx)
// rhs.tiling_(xxx).tiling_(xxx).distribute_(xx)
inline __device__ int distribute_(int x_offset, int y_offset, int x_stride,
                                  int y_stride) {
  int idx = x_offset * x_stride + y_offset * y_stride;
  return idx;
}

// convert multi-dim index of a tile-block to flatten-index
inline __device__ int tiling_(int tile_id_x, int tile_id_y, int to_stride_x,
                              int to_stride_y, int from_stride_x,
                              int from_stride_y) {
  int idx = tile_id_x * to_stride_x * from_stride_x +
            tile_id_y * to_stride_y * from_stride_y;
  return idx;
}

// template<int X, int Y>
struct Coord {
  int first;
  int second;

  // 构造函数
  constexpr __device__ Coord(int first, int second)
      : first(first), second(second) {}

  // 从 std::tuple<int, int> 构造
  constexpr __device__ Coord(const std::tuple<int, int> &t)
      : first(std::get<0>(t)), second(std::get<1>(t)) {}

  // 将 Coord 转换为 std::tuple<int, int>
  constexpr __device__ std::tuple<int, int> to_tuple() const {
    return std::make_tuple(first, second);
  }

  constexpr Coord operator+(const Coord &other) const {
    return Coord(first + other.first, second + other.second);
  }

  constexpr Coord operator-(const Coord &other) const {
    return Coord(first - other.first, second - other.second);
  }

  constexpr Coord operator*(const Coord &other) const {
    return Coord(first * other.first, second * other.second);
  }

  constexpr Coord operator/(const Coord &other) const {
    return Coord(first / other.first, second / other.second);
  }

  constexpr __device__ Coord &xor_swizzle() {
    auto i16 = (first * 32 + second) * sizeof(float) / 16;
    auto y16 = i16 / 8;
    auto x16 = i16 % 8;
    auto x16_swz = y16 ^ x16;
    auto x_swz =
        x16_swz * 16 / sizeof(float) % 32 + second % (16 / sizeof(float));
    second = x_swz % 32;
    first = x_swz / 32;
    return *this;
  }

  constexpr __device__ Coord &xor_swizzle_col() {
    auto i = (first * 16 + second);
    auto i_swz = (first * 16 + second) ^ first;
    second = i_swz % 16;
    first = i_swz / 16;
    return *this;
  }

  constexpr __device__ Coord &xor_swizzle_row() {
    auto i = (first * 16 + second);
    auto i_swz = (first * 16 + second) ^ second;
    second = i_swz % 16;
    first = i_swz / 16;
    return *this;
  }
};

template <int ROWS, int COLS>
struct CoordNightly {
  static_assert(ROWS != 0, "first dim cannot be zero.");
  static_assert(COLS != 0, "second dim cannot be zero.");
  static constexpr int first = ROWS;
  static constexpr int second = COLS;

  // 构造函数
  constexpr __device__ CoordNightly() {}

  template <int ofirst, int osecond>
  constexpr CoordNightly<first + ofirst, second + osecond>
  operator+(const CoordNightly<ofirst, osecond> &other) const {
    return CoordNightly<first + other.first, second + other.second>();
  }

  template <int ofirst, int osecond>
  constexpr CoordNightly<first - ofirst, second - osecond>
  operator-(const CoordNightly<ofirst, osecond> &other) const {
    return CoordNightly<first - other.first, second - other.second>();
  }

  template <int ofirst, int osecond>
  constexpr CoordNightly<first * ofirst, second * osecond>
  operator*(const CoordNightly<ofirst, osecond> &other) const {
    return CoordNightly<first * other.first, second * other.second>();
  }

  template <int ofirst, int osecond>
  constexpr CoordNightly<first / ofirst, second / osecond>
  operator/(const CoordNightly<ofirst, osecond> &other) const {
    static_assert(second != 0 && other.second != 0, "Division by zero");
    return CoordNightly<first / other.first, second / other.second>();
  }

  constexpr __device__ CoordNightly &xor_swizzle() {
    auto i16 = (first * 32 + second) * sizeof(float) / 16;
    auto y16 = i16 / 8;
    auto x16 = i16 % 8;
    auto x16_swz = y16 ^ x16;
    auto x_swz =
        x16_swz * 16 / sizeof(float) % 32 + second % (16 / sizeof(float));
    second = x_swz % 32;
    first = x_swz / 32;
    return *this;
  }

  constexpr __device__ CoordNightly &xor_swizzle_col() {
    constexpr auto i = (first * 16 + second);
    constexpr auto i_swz = i ^ first;
    // auto _second = i_swz % 16;
    // auto _first = i_swz / 16;
    // return *this;
    return CoordNightly<(i_swz / 16), (i_swz % 16)>();
  }

  constexpr __device__ CoordNightly &xor_swizzle_row() {
    constexpr auto i = (first * 16 + second);
    constexpr auto i_swz = i ^ second;
    // second = i_swz % 16;
    // first = i_swz / 16;
    // return *this;
    return CoordNightly<(i_swz / 16), (i_swz % 16)>();
  }
};

template <typename T, typename CoordType, typename CoordType2>
struct MatrixNightly;

template <typename T, int ROWS, int COLS, int STRD>
struct MatrixNightly<T, CoordNightly<ROWS, COLS>, CoordNightly<STRD, 1>> {
  static_assert(is_allowed_type<T>::value,
                "T must be one of the allowed types: float, half, or float4.");
  T *data;
  CoordNightly<ROWS, COLS> shape;
  CoordNightly<STRD, 1> stride;

  // the complete construction function
  constexpr __device__ MatrixNightly(T *data, CoordNightly<ROWS, COLS> shape,
                                     CoordNightly<STRD, 1> stride)
      : data(data), shape(shape), stride(stride) {}

  // copy
  constexpr __device__ MatrixNightly(const MatrixNightly &other)
      : data(other.data), shape(other.shape), stride(other.stride) {}

  // move
  constexpr __device__ MatrixNightly(MatrixNightly &&other) noexcept
      : data(other.data), shape(other.shape), stride(other.stride) {
    other.data = nullptr;
  }

  inline __device__ MatrixNightly next() {
    data += 1;
    return *this;
  }

  inline __device__ void fill(T value) {
    int flat_id = threadIdx.y * blockDim.x + threadIdx.x;
    int row_in_current = flat_id / shape.second;
    int col_in_current = flat_id % shape.second;
    int elements = shape.first * shape.second;
    int threads = blockDim.x * blockDim.y;
    for (int chunk = 0; chunk < CEIL_DIV(elements, threads); chunk++) {
      data[chunk * threads + flat_id] = value;
    }
  }

  // TODO: make it device'd code
  template <int NROWS, int NCOLS>
  constexpr inline __device__
      MatrixNightly<T, CoordNightly<NROWS, NCOLS>, CoordNightly<STRD, 1>>
      tile(Coord tile_var, CoordNightly<NROWS, NCOLS> new_shape) {
    T *new_data = data + tile_var.first * new_shape.first * stride.first +
                  tile_var.second * new_shape.second * stride.second;
    MatrixNightly<T, CoordNightly<NROWS, NCOLS>, CoordNightly<STRD, 1>> ret =
        make_matrix(new_data, new_shape, stride);
    return std::move(ret);
  }
};

// Default builder:
// (T*, CoordType, CoordType) -> Matrix
// auto matrix = make_matrix(data, Coord<16, 32>(), Coord<32, 1>())
template <typename T, typename CoordType, typename CoordType2>
__device__ MatrixNightly<T, CoordType, CoordType2>
make_matrix(T *data, CoordType shape, CoordType2 stride) {
  return MatrixNightly<T, CoordType, CoordType2>(data, shape, stride);
}

// Specialised builder for contiguous memory
// (T*, CoordType) -> Matrix
template <typename T, int ROWS, int COLS>
__device__ MatrixNightly<T, CoordNightly<ROWS, COLS>, CoordNightly<COLS, 1>>
make_matrix(T *data, CoordNightly<ROWS, COLS> shape) {
  return MatrixNightly<T, CoordNightly<ROWS, COLS>, CoordNightly<COLS, 1>>(
      data, shape, CoordNightly<COLS, 1>());
}

template <typename T>
struct Matrix {
  static_assert(is_allowed_type<T>::value,
                "T must be one of the allowed types: float, half, or float4.");

  // A, 16x32, 32x4
  T *data;
  Coord shape;
  Coord stride;

  constexpr __device__ Matrix(T *data, Coord shape)
      : data(data), shape(shape), stride(Coord(shape.second, 1)) {}

  constexpr __device__ Matrix(T *data, Coord shape, Coord stride)
      : data(data), shape(shape), stride(stride) {}

  // copy
  constexpr __device__ Matrix(const Matrix &other)
      : data(other.data), shape(other.shape), stride(other.stride) {}

  // move
  constexpr __device__ Matrix(Matrix &&other) noexcept
      : data(other.data), shape(other.shape), stride(other.stride) {
    other.data = nullptr;
  }

  __device__ Matrix next() {
    data += 1;
    return *this;
  }

  constexpr inline __device__ void fill(T value) {
    int flat_id = threadIdx.y * blockDim.x + threadIdx.x;
    int row_in_current = flat_id / shape.second;
    int col_in_current = flat_id % shape.second;
    int elements = shape.first * shape.second;
    int threads = blockDim.x * blockDim.y;
    for (int chunk = 0; chunk < CEIL_DIV(elements, threads); chunk++) {
      data[chunk * threads + flat_id] = value;
    }
  }

  // TODO: rename x, y into row, col
  template <typename U = T>
  inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      load_fragments(unsigned *loader) {
    // TODO: add loader length check
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    int lane_rank = lane_id % shape.first;
    int lane_group = lane_id / shape.first;
    const half *data_ptr = data + lane_rank * stride.first + lane_group * 8;
    if (shape.first == 16 && shape.second == 8) {
      // LHS X2
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                   : "=r"(loader[0]), "=r"(loader[1])
                   : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.first == 16 && shape.second == 16) {
      // RHS X1
      asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
          : "=r"(loader[0]), "=r"(loader[1]), "=r"(loader[2]), "=r"(loader[3])
          : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.first == 8 && shape.second == 8) {
      // LHS X4
      asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                   : "=r"(loader[0])
                   : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.first == 8 && shape.second == 16) {
      // RHS X2
      asm volatile(
          "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
          : "=r"(loader[0]), "=r"(loader[1])
          : "r"(get_smem_ptr(data_ptr)));
    }
  }

  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, float>::value, void>::type
      load_fragments_c(float *loader) {
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    if (shape.first == 16 && shape.second == 8) {
      loader[0] = data[(lane_id / 4) * stride.first + (lane_id % 4) * 2];
      loader[1] = data[(lane_id / 4) * stride.first + (lane_id % 4) * 2 + 1];
      loader[2] = data[(lane_id / 4) * stride.first + (lane_id % 4) * 2 +
                       8 * stride.first];
      loader[3] = data[(lane_id / 4) * stride.first + (lane_id % 4) * 2 +
                       8 * stride.first + 1];
    }
  }

  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, float>::value, void>::type
      store_fragments_c(float *storer) {
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    if (shape.first == 16 && shape.second == 8) {
      data[(lane_id / 4) * stride.first + (lane_id % 4) * 2] = storer[0];
      data[(lane_id / 4) * stride.first + (lane_id % 4) * 2 + 1] = storer[1];
      data[(lane_id / 4) * stride.first + (lane_id % 4) * 2 +
           8 * stride.first] = storer[2];
      data[(lane_id / 4) * stride.first + (lane_id % 4) * 2 + 8 * stride.first +
           1] = storer[3];
    }
  }

  constexpr inline __device__ Matrix tile(Coord tile_var, Coord other_shape) {
    Matrix ret =
        Matrix(data + tile_var.first * other_shape.first * stride.first +
                   tile_var.second * other_shape.second * stride.second,
               other_shape, stride);
    return std::move(ret);
  }

  constexpr inline __device__ Matrix dist_to(Coord tile_var) {
    // shape = 8 x 2
    // y in 0-4, x in 0-4
    Matrix ret = Matrix(data + tile_var.first * stride.first +
                            tile_var.second * stride.second,
                        shape, stride);
    return std::move(ret);
  }

  constexpr inline __device__ Matrix dist_to(int dist_id) {
    int row_in_current = dist_id / shape.second;
    int col_in_current = dist_id % shape.second;
    Matrix ret = Matrix(data + row_in_current * stride.first +
                            col_in_current * stride.second,
                        shape, stride);
    return std::move(ret);
  }

  // distribute the following 32 elements to a wrap
  constexpr inline __device__ Matrix dist_to_wrap() {
    // NOTE: we can have three design
    // 1. use all threads to spread
    // 2. use x threads to spread
    // 3. use a wrap to spread.
    // we prefer to take third option in currently design
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    int row_in_current = lane_id / shape.second;
    int col_in_current = lane_id % shape.second;
    Matrix ret = Matrix(data + row_in_current * stride.first +
                            col_in_current * stride.second,
                        shape, stride);
    return std::move(ret);
  }

  // distribute the following NUM_THREADS elements to all threads
  constexpr inline __device__ Matrix dist_to_thread() {
    // NOTE: we can have three design
    // 1. use all threads to spread
    // 2. use x threads to spread
    // 3. use a wrap to spread.
    // we prefer to take third option in currently design
    int lane_id = threadIdx.y * blockDim.x + threadIdx.x;
    int row_in_current = lane_id / shape.second;
    int col_in_current = lane_id % shape.second;
    Matrix ret = Matrix(data + row_in_current * stride.first +
                            col_in_current * stride.second,
                        shape, stride);
    return std::move(ret);
  }

  // operator '<=' is syntax sugar that combines dist-to-threads and '='
  // operator
  constexpr inline __device__ void operator<=(const Matrix &other) {
    // can make 11352
    int total_threads = blockDim.x * blockDim.y;
    // int total_threads = 256;
    int total_elements = shape.first * shape.second;
    // int total_elements = 128 * 32;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    int row_this = thread_id / shape.second;
    int col_this = thread_id % shape.second;
    int row_other = thread_id / other.shape.second;
    int col_other = thread_id % other.shape.second;

    // TODO: make this config more general
    // #pragma unroll 8
    // NOTE: we need constexpr for avoiding heavy runtime index calc
    // compile-time symbolic index calc, become perf drop
    // for (int i = 0; i < total_elements / total_threads; i++)
    //   (data)[i * total_threads * 32 / 32
    //          + row_this * 32 + col_this]
    //     = ((other.data))[i * total_threads *4096
    //                        /32
    //                      + row_other * 4096 + col_other];
#pragma unroll 16
    for (int i = 0; i < 16 /* total_elements / total_threads */; i++)
      (data)[i * total_threads * stride.first / shape.second +
             row_this * stride.first + col_this] =
          ((other.data))[i * total_threads * other.stride.first /
                             other.shape.second +
                         row_other * other.stride.first + col_other];
    // if (total_threads * 4 < total_elements) {
    //   total_elements = total_elements / 4;
    //   int row_this = thread_id / (shape.second / 4);
    //   int col_this = thread_id % (shape.second / 4);
    //   int row_other = thread_id / (other.shape.second / 4);
    //   int col_other = thread_id % (other.shape.second / 4);
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     ((float4*)data)[i * stride.first / shape.second + row_this *
    //     stride.first + col_this]
    //       = ((float4*)(other.data))[i * other.stride.first /
    //       other.shape.second
    //                    + row_other * other.stride.first / 4 + col_other];
    // } else {
    //   int row_this = thread_id / shape.second;
    //   int col_this = thread_id % shape.second;
    //   int row_other = thread_id / other.shape.second;
    //   int col_other = thread_id % other.shape.second;
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     (data)[i * stride.first / shape.second + row_this * stride.first +
    //     col_this]
    //       = ((other.data))[i * other.stride.first / other.shape.second
    //                    + row_other * other.stride.first + col_other];
    // }
  }

  //   for (int m = 0; m < 8/* CEIL_DIV(M_TILE, 2) */; m++) {
  //     lhs_shared_mat.tile(Coord(m, kin), Coord(2, 16)).dist_to_thread()
  //       = lhs_mat.tile(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
  //           .tile(Coord(m, kin), Coord(2, 16))
  //           .dist_to_thread();
  //   }
  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      operator<=(const Matrix<float> &other) {
    int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.first * shape.second;
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int row_this = thread_id / shape.second;
    int col_this = thread_id % shape.second;
    int row_other = thread_id / other.shape.second;
    int col_other = thread_id % other.shape.second;

    // OPTION I: normalised loop, with full symbolic index calculation
    // ~ 4180 GFLOPS
#pragma unroll 8
    for (int i = 0; i < total_elements / total_threads; i++)
      (data)[i * total_threads * stride.first / shape.second +
             row_this * stride.first + col_this] =
          ((other.data))[i * total_threads * other.stride.first /
                             other.shape.second +
                         row_other * other.stride.first + col_other];

    // OPTION II: non-normalised loop, with full symbolic index calculation
    // ~ 1940 GFLOPS
    // for (int i = 0; i < total_elements; i+=total_threads)
    //   (data)[i * stride.first / shape.second + row_this * stride.first +
    //   col_this]
    //     = ((other.data))[i * other.stride.first / other.shape.second
    //                      + row_other * other.stride.first + col_other];

    // OPTION II: non-normalised loop, with full symbolic index calculation
    // ~ 2800 GFLOPS
    // for (int i = 0; i < total_elements; i+=32)
    //   (data)[i * stride.first / shape.second + row_this * stride.first +
    //   col_this]
    //     = ((other.data))[i * other.stride.first / other.shape.second
    //                      + row_other * other.stride.first + col_other];

    // OPTION II: non-normalised loop, with more-of-literal calculation
    // ~ 2790 GFLOPS
    //
    // for (int i = 0; i < 256; i+=32)
    //   (data)[i * 16 / shape.second + row_this * 16 + col_this]
    //     = ((other.data))[i * 4096 / other.shape.second
    //                      + row_other * 4096 + col_other];

    // if (total_threads * 4 < total_elements) {
    //   total_elements = total_elements / 4;
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     ((float4*)data)[i * stride.first / shape.second + row_this *
    //     stride.first + col_this]
    //       = ((float4*)(other.data))[i * other.stride.first /
    //       other.shape.second
    //                    + row_other * other.stride.first + col_other];
    // } else {
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     (data)[i * stride.first / shape.second + row_this * stride.first +
    //     col_this]
    //       = ((other.data))[i * other.stride.first / other.shape.second
    //                    + row_other * other.stride.first + col_other];
    // }
  }

  // TODO: merge it
  // use threads in one wrap to move it,
  // replace m with threadIdx.y
  //
  // for (int n = 0; n < CEIL_DIV(N_TILE, N_REG); n++) {
  //   out_mat.tile(Coord(blockIdx.x, blockIdx.y), out_sm_tile_shape)
  //       .tile(Coord(m, n), out_reg_tile_shape)
  //     <<= out_shared_mat.tile(Coord(m, n), out_reg_tile_shape);
  // }
  // special operator that allows any same-volume copy, when the volume is
  // dividable by threads volume
  constexpr inline __device__ void operator<<=(const Matrix &other) {
    int total_threads = blockDim.x;
    // int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.first * shape.second;
    int thread_id = (threadIdx.x + threadIdx.y * blockDim.x) % 32;
    // int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    int row_this = thread_id / shape.second;
    int col_this = thread_id % shape.second;
    int row_other = thread_id / other.shape.second;
    int col_other = thread_id % other.shape.second;
    // #pragma unroll
    for (int i = 0; i < total_elements; i += total_threads) {
      data[i * stride.first / shape.second + row_this * stride.first +
           col_this] = other.data[i * other.stride.first / other.shape.second +
                                  row_other * other.stride.first + col_other];
    }
  }

  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      operator<<=(const Matrix<float> &other) {
    int total_threads = blockDim.x;
    // int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.first * shape.second;
    int thread_id = (threadIdx.x + threadIdx.y * blockDim.x) % 32;
    int row_this = thread_id / shape.second;
    int col_this = thread_id % shape.second;
    int row_other = thread_id / other.shape.second;
    int col_other = thread_id % other.shape.second;
    // #pragma unroll
    for (int i = 0; i < total_elements; i += total_threads) {
      data[i * stride.first / shape.second + row_this * stride.first +
           col_this] =
          __float2half(other.data[i * other.stride.first / other.shape.second +
                                  row_other * other.stride.first + col_other]);
    }
  }

  constexpr __device__ void operator=(const Matrix<T> &other) {
    *data = *other.data;
  }

  constexpr __device__ void operator=(const T other_scalar) {
    *data = other_scalar;
  }

  // TODO: merge generics
  // half <= float
  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      operator=(const Matrix<float> &other) {
    *data = __float2half(*other.data);
  }

  // float <= half
  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, float>::value, void>::type
      operator=(const Matrix<half> &other) {
    *data = __half2float(*other.data);
  }

  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      operator=(const float other_scalar) {
    *data = __float2half(other_scalar);
  }

  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, float>::value, void>::type
      print() const {
    for (size_t i = 0; i < shape.first; ++i) {
      for (size_t j = 0; j < shape.second; ++j) {
        std::cout << data[i * shape.first + j * shape.second] << " ";
      }
      std::cout << "\n";
    }
    std::cout << std::endl;
  }
};

// helper for shared decl
// template<Coord shape>
template <int x, int y, typename T>
__device__ Matrix<T> make_shared() {
  static_assert(is_allowed_type<T>::value,
                "T must be one of the allowed types: float, half, or float4.");

  __shared__ T _data[x * y];
  return Matrix<T>(std::move(_data), Coord(x, y));
}

template <int x, int y, typename T>
__device__ Matrix<T> make_local() {
  static_assert(is_allowed_type<T>::value,
                "T must be one of the allowed types: float, half, or float4.");
  float _data[x * y] = {0.};
  // extern __shared__ float _data[];
  return Matrix<T>(_data, Coord(x, y));
}

// TODO: consider use std::integral_constant to make Coord known at compile-time
__device__ Coord &&make_coord(int x, int y) { return std::move(Coord(x, y)); }

__device__ __forceinline__ int xor_swizzle(int o) {
  return (o ^ ((o & (7 << 5)) >> 3));
}

} // namespace catz

#endif // CATZILLA_CATZ_MATRIX_H_
