#ifndef CATZILLA_RECIPES_UTILS_INDEX_UTILS_H_
#define CATZILLA_RECIPES_UTILS_INDEX_UTILS_H_

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
#include "macros.h"

// TODO: rename to matrix.h
namespace catz
{

template <typename T>
struct is_allowed_type
    : std::disjunction<std::is_same<T, float>, std::is_same<T, half>,
                       std::is_same<T, __nv_bfloat16>,
                       std::is_same<T, float4>> {
};

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
// novel:
// lhs.tiling_(xxx).distribute_(xx)
// rhs.tiling_(xxx).tiling_(xxx).distribute_(xx)
inline __device__ int distribute_(int x_offset, int y_offset, int x_stride,
                                  int y_stride)
{
  int idx = x_offset * x_stride + y_offset * y_stride;
  return idx;
}

// convert multi-dim index of a tile-block to flatten-index
inline __device__ int tiling_(int tile_id_x, int tile_id_y, int to_stride_x,
                              int to_stride_y, int from_stride_x,
                              int from_stride_y)
{
  int idx = tile_id_x * to_stride_x * from_stride_x
            + tile_id_y * to_stride_y * from_stride_y;
  return idx;
}

// template<int X, int Y>
// TODO: rename to row, col, not x, y
struct Coord {
  int x;
  int y;

  // 构造函数
  __device__ Coord(int x, int y)
      : x(x)
      , y(y)
  {
  }

  // 从 std::tuple<int, int> 构造
  __device__ Coord(const std::tuple<int, int> &t)
      : x(std::get<0>(t))
      , y(std::get<1>(t))
  {
  }

  // 将 Coord 转换为 std::tuple<int, int>
  __device__ std::tuple<int, int> to_tuple() const
  {
    return std::make_tuple(x, y);
  }

  __device__ Coord &xor_swizzle()
  {
    auto i16 = (y * 32 + x) * sizeof(float) / 16;
    auto y16 = i16 / 8;
    auto x16 = i16 % 8;
    auto x16_swz = y16 ^ x16;
    auto x_swz = x16_swz * 16 / sizeof(float) % 32 + x % (16 / sizeof(float));
    x = x_swz % 32;
    y = x_swz / 32;
    return *this;
  }

  __device__ Coord &xor_swizzle_col()
  {
    auto i = (x * 16 + y);
    auto i_swz = (x * 16 + y) ^ x;
    y = i_swz % 16;
    x = i_swz / 16;
    return *this;
  }

  __device__ Coord &xor_swizzle_row()
  {
    auto i = (x * 16 + y);
    auto i_swz = (x * 16 + y) ^ y;
    y = i_swz % 16;
    x = i_swz / 16;
    return *this;
  }
};

template <typename T> struct Matrix {
  static_assert(is_allowed_type<T>::value,
                "T must be one of the allowed types: float, half, or float4.");

  // A, 16x32, 32x4
  T *data;
  Coord shape;
  Coord stride;

  __device__ Matrix(T *data, Coord shape)
      : data(data)
      , shape(shape)
      , stride(Coord(shape.y, 1))
  {
  }

  __device__ Matrix(T *data, Coord shape, Coord stride)
      : data(data)
      , shape(shape)
      , stride(stride)
  {
  }

  __device__ Matrix(const Matrix &other)
      : data(other.data)
      , shape(other.shape)
      , stride(other.stride)
  {
  }

  __device__ Matrix(Matrix &&other) noexcept
      : data(other.data)
      , shape(other.shape)
      , stride(other.stride)
  {
    other.data = nullptr;
  }

  __device__ Matrix next()
  {
    data += 1;
    return *this;
  }

  inline __device__ void fill(T value)
  {
    int flat_id = threadIdx.y * blockDim.x + threadIdx.x;
    int row_in_current = flat_id / shape.y;
    int col_in_current = flat_id % shape.y;
    int elements = shape.x * shape.y;
    int threads = blockDim.x * blockDim.y;
    for (int chunk = 0; chunk < CEIL_DIV(elements, threads); chunk++) {
      data[chunk * threads + flat_id] = value;
    }
  }

  // TODO: rename x, y into row, col
  template <typename U = T>
  inline __device__
    typename std::enable_if<std::is_same<U, half>::value, void>::type
    load_fragments(unsigned *loader)
  {
    // TODO: add loader length check
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    int lane_rank = lane_id % shape.x;
    int lane_group = lane_id / shape.x;
    const half *data_ptr = data + lane_rank * stride.x + lane_group * 8;
    if (shape.x == 16 && shape.y == 8) {
      // LHS X2
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                   : "=r"(loader[0]), "=r"(loader[1])
                   : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.x == 16 && shape.y == 16) {
      // RHS X1
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(loader[0]), "=r"(loader[1]), "=r"(loader[2]), "=r"(loader[3])
        : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.x == 8 && shape.y == 8) {
      // LHS X4
      asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                   : "=r"(loader[0])
                   : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.x == 8 && shape.y == 16) {
      // RHS X2
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(loader[0]), "=r"(loader[1])
        : "r"(get_smem_ptr(data_ptr)));
    }
  }

  template <typename U = T>
  inline __device__
    typename std::enable_if<std::is_same<U, float>::value, void>::type
    load_fragments_c(float *loader)
  {
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    if (shape.x == 16 && shape.y == 8) {
      loader[0] = data[(lane_id / 4) * stride.x + (lane_id % 4) * 2];
      loader[1] = data[(lane_id / 4) * stride.x + (lane_id % 4) * 2 + 1];
      loader[2]
        = data[(lane_id / 4) * stride.x + (lane_id % 4) * 2 + 8 * stride.x];
      loader[3]
        = data[(lane_id / 4) * stride.x + (lane_id % 4) * 2 + 8 * stride.x + 1];
    }
  }

  template <typename U = T>
  inline __device__
    typename std::enable_if<std::is_same<U, float>::value, void>::type
    store_fragments_c(float *storer)
  {
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    if (shape.x == 16 && shape.y == 8) {
      data[(lane_id / 4) * stride.x + (lane_id % 4) * 2] = storer[0];
      data[(lane_id / 4) * stride.x + (lane_id % 4) * 2 + 1] = storer[1];
      data[(lane_id / 4) * stride.x + (lane_id % 4) * 2 + 8 * stride.x]
        = storer[2];
      data[(lane_id / 4) * stride.x + (lane_id % 4) * 2 + 8 * stride.x + 1]
        = storer[3];
    }
  }

  inline __device__ Matrix tile(Coord tile_var, Coord other_shape)
  {
    Matrix ret = Matrix(data + tile_var.x * other_shape.x * stride.x
                          + tile_var.y * other_shape.y * stride.y,
                        other_shape, stride);
    return std::move(ret);
  }

  inline __device__ Matrix dist_to(Coord tile_var)
  {
    // shape = 8 x 2
    // y in 0-4, x in 0-4
    Matrix ret = Matrix(data + tile_var.x * stride.x + tile_var.y * stride.y,
                        shape, stride);
    return std::move(ret);
  }

  inline __device__ Matrix dist_to(int dist_id)
  {
    int row_in_current = dist_id / shape.y;
    int col_in_current = dist_id % shape.y;
    Matrix ret
      = Matrix(data + row_in_current * stride.x + col_in_current * stride.y,
               shape, stride);
    return std::move(ret);
  }

  // distribute the following 32 elements to a wrap
  inline __device__ Matrix dist_to_wrap()
  {
    // NOTE: we can have three design
    // 1. use all threads to spread
    // 2. use x threads to spread
    // 3. use a wrap to spread.
    // we prefer to take third option in currently design
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    int row_in_current = lane_id / shape.y;
    int col_in_current = lane_id % shape.y;
    Matrix ret
      = Matrix(data + row_in_current * stride.x + col_in_current * stride.y,
               shape, stride);
    return std::move(ret);
  }

  // distribute the following NUM_THREADS elements to all threads
  inline __device__ Matrix dist_to_thread()
  {
    // NOTE: we can have three design
    // 1. use all threads to spread
    // 2. use x threads to spread
    // 3. use a wrap to spread.
    // we prefer to take third option in currently design
    int lane_id = threadIdx.y * blockDim.x + threadIdx.x;
    int row_in_current = lane_id / shape.y;
    int col_in_current = lane_id % shape.y;
    Matrix ret
      = Matrix(data + row_in_current * stride.x + col_in_current * stride.y,
               shape, stride);
    return std::move(ret);
  }

  // operator '<=' is syntax sugar that combines dist-to-threads and '='
  // operator
  inline __device__ void operator<=(const Matrix &other)
  {
    int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.x * shape.y;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int row_this = thread_id / shape.y;
    int col_this = thread_id % shape.y;
    int row_other = thread_id / other.shape.y;
    int col_other = thread_id % other.shape.y;
// TODO: make this config more general
#pragma unroll 8
    for (int i = 0; i < total_elements / total_threads; i++)
      (data)[i * total_threads * stride.x / shape.y + row_this * stride.x
             + col_this]
        = ((other.data))[i * total_threads * other.stride.x / other.shape.y
                         + row_other * other.stride.x + col_other];
    // if (total_threads * 4 < total_elements) {
    //   total_elements = total_elements / 4;
    //   int row_this = thread_id / (shape.y / 4);
    //   int col_this = thread_id % (shape.y / 4);
    //   int row_other = thread_id / (other.shape.y / 4);
    //   int col_other = thread_id % (other.shape.y / 4);
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     ((float4*)data)[i * stride.x / shape.y + row_this * stride.x +
    //     col_this]
    //       = ((float4*)(other.data))[i * other.stride.x / other.shape.y
    //                    + row_other * other.stride.x / 4 + col_other];
    // } else {
    //   int row_this = thread_id / shape.y;
    //   int col_this = thread_id % shape.y;
    //   int row_other = thread_id / other.shape.y;
    //   int col_other = thread_id % other.shape.y;
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     (data)[i * stride.x / shape.y + row_this * stride.x + col_this]
    //       = ((other.data))[i * other.stride.x / other.shape.y
    //                    + row_other * other.stride.x + col_other];
    // }
  }

  //   for (int m = 0; m < 8/* CEIL_DIV(M_TILE, 2) */; m++) {
  //     lhs_shared_mat.tile(Coord(m, kin), Coord(2, 16)).dist_to_thread()
  //       = lhs_mat.tile(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
  //           .tile(Coord(m, kin), Coord(2, 16))
  //           .dist_to_thread();
  //   }
  template <typename U = T>
  inline __device__
    typename std::enable_if<std::is_same<U, half>::value, void>::type
    operator<=(const Matrix<float> &other)
  {
    int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.x * shape.y;
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int row_this = thread_id / shape.y;
    int col_this = thread_id % shape.y;
    int row_other = thread_id / other.shape.y;
    int col_other = thread_id % other.shape.y;

    // OPTION I: normalised loop, with full symbolic index calculation
    // ~ 4180 GFLOPS
#pragma unroll 8
    for (int i = 0; i < total_elements / total_threads; i++)
      (data)[i * total_threads * stride.x / shape.y + row_this * stride.x
             + col_this]
        = ((other.data))[i * total_threads * other.stride.x / other.shape.y
                         + row_other * other.stride.x + col_other];

    // OPTION II: non-normalised loop, with full symbolic index calculation
    // ~ 1940 GFLOPS
    // for (int i = 0; i < total_elements; i+=total_threads)
    //   (data)[i * stride.x / shape.y + row_this * stride.x + col_this]
    //     = ((other.data))[i * other.stride.x / other.shape.y
    //                      + row_other * other.stride.x + col_other];

    // OPTION II: non-normalised loop, with full symbolic index calculation
    // ~ 2800 GFLOPS
    // for (int i = 0; i < total_elements; i+=32)
    //   (data)[i * stride.x / shape.y + row_this * stride.x + col_this]
    //     = ((other.data))[i * other.stride.x / other.shape.y
    //                      + row_other * other.stride.x + col_other];

    // OPTION II: non-normalised loop, with more-of-literal calculation
    // ~ 2790 GFLOPS
    //
    // for (int i = 0; i < 256; i+=32)
    //   (data)[i * 16 / shape.y + row_this * 16 + col_this]
    //     = ((other.data))[i * 4096 / other.shape.y
    //                      + row_other * 4096 + col_other];

    // if (total_threads * 4 < total_elements) {
    //   total_elements = total_elements / 4;
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     ((float4*)data)[i * stride.x / shape.y + row_this * stride.x +
    //     col_this]
    //       = ((float4*)(other.data))[i * other.stride.x / other.shape.y
    //                    + row_other * other.stride.x + col_other];
    // } else {
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     (data)[i * stride.x / shape.y + row_this * stride.x + col_this]
    //       = ((other.data))[i * other.stride.x / other.shape.y
    //                    + row_other * other.stride.x + col_other];
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
  inline __device__ void operator<<=(const Matrix &other)
  {
    int total_threads = blockDim.x;
    // int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.x * shape.y;
    int thread_id = (threadIdx.x + threadIdx.y * blockDim.x) % 32;
    // int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    int row_this = thread_id / shape.y;
    int col_this = thread_id % shape.y;
    int row_other = thread_id / other.shape.y;
    int col_other = thread_id % other.shape.y;
    // #pragma unroll
    for (int i = 0; i < total_elements; i += total_threads) {
      data[i * stride.x / shape.y + row_this * stride.x + col_this]
        = other.data[i * other.stride.x / other.shape.y
                     + row_other * other.stride.x + col_other];
    }
  }
  template <typename U = T>
  inline __device__
    typename std::enable_if<std::is_same<U, half>::value, void>::type
    operator<<=(const Matrix<float> &other)
  {
    int total_threads = blockDim.x;
    // int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.x * shape.y;
    int thread_id = (threadIdx.x + threadIdx.y * blockDim.x) % 32;
    int row_this = thread_id / shape.y;
    int col_this = thread_id % shape.y;
    int row_other = thread_id / other.shape.y;
    int col_other = thread_id % other.shape.y;
    // #pragma unroll
    for (int i = 0; i < total_elements; i += total_threads) {
      data[i * stride.x / shape.y + row_this * stride.x + col_this]
        = __float2half(other.data[i * other.stride.x / other.shape.y
                                  + row_other * other.stride.x + col_other]);
    }
  }

  __device__ void operator=(const Matrix<T> &other)
  {
    *data = *other.data;
  }

  __device__ void operator=(const T other_scalar)
  {
    *data = other_scalar;
  }

  // TODO: merge generics
  // half <= float
  template <typename U = T>
  __device__ typename std::enable_if<std::is_same<U, half>::value, void>::type
  operator=(const Matrix<float> &other)
  {
    *data = __float2half(*other.data);
  }

  // float <= half
  template <typename U = T>
  __device__ typename std::enable_if<std::is_same<U, float>::value, void>::type
  operator=(const Matrix<half> &other)
  {
    *data = __half2float(*other.data);
  }

  template <typename U = T>
  __device__ typename std::enable_if<std::is_same<U, half>::value, void>::type
  operator=(const float other_scalar)
  {
    *data = __float2half(other_scalar);
  }

  template <typename U = T>
  __device__ typename std::enable_if<std::is_same<U, float>::value, void>::type
  print() const
  {
    for (size_t i = 0; i < shape.x; ++i) {
      for (size_t j = 0; j < shape.y; ++j) {
        std::cout << data[i * shape.x + j * shape.y] << " ";
      }
      std::cout << "\n";
    }
    std::cout << std::endl;
  }
};

// helper for shared decl
// template<Coord shape>
template <int x, int y, typename T> __device__ Matrix<T> make_shared()
{
  static_assert(is_allowed_type<T>::value,
                "T must be one of the allowed types: float, half, or float4.");

  __shared__ T _data[x * y];
  return Matrix<T>(std::move(_data), Coord(x, y));
}

template <int x, int y, typename T> __device__ Matrix<T> make_local()
{
  static_assert(is_allowed_type<T>::value,
                "T must be one of the allowed types: float, half, or float4.");
  float _data[x * y] = {0.};
  // extern __shared__ float _data[];
  return Matrix<T>(_data, Coord(x, y));
}

// TODO: consider use std::integral_constant to make Coord known at compile-time
__device__ Coord &&make_coord(int x, int y)
{
  return std::move(Coord(x, y));
}

__device__ __forceinline__ int xor_swizzle(int o)
{
  return (o ^ ((o & (7 << 5)) >> 3));
}

} // namespace catz

#endif // CATZILLA_RECIPES_UTILS_INDEX_UTILS_H_
