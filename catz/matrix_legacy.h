#ifndef CATZILLA_CATZ_MATRIX_LEGACY_H_
#define CATZILLA_CATZ_MATRIX_LEGACY_H_

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

#include "coord.h"
#include "coord_legacy.h"
#include "cuda_utils.h"
#include "macro.h"

// TODO: rename to matrix.h
namespace catz {

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

template <typename T, typename CoordType, typename CoordType2>
struct MatrixNightly;

template <typename T, int ROWS, int COLS, int STRD>
struct MatrixNightly<T, CoordSt<ROWS, COLS>, CoordSt<STRD, 1>> {
  static_assert(is_allowed_type<T>::value,
                "T must be one of the allowed types: float, half, or float4.");
  T *data;
  CoordSt<ROWS, COLS> shape;
  CoordSt<STRD, 1> stride;

  // constructor
  constexpr __device__ MatrixNightly(T *data, CoordSt<ROWS, COLS> shape,
                                     CoordSt<STRD, 1> stride)
      : data(data), shape(shape), stride(stride) {}

  // copy
  constexpr __device__ MatrixNightly(const MatrixNightly &other)
      : data(other.data), shape(other.shape), stride(other.stride) {}

  // move
  constexpr __device__ MatrixNightly(MatrixNightly &&other) noexcept
      : data(other.data), shape(other.shape), stride(other.stride) {
    other.data = nullptr;
  }

  ///////////////////////////////////////////////////////////
  /// dataflow related methods
  ///////////////////////////////////////////////////////////

  // TODO: make it device'd code
  template <int NROWS, int NCOLS>
  constexpr inline __device__
      MatrixNightly<T, CoordSt<NROWS, NCOLS>, CoordSt<STRD, 1>>
      tile(CoordDyn tile_var, CoordSt<NROWS, NCOLS> new_shape) {
    T *new_data = data + tile_var.rows * new_shape.rows * stride.rows +
                  tile_var.cols * new_shape.cols * stride.cols;
    MatrixNightly<T, CoordSt<NROWS, NCOLS>, CoordSt<STRD, 1>> ret =
        make_matrix_nightly(new_data, new_shape, stride);
    return std::move(ret);
  }

  constexpr inline __device__ MatrixNightly dist_to(CoordDyn tile_var) {
    // shape = 8 x 2
    // y in 0-4, x in 0-4
    MatrixNightly ret = MatrixNightly(data + tile_var.rows * stride.rows +
                                          tile_var.cols * stride.cols,
                                      shape, stride);
    return std::move(ret);
  }

  constexpr inline __device__ MatrixNightly dist_to(int dist_id) {
    int row_in_current = dist_id / shape.cols;
    int col_in_current = dist_id % shape.cols;
    MatrixNightly ret = MatrixNightly(data + row_in_current * stride.rows +
                                          col_in_current * stride.cols,
                                      shape, stride);
    return std::move(ret);
  }

  // distribute the following NUM_THREADS elements to all threads
  constexpr inline __device__ MatrixNightly dist_to_thread() {
    // NOTE: we can have three design
    // 1. use all threads to spread
    // 2. use x threads to spread
    // 3. use a wrap to spread.
    // we prefer to take third option in currently design
    int lane_id = threadIdx.y * blockDim.x + threadIdx.x;
    int row_in_current = lane_id / shape.cols;
    int col_in_current = lane_id % shape.cols;
    MatrixNightly ret = MatrixNightly(data + row_in_current * stride.rows +
                                          col_in_current * stride.cols,
                                      shape, stride);
    return std::move(ret);
  }

  // very specialised API, only consider using this for wrap-level
  // distribute the following 32 elements to a wrap
  constexpr inline __device__ MatrixNightly dist_to_wrap() {
    // NOTE: we can have three design
    // 1. use all threads to spread
    // 2. use x threads to spread
    // 3. use a wrap to spread.
    // we prefer to take third option in currently design
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    int row_in_current = lane_id / shape.cols;
    int col_in_current = lane_id % shape.cols;
    MatrixNightly ret = MatrixNightly(data + row_in_current * stride.rows +
                                          col_in_current * stride.cols,
                                      shape, stride);
    return std::move(ret);
  }

  ///////////////////////////////////////////////////////////
  /// tensor-core related methods
  ///////////////////////////////////////////////////////////

  template <typename U = T>
  inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      load_fragments(unsigned *loader) {
    // TODO: add loader length check
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    int lane_rank = lane_id % shape.rows;
    int lane_group = lane_id / shape.rows;
    const half *data_ptr = data + lane_rank * stride.rows + lane_group * 8;
    if (shape.rows == 16 && shape.cols == 8) {
      // LHS X2
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                   : "=r"(loader[0]), "=r"(loader[1])
                   : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.rows == 16 && shape.cols == 16) {
      // RHS X1
      asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
          : "=r"(loader[0]), "=r"(loader[1]), "=r"(loader[2]), "=r"(loader[3])
          : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.rows == 8 && shape.cols == 8) {
      // LHS X4
      asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                   : "=r"(loader[0])
                   : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.rows == 8 && shape.cols == 16) {
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
    if (shape.rows == 16 && shape.cols == 8) {
      loader[0] = data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2];
      loader[1] = data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2 + 1];
      loader[2] = data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2 +
                       8 * stride.rows];
      loader[3] = data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2 +
                       8 * stride.rows + 1];
    }
  }

  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, float>::value, void>::type
      store_fragments_c(float *storer) {
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    if (shape.rows == 16 && shape.cols == 8) {
      data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2] = storer[0];
      data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2 + 1] = storer[1];
      data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2 + 8 * stride.rows] =
          storer[2];
      data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2 + 8 * stride.rows +
           1] = storer[3];
    }
  }

  ///////////////////////////////////////////////////////////
  /// operators
  ///////////////////////////////////////////////////////////

  constexpr __device__ void operator=(const MatrixNightly &other) {
    *data = *other.data;
  }

  constexpr __device__ void operator=(const T other_scalar) {
    *data = other_scalar;
  }

  // TODO: merge generics
  // half <= float
  template <typename U = T, typename CoordType, typename CoordType2>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      operator=(const MatrixNightly<float, CoordType, CoordType2> &other) {
    *data = __float2half(*other.data);
  }

  // float <= half
  template <typename U = T, typename CoordType, typename CoordType2>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, float>::value, void>::type
      operator=(const MatrixNightly<half, CoordType, CoordType2> &other) {
    *data = __half2float(*other.data);
  }

  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      operator=(const float other_scalar) {
    *data = __float2half(other_scalar);
  }

  // operator '<=' is syntax sugar that combines dist-to-threads and '='
  // operator
  constexpr inline __device__ void operator<=(const MatrixNightly &other) {
    // can make 11352
    int total_threads = blockDim.x * blockDim.y;
    // int total_threads = 256;
    int total_elements = shape.rows * shape.cols;
    // int total_elements = 128 * 32;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    int row_this = thread_id / shape.cols;
    int col_this = thread_id % shape.cols;
    int row_other = thread_id / other.shape.cols;
    int col_other = thread_id % other.shape.cols;

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
      (data)[i * total_threads * stride.rows / shape.cols +
             row_this * stride.rows + col_this] =
          ((other.data))[i * total_threads * other.stride.rows /
                             other.shape.cols +
                         row_other * other.stride.rows + col_other];
    // if (total_threads * 4 < total_elements) {
    //   total_elements = total_elements / 4;
    //   int row_this = thread_id / (shape.cols / 4);
    //   int col_this = thread_id % (shape.cols / 4);
    //   int row_other = thread_id / (other.shape.cols / 4);
    //   int col_other = thread_id % (other.shape.cols / 4);
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     ((float4*)data)[i * stride.rows / shape.cols + row_this *
    //     stride.rows + col_this]
    //       = ((float4*)(other.data))[i * other.stride.rows /
    //       other.shape.cols
    //                    + row_other * other.stride.rows / 4 + col_other];
    // } else {
    //   int row_this = thread_id / shape.cols;
    //   int col_this = thread_id % shape.cols;
    //   int row_other = thread_id / other.shape.cols;
    //   int col_other = thread_id % other.shape.cols;
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     (data)[i * stride.rows / shape.cols + row_this * stride.rows +
    //     col_this]
    //       = ((other.data))[i * other.stride.rows / other.shape.cols
    //                    + row_other * other.stride.rows + col_other];
    // }
  }

  //   for (int m = 0; m < 8/* CEIL_DIV(M_TILE, 2) */; m++) {
  //     lhs_shared_mat.tile(Coord(m, kin), Coord(2, 16)).dist_to_thread()
  //       = lhs_mat.tile(Coord(blockIdx.y, ko), lhs_sm_tile_shape)
  //           .tile(Coord(m, kin), Coord(2, 16))
  //           .dist_to_thread();
  //   }
  template <typename U = T, typename CoordType, typename CoordType2>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      operator<=(const MatrixNightly<float, CoordType, CoordType2> &other) {
    int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.rows * shape.cols;
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int row_this = thread_id / shape.cols;
    int col_this = thread_id % shape.cols;
    int row_other = thread_id / other.shape.cols;
    int col_other = thread_id % other.shape.cols;

    // OPTION I: normalised loop, with full symbolic index calculation
    // ~ 4180 GFLOPS
#pragma unroll 8
    for (int i = 0; i < total_elements / total_threads; i++)
      (data)[i * total_threads * stride.rows / shape.cols +
             row_this * stride.rows + col_this] =
          ((other.data))[i * total_threads * other.stride.rows /
                             other.shape.cols +
                         row_other * other.stride.rows + col_other];

    // OPTION II: non-normalised loop, with full symbolic index calculation
    // ~ 1940 GFLOPS
    // for (int i = 0; i < total_elements; i+=total_threads)
    //   (data)[i * stride.rows / shape.cols + row_this * stride.rows +
    //   col_this]
    //     = ((other.data))[i * other.stride.rows / other.shape.cols
    //                      + row_other * other.stride.rows + col_other];

    // OPTION II: non-normalised loop, with full symbolic index calculation
    // ~ 2800 GFLOPS
    // for (int i = 0; i < total_elements; i+=32)
    //   (data)[i * stride.rows / shape.cols + row_this * stride.rows +
    //   col_this]
    //     = ((other.data))[i * other.stride.rows / other.shape.cols
    //                      + row_other * other.stride.rows + col_other];

    // OPTION II: non-normalised loop, with more-of-literal calculation
    // ~ 2790 GFLOPS
    //
    // for (int i = 0; i < 256; i+=32)
    //   (data)[i * 16 / shape.cols + row_this * 16 + col_this]
    //     = ((other.data))[i * 4096 / other.shape.cols
    //                      + row_other * 4096 + col_other];

    // if (total_threads * 4 < total_elements) {
    //   total_elements = total_elements / 4;
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     ((float4*)data)[i * stride.rows / shape.cols + row_this *
    //     stride.rows + col_this]
    //       = ((float4*)(other.data))[i * other.stride.rows /
    //       other.shape.cols
    //                    + row_other * other.stride.rows + col_other];
    // } else {
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     (data)[i * stride.rows / shape.cols + row_this * stride.rows +
    //     col_this]
    //       = ((other.data))[i * other.stride.rows / other.shape.cols
    //                    + row_other * other.stride.rows + col_other];
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
  constexpr inline __device__ void operator<<=(const MatrixNightly &other) {
    int total_threads = blockDim.x;
    // int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.rows * shape.cols;
    int thread_id = (threadIdx.x + threadIdx.y * blockDim.x) % 32;
    // int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    int row_this = thread_id / shape.cols;
    int col_this = thread_id % shape.cols;
    int row_other = thread_id / other.shape.cols;
    int col_other = thread_id % other.shape.cols;
    // #pragma unroll
    for (int i = 0; i < total_elements; i += total_threads) {
      data[i * stride.rows / shape.cols + row_this * stride.rows + col_this] =
          other.data[i * other.stride.rows / other.shape.cols +
                     row_other * other.stride.rows + col_other];
    }
  }

  template <typename U = T, typename CoordType, typename CoordType2>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      operator<<=(const MatrixNightly<float, CoordType, CoordType2> &other) {
    int total_threads = blockDim.x;
    // int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.rows * shape.cols;
    int thread_id = (threadIdx.x + threadIdx.y * blockDim.x) % 32;
    int row_this = thread_id / shape.cols;
    int col_this = thread_id % shape.cols;
    int row_other = thread_id / other.shape.cols;
    int col_other = thread_id % other.shape.cols;
    // #pragma unroll
    for (int i = 0; i < total_elements; i += total_threads) {
      data[i * stride.rows / shape.cols + row_this * stride.rows + col_this] =
          __float2half(other.data[i * other.stride.rows / other.shape.cols +
                                  row_other * other.stride.rows + col_other]);
    }
  }

  ///////////////////////////////////////////////////////////
  /// utils methods
  ///////////////////////////////////////////////////////////

  inline __device__ MatrixNightly next() {
    data += 1;
    return *this;
  }

  inline __device__ void fill(T value) {
    int flat_id = threadIdx.y * blockDim.x + threadIdx.x;
    int row_in_current = flat_id / shape.cols;
    int col_in_current = flat_id % shape.cols;
    int elements = shape.rows * shape.cols;
    int threads = blockDim.x * blockDim.y;
    for (int chunk = 0; chunk < CEIL_DIV(elements, threads); chunk++) {
      data[chunk * threads + flat_id] = value;
    }
  }

  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, float>::value, void>::type
      print() const {
    for (size_t i = 0; i < shape.rows; ++i) {
      for (size_t j = 0; j < shape.cols; ++j) {
        std::cout << data[i * shape.rows + j * shape.cols] << " ";
      }
      std::cout << "\n";
    }
    std::cout << std::endl;
  }
};

// Default builder:
// (T*, CoordType, CoordType) -> Matrix
// auto matrix = make_matrix(data, CoordSt<16, 32>(), CoordSt<32, 1>())
template <typename T, typename CoordType, typename CoordType2>
__device__ MatrixNightly<T, CoordType, CoordType2>
make_matrix_nightly(T *data, CoordType shape, CoordType2 stride) {
  return MatrixNightly<T, CoordType, CoordType2>(data, shape, stride);
}

// Specialised builder for contiguous memory
// (T*, CoordType) -> Matrix
template <typename T, int ROWS, int COLS>
__device__ MatrixNightly<T, CoordSt<ROWS, COLS>, CoordSt<COLS, 1>>
make_matrix_nightly(T *data, CoordSt<ROWS, COLS> shape) {
  return MatrixNightly<T, CoordSt<ROWS, COLS>, CoordSt<COLS, 1>>(
      data, shape, CoordSt<COLS, 1>());
}

template <typename T>
struct MatrixDyn {
  static_assert(is_allowed_type<T>::value,
                "T must be one of the allowed types: float, half, or float4.");

  // A, 16x32, 32x4
  T *data;
  CoordDyn shape;
  CoordDyn stride;

  constexpr __device__ MatrixDyn(T *data, CoordDyn shape)
      : data(data), shape(shape), stride(CoordDyn(shape.cols, 1)) {}

  constexpr __device__ MatrixDyn(T *data, CoordDyn shape, CoordDyn stride)
      : data(data), shape(shape), stride(stride) {}

  // copy
  constexpr __device__ MatrixDyn(const MatrixDyn &other)
      : data(other.data), shape(other.shape), stride(other.stride) {}

  // move
  constexpr __device__ MatrixDyn(MatrixDyn &&other) noexcept
      : data(other.data), shape(other.shape), stride(other.stride) {
    other.data = nullptr;
  }

  __device__ MatrixDyn next() {
    data += 1;
    return *this;
  }

  constexpr inline __device__ void fill(T value) {
    int flat_id = threadIdx.y * blockDim.x + threadIdx.x;
    int row_in_current = flat_id / shape.cols;
    int col_in_current = flat_id % shape.cols;
    int elements = shape.rows * shape.cols;
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
    int lane_rank = lane_id % shape.rows;
    int lane_group = lane_id / shape.rows;
    const half *data_ptr = data + lane_rank * stride.rows + lane_group * 8;
    if (shape.rows == 16 && shape.cols == 8) {
      // LHS X2
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                   : "=r"(loader[0]), "=r"(loader[1])
                   : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.rows == 16 && shape.cols == 16) {
      // RHS X1
      asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
          : "=r"(loader[0]), "=r"(loader[1]), "=r"(loader[2]), "=r"(loader[3])
          : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.rows == 8 && shape.cols == 8) {
      // LHS X4
      asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                   : "=r"(loader[0])
                   : "r"(get_smem_ptr(data_ptr)));
    } else if (shape.rows == 8 && shape.cols == 16) {
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
    if (shape.rows == 16 && shape.cols == 8) {
      loader[0] = data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2];
      loader[1] = data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2 + 1];
      loader[2] = data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2 +
                       8 * stride.rows];
      loader[3] = data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2 +
                       8 * stride.rows + 1];
    }
  }

  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, float>::value, void>::type
      store_fragments_c(float *storer) {
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    if (shape.rows == 16 && shape.cols == 8) {
      data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2] = storer[0];
      data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2 + 1] = storer[1];
      data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2 + 8 * stride.rows] =
          storer[2];
      data[(lane_id / 4) * stride.rows + (lane_id % 4) * 2 + 8 * stride.rows +
           1] = storer[3];
    }
  }

  constexpr inline __device__ MatrixDyn tile(CoordDyn tile_var,
                                             CoordDyn other_shape) {
    MatrixDyn ret =
        MatrixDyn(data + tile_var.rows * other_shape.rows * stride.rows +
                      tile_var.cols * other_shape.cols * stride.cols,
                  other_shape, stride);
    return std::move(ret);
  }

  constexpr inline __device__ MatrixDyn dist_to(CoordDyn tile_var) {
    // shape = 8 x 2
    // y in 0-4, x in 0-4
    MatrixDyn ret = MatrixDyn(data + tile_var.rows * stride.rows +
                                  tile_var.cols * stride.cols,
                              shape, stride);
    return std::move(ret);
  }

  constexpr inline __device__ MatrixDyn dist_to(int dist_id) {
    int row_in_current = dist_id / shape.cols;
    int col_in_current = dist_id % shape.cols;
    MatrixDyn ret = MatrixDyn(data + row_in_current * stride.rows +
                                  col_in_current * stride.cols,
                              shape, stride);
    return std::move(ret);
  }

  // distribute the following 32 elements to a wrap
  constexpr inline __device__ MatrixDyn dist_to_wrap() {
    // NOTE: we can have three design
    // 1. use all threads to spread
    // 2. use x threads to spread
    // 3. use a wrap to spread.
    // we prefer to take third option in currently design
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    int row_in_current = lane_id / shape.cols;
    int col_in_current = lane_id % shape.cols;
    MatrixDyn ret = MatrixDyn(data + row_in_current * stride.rows +
                                  col_in_current * stride.cols,
                              shape, stride);
    return std::move(ret);
  }

  // distribute the following NUM_THREADS elements to all threads
  constexpr inline __device__ MatrixDyn dist_to_thread() {
    // NOTE: we can have three design
    // 1. use all threads to spread
    // 2. use x threads to spread
    // 3. use a wrap to spread.
    // we prefer to take third option in currently design
    int lane_id = threadIdx.y * blockDim.x + threadIdx.x;
    int row_in_current = lane_id / shape.cols;
    int col_in_current = lane_id % shape.cols;
    MatrixDyn ret = MatrixDyn(data + row_in_current * stride.rows +
                                  col_in_current * stride.cols,
                              shape, stride);
    return std::move(ret);
  }

  // operator '<=' is syntax sugar that combines dist-to-threads and '='
  // operator
  constexpr inline __device__ void operator<=(const MatrixDyn &other) {
    // can make 11352
    int total_threads = blockDim.x * blockDim.y;
    // int total_threads = 256;
    int total_elements = shape.rows * shape.cols;
    // int total_elements = 128 * 32;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    int row_this = thread_id / shape.cols;
    int col_this = thread_id % shape.cols;
    int row_other = thread_id / other.shape.cols;
    int col_other = thread_id % other.shape.cols;

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
      (data)[i * total_threads * stride.rows / shape.cols +
             row_this * stride.rows + col_this] =
          ((other.data))[i * total_threads * other.stride.rows /
                             other.shape.cols +
                         row_other * other.stride.rows + col_other];
    // if (total_threads * 4 < total_elements) {
    //   total_elements = total_elements / 4;
    //   int row_this = thread_id / (shape.cols / 4);
    //   int col_this = thread_id % (shape.cols / 4);
    //   int row_other = thread_id / (other.shape.cols / 4);
    //   int col_other = thread_id % (other.shape.cols / 4);
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     ((float4*)data)[i * stride.rows / shape.cols + row_this *
    //     stride.rows + col_this]
    //       = ((float4*)(other.data))[i * other.stride.rows /
    //       other.shape.cols
    //                    + row_other * other.stride.rows / 4 + col_other];
    // } else {
    //   int row_this = thread_id / shape.cols;
    //   int col_this = thread_id % shape.cols;
    //   int row_other = thread_id / other.shape.cols;
    //   int col_other = thread_id % other.shape.cols;
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     (data)[i * stride.rows / shape.cols + row_this * stride.rows +
    //     col_this]
    //       = ((other.data))[i * other.stride.rows / other.shape.cols
    //                    + row_other * other.stride.rows + col_other];
    // }
  }

  //   for (int m = 0; m < 8/* CEIL_DIV(M_TILE, 2) */; m++) {
  //     lhs_shared_mat.tile(CoordDyn(m, kin), CoordDyn(2, 16)).dist_to_thread()
  //       = lhs_mat.tile(CoordDyn(blockIdx.y, ko), lhs_sm_tile_shape)
  //           .tile(CoordDyn(m, kin), CoordDyn(2, 16))
  //           .dist_to_thread();
  //   }
  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      operator<=(const MatrixDyn<float> &other) {
    int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.rows * shape.cols;
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int row_this = thread_id / shape.cols;
    int col_this = thread_id % shape.cols;
    int row_other = thread_id / other.shape.cols;
    int col_other = thread_id % other.shape.cols;

    // OPTION I: normalised loop, with full symbolic index calculation
    // ~ 4180 GFLOPS
#pragma unroll 8
    for (int i = 0; i < total_elements / total_threads; i++)
      (data)[i * total_threads * stride.rows / shape.cols +
             row_this * stride.rows + col_this] =
          ((other.data))[i * total_threads * other.stride.rows /
                             other.shape.cols +
                         row_other * other.stride.rows + col_other];

    // OPTION II: non-normalised loop, with full symbolic index calculation
    // ~ 1940 GFLOPS
    // for (int i = 0; i < total_elements; i+=total_threads)
    //   (data)[i * stride.rows / shape.cols + row_this * stride.rows +
    //   col_this]
    //     = ((other.data))[i * other.stride.rows / other.shape.cols
    //                      + row_other * other.stride.rows + col_other];

    // OPTION II: non-normalised loop, with full symbolic index calculation
    // ~ 2800 GFLOPS
    // for (int i = 0; i < total_elements; i+=32)
    //   (data)[i * stride.rows / shape.cols + row_this * stride.rows +
    //   col_this]
    //     = ((other.data))[i * other.stride.rows / other.shape.cols
    //                      + row_other * other.stride.rows + col_other];

    // OPTION II: non-normalised loop, with more-of-literal calculation
    // ~ 2790 GFLOPS
    //
    // for (int i = 0; i < 256; i+=32)
    //   (data)[i * 16 / shape.cols + row_this * 16 + col_this]
    //     = ((other.data))[i * 4096 / other.shape.cols
    //                      + row_other * 4096 + col_other];

    // if (total_threads * 4 < total_elements) {
    //   total_elements = total_elements / 4;
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     ((float4*)data)[i * stride.rows / shape.cols + row_this *
    //     stride.rows + col_this]
    //       = ((float4*)(other.data))[i * other.stride.rows /
    //       other.shape.cols
    //                    + row_other * other.stride.rows + col_other];
    // } else {
    //   for (int i = 0; i < total_elements; i += total_threads)
    //     (data)[i * stride.rows / shape.cols + row_this * stride.rows +
    //     col_this]
    //       = ((other.data))[i * other.stride.rows / other.shape.cols
    //                    + row_other * other.stride.rows + col_other];
    // }
  }

  // TODO: merge it
  // use threads in one wrap to move it,
  // replace m with threadIdx.y
  //
  // for (int n = 0; n < CEIL_DIV(N_TILE, N_REG); n++) {
  //   out_mat.tile(CoordDyn(blockIdx.x, blockIdx.y), out_sm_tile_shape)
  //       .tile(CoordDyn(m, n), out_reg_tile_shape)
  //     <<= out_shared_mat.tile(CoordDyn(m, n), out_reg_tile_shape);
  // }
  // special operator that allows any same-volume copy, when the volume is
  // dividable by threads volume
  constexpr inline __device__ void operator<<=(const MatrixDyn &other) {
    int total_threads = blockDim.x;
    // int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.rows * shape.cols;
    int thread_id = (threadIdx.x + threadIdx.y * blockDim.x) % 32;
    // int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    int row_this = thread_id / shape.cols;
    int col_this = thread_id % shape.cols;
    int row_other = thread_id / other.shape.cols;
    int col_other = thread_id % other.shape.cols;
    // #pragma unroll
    for (int i = 0; i < total_elements; i += total_threads) {
      data[i * stride.rows / shape.cols + row_this * stride.rows + col_this] =
          other.data[i * other.stride.rows / other.shape.cols +
                     row_other * other.stride.rows + col_other];
    }
  }

  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, half>::value, void>::type
      operator<<=(const MatrixDyn<float> &other) {
    int total_threads = blockDim.x;
    // int total_threads = blockDim.x * blockDim.y;
    int total_elements = shape.rows * shape.cols;
    int thread_id = (threadIdx.x + threadIdx.y * blockDim.x) % 32;
    int row_this = thread_id / shape.cols;
    int col_this = thread_id % shape.cols;
    int row_other = thread_id / other.shape.cols;
    int col_other = thread_id % other.shape.cols;
    // #pragma unroll
    for (int i = 0; i < total_elements; i += total_threads) {
      data[i * stride.rows / shape.cols + row_this * stride.rows + col_this] =
          __float2half(other.data[i * other.stride.rows / other.shape.cols +
                                  row_other * other.stride.rows + col_other]);
    }
  }

  constexpr __device__ void operator=(const MatrixDyn<T> &other) {
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
      operator=(const MatrixDyn<float> &other) {
    *data = __float2half(*other.data);
  }

  // float <= half
  template <typename U = T>
  constexpr inline __device__
      typename std::enable_if<std::is_same<U, float>::value, void>::type
      operator=(const MatrixDyn<half> &other) {
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
    for (size_t i = 0; i < shape.rows; ++i) {
      for (size_t j = 0; j < shape.cols; ++j) {
        std::cout << data[i * shape.rows + j * shape.cols] << " ";
      }
      std::cout << "\n";
    }
    std::cout << std::endl;
  }
};

// helper for shared decl
template <int x, int y, typename T>
__device__ MatrixDyn<T> make_shared() {
  static_assert(is_allowed_type<T>::value,
                "T must be one of the allowed types: float, half, or float4.");

  __shared__ T _data[x * y];
  return MatrixDyn<T>(std::move(_data), CoordDyn(x, y));
}

// template <typename T, typename RowType, typename ColType>
// __device__ auto make_shared_test(const CoordS<RowType, ColType>& shape) {
//   static_assert(is_allowed_type<T>::value,
//                 "T must be one of the allowed types: float, half, or
//                 float4.");
//
//   __shared__ T _data[shape.volume()];
//   return make_matrix(std::move(_data), shape);
// }

template <int x, int y, typename T>
__device__ MatrixDyn<T> make_local() {
  static_assert(is_allowed_type<T>::value,
                "T must be one of the allowed types: float, half, or float4.");
  float _data[x * y] = {0.};
  // extern __shared__ float _data[];
  return MatrixDyn<T>(_data, CoordDyn(x, y));
}

template <typename T, typename ShapeType>
__device__ auto make_local_test(const ShapeType &&shape) {
  static_assert(is_allowed_type<T>::value,
                "T must be one of the allowed types: float, half, or float4.");
  T _data[shape.volume()] = {0.};
  return make_matrix(_data, shape);
}

__device__ __forceinline__ int xor_swizzle(int o) {
  return (o ^ ((o & (7 << 5)) >> 3));
}

} // namespace catz

#endif // CATZILLA_CATZ_MATRIX_LEGACY_H_
