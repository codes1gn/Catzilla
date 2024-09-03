#ifndef CATZILLA_RECIPES_UTILS_INDEX_UTILS_H_
#define CATZILLA_RECIPES_UTILS_INDEX_UTILS_H_

#include <tuple>
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>


// convert multi-dim index to flatten-index
// make it stream-style coding
//
// legacy:
// output[tiling_(bid_y, bid_x, M_TILE_SM, N_TILE_SM, N, 1) + distribute_(tid_x*N_TILE_REG, tid_y*M_TILE_REG, 1, N) + distribute_(m_reg, n_reg, N, 1)]
//
// novel:
// lhs.tiling_(xxx).distribute_(xx)
// rhs.tiling_(xxx).tiling_(xxx).distribute_(xx)
inline __device__ int distribute_(int x_offset, int y_offset, int x_stride, int y_stride) {
  int idx = x_offset * x_stride + y_offset * y_stride;
  return idx;
}

// convert multi-dim index of a tile-block to flatten-index
inline __device__ int tiling_(int tile_id_x, int tile_id_y, int to_stride_x, int to_stride_y, int from_stride_x, int from_stride_y) {
  int idx = tile_id_x * to_stride_x * from_stride_x + tile_id_y * to_stride_y * from_stride_y;
  return idx;
}


struct Coord {
  int x;
  int y;

  // 构造函数
  __device__ Coord(int x, int y) : x(x), y(y) {}

  // 从 std::tuple<int, int> 构造
  __device__ Coord(const std::tuple<int, int>& t) : x(std::get<0>(t)), y(std::get<1>(t)) {}

  // 将 Coord 转换为 std::tuple<int, int>
  __device__ std::tuple<int, int> to_tuple() const {
    return std::make_tuple(x, y);
  }
};

struct Matrix {
  float* data;
  Coord size;
  bool row_major;

  __device__ Matrix(float* data, Coord size, bool is_row_major=true) : 
    data(data), 
    size(size), 
    row_major(is_row_major) {}

  __device__ Matrix distribute(Coord id, Coord stride) {
    data += id.x * stride.x + id.y * stride.y;
    return *this;
  }

  __device__ Matrix tile(Coord id, Coord from_stride, Coord to_stride) {
    data += id.x * from_stride.x * to_stride.x + id.y * from_stride.y * to_stride.y;
    return *this;
  }

  __device__ Matrix& operator=(const Matrix& other) {
    data[0] = other.data[0];
  }

  __device__ Matrix& operator=(const float other_scalar) {
    data[0] = other_scalar;
  }

  __device__ bool is_row_major() {
    return row_major;
  }

  __device__ bool is_col_major() {
    return !row_major;
  }

  __device__ void print() const {
    if (row_major) {
      for (size_t i = 0; i < size.x; ++i) {
        for (size_t j = 0; j < size.y; ++j) {
          std::cout << data[i * size.x + j * size.y] << " ";
        }
        std::cout << "\n";
      }
    }
    else {
      for (size_t j = 0; j < size.y; ++j) {
        for (size_t i = 0; i < size.x; ++i) {
          std::cout << data[i * size.x + j * size.y] << " ";
        }
        std::cout << "\n";
      }
    }
    std::cout << std::endl;
  }
};

#endif // CATZILLA_RECIPES_UTILS_INDEX_UTILS_H_
