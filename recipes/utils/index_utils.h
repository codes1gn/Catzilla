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

  __device__ Coord& xor_swizzle() {
    auto i16 = (y*32 + x) * sizeof(float) / 16;
    auto y16 = i16 / 8;
    auto x16 = i16 % 8;
    auto x16_swz = y16 ^ x16;
    auto x_swz = x16_swz * 16 / sizeof(float) % 32 + x % (16 / sizeof(float));
    x = x_swz % 32;
    y = x_swz / 32;
    return *this;
  }
  
};

// keep in mind, always row major
struct Matrix {
  float* data;
  Coord shape;
  Coord stride;

  __device__ Matrix(float* data, Coord shape) : 
    data(data), 
    shape(shape),
    stride(Coord(shape.y, 1)){
  }


  __device__ Matrix(float* data, Coord shape, Coord stride) : 
    data(data), 
    shape(shape),
    stride(stride) {}

  __device__ Matrix(const Matrix& other) : data(other.data), shape(other.shape), stride(other.stride) {}

  __device__ Matrix(Matrix&& other) noexcept : data(other.data), shape(other.shape), stride(other.stride) {
    other.data = nullptr;
  }

  __device__ Matrix inc(int value) {
    data += value;
    return *this;
  }

  __device__ Matrix distribute(Coord id, Coord stride) {
    Matrix ret = Matrix(data + id.x * stride.x + id.y * stride.y, shape);
    return std::move(ret);
  }

  __device__ Matrix tile(Coord id, Coord from_stride, Coord to_stride) {
    Matrix ret = Matrix(data + id.x * from_stride.x * to_stride.x + id.y * from_stride.y * to_stride.y, shape);
    return std::move(ret);
  }

  __device__ Matrix tile_ex(Coord tile_var, Coord other_shape) {
    Matrix ret = Matrix(data + tile_var.x * other_shape.x * stride.x + tile_var.y * other_shape.y * stride.y, other_shape, stride);
    return std::move(ret);
  }

  __device__ Matrix dist_ex(Coord tile_var) {
    Matrix ret = Matrix(data + tile_var.x * stride.x + tile_var.y * stride.y, shape, stride);
    return std::move(ret);
  }

  __device__ void operator=(const Matrix& other) {
    *data = *(other.data);
  }

  __device__ void operator=(const float other_scalar) {
    *data = other_scalar;
  }

  __device__ void print() const {
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
__device__ Matrix& make_shared(Coord shape) {
  __shared__ float _data[shape.x * shape.y];
  data = std::move(_data);
  return Matrix(std::move(_data), shape)
}

#endif // CATZILLA_RECIPES_UTILS_INDEX_UTILS_H_
