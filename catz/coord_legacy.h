#ifndef CATZILLA_CATZ_COORD_LEGACY_H_
#define CATZILLA_CATZ_COORD_LEGACY_H_

#include <string>
#include <type_traits>

#include "index.h"
#include "trait.h"

namespace catz {

template <int ROWS, int COLS>
struct CoordSt {
  static constexpr int rows = ROWS;
  static constexpr int cols = COLS;

  // 构造函数
  constexpr __device__ CoordSt() {}

  //////////////////////////////////////
  /// utils
  //////////////////////////////////////

  std::string str() const {
    char buffer[20]; // 提供足够大的缓冲区
    std::sprintf(buffer, "CoordSt<%d, %d>", rows, cols);
    return std::string(buffer);
  }

  constexpr bool isStatic() { return true; }
  constexpr bool isDynamic() { return false; }

  //////////////////////////////////////
  /// operators
  //////////////////////////////////////

  // template <int orows, int ocols>
  // constexpr CoordSt<rows + orows, cols + ocols>
  // operator+(const CoordSt<orows, ocols> &other) const {
  //   return CoordSt<rows + other.rows, cols + other.cols>();
  // }

  // template <int orows, int ocols>
  // constexpr CoordSt<rows - orows, cols - ocols>
  // operator-(const CoordSt<orows, ocols> &other) const {
  //   return CoordSt<rows - other.rows, cols - other.cols>();
  // }
  //
  // template <int orows, int ocols>
  // constexpr CoordSt<rows * orows, cols * ocols>
  // operator*(const CoordSt<orows, ocols> &other) const {
  //   return CoordSt<rows * other.rows, cols * other.cols>();
  // }
  //
  // template <int orows, int ocols>
  // constexpr CoordSt<rows / orows, cols / ocols>
  // operator/(const CoordSt<orows, ocols> &other) const {
  //   static_assert(cols != 0 && other.cols != 0, "Division by zero");
  //   return CoordSt<rows / other.rows, cols / other.cols>();
  // }
  //
  // constexpr __device__ CoordSt &xor_swizzle() {
  //   auto i16 = (rows * 32 + cols) * sizeof(float) / 16;
  //   auto y16 = i16 / 8;
  //   auto x16 = i16 % 8;
  //   auto x16_swz = y16 ^ x16;
  //   auto x_swz =
  //       x16_swz * 16 / sizeof(float) % 32 + cols % (16 / sizeof(float));
  //   cols = x_swz % 32;
  //   rows = x_swz / 32;
  //   return *this;
  // }
  //
  // constexpr __device__ CoordSt &xor_swizzle_col() {
  //   constexpr auto i = (rows * 16 + cols);
  //   constexpr auto i_swz = i ^ rows;
  //   // auto _cols = i_swz % 16;
  //   // auto _rows = i_swz / 16;
  //   // return *this;
  //   return CoordSt<(i_swz / 16), (i_swz % 16)>();
  // }
  //
  // constexpr __device__ CoordSt &xor_swizzle_row() {
  //   constexpr auto i = (rows * 16 + cols);
  //   constexpr auto i_swz = i ^ cols;
  //   // cols = i_swz % 16;
  //   // rows = i_swz / 16;
  //   // return *this;
  //   return CoordSt<(i_swz / 16), (i_swz % 16)>();
  // }
};

// template<int X, int Y>
struct CoordDyn {
  int rows;
  int cols;

  // 构造函数
  constexpr __device__ CoordDyn(int rows, int cols) : rows(rows), cols(cols) {}

  //////////////////////////////////////
  /// utils
  //////////////////////////////////////

  std::string str() const {
    char buffer[20]; // 提供足够大的缓冲区
    std::sprintf(buffer, "CoordDyn<%d, %d>", rows, cols);
    return std::string(buffer);
  }

  constexpr bool isStatic() { return false; }
  constexpr bool isDynamic() { return true; }

  //////////////////////////////////////
  /// operators
  //////////////////////////////////////

  // constexpr CoordDyn operator+(const CoordDyn &other) const {
  //   return CoordDyn(rows + other.rows, cols + other.cols);
  // }

  // constexpr CoordDyn operator-(const CoordDyn &other) const {
  //   return CoordDyn(rows - other.rows, cols - other.cols);
  // }
  //
  // constexpr CoordDyn operator*(const CoordDyn &other) const {
  //   return CoordDyn(rows * other.rows, cols * other.cols);
  // }
  //
  // constexpr CoordDyn operator/(const CoordDyn &other) const {
  //   return CoordDyn(rows / other.rows, cols / other.cols);
  // }
  //
  // constexpr __device__ CoordDyn &xor_swizzle() {
  //   auto i16 = (rows * 32 + cols) * sizeof(float) / 16;
  //   auto y16 = i16 / 8;
  //   auto x16 = i16 % 8;
  //   auto x16_swz = y16 ^ x16;
  //   auto x_swz =
  //       x16_swz * 16 / sizeof(float) % 32 + cols % (16 / sizeof(float));
  //   cols = x_swz % 32;
  //   rows = x_swz / 32;
  //   return *this;
  // }
  //
  // constexpr __device__ CoordDyn &xor_swizzle_col() {
  //   auto i = (rows * 16 + cols);
  //   auto i_swz = (rows * 16 + cols) ^ rows;
  //   cols = i_swz % 16;
  //   rows = i_swz / 16;
  //   return *this;
  // }
  //
  // constexpr __device__ CoordDyn &xor_swizzle_row() {
  //   auto i = (rows * 16 + cols);
  //   auto i_swz = (rows * 16 + cols) ^ cols;
  //   cols = i_swz % 16;
  //   rows = i_swz / 16;
  //   return *this;
  // }
};

} // namespace catz

#endif // CATZILLA_CATZ_COORD_LEGACY_H_
