#ifndef CATZILLA_CATZ_COORD_H_
#define CATZILLA_CATZ_COORD_H_

#include <string>
#include <type_traits>

#include "index.h"
#include "trait.h"

namespace catz {

////////////////////////////////////////////////////
#define DEFINE_BINARY_OPERATOR_FOR_COORD(op)                                   \
  template <typename OtherRowType, typename OtherColType>                      \
  constexpr auto operator op(const CoordS<OtherRowType, OtherColType> &other)  \
      const {                                                                  \
    auto new_rows = rows op other.rows;                                        \
    auto new_cols = cols op other.cols;                                        \
    return CoordS<decltype(new_rows), decltype(new_cols)>(new_rows, new_cols); \
  }

template <typename RowType, typename ColType>
struct CoordS {
  RowType rows;
  ColType cols;

  constexpr CoordS(RowType r, ColType c) : rows(r), cols(c) {}

  constexpr bool isStatic() { return (rows.isStatic() && cols.isStatic()); }

  DEFINE_BINARY_OPERATOR_FOR_COORD(+)
  DEFINE_BINARY_OPERATOR_FOR_COORD(-)
  DEFINE_BINARY_OPERATOR_FOR_COORD(*)
  DEFINE_BINARY_OPERATOR_FOR_COORD(/)
  DEFINE_BINARY_OPERATOR_FOR_COORD(%)

  template <typename OtherRowType, typename OtherColType>
  constexpr auto
  ceil_div(const CoordS<OtherRowType, OtherColType> &other) const {
    auto new_rows = (rows + other.rows - I1()) / other.rows;
    auto new_cols = (cols + other.cols - I1()) / other.cols;
    return CoordS<decltype(new_rows), decltype(new_cols)>(new_rows, new_cols);
  }
};

#define make_coord(R, C) CoordS(Index<R>(), Index<C>())

#define make_coord_dyn(R, C) CoordS(IndexDyn(R), IndexDyn(C))

template <int ROWS, int COLS>
struct Coord {
  static constexpr int rows = ROWS;
  static constexpr int cols = COLS;

  // 构造函数
  constexpr __device__ Coord() {}

  //////////////////////////////////////
  /// utils
  //////////////////////////////////////

  std::string str() const {
    char buffer[20]; // 提供足够大的缓冲区
    std::sprintf(buffer, "Coord<%d, %d>", rows, cols);
    return std::string(buffer);
  }

  constexpr bool isStatic() { return true; }
  constexpr bool isDynamic() { return false; }

  //////////////////////////////////////
  /// operators
  //////////////////////////////////////

  // template <int orows, int ocols>
  // constexpr Coord<rows + orows, cols + ocols>
  // operator+(const Coord<orows, ocols> &other) const {
  //   return Coord<rows + other.rows, cols + other.cols>();
  // }

  // template <int orows, int ocols>
  // constexpr Coord<rows - orows, cols - ocols>
  // operator-(const Coord<orows, ocols> &other) const {
  //   return Coord<rows - other.rows, cols - other.cols>();
  // }
  //
  // template <int orows, int ocols>
  // constexpr Coord<rows * orows, cols * ocols>
  // operator*(const Coord<orows, ocols> &other) const {
  //   return Coord<rows * other.rows, cols * other.cols>();
  // }
  //
  // template <int orows, int ocols>
  // constexpr Coord<rows / orows, cols / ocols>
  // operator/(const Coord<orows, ocols> &other) const {
  //   static_assert(cols != 0 && other.cols != 0, "Division by zero");
  //   return Coord<rows / other.rows, cols / other.cols>();
  // }
  //
  // constexpr __device__ Coord &xor_swizzle() {
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
  // constexpr __device__ Coord &xor_swizzle_col() {
  //   constexpr auto i = (rows * 16 + cols);
  //   constexpr auto i_swz = i ^ rows;
  //   // auto _cols = i_swz % 16;
  //   // auto _rows = i_swz / 16;
  //   // return *this;
  //   return Coord<(i_swz / 16), (i_swz % 16)>();
  // }
  //
  // constexpr __device__ Coord &xor_swizzle_row() {
  //   constexpr auto i = (rows * 16 + cols);
  //   constexpr auto i_swz = i ^ cols;
  //   // cols = i_swz % 16;
  //   // rows = i_swz / 16;
  //   // return *this;
  //   return Coord<(i_swz / 16), (i_swz % 16)>();
  // }
};

// template<int X, int Y>
struct CoordDyn {
  int rows;
  int cols;

  // 构造函数
  constexpr __device__ CoordDyn(int rows, int cols) : rows(rows), cols(cols) {}

  // 从 std::tuple<int, int> 构造
  constexpr __device__ CoordDyn(const std::tuple<int, int> &t)
      : rows(std::get<0>(t)), cols(std::get<1>(t)) {}

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

  // 将 CoordDyn 转换为 std::tuple<int, int>
  constexpr __device__ std::tuple<int, int> to_tuple() const {
    return std::make_tuple(rows, cols);
  }

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

#endif // CATZILLA_CATZ_COORD_H_
