#ifndef CATZILLA_CATZ_COORD_H_
#define CATZILLA_CATZ_COORD_H_

#include <string>
#include <type_traits>

#include "index.h"
#include "trait.h"

namespace catz {

// NOTE: explicit design of Coord
// // Coord 结构体，包含 rows 和 cols 两个成员
// template <typename T1, typename T2, typename=void>
// struct CoordNightly {
//     IndexWrapper<T1> rows;
//     IndexWrapper<T2> cols;
//
//     constexpr CoordNightly(T1 r, T2 c) : rows(make_index(r)),
//     cols(make_index(c)) {}
//
//     constexpr auto area() const {
//         return rows.get() * cols.get();
//     }
// };
//
// // 特化版本，用于编译期常量
// template <typename T1, typename T2>
// struct CoordNightly<T1, T2,
// std::enable_if_t<is_compile_time_constant<T1>::value &&
// is_compile_time_constant<T2>::value>> {
//   static constexpr T1 rows = T1{};
//   static constexpr T2 cols = T2{};
//
// };
//
// template <typename T1, typename T2, typename U1, typename U2>
// constexpr auto operator+(const CoordNightly<T1, T2>& lhs, const
// CoordNightly<U1, U2>& rhs) {
//     using ResultT1 = std::conditional_t<is_compile_time_constant_v<T1> &&
//     is_compile_time_constant_v<U1>, T1, std::decay_t<T1>>; using ResultT2 =
//     std::conditional_t<is_compile_time_constant_v<T2> &&
//     is_compile_time_constant_v<U2>, T2, std::decay_t<T2>>;
//
//     return CoordNightly<ResultT1, ResultT2>(lhs.rows.get() + rhs.rows.get(),
//     lhs.cols.get() + rhs.cols.get());
// }
//
// template <typename T1, typename T2>
// using CoordWrapper = CoordNightly<std::decay_t<T1>, std::decay_t<T2>>;
//
// // make_coord 函数模板，用于创建 CoordNightly 实例
// template <typename T1, typename T2>
// constexpr CoordWrapper<T1, T2> make_coord(T1 &&r, T2 &&c) {
//     return CoordWrapper<T1, T2>(std::forward<T1>(r), std::forward<T2>(c));
// }

// // NOTE: implicit design of Coord 类，使用 Index 作为参数封装行数和列数
// template <typename RowIndex, typename ColIndex>
// struct CoordNightly {
//   RowIndex rows;
//   ColIndex cols;
//
//   constexpr CoordNightly(RowIndex r, ColIndex c) : rows(r), cols(c) {}
//
//   constexpr auto eval_rows() const { return rows(); }
//   constexpr auto eval_cols() const { return cols(); }
//
//   // constexpr auto isStatic() { return rows::isStatic() && cols::isStatic();
//   }
//   // constexpr auto isDynamic() { return rows::isDynamic() ||
//   cols::isDynamic();
//   // }
//
//   std::string str() const {
//     return "Coord<" + std::to_string(rows()) + ", " + std::to_string(cols())
//     +
//            ">";
//   }
//
//   template <typename OtherRowIndex, typename OtherColIndex>
//   constexpr auto
//   operator+(const CoordNightly<OtherRowIndex, OtherColIndex> &other) const {
//     auto new_row_index = rows + other.rows;
//     auto new_col_index = cols + other.cols;
//     return CoordNightly<decltype(new_row_index), decltype(new_col_index)>(
//         new_row_index, new_col_index);
//   }
//
//   template <typename OtherRowIndex, typename OtherColIndex>
//   constexpr auto
//   operator-(const CoordNightly<OtherRowIndex, OtherColIndex> &other) const {
//     auto new_row_index = rows - other.rows;
//     auto new_col_index = cols - other.cols;
//     return CoordNightly<decltype(new_row_index), decltype(new_col_index)>(
//         new_row_index, new_col_index);
//   }
//
//   template <typename OtherRowIndex, typename OtherColIndex>
//   constexpr auto
//   operator*(const CoordNightly<OtherRowIndex, OtherColIndex> &other) const {
//     auto new_row_index = rows * other.rows;
//     auto new_col_index = cols * other.cols;
//     return CoordNightly<decltype(new_row_index), decltype(new_col_index)>(
//         new_row_index, new_col_index);
//   }
//
//   template <typename OtherRowIndex, typename OtherColIndex>
//   constexpr auto
//   operator/(const CoordNightly<OtherRowIndex, OtherColIndex> &other) const {
//     auto new_row_index = rows / other.rows;
//     auto new_col_index = cols / other.cols;
//     return CoordNightly<decltype(new_row_index), decltype(new_col_index)>(
//         new_row_index, new_col_index);
//   }
//
//   template <typename OtherRowIndex, typename OtherColIndex>
//   constexpr auto
//   operator%(const CoordNightly<OtherRowIndex, OtherColIndex> &other) const {
//     auto new_row_index = rows % other.rows;
//     auto new_col_index = cols % other.cols;
//     return CoordNightly<decltype(new_row_index), decltype(new_col_index)>(
//         new_row_index, new_col_index);
//   }
//
//   template <typename OtherRowIndex, typename OtherColIndex>
//   constexpr auto
//   ceil_div(const CoordNightly<OtherRowIndex, OtherColIndex> &other) const {
//     auto new_row_index = rows.ceil_div(other.rows);
//     auto new_col_index = cols.ceil_div(other.cols);
//     return CoordNightly<decltype(new_row_index), decltype(new_col_index)>(
//         new_row_index, new_col_index);
//   }
// };
//
// // 工厂函数：根据参数类型构造 Coord
// template <typename T1, typename T2>
// constexpr auto make_coord(T1 &&row_value, T2 &&col_value) {
//   auto row_index = make_index(std::forward<T1>(row_value));
//   auto col_index = make_index(std::forward<T2>(col_value));
//   return CoordNightly<decltype(row_index), decltype(col_index)>(row_index,
//                                                                 col_index);
// }

////////////////////////////////////////////////////

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

template <int ROWS, int COLS>
// typename = std::enable_if_t<(ROWS >= 0 && COLS >= 0 &&
// std::is_const_v<decltype(ROWS)>)>>
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

} // namespace catz

#endif // CATZILLA_CATZ_COORD_H_
