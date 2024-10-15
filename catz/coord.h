#ifndef CATZILLA_CATZ_COORD_H_
#define CATZILLA_CATZ_COORD_H_

#include <string>
#include <type_traits>

#include "index.h"
#include "trait.h"

namespace catz {

// Coord 类，使用 Index 作为参数封装行数和列数
template <typename RowIndex, typename ColIndex>
struct CoordNightly {
  RowIndex rows;
  ColIndex cols;

  constexpr CoordNightly(RowIndex r, ColIndex c) : rows(r), cols(c) {}

  constexpr auto eval_rows() const { return rows(); }
  constexpr auto eval_cols() const { return cols(); }

  std::string str() const {
    return "Coord<" + std::to_string(rows()) + ", " + std::to_string(cols()) +
           ">";
  }
};

// 工厂函数：根据参数类型构造 Coord
template <typename T1, typename T2>
constexpr auto make_coord(T1 &&row_value, T2 &&col_value) {
  auto row_index = make_index(std::forward<T1>(row_value));
  auto col_index = make_index(std::forward<T2>(col_value));
  return CoordNightly<decltype(row_index), decltype(col_index)>(row_index,
                                                                col_index);
}

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

// 通用模板：用于 Coord + Coord，相加返回 Coord
template <
    typename T1, typename T2,
    typename = std::enable_if_t<std::is_same_v<T1, Coord<T1::rows, T1::cols>> &&
                                std::is_same_v<T2, Coord<T2::rows, T2::cols>>>>
constexpr auto operator+(const T1 &, const T2 &) {
  return Coord<T1::rows + T2::rows, T1::cols + T2::cols>{};
}

// 通用模板：用于 CoordDyn + CoordDyn，相加返回 CoordDyn
template <typename T1, typename T2,
          typename = std::enable_if_t<std::is_same_v<T1, CoordDyn> &&
                                      std::is_same_v<T2, CoordDyn>>>
CoordDyn operator+(const T1 &lhs, const T2 &rhs) {
  return CoordDyn(lhs.rows + rhs.rows, lhs.cols + rhs.cols);
}

// // 通用模板：用于混合类型加法（Coord + CoordDyn 或 CoordDyn + Coord），返回
// CoordDyn template <typename T1, typename T2,
//           typename = std::enable_if_t<
//               (std::is_same_v<T1, CoordDyn> && std::is_same_v<T2,
//               Coord<T2::rows, T2::cols>>) || (std::is_same_v<T2, CoordDyn> &&
//               std::is_same_v<T1, Coord<T1::rows, T1::cols>>)>>
// CoordDyn operator+(const T1& lhs, const T2& rhs) {
//     int new_rows = lhs.rows + rhs.rows;
//     int new_cols = lhs.cols + rhs.cols;
//     return CoordDyn(new_rows, new_cols);
// }

// 4. make_coord 函数：支持静态和动态版本
// template <typename T1, typename T2>
// auto make_coord(T1 rows, T2 cols) {
//     // 提升参数
//     auto promoted_rows = promote(rows);
//     auto promoted_cols = promote(cols);
//
//     if constexpr (std::is_same_v<decltype(promoted_rows),
//     std::integral_constant<int, promoted_rows()>> &&
//                   std::is_same_v<decltype(promoted_cols),
//                   std::integral_constant<int, promoted_cols()>>) {
//         // 如果参数是编译期常量，则使用 Coord 模板
//         return Coord<promoted_rows(), promoted_cols()>{};
//     } else {
//         // 否则使用 CoordDyn（运行时版本）
//         return CoordDyn(rows, cols);
//     }
// }

#define make_coord_nightly(rows, cols)                                         \
  ([&]() {                                                                     \
    if constexpr (std::is_integral_v<decltype(rows)> &&                        \
                  std::is_integral_v<decltype(cols)> &&                        \
                  IS_CONSTANT_EVALUATED()) {                                   \
      return Coord<std::integral_constant<int, rows>(),                        \
                   std::integral_constant<int, cols>()>();                     \
    } else {                                                                   \
      return CoordDyn(rows, cols);                                             \
    }                                                                          \
  }())
// // #define Coord(rows, cols) Coord<rows, cols>()
// template <int ROWS, int COLS>
// constexpr Coord<ROWS, COLS> make_coord(std::integral_constant<int, ROWS>,
//                                        std::integral_constant<int, COLS>) {
//     return Coord<ROWS, COLS>();
// }
//
// CoordDyn make_coord(int rows, int cols) {
//     return CoordDyn(rows, cols);
// }

} // namespace catz

#endif // CATZILLA_CATZ_COORD_H_
