#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "matrix.h"
#include "macro.h"

using namespace catz;


// TEST_CASE("coord construction", "[coord][construction]") {
//   constexpr int r = 2;
//   constexpr int c = 3;
//   constexpr auto static_coord = make_coord(r, c);  // 使用编译期常量
//   SCHECK(static_coord.rows() == 2);
//   SCHECK(static_coord.cols() == 3);
//
//   // 运行时变量构造 Coord
//   int r_dyn = 5;
//   int c_dyn = 6;
//   auto dynamic_coord = make_coord(r_dyn, c_dyn);  // 使用运行时变量
//   CHECK(dynamic_coord.rows() == 5);
//   CHECK(dynamic_coord.cols() == 6);
// }
//
//
// TEST_CASE("coord add", "[coord][add]") {
//   constexpr int r = 2;
//   constexpr int c = 3;
//   constexpr auto coord1 = make_coord(r, c);  // 使用编译期常量
//   constexpr auto coord2 = make_coord(4, 5);
//   constexpr auto coord3 = coord1 + coord2;
//
//   // BUG: the sta/dyn properties does not shown as expected.
//   // SCHECK(coord1.rows.isStatic() == false);
//
//   SCHECK(coord3.rows() == 6);
//   SCHECK(coord3.cols() == 8);
// }
//
// TEST_CASE("coord sub", "[coord][sub]") {
//   constexpr int r = 2;
//   constexpr int c = 3;
//   constexpr auto coord1 = make_coord(r, c);  // 使用编译期常量
//   constexpr auto coord2 = make_coord(4, 5);
//   constexpr auto coord3 = coord1 - coord2;
//
//   // SCHECK(coord3.rows.isStatic() == true);
//
//   SCHECK(coord3.rows() == -2);
//   SCHECK(coord3.cols() == -2);
// }
//
// TEST_CASE("coord mul", "[coord][mul]") {
//   constexpr int r = 2;
//   constexpr int c = 3;
//   constexpr auto coord1 = make_coord(r, c);  // 使用编译期常量
//   constexpr auto coord2 = make_coord(4, 5);
//   constexpr auto coord3 = coord1 * coord2;
//
//   // SCHECK(coord3.rows.isStatic() == true);
//
//   SCHECK(coord3.rows() == 8);
//   SCHECK(coord3.cols() == 15);
// }
//
// TEST_CASE("coord mod", "[coord][mod]") {
//   constexpr int r = 2;
//   constexpr int c = 3;
//   constexpr auto coord1 = make_coord(r, c);  // 使用编译期常量
//   constexpr auto coord2 = make_coord(4, 5);
//   constexpr auto coord3 = coord1 % coord2;
//
//   // SCHECK(coord3.rows.isStatic() == true);
//
//   SCHECK(coord3.rows() == 2);
//   SCHECK(coord3.cols() == 3);
// }
