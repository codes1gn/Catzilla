#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "matrix.h"
#include "macro.h"

using namespace catz;


TEST_CASE("coord construction", "[coord][construction]") {
  constexpr int r = 2;
  constexpr int c = 3;
  constexpr auto static_coord = make_coord(r, c);  // 使用编译期常量
  SCHECK(static_coord.rows() == 2);
  SCHECK(static_coord.cols() == 3);

  // 运行时变量构造 Coord
  int r_dyn = 5;
  int c_dyn = 6;
  auto dynamic_coord = make_coord(r_dyn, c_dyn);  // 使用运行时变量
  CHECK(dynamic_coord.rows() == 5);
  CHECK(dynamic_coord.cols() == 6);
}


// TEST_CASE("coord add", "[coord][add]") {
//   CoordDyn coord1 = CoordDyn(1, 2);
//   CoordDyn coord2 = CoordDyn(4, 5);
//   CoordDyn coord3 = coord1 + coord2;
//
//   REQUIRE(coord1.isDynamic() == true);
//
//   CHECK(coord3.rows == 5);
//   CHECK(coord3.cols == 7);
// }

// TEST_CASE("coord sub", "[coord][sub]") {
//   CoordDyn coord1 = CoordDyn(1, 2);
//   CoordDyn coord2 = CoordDyn(4, 5);
//   CoordDyn coord3 = coord1 - coord2;
//
//   REQUIRE(coord1.isDynamic() == true);
//
//   CHECK(coord3.rows == -3);
//   CHECK(coord3.cols == -3);
// }
//
// TEST_CASE("coord div", "[coord][div]") {
//   CoordDyn coord1 = CoordDyn(1, 2);
//   CoordDyn coord2 = CoordDyn(4, 8);
//   CoordDyn coord3 = coord2 / coord1;
//
//   REQUIRE(coord1.isDynamic() == true);
//
//   CHECK(coord3.rows == 4);
//   CHECK(coord3.cols == 4);
// }
//
// TEST_CASE("coord construction nightly", "[coord][construction]") {
//   Coord coord1 = Coord<1, 2>();
//
//   REQUIRE(coord1.isDynamic() == false);
//
//   SCHECK(coord1.rows == 1);
//   SCHECK(coord1.cols == 2);
// }
//
// TEST_CASE("coord add nightly", "[coord][add]") {
//   auto coord1 = Coord<1, 2>();
//   auto coord2 = Coord<4, 5>();
//   auto coord3 = coord1 + coord2;
//
//   REQUIRE(coord1.isDynamic() == false);
//
//   SCHECK(coord3.rows == 5);
//   SCHECK(coord3.cols == 7);
// }
//
// TEST_CASE("coord sub nightly", "[coord][sub]") {
//   auto coord1 = Coord<1, 2>();
//   auto coord2 = Coord<4, 5>();
//   auto coord3 = coord1 - coord2;
//
//   REQUIRE(coord1.isDynamic() == false);
//
//   SCHECK(coord3.rows == -3);
//   SCHECK(coord3.cols == -3);
// }
//
// TEST_CASE("coord div nightly", "[coord][div]") {
//   auto coord1 = Coord<1, 2>();
//   auto coord2 = Coord<4, 8>();
//   auto coord3 = coord2 / coord1;
//
//   REQUIRE(coord1.isDynamic() == false);
//
//   SCHECK(coord3.rows == 4);
//   SCHECK(coord3.cols == 4);
// }
//
