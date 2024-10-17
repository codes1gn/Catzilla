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
  auto r = make_index<2>();
  auto c = make_index<3>();
  auto static_coord = CoordS(r, c);
  SCHECK(static_coord.rows() == 2);
  SCHECK(static_coord.cols() == 3);

  auto r_dyn = make_index(5);
  auto c_dyn = make_index(6);
  auto dynamic_coord = CoordS(r_dyn, c_dyn);  // 使用运行时变量
  CHECK(dynamic_coord.rows() == 5);
  CHECK(dynamic_coord.cols() == 6);
}

TEST_CASE("coord construction builder", "[coord][construction][builder]") {
  auto static_coord = make_coord(2, 3);
  SCHECK(static_coord.rows() == 2);
  SCHECK(static_coord.cols() == 3);
  SCHECK(static_coord.isStatic() == true);

  int r_dyn = 5;
  int c_dyn = 6;
  auto dynamic_coord = make_coord_dyn(r_dyn, c_dyn);  // 使用运行时变量
  CHECK(dynamic_coord.rows() == 5);
  CHECK(dynamic_coord.cols() == 6);
  SCHECK(dynamic_coord.isStatic() == false);

  const int r_2 = 5;
  const int c_2 = 6;
  auto static_coord_2 = make_coord(r_2, c_2);  // 使用运行时变量
  SCHECK(static_coord_2.rows() == 5);
  SCHECK(static_coord_2.cols() == 6);
  SCHECK(static_coord_2.isStatic() == true);

  constexpr int r_3 = 5;
  constexpr int c_3 = 6;
  auto static_coord_3 = make_coord(r_3, c_3);  // 使用运行时变量
  SCHECK(static_coord_3.rows() == 5);
  SCHECK(static_coord_3.cols() == 6);
  SCHECK(static_coord_3.isStatic() == true);
}

TEST_CASE("coord add", "[coord][add]") {
  const int r = 2;
  const int c = 3;
  auto coord1 = make_coord(r, c);  // 使用编译期常量
  auto coord2 = make_coord(4, 5);
  auto coord3 = coord1 + coord2;

  SCHECK(coord1.isStatic() == true);

  SCHECK(coord3.rows() == 6);
  SCHECK(coord3.cols() == 8);
}

TEST_CASE("coord sub", "[coord][sub]") {
  constexpr int r = 2;
  constexpr int c = 3;
  auto coord1 = make_coord(r, c);  // 使用编译期常量
  auto coord2 = make_coord(4, 5);
  auto coord3 = coord1 - coord2;

  SCHECK(coord3.isStatic() == true);

  SCHECK(coord3.rows() == -2);
  SCHECK(coord3.cols() == -2);
}

TEST_CASE("coord mul", "[coord][mul]") {
  constexpr int r = 2;
  constexpr int c = 3;
  auto coord1 = make_coord(r, c);  // 使用编译期常量
  auto coord2 = make_coord(4, 5);
  auto coord3 = coord1 * coord2;

  SCHECK(coord3.isStatic() == true);

  SCHECK(coord3.rows() == 8);
  SCHECK(coord3.cols() == 15);
}

TEST_CASE("coord mod", "[coord][mod]") {
  constexpr int r = 2;
  constexpr int c = 3;
  auto coord1 = make_coord(r, c);  // 使用编译期常量
  auto coord2 = make_coord(4, 5);
  auto coord3 = coord1 % coord2;

  SCHECK(coord3.isStatic() == true);

  SCHECK(coord3.rows() == 2);
  SCHECK(coord3.cols() == 3);
}

TEST_CASE("coord ceil", "[coord][ceil]") {
  constexpr int r = 2;
  constexpr int c = 3;
  auto coord1 = make_coord(r, c);  // 使用编译期常量
  auto coord2 = make_coord(4, 5);
  auto coord3 = coord1.ceil_div(coord2);

  SCHECK(coord1.isStatic() == true);
  SCHECK(coord2.isStatic() == true);
  SCHECK(coord3.isStatic() == true);

  SCHECK(coord3.rows() == 1);
  SCHECK(coord3.cols() == 1);
}
