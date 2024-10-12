#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "matrix.h"

using namespace catz;

TEST_CASE("coord construction", "[coord][construction]") {
  constexpr CoordDyn coord1 = CoordDyn(1, 2);

  static_assert(coord1.first == 1, "coord.first wrong");
  static_assert(coord1.second == 2, "coord.second wrong");
}

TEST_CASE("coord add", "[coord][add]") {
  constexpr CoordDyn coord1 = CoordDyn(1, 2);
  constexpr CoordDyn coord2 = CoordDyn(4, 5);
  constexpr CoordDyn coord3 = coord1 + coord2;

  static_assert(coord3.first == 5, "coord.first wrong");
  static_assert(coord3.second == 7, "coord.second wrong");
}

TEST_CASE("coord sub", "[coord][sub]") {
  constexpr CoordDyn coord1 = CoordDyn(1, 2);
  constexpr CoordDyn coord2 = CoordDyn(4, 5);
  constexpr CoordDyn coord3 = coord1 - coord2;

  static_assert(coord3.first == -3, "coord.first wrong");
  static_assert(coord3.second == -3, "coord.second wrong");
}

TEST_CASE("coord div", "[coord][div]") {
  constexpr CoordDyn coord1 = CoordDyn(1, 2);
  constexpr CoordDyn coord2 = CoordDyn(4, 8);
  constexpr CoordDyn coord3 = coord2 / coord1;

  static_assert(coord3.first == 4, "coord.first wrong");
  static_assert(coord3.second == 4, "coord.second wrong");
}

TEST_CASE("coord construction nightly", "[coord][construction]") {
  Coord coord1 = Coord<1, 2>();

  static_assert(coord1.first == 1, "coord.first wrong");
  static_assert(coord1.second == 2, "coord.second wrong");
}

TEST_CASE("coord add nightly", "[coord][add]") {
  auto coord1 = Coord<1, 2>();
  auto coord2 = Coord<4, 5>();
  auto coord3 = coord1 + coord2;

  static_assert(coord3.first == 5, "coord.first wrong");
  static_assert(coord3.second == 7, "coord.second wrong");
}

TEST_CASE("coord sub nightly", "[coord][sub]") {
  auto coord1 = Coord<1, 2>();
  auto coord2 = Coord<4, 5>();
  auto coord3 = coord1 - coord2;

  static_assert(coord3.first == -3, "coord.first wrong");
  static_assert(coord3.second == -3, "coord.second wrong");
}

TEST_CASE("coord div nightly", "[coord][div]") {
  auto coord1 = Coord<1, 2>();
  auto coord2 = Coord<4, 8>();
  auto coord3 = coord2 / coord1;

  static_assert(coord3.first == 4, "coord.first wrong");
  static_assert(coord3.second == 4, "coord.second wrong");
}

