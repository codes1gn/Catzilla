#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "matrix_utils.h"

using namespace catz;

TEST_CASE("coord construction", "[coord][construction]") {
  constexpr Coord coord1 = Coord(1, 2);

  static_assert(coord1.first == 1, "coord.first wrong");
  static_assert(coord1.second == 2, "coord.second wrong");
}

TEST_CASE("coord add", "[coord][add]") {
  constexpr Coord coord1 = Coord(1, 2);
  constexpr Coord coord2 = Coord(4, 5);
  constexpr Coord coord3 = coord1 + coord2;

  static_assert(coord3.first == 5, "coord.first wrong");
  static_assert(coord3.second == 7, "coord.second wrong");
}

TEST_CASE("coord sub", "[coord][sub]") {
  constexpr Coord coord1 = Coord(1, 2);
  constexpr Coord coord2 = Coord(4, 5);
  constexpr Coord coord3 = coord1 - coord2;

  static_assert(coord3.first == -3, "coord.first wrong");
  static_assert(coord3.second == -3, "coord.second wrong");
}

TEST_CASE("coord div", "[coord][div]") {
  constexpr Coord coord1 = Coord(1, 2);
  constexpr Coord coord2 = Coord(4, 8);
  constexpr Coord coord3 = coord2 / coord1;

  static_assert(coord3.first == 4, "coord.first wrong");
  static_assert(coord3.second == 4, "coord.second wrong");
}
