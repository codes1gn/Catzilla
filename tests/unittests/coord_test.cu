#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "matrix.h"

using namespace catz;

TEST_CASE("coord construction", "[coord][construction]") {
  auto coord1 = make_coord(1, 2);

  REQUIRE(coord1.isDynamic() == true);

  CHECK(coord1.first == 1);
  CHECK(coord1.second == 2);
}

TEST_CASE("coord add", "[coord][add]") {
  CoordDyn coord1 = CoordDyn(1, 2);
  CoordDyn coord2 = CoordDyn(4, 5);
  CoordDyn coord3 = coord1 + coord2;

  REQUIRE(coord1.isDynamic() == true);

  CHECK(coord3.first == 5);
  CHECK(coord3.second == 7);
}

TEST_CASE("coord sub", "[coord][sub]") {
  CoordDyn coord1 = CoordDyn(1, 2);
  CoordDyn coord2 = CoordDyn(4, 5);
  CoordDyn coord3 = coord1 - coord2;

  REQUIRE(coord1.isDynamic() == true);

  CHECK(coord3.first == -3);
  CHECK(coord3.second == -3);
}

TEST_CASE("coord div", "[coord][div]") {
  CoordDyn coord1 = CoordDyn(1, 2);
  CoordDyn coord2 = CoordDyn(4, 8);
  CoordDyn coord3 = coord2 / coord1;

  REQUIRE(coord1.isDynamic() == true);

  CHECK(coord3.first == 4);
  CHECK(coord3.second == 4);
}

TEST_CASE("coord construction nightly", "[coord][construction]") {
  Coord coord1 = Coord<1, 2>();

  REQUIRE(coord1.isDynamic() == false);

  SCHECK(coord1.first == 1);
  SCHECK(coord1.second == 2);
}

TEST_CASE("coord add nightly", "[coord][add]") {
  auto coord1 = Coord<1, 2>();
  auto coord2 = Coord<4, 5>();
  auto coord3 = coord1 + coord2;

  REQUIRE(coord1.isDynamic() == false);

  SCHECK(coord3.first == 5);
  SCHECK(coord3.second == 7);
}

TEST_CASE("coord sub nightly", "[coord][sub]") {
  auto coord1 = Coord<1, 2>();
  auto coord2 = Coord<4, 5>();
  auto coord3 = coord1 - coord2;

  REQUIRE(coord1.isDynamic() == false);

  SCHECK(coord3.first == -3);
  SCHECK(coord3.second == -3);
}

TEST_CASE("coord div nightly", "[coord][div]") {
  auto coord1 = Coord<1, 2>();
  auto coord2 = Coord<4, 8>();
  auto coord3 = coord2 / coord1;

  REQUIRE(coord1.isDynamic() == false);

  SCHECK(coord3.first == 4);
  SCHECK(coord3.second == 4);
}

