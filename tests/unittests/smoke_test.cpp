#define CATCH_CONFIG_MAIN // 让 Catch 提供 main()
#include <catch2/catch_test_macros.hpp>

// 一个简单的函数，用于测试
int add(int a, int b) { return a + b; }

// 测试 add 函数的测试用例
TEST_CASE("Addition tests", "[add]") {
  REQUIRE(add(1, 1) == 2);
  REQUIRE(add(-1, 1) == 0);
  REQUIRE(add(-1, -1) == -2);

  SECTION("Adding zero") {
    REQUIRE(add(0, 0) == 0);
    REQUIRE(add(0, 1) == 1);
    REQUIRE(add(1, 0) == 1);
  }

  SECTION("Adding negative numbers") {
    REQUIRE(add(-2, -2) == -4);
    REQUIRE(add(-2, 2) == 0);
  }
}

// 另一个示例测试用例
TEST_CASE("Multiplication tests", "[multiply]") {
  REQUIRE(2 * 2 == 4);
  REQUIRE(-2 * 2 == -4);
  REQUIRE(2 * -2 == -4);

  SECTION("Multiplication by zero") {
    REQUIRE(2 * 0 == 0);
    REQUIRE(-2 * 0 == 0);
    REQUIRE(0 * 2 == 0);
  }
}
