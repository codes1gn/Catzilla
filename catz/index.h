#ifndef CATZILLA_CATZ_INDEX_H_
#define CATZILLA_CATZ_INDEX_H_

#include "trait.h"
#include <type_traits>

template <int Value>
struct Index;

struct IndexDyn {
  int value;

  constexpr IndexDyn(int v) : value(v) {}
  constexpr IndexDyn() : value(0) {}

  constexpr int operator()() const { return value; }

  static constexpr bool isStatic() { return false; }
};

template <int Value>
struct Index {
  static constexpr int value = Value;

  constexpr Index() {}
  constexpr Index(int v) {}

  constexpr int operator()() const { return value; }

  static constexpr bool isStatic() { return true; }
};

#define DEFINE_BINARY_OPERATOR_FOR_INDEX(op)                                   \
  template <int Value, int OtherValue>                                         \
  constexpr auto operator op(const Index<Value> &lhs,                          \
                             const Index<OtherValue> &rhs) {                   \
    return Index<Value op OtherValue>();                                       \
  }                                                                            \
                                                                               \
  template <int Value>                                                         \
  constexpr auto operator op(const IndexDyn &lhs, const Index<Value> &rhs) {   \
    return IndexDyn(lhs.value op Value);                                       \
  }                                                                            \
                                                                               \
  template <int Value>                                                         \
  constexpr auto operator op(const Index<Value> &lhs, const IndexDyn &rhs) {   \
    return IndexDyn(lhs.value op rhs.value);                                   \
  }                                                                            \
                                                                               \
  constexpr auto operator op(const IndexDyn &lhs, const IndexDyn &rhs) {       \
    return IndexDyn(lhs.value op rhs.value);                                   \
  }

DEFINE_BINARY_OPERATOR_FOR_INDEX(+)
DEFINE_BINARY_OPERATOR_FOR_INDEX(-)
DEFINE_BINARY_OPERATOR_FOR_INDEX(*)
DEFINE_BINARY_OPERATOR_FOR_INDEX(/)
DEFINE_BINARY_OPERATOR_FOR_INDEX(%)

// define special ceil-div
template <int Value, int OtherValue>
constexpr auto ceil_div(const Index<Value> &lhs, const Index<OtherValue> &rhs) {
  return Index<(Value + OtherValue - 1) / OtherValue>();
}

constexpr auto ceil_div(const IndexDyn &lhs, const IndexDyn &rhs) {
  return IndexDyn((lhs.value + rhs.value - 1) / rhs.value);
}

template <int Value>
constexpr auto ceil_div(const Index<Value> &lhs, const IndexDyn &rhs) {
  return IndexDyn((lhs.value + rhs.value - 1) / rhs.value);
}

template <int Value>
constexpr auto ceil_div(const IndexDyn &lhs, const Index<Value> &rhs) {
  return IndexDyn((lhs.value + rhs - 1) / rhs);
}

template <int Value>
constexpr auto make_index() {
  return Index<Value>();
}

constexpr auto make_index(int value) {
  return IndexDyn(value); // 返回 IndexDyn 实例
}

/////////// macros for convinience
///

#define DEFINE_INDEX(n) using I##n = Index<n>;

// 展开所有 2 的幂次别名，从 1 到 1024
DEFINE_INDEX(1)
DEFINE_INDEX(2)
DEFINE_INDEX(4)
DEFINE_INDEX(8)
DEFINE_INDEX(16)
DEFINE_INDEX(32)
DEFINE_INDEX(64)
DEFINE_INDEX(128)
DEFINE_INDEX(256)
DEFINE_INDEX(512)
DEFINE_INDEX(1024)

#endif // CATZILLA_CATZ_INDEX_H_
