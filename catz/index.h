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

  static constexpr bool isStatic() { return false; }
};

template <int Value>
struct Index {
  static constexpr int value = Value;

  constexpr Index() {}
  constexpr Index(int v) {}

  static constexpr bool isStatic() { return true; }
};

#define DEFINE_BINARY_OPERATOR(op)                                             \
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

DEFINE_BINARY_OPERATOR(+)
DEFINE_BINARY_OPERATOR(-)
DEFINE_BINARY_OPERATOR(*)
DEFINE_BINARY_OPERATOR(/)
DEFINE_BINARY_OPERATOR(%)

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

#endif // CATZILLA_CATZ_INDEX_H_
