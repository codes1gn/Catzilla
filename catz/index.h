#ifndef CATZILLA_CATZ_INDEX_H_
#define CATZILLA_CATZ_INDEX_H_

#include "trait.h"
#include <type_traits>

template <typename T, typename = void>
struct Index {
  T value;

  constexpr Index(T v) : value(v) {}

  constexpr T operator()() const { return value; }

  constexpr bool isStatic() { return std::false_type::value; }
  constexpr bool isDynamic() { return std::true_type::value; }

  constexpr Index operator+(const Index &other) const {
    return Index(value + other.value);
  }

  constexpr Index operator-(const Index &other) const {
    return Index(value - other.value);
  }

  constexpr Index operator*(const Index &other) const {
    return Index(value * other.value);
  }

  constexpr Index operator/(const Index &other) const {
    return Index(value / other.value);
  }

  constexpr Index operator%(const Index &other) const {
    return Index(value % other.value);
  }

  constexpr Index ceil_div(const Index &other) const {
    return Index((value + other.value - 1) % other.value);
  }
};

template <typename T>
struct Index<T, std::enable_if_t<is_compile_time_constant<T>::value>> {
  static constexpr T value = T{};

  constexpr T operator()() const { return value; }

  constexpr bool isStatic() { return std::true_type::value; }
  constexpr bool isDynamic() { return std::false_type::value; }

  template <typename U>
  constexpr Index operator+(const Index<U> &other) const {
    using ResultType =
        std::conditional_t<is_compile_time_constant<T>::value &&
                               is_compile_time_constant<U>::value,
                           Index<std::integral_constant<int, T{} + U{}>>,
                           Index<decltype(value + other.value)>>;
    return ResultType(value + other.value);
  }

  template <typename U>
  constexpr Index operator-(const Index<U> &other) const {
    using ResultType =
        std::conditional_t<is_compile_time_constant<T>::value &&
                               is_compile_time_constant<U>::value,
                           Index<std::integral_constant<int, T{} - U{}>>,
                           Index<decltype(value - other.value)>>;
    return ResultType(value - other.value);
  }

  template <typename U>
  constexpr Index operator*(const Index<U> &other) const {
    using ResultType =
        std::conditional_t<is_compile_time_constant<T>::value &&
                               is_compile_time_constant<U>::value,
                           Index<std::integral_constant<int, T{} * U{}>>,
                           Index<decltype(value * other.value)>>;
    return ResultType(value * other.value);
  }

  template <typename U>
  constexpr Index operator/(const Index<U> &other) const {
    using ResultType =
        std::conditional_t<is_compile_time_constant<T>::value &&
                               is_compile_time_constant<U>::value,
                           Index<std::integral_constant<int, T{} / U{}>>,
                           Index<decltype(value / other.value)>>;
    return ResultType(value / other.value);
  }

  template <typename U>
  constexpr Index operator%(const Index<U> &other) const {
    using ResultType =
        std::conditional_t<is_compile_time_constant<T>::value &&
                               is_compile_time_constant<U>::value,
                           Index<std::integral_constant<int, T{} % U{}>>,
                           Index<decltype(value % other.value)>>;
    return ResultType(value % other.value);
  }

  template <typename U>
  constexpr auto ceil_div(const Index<U> &other) const {
    if constexpr (is_compile_time_constant<U>::value && U{} == 0) {
      throw std::runtime_error("Division by zero");
    }
    using ResultType = std::conditional_t<
        is_compile_time_constant<T>::value &&
            is_compile_time_constant<U>::value,
        Index<std::integral_constant<int, (T{} + U{} - 1) / U{}>>,
        Index<decltype((value + other.value - 1) / other.value)>>;
    return ResultType((value + other.value - 1) / other.value);
  }
};

template <typename T>
using IndexWrapper = Index<std::decay_t<T>>;

template <typename T>
constexpr IndexWrapper<T> make_index(T &&value) {
  return IndexWrapper<T>(std::forward<T>(value));
}

#endif // CATZILLA_CATZ_INDEX_H_
