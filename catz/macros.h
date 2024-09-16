#ifndef CATZILLA_RECIPES_UTILS_MACROS_H_
#define CATZILLA_RECIPES_UTILS_MACROS_H_

#define CEIL_DIV(dividend, divisor) (((dividend) + (divisor)-1) / (divisor))

#define MAKE_SHARED(matrixVar, size, type)                                     \
  __shared__ type matrixVar##_data[size];                                      \
  Matrix<type> matrixVar = Matrix<type>(matrixVar##_data, Coord(1, size))

#define MAKE_SHARED(matrixVar, size_x, size_y, type)                           \
  __shared__ type matrixVar##_data[size_x * size_y];                           \
  Matrix<type> matrixVar = Matrix<type>(matrixVar##_data, Coord(size_x, size_y))

#endif // CATZILLA_RECIPES_UTILS_MACROS_H_
