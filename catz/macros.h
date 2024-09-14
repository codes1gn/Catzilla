#ifndef CATZILLA_RECIPES_UTILS_MACROS_H_
#define CATZILLA_RECIPES_UTILS_MACROS_H_

#define CEIL_DIV(dividend, divisor) (((dividend) + (divisor)-1) / (divisor))

#define MAKE_SHARED(matrixVar, shape)                                          \
  __shared__ float matrixVar##_data[shape.x * shape.y];                        \
  Matrix matrixVar(matrixVar##_data, shape)

#endif // CATZILLA_RECIPES_UTILS_MACROS_H_
