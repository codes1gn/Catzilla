
#ifndef CONV_UTILS_H
#define CONV_UTILS_H

template<int M_, int K_, int N_>
struct ShapeBase {
    static constexpr int M = M_, K = K_, N = N_;
};

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      printf("*** CUDA Error *** at [%s:%d] error=%d, reason:%s \n",    \
            __FILE__, __LINE__, error, cudaGetErrorString(error));      \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
}

#endif //CONV_UTILS_H
