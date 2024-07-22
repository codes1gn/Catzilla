#ifndef CONV_STORAGE_TRANSFER_H
#define CONV_STORAGE_TRANSFER_H

#include "utils.h"

template<typename T>
void input_h2d(param_t<T> &param) {
    CUDA_CHECK(cudaMalloc((void **) &param.input_d, param.get_input_size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(param.input_d, param.input_h, param.get_input_size() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void weight_h2d(param_t<T> &param) {
    CUDA_CHECK(cudaMalloc((void **) &param.weight_d, param.get_weight_size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(param.weight_d, param.weight_h, param.get_weight_size() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void bias_h2d(param_t<T> &param) {
    CUDA_CHECK(cudaMalloc((void **) &param.bias_d, param.get_bias_size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(param.bias_d, param.bias_h, param.get_bias_size() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void output_d2h(param_t<T> &param) {
    CUDA_CHECK(cudaMemcpy(param.output_h, param.output_d, param.get_output_size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void sync_to_device(param_t<T> &param) {
    input_h2d(param);
    weight_h2d(param);
    bias_h2d(param);
    // output on device should be allocated
    CUDA_CHECK(cudaMalloc((void **) &param.output_d, param.get_output_size() * sizeof(T)));
}

template<typename T>
void sync_to_host(param_t<T> &param) {
    output_d2h(param);
}

#endif //CONV_STORAGE_TRANSFER_H
