#ifndef CONV_STORAGE_CREATE_H
#define CONV_STORAGE_CREATE_H

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "storage_structure.h"

template<typename T>
void create_storage_random(param_t<T> &param) {
    T *input = (T *) malloc(param.get_input_size() * sizeof(T));
    T *weight = (T *) malloc(param.get_weight_size() * sizeof(T));
    T *bias = (T *) malloc(param.get_bias_size() * sizeof(T));
    T *output = (T *) malloc(param.get_output_size() * sizeof(T));

    for (int i = 0; i < param.get_input_size(); i++) {
        std::is_integral<T>::value ? input[i] = static_cast<T>(rand() % 10) : input[i] = static_cast<T>(double(rand()) / RAND_MAX);
    }
    for (int i = 0; i < param.get_weight_size(); i++) {
        std::is_integral<T>::value ? weight[i] = static_cast<T>(rand() % 10) : weight[i] = static_cast<T>(double(rand()) / RAND_MAX);
    }
    for (int i = 0; i < param.get_bias_size(); i++) {
        // std::is_integral<T>::value ? bias[i] = static_cast<T>(rand() % 10) : bias[i] = static_cast<T>(double(rand()) / RAND_MAX);
        bias[i] = static_cast<T>(0);
    }

    param.input_h = input;
    param.weight_h = weight;
    param.bias_h = bias;
    param.output_h = output;
    param.is_NCWH = true;
    param.is_KCRS = true;
}

template<typename T>
void create_storage_order(param_t<T> param) {
    T *input = (T *) malloc(param.get_input_size() * sizeof(T));
    T *weight = (T *) malloc(param.get_weight_size() * sizeof(T));
    T *bias = (T *) malloc(param.get_bias_size() * sizeof(T));
    T *output = (T *) malloc(param.get_output_size() * sizeof(T));

    for (int i = 0; i < param.get_input_size(); i++) {
        std::is_integral<T>::value ? input[i] = static_cast<T>(rand() % 10) : input[i] = static_cast<T>(double(rand()) / RAND_MAX);
    }
    for (int i = 0; i < param.get_weight_size(); i++) {
        std::is_integral<T>::value ? weight[i] = static_cast<T>(rand() % 10) : weight[i] = static_cast<T>(double(rand()) / RAND_MAX);
    }
    for (int i = 0; i < param.get_bias_size(); i++) {
        std::is_integral<T>::value ? bias[i] = static_cast<T>(rand() % 10) : bias[i] = static_cast<T>(double(rand()) / RAND_MAX);
    }

    param.input_h = input;
    param.weight_h = weight;
    param.bias_h = bias;
    param.output_h = output;
    param.is_NCWH = true;
    param.is_KCRS = true;
}

#endif //CONV_STORAGE_CREATE_H
