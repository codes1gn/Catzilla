#ifndef STORAGE_CONVERT_H
#define STORAGE_CONVERT_H

#include "storage_structure.h"

template <typename T>
void _convert_NCHW_to_NHWC(T* input, T* output, int n, int c, int h, int w) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                for (int l = 0; l < c; l++) {
                    output[i * h * w * c + j * w * c + k * c + l] = input[i * c * h * w + l * h * w + j * w + k];
                }
            }
        }
    }
}

template <typename T>
void _convert_NHWC_to_NCHW(T* input, T* output, int n, int c, int h, int w) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                for (int l = 0; l < c; l++) {
                    output[i * c * h * w + l * h * w + j * w + k] = input[i * h * w * c + j * w * c + k * c + l];
                }
            }
        }
    }
}

template <typename T>
void _convert_KCRS_to_KRSC(T* input, T* output, int k, int c, int r, int s) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < r; j++) {
            for (int k = 0; k < s; k++) {
                for (int l = 0; l < c; l++) {
                    output[i * r * s * c + j * s * c + k * c + l] = input[i * c * r * s + l * r * s + j * s + k];
                }
            }
        }
    }
}

template <typename T>
void _convert_KRSC_to_KCRS(T* input, T* output, int k, int c, int r, int s) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < r; j++) {
            for (int k = 0; k < s; k++) {
                for (int l = 0; l < c; l++) {
                    output[i * c * r * s + l * r * s + j * s + k] = input[i * r * s * c + j * s * c + k * c + l];
                }
            }
        }
    }
}

template <typename T>
void convert_NCHW_to_NHWC(param_t<T> &param) {
    if (!param.is_NCWH) return;
    T *temp = (T *) malloc(param.get_input_size() * sizeof(T));
    _convert_NCHW_to_NHWC(param.input_h, temp, param.n, param.c, param.h, param.w);
    free(param.input_h);
    param.input_h = temp;
    param.is_NCWH = false;
}

template <typename T>
void convert_NHWC_to_NCHW(param_t<T> &param) {
    if (param.is_NCWH) return;
    T *temp = (T *) malloc(param.get_input_size() * sizeof(T));
    _convert_NHWC_to_NCHW(param.input_h, temp, param.n, param.c, param.h, param.w);
    free(param.input_h);
    param.input_h = temp;
    param.is_NCWH = true;
}

template <typename T>
void convert_KCRS_to_KRSC(param_t<T> &param) {
    if (!param.is_KCRS) return;
    T *temp = (T *) malloc(param.get_weight_size() * sizeof(T));
    _convert_KCRS_to_KRSC(param.weight_h, temp, param.k, param.c, param.r, param.s);
    free(param.weight_h);
    param.weight_h = temp;
    param.is_KCRS = false;
}

template <typename T>
void convert_KRSC_to_KCRS(param_t<T> &param) {
    if (param.is_KCRS) return;
    T *temp = (T *) malloc(param.get_weight_size() * sizeof(T));
    _convert_KRSC_to_KCRS(param.weight_h, temp, param.k, param.c, param.r, param.s);
    free(param.weight_h);
    param.weight_h = temp;
    param.is_KCRS = true;
}

#endif // STORAGE_CONVERT_H