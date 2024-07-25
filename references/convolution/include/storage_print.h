#ifndef CONV_STORAGE_PRINT_H
#define CONV_STORAGE_PRINT_H

#include <iostream>
#include<iomanip>
#include <stdio.h>
#include "storage_structure.h"

template <typename T>
void print_input_NCHW(param_t<T> param) {
    for (int i = 0; i < param.n; i++) {
        for (int j = 0; j < param.c; j++) {
            for (int k = 0; k < param.h; k++) {
                i == 0 && j == 0 && k == 0 ? std::cout << "[ " : std::cout << "  ";
                j == 0 && k == 0 ? std::cout << "[ " : std::cout << "  ";
                k == 0 ? std::cout << "[ " : std::cout << "  ";
                std::cout << "[ ";
                for (int l = 0; l < param.w; l++) {
                    if (std::is_integral<T>::value) {
                        std::cout << static_cast<int>(
                                param.input_h[i * param.c * param.h * param.w +
                                            j * param.h * param.w + k * param.w + l]) << " ";
                    } else {
                        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << static_cast<float>(
                                param.input_h[i * param.c * param.h * param.w +
                                            j * param.h * param.w + k * param.w + l]) << " ";
                    }
                }
                std::cout << "] ";
                k == param.h - 1 ? std::cout << "] " : std::cout << "  ";
                j == param.c - 1 && k == param.h - 1 ? std::cout << "] " : std::cout << "  ";
                i == param.n - 1 && j == param.c - 1 && k == param.h - 1 ? std::cout << "] " : std::cout << "  ";
                std::cout << std::endl;
            }
        }
    }
}

template <typename T>
void print_input_NHWC(param_t<T> param) {
    for (int i = 0; i < param.n; i++) {
        for (int j = 0; j < param.h; j++) {
            for (int k = 0; k < param.w; k++) {
                i == 0 && j == 0 && k == 0 ? std::cout << "[ " : std::cout << "  ";
                j == 0 && k == 0 ? std::cout << "[ " : std::cout << "  ";
                k == 0 ? std::cout << "[ " : std::cout << "  ";
                std::cout << "[ ";
                for (int l = 0; l < param.c; l++) {
                    if (std::is_integral<T>::value) {
                        std::cout << static_cast<int>(
                                param.input_h[i * param.h * param.w * param.c +
                                            j * param.w * param.c + k * param.c + l]) << " ";
                    } else {
                        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << static_cast<float>(
                                                    param.input_h[i * param.h * param.w * param.c +
                                                                j * param.w * param.c + k * param.c + l]) << " ";
                    }
                }
                std::cout << "] ";
                k == param.w - 1 ? std::cout << "] " : std::cout << "  ";
                j == param.h - 1 && k == param.w - 1 ? std::cout << "] " : std::cout << "  ";
                i == param.n - 1 && j == param.h - 1 && k == param.w - 1 ? std::cout << "] " : std::cout << "  ";
                std::cout << std::endl;
            }
        }
    }
}

template <typename T>
void print_input_NHWC(param_t<T> param, int print_n, int print_h, int print_w, int print_c) {
    for (int i = 0; i < print_n; i++) {
        for (int j = 0; j < print_h; j++) {
            for (int k = 0; k < print_w; k++) {
                i == 0 && j == 0 && k == 0 ? std::cout << "[ " : std::cout << "  ";
                j == 0 && k == 0 ? std::cout << "[ " : std::cout << "  ";
                k == 0 ? std::cout << "[ " : std::cout << "  ";
                std::cout << "[ ";
                for (int l = 0; l < print_c; l++) {
                    if (std::is_integral<T>::value) {
                        std::cout << static_cast<int>(
                                param.input_h[i * param.h * param.w * param.c +
                                              j * param.w * param.c + k * param.c + l]) << " ";
                    } else {
                        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << static_cast<float>(
                                param.input_h[i * param.h * param.w * param.c +
                                              j * param.w * param.c + k * param.c + l]) << " ";
                    }
                }
                std::cout << "] ";
                k == param.w - 1 ? std::cout << "] " : std::cout << "  ";
                j == param.h - 1 && k == param.w - 1 ? std::cout << "] " : std::cout << "  ";
                i == param.n - 1 && j == param.h - 1 && k == param.w - 1 ? std::cout << "] " : std::cout << "  ";
                std::cout << std::endl;
            }
        }
    }
}

template <typename T>
void print_weight_KCRS(param_t<T> param) {
    for (int i = 0; i < param.k; i++) {
        for (int j = 0; j < param.c; j++) {
            for (int k = 0; k < param.r; k++) {
                i == 0 && j == 0 && k == 0 ? std::cout << "[ " : std::cout << "  ";
                j == 0 && k == 0 ? std::cout << "[ " : std::cout << "  ";
                k == 0 ? std::cout << "[ " : std::cout << "  ";
                std::cout << "[ ";
                for (int l = 0; l < param.s; l++) {
                    if (std::is_integral<T>::value) {
                        std::cout << static_cast<int>(
                                param.weight_h[i * param.c * param.r * param.s +
                                               j * param.r * param.s + k * param.s + l]) << " ";
                    } else {
                        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << static_cast<float>(
                                param.weight_h[i * param.c * param.r * param.s +
                                               j * param.r * param.s + k * param.s + l]) << " ";
                    }
                }
                std::cout << "] ";
                k == param.r - 1 ? std::cout << "] " : std::cout << "  ";
                j == param.c - 1 && k == param.r - 1 ? std::cout << "] " : std::cout << "  ";
                i == param.k - 1 && j == param.c - 1 && k == param.r - 1 ? std::cout << "] " : std::cout << "  ";
                std::cout << std::endl;
            }
        }
    }
}

template <typename T>
void print_weight_KRSC(param_t<T> param) {
    for (int i = 0; i < param.k; i++) {
        for (int j = 0; j < param.r; j++) {
            for (int k = 0; k < param.s; k++) {
                i == 0 && j == 0 && k == 0 ? std::cout << "[ " : std::cout << "  ";
                j == 0 && k == 0 ? std::cout << "[ " : std::cout << "  ";
                k == 0 ? std::cout << "[ " : std::cout << "  ";
                std::cout << "[ ";
                for (int l = 0; l < param.c; l++) {
                    if (std::is_integral<T>::value) {
                        std::cout << static_cast<int>(
                                param.weight_h[i * param.r * param.s * param.c +
                                               j * param.s * param.c + k * param.c + l]) << " ";
                    } else {
                        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << static_cast<float>(
                                param.weight_h[i * param.r * param.s * param.c +
                                               j * param.s * param.c + k * param.c + l]) << " ";
                    }
                }
                std::cout << "] ";
                k == param.s - 1 ? std::cout << "] " : std::cout << "  ";
                j == param.r - 1 && k == param.s - 1 ? std::cout << "] " : std::cout << "  ";
                i == param.k - 1 && j == param.r - 1 && k == param.s - 1 ? std::cout << "] " : std::cout << "  ";
                std::cout << std::endl;
            }
        }
    }
}

template <typename T>
void print_weight_KRSC(param_t<T> param, int start_k, int start_r, int start_s, int start_c,
                       int print_k, int print_r, int print_s, int print_c) {
    for (int i = start_k; i < start_k + print_k; i++) {
        for (int j = start_r; j < start_r + print_r; j++) {
            for (int k = start_s; k < start_s + print_s; k++) {
                i == 0 && j == 0 && k == 0 ? std::cout << "[ " : std::cout << "  ";
                j == 0 && k == 0 ? std::cout << "[ " : std::cout << "  ";
                k == 0 ? std::cout << "[ " : std::cout << "  ";
                std::cout << "[ ";
                for (int l = start_c; l < start_c + print_c; l++) {
                    if (std::is_integral<T>::value) {
                        std::cout << static_cast<int>(
                                param.weight_h[i * param.r * param.s * param.c +
                                               j * param.s * param.c + k * param.c + l]) << " ";
                    } else {
                        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << static_cast<float>(
                                param.weight_h[i * param.r * param.s * param.c +
                                               j * param.s * param.c + k * param.c + l]) << " ";
                    }
                }
                std::cout << "] ";
                k == param.s - 1 ? std::cout << "] " : std::cout << "  ";
                j == param.r - 1 && k == param.s - 1 ? std::cout << "] " : std::cout << "  ";
                i == param.k - 1 && j == param.r - 1 && k == param.s - 1 ? std::cout << "] " : std::cout << "  ";
                std::cout << std::endl;
            }
        }
    }
}

template <typename T>
void print_input(param_t<T> param) {
    std::cout << "Input: " << std::endl;
    if (param.is_NCWH) {
        print_input_NCHW(param);
    } else {
        print_input_NHWC(param);
    }
}

template <typename T>
void print_weight(param_t<T> param) {
    std::cout << "Weight: " << std::endl;
    if (param.is_KCRS) {
        print_weight_KCRS(param);
    } else {
        print_weight_KRSC(param);
    }
}

template <typename T>
void print_partial_input(param_t<T> param, int print_n, int print_h, int print_w, int print_c) {
    std::cout << "Input: " << std::endl;
    if (param.is_NCWH) {
        std::cerr << "Not support yet" << std::endl;
    } else {
        print_input_NHWC(param, print_n, print_h, print_w, print_c);
    }
}



template <typename T>
void print_partial_weight(param_t<T> param, int print_k, int print_r, int print_s, int print_c) {
    std::cout << "Weight: " << std::endl;
    if (param.is_KCRS) {
        std::cerr << "Not support yet" << std::endl;
    } else {
        print_weight_KRSC(param, 0, 0, 0, 0, print_k, print_r, print_s, print_c);
    }
}

template <typename T>
void print_partial_weight(param_t<T> param, int start_k, int start_r, int start_s, int start_c,
                          int print_k, int print_r, int print_s, int print_c) {
    std::cout << "Weight: " << std::endl;
    if (param.is_KCRS) {
        std::cerr << "Not support yet" << std::endl;
    } else {
        print_weight_KRSC(param, start_k, start_r, start_s, start_c, print_k, print_r, print_s, print_c);
    }
}

template <typename T>
void print_output(param_t<T> param) {
    std::cout << "Output: " << std::endl;
    for (int i = 0; i < param.n; i++) {
        for (int j = 0; j < param.k; j++) {
            for (int k = 0; k < param.Oh; k++) {
                i == 0 && j == 0 && k == 0 ? std::cout << "[ " : std::cout << "  ";
                j == 0 && k == 0 ? std::cout << "[ " : std::cout << "  ";
                k == 0 ? std::cout << "[ " : std::cout << "  ";
                std::cout << "[ ";
                for (int l = 0; l < param.Ow; l++) {
                    if (std::is_integral<T>::value) {
                        std::cout << static_cast<int>(
                                param.output_h[i * param.k * param.Oh * param.Ow +
                                               j * param.Oh * param.Ow + k * param.Ow + l]) << " ";
                    } else {
                        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << static_cast<float>(
                                param.output_h[i * param.k * param.Oh * param.Ow +
                                               j * param.Oh * param.Ow + k * param.Ow + l]) << " ";
                    }
                }
                std::cout << "] ";
                k == param.Oh - 1 ? std::cout << "] " : std::cout << "  ";
                j == param.k - 1 && k == param.Oh - 1 ? std::cout << "] " : std::cout << "  ";
                i == param.n - 1 && j == param.k - 1 && k == param.Oh - 1 ? std::cout << "] " : std::cout << "  ";
                std::cout << std::endl;
            }
        }
    }
}

#endif //CONV_STORAGE_PRINT_H
