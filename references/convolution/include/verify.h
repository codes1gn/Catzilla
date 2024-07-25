#ifndef VERIFY_H
#define VERIFY_H

#include "storage.h"

#define PRECISION 1.0e-1

float getPrecision(float tmp) {
    int tmpInt = (int) tmp;
    float eNum = 1.0e-6;
    if (abs(tmpInt) > 0) {
        while (tmpInt != 0) {
            tmpInt = (int) (tmpInt / 10);
            eNum *= 10;
        }
    } else {

        if (tmp == 0)
            return eNum;

        eNum = 1.0e-5;

        while (tmpInt == 0) {
            tmp *= 10;
            tmpInt = (int) (tmp);
            eNum /= 10;
        }
    }
    return eNum;
}

// template<typename T>
// T getPrecision(T tmp) {
//
// }

template<typename T>
void conv2d_CPU_NCHW(param_t<T> param, T *output_cpu) {
    if (std::is_integral<T>::value) {
        for (int n = 0; n < param.n; n++) {
            for (int k = 0; k < param.k; k++) {
                for (int oh = 0; oh < param.Oh; oh++) {
                    for (int ow = 0; ow < param.Ow; ow++) {
                        int sum = 0;
                        for (int c = 0; c < param.c; c++) {
                            for (int r = 0; r < param.r; r++) {
                                for (int s = 0; s < param.s; s++) {
                                    int ih = oh * param.u - param.p + r;
                                    int iw = ow * param.v - param.q + s;
                                    if (iw >= 0 && ih >= 0 && iw < param.w && ih < param.h) {
                                        sum += static_cast<int>(param.input_h[n * param.c * param.h * param.w +
                                                                              c * param.h * param.w +
                                                                              ih * param.w + iw]) *
                                               static_cast<int>(param.weight_h[k * param.r * param.s * param.c +
                                                                               c * param.r * param.s +
                                                                               r * param.s + s]);
                                    }
                                }
                            }
                        }
                        output_cpu[n * param.k * param.Oh * param.Ow + k * param.Oh * param.Ow + oh * param.Ow + ow] =
                                static_cast<T>(sum + static_cast<int>(param.bias_h[k]));
                    }
                }
            }
        }
    } else {
        for (int n = 0; n < param.n; n++) {
            for (int k = 0; k < param.k; k++) {
                for (int oh = 0; oh < param.Oh; oh++) {
                    for (int ow = 0; ow < param.Ow; ow++) {
                        float sum = 0;
                        for (int c = 0; c < param.c; c++) {
                            for (int r = 0; r < param.r; r++) {
                                for (int s = 0; s < param.s; s++) {
                                    int ih = oh * param.u - param.p + r;
                                    int iw = ow * param.v - param.q + s;
                                    if (iw >= 0 && ih >= 0 && iw < param.w && ih < param.h) {
                                        sum += static_cast<float>(param.input_h[n * param.c * param.h * param.w +
                                                                                c * param.h * param.w +
                                                                                ih * param.w + iw]) *
                                               static_cast<float>(param.weight_h[k * param.r * param.s * param.c +
                                                                                 c * param.r * param.s +
                                                                                 r * param.s + s]);
                                    }
                                }
                            }
                        }
                        output_cpu[n * param.k * param.Oh * param.Ow + k * param.Oh * param.Ow + oh * param.Ow + ow] =
                                static_cast<T>(sum + static_cast<float>(param.bias_h[k]));
                    }
                }
            }
        }
    }
}

template<typename T>
void conv2d_CPU_NHWC(param_t<T> param, T *output_cpu) {
    if (std::is_integral<T>::value) {
        for (int n = 0; n < param.n; n++) {
            for (int k = 0; k < param.k; k++) {
                for (int oh = 0; oh < param.Oh; oh++) {
                    for (int ow = 0; ow < param.Ow; ow++) {
                        int sum = 0.0;
                        for (int c = 0; c < param.c; c++) {
                            for (int r = 0; r < param.r; r++) {
                                for (int s = 0; s < param.s; s++) {
                                    int ih = oh * param.u - param.p + r;
                                    int iw = ow * param.v - param.q + s;
                                    if (iw >= 0 && ih >= 0 && iw < param.w && ih < param.h) {
                                        sum += static_cast<int>(param.input_h[n * param.h * param.w * param.c +
                                                                              ih * param.w * param.c +
                                                                              iw * param.c + c]) *
                                               static_cast<int>(param.weight_h[k * param.r * param.s * param.c +
                                                                               r * param.s * param.c +
                                                                               s * param.c + c]);
                                    }
                                }
                            }
                        }
                        output_cpu[n * param.k * param.Oh * param.Ow + k * param.Oh * param.Ow + oh * param.Ow + ow] =
                                static_cast<T>(sum + static_cast<int>(param.bias_h[k]));
                    }
                }
            }
        }
    } else {
        for (int n = 0; n < param.n; n++) {
            for (int k = 0; k < param.k; k++) {
                for (int oh = 0; oh < param.Oh; oh++) {
                    for (int ow = 0; ow < param.Ow; ow++) {
                        float sum = 0.0;
                        for (int c = 0; c < param.c; c++) {
                            for (int r = 0; r < param.r; r++) {
                                for (int s = 0; s < param.s; s++) {
                                    int ih = oh * param.u - param.p + r;
                                    int iw = ow * param.v - param.q + s;
                                    if (iw >= 0 && ih >= 0 && iw < param.w && ih < param.h) {
                                        sum += static_cast<float>(param.input_h[n * param.h * param.w * param.c +
                                                                                ih * param.w * param.c +
                                                                                iw * param.c + c]) *
                                               static_cast<float>(param.weight_h[k * param.r * param.s * param.c +
                                                                                 r * param.s * param.c +
                                                                                 s * param.c + c]);
                                    }
                                }
                            }
                        }
                        output_cpu[n * param.k * param.Oh * param.Ow + k * param.Oh * param.Ow + oh * param.Ow + ow] =
                                static_cast<T>(sum + static_cast<float>(param.bias_h[k]));
                    }
                }
            }
        }
    }
}

template<typename T>
void verify(param_t<T> param) {
    bool correct = true;
    T *output_cpu = (T *) malloc(param.get_output_size() * sizeof(T));
    std::cout << "Computing CPU..." << std::endl;
    if (param.is_NCWH) {
        conv2d_CPU_NCHW(param, output_cpu);
    } else {
        conv2d_CPU_NHWC(param, output_cpu);
    }
    std::cout << "Verifying..." << std::endl;
    for (int i = 0; i < param.get_output_size(); i++) {
        int n = i / (param.k * param.Oh * param.Ow);
        int k = (i % (param.k * param.Oh * param.Ow)) / (param.Oh * param.Ow);
        int oh = (i % (param.Oh * param.Ow)) / param.Ow;
        int ow = i % param.Ow;
        if (std::is_integral<T>::value) {
            if (static_cast<int>(output_cpu[i]) != static_cast<int>(param.output_h[i])) {
                std::cout << "Error: " << i << "(" << n << "*" << k << "*" << oh << "*" << ow << ")" << " CPU result: " << static_cast<int>(output_cpu[i]) << " GPU result: "
                          << static_cast<int>(param.output_h[i]) << std::endl;
                correct = false;
                break;
            }
        } else {
            // if (abs(static_cast<float>(output_cpu[i]) - static_cast<float>(param.output_h[i])) >
            //     getPrecision(static_cast<float>(output_cpu[i]))) {
            if (abs(static_cast<float>(output_cpu[i]) - static_cast<float>(param.output_h[i])) >
                PRECISION) {
                std::cout << "Error: " << i << "(" << n << "*" << k << "*" << oh << "*" << ow << ")" << " CPU result: " << static_cast<float>(output_cpu[i]) << " GPU result: "
                          << static_cast<float>(param.output_h[i]) << std::endl;
                correct = false;
                break;
            }
        }
    }
    if (correct)
        std::cout << "Correct!" << std::endl;
}

#endif //VERIFY_H