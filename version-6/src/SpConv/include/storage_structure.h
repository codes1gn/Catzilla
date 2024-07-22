#ifndef CONV2D_H
#define CONV2D_H

#include <cuda_fp16.h>

template<typename T>
class param_t {
public:
    T *input_h;                                   //输入数据地址
    T *weight_h;                                  //权值数据地址
    T *bias_h;                                    //偏置值数据地址
    T *output_h;                                  //输出数据地址
    T *input_d;                                   //输入数据地址
    T *weight_d;                                  //权值数据地址
    T *bias_d;                                    //偏置值数据地址
    T *output_d;                                  //输出数据地址
    unsigned int n;                              //batch szie
    unsigned int c;                              //channel number
    unsigned int h;                              //数据高
    unsigned int w;                              //数据宽
    unsigned int k;                              //卷积核数量
    unsigned int r;                              //卷积核高
    unsigned int s;                              //卷积核宽
    unsigned int u;                              //卷积在高方向上的步长
    unsigned int v;                              //卷积在宽方向上的步长
    unsigned int p;                              //卷积在高方向上的补边
    unsigned int q;                              //卷积在宽方向上的补边
    unsigned int Oh;                             //卷积结果高
    unsigned int Ow;                             //卷积结果宽
    bool is_NCWH;                                   //是否是NCWH格式,true表示是NCWH格式，false表示是NCHW格式
    bool is_KCRS;                                   //是否是KCRS格式,true表示是KCRS格式，false表示是KRSC格式

    param_t(int n, int c, int h, int w, int k, int r, int s, int u, int v, int p, int q)
            : input_h(NULL), weight_h(NULL), bias_h(NULL), output_h(NULL),
              input_d(NULL), weight_d(NULL), bias_d(NULL), output_d(NULL),
              is_NCWH(true), is_KCRS(true),
              n(n), c(c), h(h), w(w), k(k), r(r), s(s), u(u), v(v), p(p), q(q) {
        Oh = (h - r + 2 * p) / u + 1;
        Ow = (w - s + 2 * q) / v + 1;
    }

    unsigned int get_input_size() {
        return n * c * h * w;
    }

    unsigned int get_weight_size() {
        return k * c * r * s;
    }

    unsigned int get_bias_size() {
        return k;
    }

    unsigned int get_output_size() {
        return n * k * Oh * Ow;
    }

};
// void launch_implgemm(param_t param);

#endif //CONV2D_H