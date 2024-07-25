//
// Created by kiki on 24-7-1.
//

#ifndef CONV_DATA_STRUCTURE_HELPER_H
#define CONV_DATA_STRUCTURE_HELPER_H

#include <cuda_fp16.h>

#include "utils.h"

typedef ShapeBase<16, 8, 8> Shape_16_8_8;

template<typename Shape>
struct A_fragment;
template<typename Shape>
struct B_fragment;
template<typename Shape, typename AccumulatorType>
struct C_fragment;
template<typename Shape>
struct Meta_fragment;

template<>
struct A_fragment<Shape_16_8_8> {
    float x[4];
};
template<>
struct B_fragment<Shape_16_8_8> {
    float x[2];
};
template<>
struct C_fragment<Shape_16_8_8, float> {
    float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

#endif //CONV_DATA_STRUCTURE_HELPER_H
