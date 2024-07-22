/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "utils/helpers.h"

#include <cudnn_frontend.h>
#include "utils/helpers.h"

#include "argparse/argparse.hpp"

int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("storage_convert_test");
    parser.add_argument("-N").help("Dimension N of the input matrix").default_value(2).scan<'i', int>();
    parser.add_argument("-C").help("Dimension C of the input matrix").default_value(32).scan<'i', int>();
    parser.add_argument("-H").help("Dimension H of the input matrix").default_value(128).scan<'i', int>();
    parser.add_argument("-W").help("Dimension W of the input matrix").default_value(128).scan<'i', int>();
    parser.add_argument("-K").help("Dimension K of the weight matrix").default_value(128).scan<'i', int>();
    parser.add_argument("-R").help("Dimension R of the weight matrix").default_value(3).scan<'i', int>();
    parser.add_argument("-S").help("Dimension S of the weight matrix").default_value(3).scan<'i', int>();
    parser.add_argument("-U").help("Stride on dimension H").default_value(1).scan<'i', int>();
    parser.add_argument("-V").help("Stride on dimension W").default_value(1).scan<'i', int>();
    parser.add_argument("-P").help("Padding on dimension H").default_value(1).scan<'i', int>();
    parser.add_argument("-Q").help("Padding on dimension W").default_value(1).scan<'i', int>();

    parser.add_argument("-Warmup").help("Warmup iteration(default:5)").default_value(5).scan<'i', int>();
    parser.add_argument("-I").help("Iteration(default:1000)").default_value(1000).scan<'i', int>();

    parser.parse_args(argc, argv);

    int n = parser.get<int>("-N");
    int c = parser.get<int>("-C");
    int h = parser.get<int>("-H");
    int w = parser.get<int>("-W");
    int k = parser.get<int>("-K");
    int r = parser.get<int>("-R");
    int s = parser.get<int>("-S");
    int u = parser.get<int>("-U");
    int v = parser.get<int>("-V");
    int p = parser.get<int>("-P");
    int q = parser.get<int>("-Q");
    int warmup = parser.get<int>("-Warmup");
    int iteration = parser.get<int>("-I");


    namespace fe = cudnn_frontend;

    if (is_arch_supported_by_cudnn() == false) {
        std::cerr << "Architecture is not supported by currend cudnn version" << std::endl;
        return 1;
    }

    int oh = (h - r + 2 * p) / u + 1;
    int ow = (w - s + 2 * q) / v + 1;

    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

        auto X = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("image")
                                       .set_dim({n, c, h, w})
                                       .set_stride({c * h * w, 1, c * w, c}));

        auto W = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("filter")
                                       .set_dim({k, c, r, s})
                                       .set_stride({c * r * s, 1, c * s, c}));

        auto conv_options =
                fe::graph::Conv_fprop_attributes().set_padding({p, q}).set_stride({u, v}).set_dilation({1, 1});
        auto Y = graph->conv_fprop(X, W, conv_options);

        Y->set_output(true);

        REQUIRE(graph->validate().is_good());

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        REQUIRE(graph->build_plans(handle).is_good());

        return std::make_tuple(graph, X, W, Y);
    };

    cudnnHandle_t handle;

    checkCudnnErr(cudnnCreate(&handle));

    auto [graph, X, W, Y] = build_new_graph(handle);

    Surface<float> x_tensor(n * c * h * w, false);
    Surface<float> w_tensor(k * c * r * s, false);
    Surface<float> y_tensor(n * k * oh * ow, false);  // Should be p, q.

    std::unordered_map<int64_t, void*> variant_pack = {
            {X->get_uid(), x_tensor.devPtr}, {W->get_uid(), w_tensor.devPtr}, {Y->get_uid(), y_tensor.devPtr}};

    Surface<int8_t> workspace(graph->get_workspace_size(), false);

//    std::cout << *graph << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup; ++i) {
        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    for (int i = 0; i < iteration; ++i) {
        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    }
    cudaEventRecord(stop, 0);
    float time_elapsed = 0.0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    double flopsPerConv = n * oh * ow * 1e-9f * k * c * r * s * 2.0;
    float timePerConv = time_elapsed / iteration;
    double gflops = flopsPerConv / (timePerConv / 1000.0f);
    printf("N,C,H,W,K,R,S,U,V,P,Q,Average_elapsed_time(ms),GFLOPS\n");
    printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f\n", n, c, h, w, k, r, s, u, v, p, q, timePerConv, gflops);

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}
