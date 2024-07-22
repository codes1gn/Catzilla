#include "argparse/argparse.hpp"
#include "storage.h"
#include "utils.h"
#include "implgemm_TC.h"
#include "verify.h"

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

    printf("n: %d, c: %d, h: %d, w: %d, k: %d, r: %d, s: %d, u: %d, v: %d, p: %d, q: %d\n", n, c, h, w, k, r, s, u, v, p, q);

    using type = float;
    param_t<type> param(n, c, h, w, k, r, s, u, v, p, q);

    printf("Ow: %d, Oh: %d\n", param.Ow, param.Oh);

    create_storage_random<type>(param);
    // sync_to_device(param);
    // print_input(param);
    // print_weight(param);

    using BlockShape = ShapeBase<128, 32, 16*4>;
    using WarpShape = ShapeBase<BlockShape::M/4, BlockShape::K, BlockShape::N>;
    using MmaShape = ShapeBase<16, 8, 8>;

    // run once for correctness verification
    convert_NCHW_to_NHWC(param);
    convert_KCRS_to_KRSC(param);
    sync_to_device(param);
    // print_input(param);
    // print_weight(param);
    // print_partial_input(param, 1, 2, param.w, param.c);
    // print_partial_weight(param, 16, 0, 0, 0, 1, 1, 1, param.c);

    implgemm_TC_exec<type, BlockShape, WarpShape, MmaShape, 2>(param);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    sync_to_host(param);
    // print_output(param);

    verify(param);

    // run multiple times for performance measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms = 0.0f;
    // warm up
    for (int i = 0; i < warmup; i++) {
        implgemm_TC_exec<type, BlockShape, WarpShape, MmaShape, 2>(param);
    }
    cudaDeviceSynchronize();
    // measure time
    cudaEventRecord(start, 0);
    for (int i = 0; i < iteration; i++) {
        implgemm_TC_exec<type, BlockShape, WarpShape, MmaShape, 2>(param);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CHECK(cudaGetLastError());
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Type,N,C,H,W,K,R,S,U,V,P,Q,Average_elapsed_time(ms),GFLOPS\n");
    printf("ImplGEMM,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f\n", n, c, h, w, k, r, s, u, v, p, q, time_ms / iteration, 2.0 * param.Oh * param.Ow * n * c * k * r * s / (time_ms / iteration) / 1e6);
}