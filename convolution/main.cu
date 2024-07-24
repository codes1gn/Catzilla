#include "storage.h"
#include "utils.h"
#include "implgemm_TC.h"
#include "verify.h"

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

struct Arguments {
    int N = 2;
    int C = 32;
    int H = 128;
    int W = 128;
    int K = 128;
    int R = 3;
    int S = 3;
    int U = 1;
    int V = 1;
    int P = 1;
    int Q = 1;
    int Warmup = 5;
    int I = 1000;
};

void parse_arguments(int argc, char* argv[], Arguments& args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-N" && i + 1 < argc) {
            std::istringstream(argv[++i]) >> args.N;
        } else if (arg == "-C" && i + 1 < argc) {
            std::istringstream(argv[++i]) >> args.C;
        } else if (arg == "-H" && i + 1 < argc) {
            std::istringstream(argv[++i]) >> args.H;
        } else if (arg == "-W" && i + 1 < argc) {
            std::istringstream(argv[++i]) >> args.W;
        } else if (arg == "-K" && i + 1 < argc) {
            std::istringstream(argv[++i]) >> args.K;
        } else if (arg == "-R" && i + 1 < argc) {
            std::istringstream(argv[++i]) >> args.R;
        } else if (arg == "-S" && i + 1 < argc) {
            std::istringstream(argv[++i]) >> args.S;
        } else if (arg == "-U" && i + 1 < argc) {
            std::istringstream(argv[++i]) >> args.U;
        } else if (arg == "-V" && i + 1 < argc) {
            std::istringstream(argv[++i]) >> args.V;
        } else if (arg == "-P" && i + 1 < argc) {
            std::istringstream(argv[++i]) >> args.P;
        } else if (arg == "-Q" && i + 1 < argc) {
            std::istringstream(argv[++i]) >> args.Q;
        } else if (arg == "-Warmup" && i + 1 < argc) {
            std::istringstream(argv[++i]) >> args.Warmup;
        } else if (arg == "-I" && i + 1 < argc) {
            std::istringstream(argv[++i]) >> args.I;
        }
    }
}

int main(int argc, char **argv) {
    Arguments args;
    parse_arguments(argc, argv, args);

    int n = args.N;
    int c = args.C;
    int h = args.H;
    int w = args.W;
    int k = args.K;
    int r = args.R;
    int s = args.S;
    int u = args.U;
    int v = args.V;
    int p = args.P;
    int q = args.Q;
    int warmup = args.Warmup;
    int iteration = args.I;

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
