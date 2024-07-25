#include "argparse/argparse.hpp"
#include "storage.h"
#include "utils.h"
#include "../../implgemm_temp.h"
#include "verify.h"

int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("storage_convert_test");
    parser.add_argument("-N").help("Dimension N of the input matrix").default_value(2).scan<'i', int>();
    parser.add_argument("-C").help("Dimension C of the input matrix").default_value(2).scan<'i', int>();
    parser.add_argument("-H").help("Dimension H of the input matrix").default_value(4).scan<'i', int>();
    parser.add_argument("-W").help("Dimension W of the input matrix").default_value(4).scan<'i', int>();
    parser.add_argument("-K").help("Dimension K of the weight matrix").default_value(2).scan<'i', int>();
    parser.add_argument("-R").help("Dimension R of the weight matrix").default_value(3).scan<'i', int>();
    parser.add_argument("-S").help("Dimension S of the weight matrix").default_value(3).scan<'i', int>();
    parser.add_argument("-U").help("Stride on dimension H").default_value(1).scan<'i', int>();
    parser.add_argument("-V").help("Stride on dimension W").default_value(1).scan<'i', int>();
    parser.add_argument("-P").help("Padding on dimension H").default_value(1).scan<'i', int>();
    parser.add_argument("-Q").help("Padding on dimension W").default_value(1).scan<'i', int>();

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

    using type = float;
    param_t<type> param(n, c, h, w, k, r, s, u, v, p, q);

    create_storage_random<type>(param);
    sync_to_device(param);
    print_input(param);
    print_weight(param);

    using BlockShape = ShapeBase<128, 32, 64>;
    using WarpShape = ShapeBase<32, 32, 64>;
    using MmaShape = ShapeBase<16, 32, 8>;

    // run once for correctness verification
    implgemm_temp_exec<type, BlockShape, WarpShape, MmaShape, 2>(param);
    CUDA_CHECK(cudaDeviceSynchronize());
    sync_to_host(param);
    print_output(param);
    verify(param);

    convert_NCHW_to_NHWC(param);
    convert_KCRS_to_KRSC(param);
    print_input(param);
    print_weight(param);
    verify(param);
}