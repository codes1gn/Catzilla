import argparse
import time

import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument('-N', '--batch_size', default=1, type=int,
                    help='number of batch size(default: 1)')
parser.add_argument('-C', '--in_channel', default=3, type=int,
                    help='number of in channel(default:3)')
parser.add_argument('-H', '--height', default=128, type=int,
                    help='height of input data(default:128)')
parser.add_argument('-W', '--width', default=128, type=int,
                    help='width of input data(default:128)')
parser.add_argument('-K', '--out_channel', default=64, type=int,
                    help='number of out channel(default:64)')
parser.add_argument('-R', '--kernel_size_h', default=3, type=int,
                    help='height of kernel(default:3)')
parser.add_argument('-S', '--kernel_size_w', default=3, type=int,
                    help='width of kernel(default:3)')
parser.add_argument('-U', '--stride_h', default=1, type=int,
                    help='stride on dimension height(default:1)')
parser.add_argument('-V', '--stride_w', default=1, type=int,
                    help='stride on dimension width(default:1)')
parser.add_argument('-P', '--padding_h', default=1, type=int,
                    help='padding on dimension height(default:1)')
parser.add_argument('-Q', '--padding_w', default=1, type=int,
                    help='padding on dimension width(default:1)')

parser.add_argument('--warmup', default=5, type=int,
                    help='warmup iteration(default:5)')
parser.add_argument('-I', '--iteration', default=1000, type=int,
                    help='iteration(default:1000)')

args = parser.parse_args()


input = torch.rand(args.batch_size, args.in_channel, args.height, args.width).float().cuda()
filter = torch.rand(args.out_channel, args.in_channel, args.kernel_size_h, args.kernel_size_w).float().cuda()

output = F.conv2d(input, filter, padding=args.padding_h, stride=args.stride_h)

for i in range(args.warmup):
    output = F.conv2d(input, filter, padding=args.padding_h, stride=args.stride_h)

start_time = time.time()
for i in range(args.iteration):
    output = F.conv2d(input, filter, padding=args.padding_h, stride=args.stride_h)
torch.cuda.synchronize()
end_time = time.time()

average_elapsed_time_ms = (end_time - start_time) * 1000.0 / args.iteration

# 计算GFLOPS
# 计算输出的高度和宽度
output_height = (args.height + 2 * args.padding_h - args.kernel_size_h) // args.stride_h + 1
output_width = (args.width + 2 * args.padding_w - args.kernel_size_w) // args.stride_w + 1

# 计算总操作数
total_operations = 2.0 * output_height * output_width * (args.kernel_size_h * args.kernel_size_w * args.in_channel * args.out_channel) * args.batch_size

# 计算TOPS
gflops = total_operations / ((end_time - start_time) / args.iteration) / 1e9

print("N,C,H,W,K,R,S,U,V,P,Q,Average_elapsed_time(ms),GFLOPS")
print(str(args.batch_size) + ',' + str(args.in_channel) + ',' + str(args.height) + ',' + str(args.width) + ',' + str(args.out_channel) + ',' + str(args.kernel_size_h) + ',' + str(args.kernel_size_w) + ',' + str(args.stride_h) + ',' + str(args.stride_w) + ',' + str(args.padding_h) + ',' + str(args.padding_w) + ',' + str(average_elapsed_time_ms) + ',' + str(gflops))
