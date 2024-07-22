## Install

### Download
```shell
git clone git@github.com:KevinWu2017/SpConv.git
cd SpConv
git submodule update --init --recursive
```

### Install dependencies
#### cuDNN
If your system has no cudnn installed, you can download it from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) and install it.

#### PyTorch
You can create a python environment with conda and install pytorch with the following command. 
(Make sure the installed Pytorch version is compatible with your system.)
```shell
conda create -n SpConv python=3.11
conda activate SpConv
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Build
Run the following command in project root to build the project.
```shell
./scripts/build.sh <your-gen-code>
# For example, ./scripts/build.sh 80
```

### Benchmark
Run the following command in project root to benchmark the project.
```shell
./scripts/run_baselines.sh
```
