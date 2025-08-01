.PHONY: all build debug clean profile report-profile bench bench-all cuobjdump todo

CMAKE := cmake

BUILD_DIR := build
BENCH_DIR := __bench_cache__
BENCHMARK_DIR := __profile_cache__
CUDA_COMPUTE_CAPABILITY ?= sm_35
DEVICE_IDX ?= 0
KERNEL ?= 1

# 根据不同的计算能力版本设置CUDA_COMPUTE_CAPABILITY
GPU_CC=$(shell nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader)
ifeq ($(GPU_CC),3.0)
    CUDA_COMPUTE_CAPABILITY := 30
else ifeq ($(GPU_CC),3.5)
    CUDA_COMPUTE_CAPABILITY := 35
else ifeq ($(GPU_CC),3.7)
    CUDA_COMPUTE_CAPABILITY := 37
else ifeq ($(GPU_CC),5.0)
    CUDA_COMPUTE_CAPABILITY := 50
else ifeq ($(GPU_CC),5.2)
    CUDA_COMPUTE_CAPABILITY := 52
else ifeq ($(GPU_CC),5.3)
    CUDA_COMPUTE_CAPABILITY := 53
else ifeq ($(GPU_CC),6.0)
    CUDA_COMPUTE_CAPABILITY := 60
else ifeq ($(GPU_CC),6.1)
    CUDA_COMPUTE_CAPABILITY := 61
else ifeq ($(GPU_CC),6.2)
    CUDA_COMPUTE_CAPABILITY := 62
else ifeq ($(GPU_CC),7.0)
    CUDA_COMPUTE_CAPABILITY := 70
else ifeq ($(GPU_CC),7.2)
    CUDA_COMPUTE_CAPABILITY := 72
else ifeq ($(GPU_CC),7.5)
    CUDA_COMPUTE_CAPABILITY := 75
else ifeq ($(GPU_CC),8.0)
    CUDA_COMPUTE_CAPABILITY := 80
else ifeq ($(GPU_CC),8.6)
    CUDA_COMPUTE_CAPABILITY := 86
else ifeq ($(GPU_CC),8.9)
    CUDA_COMPUTE_CAPABILITY := 89
else ifeq ($(GPU_CC),9.0)
    CUDA_COMPUTE_CAPABILITY := 90
else
    $(error Unsupported GPU compute capability: $(GPU_CC))
endif

all: build

GPU_NAMES := $(shell nvidia-smi --query-gpu=gpu_name --format=csv,noheader)

query-gpu-arch:
	@echo "Checking GPU architectures..."
	@echo "Found GPU-CC = "$(CUDA_COMPUTE_CAPABILITY)

build: query-gpu-arch
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Release .. \
		-DCUDA_COMPUTE_CAPABILITY=$(CUDA_COMPUTE_CAPABILITY) \
		-DCMAKE_C_COMPILER_LAUNCHER=ccache \
		-DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DCMAKE_CUDA_COMPILER=nvcc \
		-DCMAKE_CUDA_FLAGS="-O3 -maxrregcount=128 --ptxas-options=-v --expt-relaxed-constexpr" \
		-GNinja
	@ninja -C $(BUILD_DIR)

debug: query-gpu-arch
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=RelWithDebInfo .. \
		-DCUDA_COMPUTE_CAPABILITY=$(CUDA_COMPUTE_CAPABILITY) \
		-DCMAKE_C_COMPILER_LAUNCHER=ccache \
		-DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DCMAKE_CUDA_COMPILER=nvcc \
		-DCMAKE_CUDA_FLAGS="-maxrregcount=128 --ptxas-options=-v --expt-relaxed-constexpr" \
		-GNinja
	@ninja -C $(BUILD_DIR)

clean:
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BENCH_DIR)

FUNCTION := $$(cuobjdump -symbols build/matmul | grep -i Warptiling | awk '{print $$NF}')

cuobjdump: build
	@cuobjdump -arch sm_86 -sass -fun $(FUNCTION) $(BUILD_DIR)/bin/catzilla-matmul | c++filt > build/cuobjdump.sass
	@cuobjdump -arch sm_86 -ptx -fun $(FUNCTION) $(BUILD_DIR)/bin/catzilla-matmul | c++filt > build/cuobjdump.ptx

# Usage: make profile KERNEL=<integer> PREFIX=<optional string>
profile: build
	@mkdir -p $(BENCHMARK_DIR)
	@ncu --set full --export $(BENCHMARK_DIR)/catzilla-kernel-$(KERNEL) --force-overwrite $(BUILD_DIR)/bin/catzilla-matmul -version $(KERNEL) -device $(DEVICE_IDX) -repeat 1 -warmup 0 -profile 1

summary:
	@ncu --import $(BENCHMARK_DIR)/catzilla-kernel-$(KERNEL).ncu-rep --print-summary per-kernel > \
		$(BENCHMARK_DIR)/catzilla-kernel-$(KERNEL).summary && \
		vim $(BENCHMARK_DIR)/catzilla-kernel-$(KERNEL).summary

analysis:
	@ncu --import $(BENCHMARK_DIR)/catzilla-kernel-$(KERNEL).ncu-rep --page details > \
		$(BENCHMARK_DIR)/catzilla-kernel-$(KERNEL).details && \
		vim $(BENCHMARK_DIR)/catzilla-kernel-$(KERNEL).details

bench%: build
	@./$(BUILD_DIR)/bin/catzilla-matmul -version ${KERNEL} -device 0 -repeat 100 -warmup 10 -size-m $* -size-n $* -size-k $*

dev-m16n8k16: build
	@./$(BUILD_DIR)/bin/catzilla-matmul -version ${KERNEL} -device 0 -repeat 1 -warmup 0 -size-m 16 -size-n 8 -size-k 16

dev-m16n8k8: build
	@./$(BUILD_DIR)/bin/catzilla-matmul -version ${KERNEL} -device 0 -repeat 1 -warmup 0 -size-m 16 -size-n 8 -size-k 8

dev%: build
	@./$(BUILD_DIR)/bin/catzilla-matmul -version $(KERNEL) -device 0 -repeat 1 -warmup 0 -size-m $* -size-n $* -size-k $*

test: build 
	@cd $(BUILD_DIR)/tests && ctest -V

format:
	@cd $(BUILD_DIR) && ninja format-code

# Define the git target, which depends on format
git: format
	@changed_files=$$(git status --porcelain | grep 'M' | awk '{print $$2}'); \
		if [ -n "$$changed_files" ]; then \
			echo "Adding changed files to git..."; \
    	git add $$changed_files; \
    else \
    	echo "No changes detected."; \
    fi

ifneq ($(wildcard $(BENCHMARK_DIR)/$(PREFIX)catzilla-kernel-$(KERNEL).ncu-rep),)
    FILE_EXISTS := 1
else
    FILE_EXISTS := 0
endif

# 定义源代码目录
SRC_DIR := catz recipes benchmark

# 定义文件扩展名（可以根据需要扩展）
SRC_EXTENSIONS := c cpp h py hpp cu

# 构建 grep 的文件匹配模式
SRC_FILES := $(foreach ext,$(SRC_EXTENSIONS),$(shell find $(SRC_DIR) -type f -name "*.$(ext)"))

# ANSI 颜色代码
BLUE := \\x1b[34m
RESET := \\x1b[0m

todo:
	@echo "Searching for TODOs in source files..."
	@echo "======================================"
	@for file in $(SRC_FILES); do \
            echo -e "\nFile: $$file"; \
            echo -e "----------------"; \
	    grep -n -A 2 "TODO" $$file | sed "s/TODO/$$(echo -e '$(BLUE)')TODO$$(echo -e '$(RESET)')/g" | sed 's/^/  /'; \
        done
	@echo "======================================"
	@echo "Done."


