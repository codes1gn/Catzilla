.PHONY: all build debug clean profile bench cuobjdump

CMAKE := cmake

BUILD_DIR := build
BENCHMARK_DIR := benchmark_results
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
		-DCMAKE_CUDA_COMPILER=nvcc \
		-DCMAKE_CUDA_FLAGS="-O3 -maxrregcount=128 --ptxas-options=-v" \
		-GNinja
	@ninja -C $(BUILD_DIR)

debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Debug .. -DCMAKE_CUDA_COMPILER=nvcc -GNinja
	@ninja -C $(BUILD_DIR)

clean:
	@rm -rf $(BUILD_DIR)

FUNCTION := $$(cuobjdump -symbols build/sgemm | grep -i Warptiling | awk '{print $$NF}')

cuobjdump: build
	@cuobjdump -arch sm_86 -sass -fun $(FUNCTION) build/sgemm/sgemm | c++filt > build/cuobjdump.sass
	@cuobjdump -arch sm_86 -ptx -fun $(FUNCTION) build/sgemm/sgemm | c++filt > build/cuobjdump.ptx

# Usage: make profile KERNEL=<integer> PREFIX=<optional string>
profile: build
	@mkdir -p $(BENCHMARK_DIR)
	@DEVICE=$(DEVICE_IDX) ncu --set full --export $(BENCHMARK_DIR)/$(PREFIX)kernel_$(KERNEL) --force-overwrite $(BUILD_DIR)/sgemm/sgemm $(KERNEL)

bench: build
	@DEVICE=$(DEVICE_IDX) ./$(BUILD_DIR)/sgemm/sgemm ${KERNEL}

ifneq ($(wildcard $(BENCHMARK_DIR)/$(PREFIX)kernel_$(KERNEL).ncu-rep),)
    FILE_EXISTS := 1
else
    FILE_EXISTS := 0
endif

report-profile:
	@if [ $(FILE_EXISTS) -eq 0 ]; then \
		$(MAKE) profile; \
	fi
	@ncu --import $(BENCHMARK_DIR)/kernel_$(KERNEL).ncu-rep --page details
