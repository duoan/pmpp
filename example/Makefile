NVCC      := nvcc
NVCCFLAGS := -O3 -std=c++17 --use_fast_math
LDFLAGS   := -lcublas

CUTLASS_DIR := ../third_party/cutlass
NVCCFLAGS   += -I$(CUTLASS_DIR)/include

# auto-detect GPU arch, fallback to sm_80
GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(GPU_ARCH),)
  GPU_ARCH := 80
endif
NVCCFLAGS += -arch=sm_$(GPU_ARCH)

TARGETS := cutlass_gemm_basic

.PHONY: all clean

all: $(TARGETS)

cutlass_gemm_basic: cutlass_gemm_basic.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS)
