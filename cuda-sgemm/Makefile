NVCC      := nvcc
NVCCFLAGS := -O3 -std=c++17 --use_fast_math
LDFLAGS   := -lcublas

# auto-detect GPU arch, fallback to sm_80
GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(GPU_ARCH),)
  GPU_ARCH := 80
endif
NVCCFLAGS += -arch=sm_$(GPU_ARCH)

TARGET := sgemm
SRCS   := main.cu runner.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS) runner.cuh kernels.cuh kernels/*.cuh
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) 1

clean:
	rm -f $(TARGET)
