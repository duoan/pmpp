NVCC      := nvcc
NVCCFLAGS := -O3 -std=c++17

GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(GPU_ARCH),)
  GPU_ARCH := 80
endif
NVCCFLAGS += -arch=sm_$(GPU_ARCH)

TARGET := stride

.PHONY: all clean run run1 run2 run3 run4 reset

all: $(TARGET)

$(TARGET): main.cu exercises/ex1_vector_add.cuh exercises/ex2_matrix_fill.cuh exercises/ex3_stride_loop.cuh exercises/ex4_2d_stride.cuh
	$(NVCC) $(NVCCFLAGS) main.cu -o $@

run: $(TARGET)
	./$(TARGET)

run1: $(TARGET)
	./$(TARGET) 1

run2: $(TARGET)
	./$(TARGET) 2

run3: $(TARGET)
	./$(TARGET) 3

run4: $(TARGET)
	./$(TARGET) 4

reset:
	cp templates/*.cuh exercises/
	@echo "Exercises reset to blank templates."

clean:
	rm -f $(TARGET)
