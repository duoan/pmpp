export PATH := /usr/local/cuda/bin:$(PATH)
export LD_LIBRARY_PATH := /usr/local/cuda/lib64:$(LD_LIBRARY_PATH)

all: run clean

run: matrix_mul_benchmark
	./matrix_mul_benchmark

matrix_mul_benchmark: matrix_mul_benchmark.cu
	nvcc --ptxas-options=-v -arch=sm_120 -o matrix_mul_benchmark matrix_mul_benchmark.cu

clean:
	rm -f matrix_mul_benchmark