export PATH := /usr/local/cuda/bin:$(PATH)
export LD_LIBRARY_PATH := /usr/local/cuda/lib64:$(LD_LIBRARY_PATH)

all: run clean

run: softmax
	./softmax

run_shlf_down:
	./shlf_down

softmax: softmax.cu
	nvcc --ptxas-options=-v -o softmax softmax.cu

shlf_down: shlf_down.cu
	nvcc --ptxas-options=-v -o shlf_down shlf_down.cu

clean:
	rm -f softmax