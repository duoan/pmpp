#include <cuda_runtime.h>
#include <cstdio>

__global__ void reduce_sum(int* input, int* output, int N) {
    extern __shared__ int blockSum[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < N) {
        blockSum[tid] = input[idx];
    } else {
        blockSum[tid] = 0;
    }
    __syncthreads();
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            blockSum[tid] += blockSum[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output, blockSum[0]);
    }
}

int main() {
    const int N = 1024;
    const int bytes = N * sizeof(int);
    
    // Allocate host memory
    int* h_input = (int*)malloc(bytes);
    int* h_output = (int*)malloc(sizeof(int));

    // Initialize input dat
    for (int i = 0; i < N; ++i) {
        h_input[i] = i; // Filling the array with 1s
    }

    h_output[0] = 0;

    // Allocate device memory
    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, sizeof(int), cudaMemcpyHostToDevice);

    // Kernel execution configuration
    int blockSize = 256;
    int grideSize = (N + blockSize - 1) / blockSize;
    reduce_sum<<<grideSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, N);

    // Synchronize device
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Reduced Sum: %d\n", h_output[0]);
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}