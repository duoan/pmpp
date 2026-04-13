#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <cstdlib>

__global__ void kernel1(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] *= 2.0f;
}

__global__ void kernel2(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] += 1.0f;
}

int main() {
    const int N = 1 << 20; // 1 million elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_A, *d_B;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Asynchronous memory copy and kernel launch in stream1
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
    kernel1<<<(N + 255) / 256, 256, 0, stream1>>>(d_A);

    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2);
    kernel2<<<(N + 255) / 256, 256, 0, stream2>>>(d_B);

    // Synchronize stream
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Copy results back to host
    cudaMemcpyAsync(h_A, d_A, size, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_B, d_B, size, cudaMemcpyDeviceToHost, stream2);

    // Synchronize again after memory copy to ensure completion
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Verify result
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_A[i] != static_cast<float>(i * 2) || h_B[i] != static_cast<float>(i + 1)) {
            success = false;
            break;
        }
    }

    if (success) {
        printf("Asynchronous operations completed successfully.\n");
    } else {
        printf("Error in asynchronous operations.\n");
    }

    // Clean up resources
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
