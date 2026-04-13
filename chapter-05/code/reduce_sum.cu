#include <cuda_runtime.h>
#include <stdio.h>

__global__ void reduce_sum(double* input, double* output, int N) {
    extern __shared__ double blockSum[];
    unsigned int tid = threadIdx.x;

    unsigned int idx = blockIdx.x * (blockDim.x * 2) + tid;
    if (idx < N) {
        blockSum[tid] = input[idx] + (idx + blockDim.x < N ? input[idx + blockDim.x] : 0.0);
    } else {
        blockSum[tid] = 0.0;
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            blockSum[tid] += blockSum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, blockSum[0]);
    }
}

int main() {
    const int N = 1 << 20;
    const int bytes = N * sizeof(double);

    double* h_input = (double*)malloc(bytes);
    double h_output = 0.0;

    for (int i = 0; i < N; ++i) {
        h_input[i] = (double)(i + 1);
    }

    double *d_input, *d_output;
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, sizeof(double));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(double));

    int threadsPerBlock = 512;
    int blockPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    reduce_sum<<<blockPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_input, d_output, N);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_output, d_output, sizeof(double), cudaMemcpyDeviceToHost);
    double expected = (double)N * (N + 1) / 2.0;
    printf("Reduced Sum: %.0f, expected %.0f\n", h_output, expected);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}
