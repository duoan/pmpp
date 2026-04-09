#include <cuda_runtime.h>
#include <iostream>


// Definitions for utility functions
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << "(" << __LINE__ << "):" \
        << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel for demonstrating memory coalescing
__global__ void coalescesAccess(float* d_out, float* d_in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = d_in[idx] * 2.0f;
    }
}

// Kernel for using shared memory
__global__ void sharedMemoryUsage(float* d_out, float* d_in, int N) {
    extern __shared__ float s_data[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (index < N) s_data[tid] = d_in[index];
    __syncthreads();

    // Perform reduction to demonstrate shared memory access
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            if (tid + stride < blockDim.x) {
                s_data[tid] += s_data[tid + stride];
            }
        }
        __syncthreads();
    }

    // Write result for first thread of each block
    if (tid == 0) {
        d_out[blockIdx.x] = s_data[0];
    }
}

// Function to illustrate constant memory use
__constant__ float c_multiplier;

__global__ void constantMemoryUsage(float* d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_data[idx] *= c_multiplier;
    }
}

// Kernel utilizing texture memory via texture object API
__global__ void textureMemoryUsage(cudaTextureObject_t texObj, float* d_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = tex1Dfetch<float>(texObj, idx) * 2.0f;
    }
}


// Function to initialize data
void initializeData(float* data, int N) {
    for (int i = 0; i < N; ++i) {
        data[i] = static_cast<float>(i);
    }
}

int main() {
    const int N = 1024;
    const size_t size = N * sizeof(float);

    // Host and device arrays
    float* h_in = new float[N];
    float* h_out = new float[N];
    float *d_in, *d_out;
    initializeData(h_in, N);

    // Constant memory variable
    const float h_multiplier = 3.0f;
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_multiplier, &h_multiplier, sizeof(float)));

    // Device memory allocation
    CHECK_CUDA_ERROR(cudaMalloc(&d_in, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Kernal launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock) / threadsPerBlock;

    // Coalescing access demonstration with dummy operation
    coalescesAccess<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Shared memory optimization using reduction
    sharedMemoryUsage<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_out, d_in, N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Using constant memory in a kernel
    constantMemoryUsage<<<blocksPerGrid, threadsPerBlock>>>(d_in, N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Create texture object bound to device memory
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_in;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.sizeInBytes = size;

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj = 0;
    CHECK_CUDA_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    // Use texture memory in computation
    textureMemoryUsage<<<blocksPerGrid, threadsPerBlock>>>(texObj, d_out, N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Device to host memory copy
    CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Output some results for validation
    std::cout << "Sample output " << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << h_out[i] << " ";
    } 
    std::cout << std::endl;

    // Cleanup
    cudaDestroyTextureObject(texObj);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;

}