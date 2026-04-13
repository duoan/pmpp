#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

/**
 * B is a matrix
 * c is a vector
 */
__global__ void matrixVecMulKernel(float* B, float* c, float* result, int n_cols, int n_rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rows) {
        float sum = 0;
        for (int j = 0; j < n_cols; ++j) {
            sum += B[i * n_cols + j] * c[j];
        }
        result[i] = sum;
    }
}

torch::Tensor matrix_vector_multiplication(torch::Tensor B, torch::Tensor c) {
    assert(B.device().type() == torch::kCUDA && c.device().type() == torch::kCUDA);
    assert(B.dtype() == torch::kFloat32 && c.dtype() == torch::kFloat32);
    assert(B.size(1) == c.size(0));

    int n_cols = c.size(0);
    int n_rows = B.size(0);

    torch::Tensor a = torch::empty({n_rows}, torch::TensorOptions().dtype(torch::kFloat32).device(B.device()));

    // // Number of threads and blocks
    int threads_per_block = 16;
    int number_of_blocks = (n_rows + threads_per_block - 1) / threads_per_block;

    matrixVecMulKernel<<<number_of_blocks, threads_per_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        B.data_ptr<float>(), c.data_ptr<float>(), a.data_ptr<float>(), n_cols, n_rows);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return a;
}
