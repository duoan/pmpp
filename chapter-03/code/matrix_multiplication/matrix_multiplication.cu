#include <torch/all.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

// assign a thread for each of output element.
// output element will be sum(row_i_j * col_j_i)
__global__ void matrixMulKernel(float* M, float* N, float* P, int m, int n, int o) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < o) {
        float sum = 0;
        // compute: 2 * N
        for (unsigned int i = 0; i < n; ++i) {
            // M: (m x n); N(n x o), get everything from the row and everything from the column
            // memory: 2 x 4 bytes
            // compute: 2 float operations, and 4 int operations
            // 2 / 8 = 0.25 OP/B
            sum += M[row * n + i] * N[i * o + col];
        }
        // the resulting one is m x o
        P[row * o + col] = sum;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor matrixMul(torch::Tensor M, torch::Tensor N) {
    assert(M.device().type() == torch::kCUDA && N.device().type() == torch::kCUDA);
    assert(M.dtype() == torch::kFloat32 && N.dtype() == torch::kFloat32);
    assert(M.size(1) == N.size(0));

    // matrices are m x n and n x o
    const auto m = M.size(0);
    const auto n = M.size(1); // N is the reduction axis
    const auto o = N.size(1);

    // data loaded: (MN + NO) x 4 bytes = 8 MNNO
    // operations: OM (output elements) * N * 2 (mul + add) = 2MNO
    // Potentional compute-to-memory ratio: 2MNO / 8MNNO = > 0.25N OP/B

    auto P = torch::empty({m, o}, torch::TensorOptions().dtype(N.dtype()).device(N.device()));

    dim3 dimBlock(16, 16);
    dim3 dimGrid(cdiv(o, dimBlock.x), cdiv(m, dimBlock.y));

    matrixMulKernel<<<dimGrid,dimBlock, 0, c10::cuda::getCurrentCUDAStream()>>>(
        M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), m, n, o);

    return P;
}
