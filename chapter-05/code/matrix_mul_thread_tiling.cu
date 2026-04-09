// nvcc -o matrix_mul_thread_tiling matrix_mul_thread_tiling.cu
//
// 寄存器级线程分块 (Thread Tiling / Register Tiling) 矩阵乘法
// C(N×N) = A(N×N) × B(N×N)
//
// 核心思想：每个线程不再只计算 C 中的 1 个元素，而是计算一个 V×V 的子块。
// 对于 k 维的每一步迭代，线程从 A 加载一列向量 (V×1)、从 B 加载一行向量 (1×V)，
// 然后做外积 (outer product) 累加到 V×V 的寄存器数组中。
//
// 优势：
//   - 每次从全局内存加载 2V 个元素，却完成 V² 次乘累加（算术强度 ≈ V/2）
//   - 所有中间结果存在寄存器中，无需 shared memory
//
// Grid 布局（BK 为 block 每维的线程数）：
//   blockDim = (BK, BK)
//   gridDim  = (ceil(N / (BK*V)), ceil(N / (BK*V)))
//   每个 block 覆盖 C 中 (BK*V) × (BK*V) 的输出区域

#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>

#define V 4
#define BK 16

__global__ void mm_thread_tiling(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    int N
) {
    // 每个线程的全局索引；每个线程负责 C 中一个 V×V 子块
    int xbase = blockIdx.x * blockDim.x + threadIdx.x;
    int ybase = blockIdx.y * blockDim.y + threadIdx.y;

    int row_start = xbase * V;
    int col_start = ybase * V;

    // V×V 累加器，全部驻留在寄存器中
    float c[V][V] = {0.0f};

    // 沿 k 维逐步迭代，每步做一次外积
    for (int k = 0; k < N; ++k) {
        float a_reg[V] = {0.0f};
        float b_reg[V] = {0.0f};

        // 加载 A 的一列片段: A[row_start..row_start+V-1][k]
        for (int x = 0; x < V; ++x) {
            if (row_start + x < N) {
                a_reg[x] = A[(row_start + x) * N + k];
            }
        }

        // 加载 B 的一行片段: B[k][col_start..col_start+V-1]
        for (int y = 0; y < V; ++y) {
            if (col_start + y < N) {
                b_reg[y] = B[k * N + (col_start + y)];
            }
        }

        // 外积 (outer product): a_reg (V×1) × b_reg (1×V) → V×V，累加到 c
        for (int x = 0; x < V; ++x) {
            for (int y = 0; y < V; ++y) {
                c[x][y] += a_reg[x] * b_reg[y];
            }
        }
    }

    // 写回 V×V 子块到全局内存
    for (int x = 0; x < V; ++x) {
        for (int y = 0; y < V; ++y) {
            int global_row = row_start + x;
            int global_col = col_start + y;
            if (global_row < N && global_col < N) {
                C[global_row * N + global_col] = c[x][y];
            }
        }
    }
}


inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand() % 100);
        h_B[i] = static_cast<float>(rand() % 100);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // blockDim = (BK, BK)，每个 block 有 BK×BK 个线程
    // 每个线程处理 V×V 个输出元素，所以每个 block 覆盖 (BK*V) × (BK*V) 的 C 子区域
    // gridDim = ceil(N/(BK*V)) × ceil(N/(BK*V))
    dim3 dimBlock(BK, BK);
    dim3 dimGrid(cdiv(N, BK * V), cdiv(N, BK * V));

    cudaEventRecord(start);
    mm_thread_tiling<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms" << std::endl;

    // 矩阵乘法浮点运算量: 2*N³ (N³ 次乘法 + N³ 次加法)
    double gflops = 2.0 * N * N * N / (ms * 1e6);
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // CPU 端验证正确性
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += h_A[i * N + k] * h_B[k * N + j];
            }
            assert(fabs(h_C[i * N + j] - sum) < 1e-3);
        }
    }
    std::cout << "Matrix multiplication completed successfully." << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}