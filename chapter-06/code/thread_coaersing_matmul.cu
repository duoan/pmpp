// nvcc -o thread_coaersing_matmul thread_coaersing_matmul.cu
//
// 线程粗化 (Thread Coarsening) 的分块矩阵乘法
// P(m×o) = M(m×n) × N(n×o)
//
// 核心思想：每个线程不再只计算 P 中的 1 个元素，而是沿列方向计算 COARSE_FACTOR 个元素。
// 这样可以让同一行的 M tile 被复用 COARSE_FACTOR 次（只加载一次到 shared memory），
// 从而减少对全局内存的访问总量和 __syncthreads 的调用次数（相对于不粗化但 block 更多的方案）。
//
// Grid 布局：
//   gridDim.x = ceil(o / (TILE_WIDTH * COARSE_FACTOR))  ← x 方向每个 block 覆盖 COARSE_FACTOR 个 tile 宽度
//   gridDim.y = ceil(m / TILE_WIDTH)
//   blockDim  = (TILE_WIDTH, TILE_WIDTH)

#include <cuda_runtime.h>

#include <cmath>
#include <stdio.h>
#include <cstdlib>
#define TILE_WIDTH 32
#define COARSE_FACTOR 4

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

// 通过写满 L2 缓存来清除其内容，确保 benchmark 每次迭代从相同的缓存状态开始
void clear_l2() {
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0);
        l2_clear_size *= 2;
        gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}

// 朴素矩阵乘法：每个线程计算 P 中的一个元素，无 shared memory 优化
__global__ void NaiveMatrixMulKernel(float* M, float* N, float* P, int m, int n, int o) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < o) {
        float sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += M[row * n + i] * N[i * o + col];
        }
        P[row * o + col] = sum;
    }
}

// 带线程粗化的分块矩阵乘法
//
// 每个 block 负责 P 中一个 TILE_WIDTH × (TILE_WIDTH * COARSE_FACTOR) 的输出区域。
// 每个线程计算该区域中同一行、间隔 TILE_WIDTH 的 COARSE_FACTOR 个输出元素。
//
// 对于每个 phase（沿 n 维的 tile 迭代）：
//   1. 所有线程协作加载一块 M tile (TILE_WIDTH × TILE_WIDTH) 到 Mds —— 只加载一次
//   2. 依次对 COARSE_FACTOR 个 N tile 各加载一块到 Nds，并立即与 Mds 做局部乘累加
//   这样 Mds 在一个 phase 内被复用 COARSE_FACTOR 次，是粗化带来收益的关键
__global__ void TiledMatrixMulKernelWithThreadCoarsening(float* M, float* N, float* P, int m, int n, int o) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;
    // 本线程负责的第一个输出列；后续列间隔 TILE_WIDTH
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    // 每个线程维护 COARSE_FACTOR 个累加器，对应 COARSE_FACTOR 个输出元素
    float Pvalue[COARSE_FACTOR] = {0.0f};

    // 沿 n 维按 TILE_WIDTH 步长迭代（phase loop）
    for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        // 加载 M 的一个 tile —— 所有 COARSE_FACTOR 轮共享同一份 Mds
        if (row < m && (ph * TILE_WIDTH + tx) < n) {
            Mds[ty][tx] = M[row * n + ph * TILE_WIDTH + tx];
        } else {
            Mds[ty][tx] = 0.0f;
        }

        // 依次处理 COARSE_FACTOR 个 N tile
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = colStart + c * TILE_WIDTH;

            // 加载 N 的对应 tile 到 Nds
            // 注意：用 ty 作为 tile 内的行索引来实现合并访存（coalesced access）
            if ((ph * TILE_WIDTH + ty) < n && (col < o)) {
                Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * o + col];
            } else {
                Nds[ty][tx] = 0.0f;
            }
            __syncthreads();

            // 对当前 tile 做乘累加: Pvalue[c] += Mds[ty][k] * Nds[k][tx]
            for (int k = 0; k < TILE_WIDTH; k++) {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();
        }
    }

    // 将 COARSE_FACTOR 个结果写回全局内存
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = colStart + c * TILE_WIDTH;
        if (row < m && col < o) {
            P[row * o + col] = Pvalue[c];
        }
    }
}

// 朴素矩阵乘法的 host 端封装：分配显存、拷贝数据、启动 kernel、拷回结果
void matrixMul(float* M, float* N, float* P, int m, int n, int o) {
    float *d_M, *d_N, *d_P;

    cudaMalloc((void**)&d_M, m * n * sizeof(float));
    cudaMalloc((void**)&d_N, n * o * sizeof(float));
    cudaMalloc((void**)&d_P, m * o * sizeof(float));

    cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    NaiveMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);

    cudaMemcpy(P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// 线程粗化分块矩阵乘法的 host 端封装
// 注意 gridDim.x 缩小了 COARSE_FACTOR 倍，因为每个 block 现在覆盖更宽的列范围
void matrixMulTilingWithThreadCoarsing(float* M, float* N, float* P, int m, int n, int o) {
    float *d_M, *d_N, *d_P;

    cudaMalloc((void**)&d_M, m * n * sizeof(float));
    cudaMalloc((void**)&d_N, n * o * sizeof(float));
    cudaMalloc((void**)&d_P, m * o * sizeof(float));

    cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(cdiv(o, dimBlock.x * COARSE_FACTOR), cdiv(m, dimBlock.y));

    TiledMatrixMulKernelWithThreadCoarsening<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);

    cudaMemcpy(P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

// 用 CUDA event 计时的 benchmark 工具：先 warmup 若干轮，再取 reps 轮的平均耗时
// 每轮之间清除 L2 缓存以获得更稳定、可比的结果
float benchmark(void (*func)(float*, float*, float*, int, int, int), float* M, float* N, float* P, int m, int n, int o,
                int warmup = 25, int reps = 100) {
    for (int i = 0; i < warmup; ++i) {
        func(M, N, P, m, n, o);
    }

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);

    float totalTime_ms = 0.0f;

    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(iterStart);
        func(M, N, P, m, n, o);
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);

        float iterTime = 0.0f;
        cudaEventElapsedTime(&iterTime, iterStart, iterStop);
        totalTime_ms += iterTime;

        clear_l2();
    }

    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);

    return totalTime_ms / reps;
}

bool allclose(float* M, float* N, int m, int n, float tol = 1e-5) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fabs(M[i * n + j] - N[i * n + j]) > tol) {
                return false;
            }
        }
    }
    return true;
}

void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%6g ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int m = 4096, n = 4096, o = 4096;

    float* M = (float*)malloc(m * n * sizeof(float));
    float* N = (float*)malloc(n * o * sizeof(float));
    float* P1 = (float*)malloc(m * o * sizeof(float));
    float* P2 = (float*)malloc(m * o * sizeof(float));

    for (int i = 0; i < m * n; ++i) {
        M[i] = (float)1;
    }
    for (int i = 0; i < n * o; ++i) {
        N[i] = (float)1.5;
    }

    float avgTimeMatrixMulTiling = benchmark(matrixMulTilingWithThreadCoarsing, M, N, P1, m, n, o);
    printf("Average time for matrixMulTilingWithThreadCoarsing: %f ms\n", avgTimeMatrixMulTiling);

    float avgTimeMatrixMul = benchmark(matrixMul, M, N, P2, m, n, o);
    printf("Average time for matrixMul: %f ms\n", avgTimeMatrixMul);

    bool same = allclose(P1, P2, m, o);
    printf("Outputs are %s\n", same ? "approximately the same" : "different");

    // printf("\n");
    // printMatrix(P1, m, o);
    // printf("\n");
    // printMatrix(P2, m, o);

    free(M);
    free(N);
    free(P1);
    free(P2);

    return 0;
}
