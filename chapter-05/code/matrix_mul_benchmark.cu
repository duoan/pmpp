// nvcc -o matrix_mul matrix_mul_benchmark.cu

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#define TILE_WIDTH 64

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

void clear_l2() {
    // Get actual L2 size via CUDA on first call of this function
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0);
        l2_clear_size *= 2;  // just to be extra safe (cache is not necessarily strict LRU)
        gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    // Clear L2 cache (this is run on every call unlike the above code)
    gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}

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

__global__ void TiledMatrixMulKernel(float* M, float* N, float* P, int m, int n, int o) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // we use this to identify the current P element
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float PValue = 0;
    for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        if (row < m && (ph * TILE_WIDTH + tx) < n) {
            Mds[ty][tx] = M[row * n + ph * TILE_WIDTH + tx];  // row + phase + right row in a phase
        } else {
            Mds[ty][tx] = 0.0f;
        }

        if ((ph * TILE_WIDTH + ty) < n && (col < o)) {
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * o + col];  // col is from ty + phase + actual col in the phase
        } else {
            Nds[ty][tx] = 0.0f;
        }

        __syncthreads();  // make sure everything is loaded to both tile matrices

        for (int i = 0; i < TILE_WIDTH; i++) {
            PValue += Mds[ty][i] * Nds[i][tx];
        }
        __syncthreads();  // make sure we update this for every thread and we can start overwriting
    }

    if (row < m && col < o) {
        P[row * o + col] = PValue;
    }
}

void launchNaiveMatrixMul(float* d_M, float* d_N, float* d_P, int m, int n, int o) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);
    NaiveMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);
}

void launchTiledMatrixMul(float* d_M, float* d_N, float* d_P, int m, int n, int o) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);
    TiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);
}

// Times only kernel execution: alloc, H2D, D2H, and free are outside the timed region.
float benchmarkKernel(void (*launch)(float*, float*, float*, int, int, int), float* h_M, float* h_N, float* h_P_out,
                      int m, int n, int o, int warmup = 25, int reps = 100) {
    float *d_M, *d_N, *d_P;
    gpuErrchk(cudaMalloc((void**)&d_M, m * n * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_N, n * o * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_P, m * o * sizeof(float)));

    gpuErrchk(cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_N, h_N, n * o * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i) {
        launch(d_M, d_N, d_P, m, n, o);
        gpuErrchk(cudaDeviceSynchronize());
        clear_l2();
    }

    gpuErrchk(cudaDeviceSynchronize());

    cudaEvent_t iterStart, iterStop;
    gpuErrchk(cudaEventCreate(&iterStart));
    gpuErrchk(cudaEventCreate(&iterStop));

    float totalTime_ms = 0.0f;
    for (int i = 0; i < reps; ++i) {
        gpuErrchk(cudaEventRecord(iterStart));
        launch(d_M, d_N, d_P, m, n, o);
        gpuErrchk(cudaEventRecord(iterStop));
        gpuErrchk(cudaEventSynchronize(iterStop));

        float iterTime = 0.0f;
        cudaEventElapsedTime(&iterTime, iterStart, iterStop);
        totalTime_ms += iterTime;

        clear_l2();
    }

    gpuErrchk(cudaEventDestroy(iterStart));
    gpuErrchk(cudaEventDestroy(iterStop));

    gpuErrchk(cudaMemcpy(h_P_out, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return totalTime_ms / reps;
}

inline bool allclose(const float* M, const float* N, const int m, const int n, const float tol = 1e-5) {
    for (unsigned int i = 0; i < m; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            if (fabs(M[i * n + j] - N[i * n + j]) > tol) {
                return false;
            }
        }
    }
    return true;
}

void printMatrix(const float* matrix, const int rows, const int cols) {
    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            printf("%6g ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    // change these to experiment with sizes, here I get a substantial boost just via using TILING
    int m = 8192, n = 8192, o = 8192;

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

    float avgTimeMatrixMulTiled = benchmarkKernel(launchTiledMatrixMul, M, N, P1, m, n, o);
    printf("Average kernel time (tiled): %f ms\n", avgTimeMatrixMulTiled);

    float avgTimeMatrixMulNaive = benchmarkKernel(launchNaiveMatrixMul, M, N, P2, m, n, o);
    printf("Average kernel time (naive): %f ms\n", avgTimeMatrixMulNaive);

    bool same = allclose(P1, P2, m, o);
    printf("Outputs are %s\n", same ? "approximately the same" : "different");

    free(M);
    free(N);
    free(P1);
    free(P2);

    return 0;
}
