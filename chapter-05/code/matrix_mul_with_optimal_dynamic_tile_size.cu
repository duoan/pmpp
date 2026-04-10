// nvcc -o matrix_mul_dynamic_tile matrix_mul_with_optimal_dynamic_tile_size.cu
// here we make the tile dynamic, the tile size is calculated based on the hardware specyfication not hardcodedd as
// before

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

static int min_int(int a, int b) {
    return a < b ? a : b;
}

static int max_int(int a, int b) {
    return a > b ? a : b;
}

int calculateOptimalTileWidth(int m, int n, int o) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Get hardware limits
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxBlockDimX = prop.maxThreadsDim[0];
    int maxBlockDimY = prop.maxThreadsDim[1];
    int sharedMemPerBlock = prop.sharedMemPerBlock;

    // Calculate maximum possible tile size based on hardware constraints

    // 1. Based on max threads per block (square tiles)
    int tileWidth = (int)sqrt((double)maxThreadsPerBlock);

    // 2. Based on max block dimensions
    tileWidth = min_int(tileWidth, min_int(maxBlockDimX, maxBlockDimY));

    // 3. Based on shared memory (we need 2 tiles worth of shared memory)
    int maxTileWidthBySharedMem = (int)sqrt((double)sharedMemPerBlock / (2.0 * sizeof(float)));
    tileWidth = min_int(tileWidth, maxTileWidthBySharedMem);

    // 4. Based on matrix dimensions (no point in having tiles larger than matrices)
    tileWidth = min_int(tileWidth, min_int(m, min_int(n, o)));

    // 5. Round down to nearest power of 2 for better memory alignment
    tileWidth = 1 << (int)log2((double)tileWidth);

    // 6. Ensure minimum practical size
    tileWidth = max_int(16, tileWidth);  // minimum tile size of 16

    // Print diagnostic information
    // printf("Calculated optimal tile width: %d\n", tileWidth);
    // printf("Based on:\n");
    // printf("- Max threads per block: %d\n", maxThreadsPerBlock);
    // printf("- Max block dimensions: %dx%d\n", maxBlockDimX, maxBlockDimY);
    // printf("- Shared memory per block: %d bytes\n", sharedMemPerBlock);
    // printf("- Matrix dimensions: %dx%dx%d\n", m, n, o);

    return tileWidth;
}

__device__ void printDeviceMatrix(float* matrix, int width, int height, const char* matrixName) {
    printf("%s:\n", matrixName);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%f ", matrix[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
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

__global__ void TiledMatrixMulKernel(float* M, float* N, float* P, int m, int n, int o, int tileWidth) {
    extern __shared__ float sharedMem[];
    // Split shared memory into two parts, one for Mds and one for Nds
    float* Mds = sharedMem;
    float* Nds = &sharedMem[tileWidth * tileWidth];

    // let's save these for convenience
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // we use this to identify the current P element
    int row = by * tileWidth + ty;
    int col = bx * tileWidth + tx;

    float PValue = 0.0;
    for (int ph = 0; ph < (n + tileWidth - 1) / tileWidth; ph++) {
        if (row < m && (ph * tileWidth + tx) < n) {
            Mds[ty * tileWidth + tx] = M[row * n + ph * tileWidth + tx];  // row + phase + right row in a phase
        } else {
            Mds[ty * tileWidth + tx] = 0.0f;
        }

        if ((ph * tileWidth + ty) < n && (col < o)) {
            Nds[ty * tileWidth + tx] =
                N[(ph * tileWidth + ty) * o + col];  // col is from ty + phase + actual col in the phase
        } else {
            Nds[ty * tileWidth + tx] = 0.0f;
        }

        __syncthreads();  // make sure everything is loaded to both tile matrices

        for (int k = 0; k < tileWidth; k++) {
            PValue += Mds[ty * tileWidth + k] * Nds[k * tileWidth + tx];
        }
        __syncthreads();  // make sure we update this for every thread and we can start overwriting
    }

    if (row < m && col < o) {
        P[row * o + col] = PValue;
    }
}

void matrixMulNaive(float* M, float* N, float* P, int m, int n, int o) {
    float *d_M, *d_N, *d_P;

    cudaMalloc((void**)&d_M, m * n * sizeof(float));
    cudaMalloc((void**)&d_N, n * o * sizeof(float));
    cudaMalloc((void**)&d_P, m * o * sizeof(float));

    cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    NaiveMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);

    cudaMemcpy(P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

void matrixMulTiled(float* M, float* N, float* P, int m, int n, int o) {
    float *d_M, *d_N, *d_P;
    // int tileWidth = calculateOptimalTileWidth();
    int tileWidth = calculateOptimalTileWidth(m, n, o);

    // for now we work just with the square matrices
    // int width = m;

    cudaMalloc((void**)&d_M, m * n * sizeof(float));
    cudaMalloc((void**)&d_N, n * o * sizeof(float));
    cudaMalloc((void**)&d_P, m * o * sizeof(float));

    cudaMemcpy(d_M, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(tileWidth, tileWidth);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    int sharedMemSize = 2 * tileWidth * tileWidth * sizeof(float);
    TiledMatrixMulKernel<<<dimGrid, dimBlock, sharedMemSize>>>(d_M, d_N, d_P, m, n, o, tileWidth);

    cudaMemcpy(P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

float benchmark(void (*func)(float*, float*, float*, int, int, int), float* M, float* N, float* P, int m, int n, int o,
                int warmup, int reps) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        func(M, N, P, m, n, o);
    }

    // Timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);
    for (int i = 0; i < reps; ++i) {
        func(M, N, P, m, n, o);
    }
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Return the average time per run
    return milliseconds / reps;
}

bool allclose(float* M, float* N, int m, int n, float tol) {
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
    // change these to experiment with sizes, here I get a substantial boost just via using TILING
    int m = 2010, n = 3200, o = 9111;

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

    // Benchmark matrixMulNaive function
    float avgTimeMatrixMulTiled = benchmark(matrixMulTiled, M, N, P1, m, n, o, 25, 100);
    printf("Average time for matrixMulTiled: %f ms\n", avgTimeMatrixMulTiled);

    float avgTimeMatrixMulNavie = benchmark(matrixMulNaive, M, N, P2, m, n, o, 25, 100);
    printf("Average time for matrixMulNaive: %f ms\n", avgTimeMatrixMulNavie);
    bool same = allclose(P1, P2, m, o, 1e-5f);
    printf("Outputs are %s\n", same ? "approximately the same" : "different");

    if (true && !same) {
        printf("\nMatrix P1 (from matrixMulTiling):\n");
        printMatrix(P1, m, o);

        printf("\nMatrix P2 (from matrixMulNaive):\n");
        printMatrix(P2, m, o);
    }

    free(M);
    free(N);
    free(P1);
    free(P2);

    return 0;
}
