#include <cuda_runtime.h>

#include <stdio.h>

struct JDSMatrix {
    int numRows;
    int numCols;
    int numTiles;
    int* colIdx;
    float* values;
    int* rowPerm;
    int* iterPtr;
};

__global__ void spmv_jds_kernel(JDSMatrix jdsMatrix, float* x, float* y) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= jdsMatrix.numRows) {
        return;
    }

    float sum = 0.0f;
    for (int t = 0; t < jdsMatrix.numTiles; ++t) {
        int i = jdsMatrix.iterPtr[t] + tid;
        if (i < jdsMatrix.iterPtr[t + 1]) {
            int col = jdsMatrix.colIdx[i];
            float value = jdsMatrix.values[i];
            sum += x[col] * value;
        }
    }
    y[jdsMatrix.rowPerm[tid]] = sum;
}

void spmv_jds(JDSMatrix jdsMatrix, float* d_x, float* d_y) {
    int blockSize = 256;
    int gridSize = (jdsMatrix.numRows + blockSize - 1) / blockSize;
    spmv_jds_kernel<<<gridSize, blockSize>>>(jdsMatrix, d_x, d_y);
}

int main() {
    const int numRows = 6;
    const int numCols = 6;

    int h_colIdx[] = {0, 0, 2, 0, 1, 3, 2, 4, 3, 4, 5};
    float h_values[] = {1, 5, 3, 7, 2, 8, 6, 4, 9, 10, 11};
    int h_rowPerm[] = {1, 3, 5, 2, 4, 0};
    int h_iterPtr[] = {0, 6, 11, 14, 15};
    float h_x[] = {1, 2, 3, 4, 5, 6};
    float h_y[numRows] = {0};

    int* d_colIdx;
    float* d_values;
    int* d_rowPerm;
    int* d_iterPtr;
    float* d_x;
    float* d_y;

    cudaMalloc(&d_colIdx, sizeof(h_colIdx));
    cudaMalloc(&d_values, sizeof(h_values));
    cudaMalloc(&d_rowPerm, sizeof(h_rowPerm));
    cudaMalloc(&d_iterPtr, sizeof(h_iterPtr));
    cudaMalloc(&d_x, sizeof(h_x));
    cudaMalloc(&d_y, sizeof(h_y));

    cudaMemcpy(d_colIdx, h_colIdx, sizeof(h_colIdx), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, sizeof(h_values), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPerm, h_rowPerm, sizeof(h_rowPerm), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iterPtr, h_iterPtr, sizeof(h_iterPtr), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeof(h_y), cudaMemcpyHostToDevice);

    int numTiles = (int)(sizeof(h_iterPtr) / sizeof(h_iterPtr[0])) - 1;
    JDSMatrix d_jdsMatrix = {numRows, numCols, numTiles, d_colIdx, d_values, d_rowPerm, d_iterPtr};

    spmv_jds(d_jdsMatrix, d_x, d_y);

    cudaMemcpy(h_y, d_y, sizeof(h_y), cudaMemcpyDeviceToHost);

    printf("Result y: ");
    for (int i = 0; i < numRows; ++i) {
        printf("%f ", h_y[i]);
    }
    printf("\n");

    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_rowPerm);
    cudaFree(d_iterPtr);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
