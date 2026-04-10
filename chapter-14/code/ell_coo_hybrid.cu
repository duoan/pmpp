#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

typedef struct {
    int numRows;
    int numCols;
    int maxNonZerosPerRow;
    int* colIdx;
    float* values;
} ELLMatrix;

typedef struct {
    int numRows;
    int numCols;
    int numNonZeros;
    int* rowIdx;
    int* colIdx;
    float* values;
} COOMatrix;

typedef struct {
    ELLMatrix ellPart;
    COOMatrix cooPart;
} HybridMatrix;

typedef struct {
    int col;
    float value;
} SparseElement;

typedef struct {
    const SparseElement* elements;
    int count;
} SparseRow;

__global__ void spmv_ell_kernel(ELLMatrix ellMatrix, float* x, float* y) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < (unsigned int)ellMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int t = 0; t < (unsigned int)ellMatrix.maxNonZerosPerRow; t++) {
            unsigned int i = t * ellMatrix.numRows + row;
            int col = ellMatrix.colIdx[i];
            float value = ellMatrix.values[i];
            if (col >= 0) {
                sum += x[col] * value;
            }
        }
        y[row] = sum;
    }
}

void spmv_coo(COOMatrix cooMatrix, const float* x, float* y) {
    for (int i = 0; i < cooMatrix.numNonZeros; i++) {
        y[cooMatrix.rowIdx[i]] += cooMatrix.values[i] * x[cooMatrix.colIdx[i]];
    }
}

HybridMatrix convertToHybrid(const SparseRow* matrix, int numRows, int maxEllNonZeros) {
    int numCols = 0;
    int cooNonZeros = 0;
    int ellEntries = numRows * maxEllNonZeros;

    for (int row = 0; row < numRows; ++row) {
        for (int elemIdx = 0; elemIdx < matrix[row].count; ++elemIdx) {
            int col = matrix[row].elements[elemIdx].col;
            if (col + 1 > numCols) {
                numCols = col + 1;
            }
        }
        if (matrix[row].count > maxEllNonZeros) {
            cooNonZeros += matrix[row].count - maxEllNonZeros;
        }
    }

    HybridMatrix hybridMatrix;
    hybridMatrix.ellPart.numRows = numRows;
    hybridMatrix.ellPart.numCols = numCols;
    hybridMatrix.ellPart.maxNonZerosPerRow = maxEllNonZeros;

    hybridMatrix.cooPart.numRows = numRows;
    hybridMatrix.cooPart.numCols = numCols;
    hybridMatrix.cooPart.numNonZeros = cooNonZeros;
    hybridMatrix.cooPart.rowIdx = cooNonZeros > 0 ? (int*)malloc(cooNonZeros * sizeof(int)) : NULL;
    hybridMatrix.cooPart.colIdx = cooNonZeros > 0 ? (int*)malloc(cooNonZeros * sizeof(int)) : NULL;
    hybridMatrix.cooPart.values = cooNonZeros > 0 ? (float*)malloc(cooNonZeros * sizeof(float)) : NULL;

    int* ellColIdx = (int*)malloc(ellEntries * sizeof(int));
    float* ellValues = (float*)malloc(ellEntries * sizeof(float));
    for (int i = 0; i < ellEntries; ++i) {
        ellColIdx[i] = -1;
        ellValues[i] = 0.0f;
    }

    int cooWriteIdx = 0;
    for (int row = 0; row < numRows; ++row) {
        int ellCount = 0;
        for (int elemIdx = 0; elemIdx < matrix[row].count; ++elemIdx) {
            SparseElement elem = matrix[row].elements[elemIdx];
            if (ellCount < maxEllNonZeros) {
                ellColIdx[ellCount * numRows + row] = elem.col;
                ellValues[ellCount * numRows + row] = elem.value;
                ellCount++;
            } else {
                hybridMatrix.cooPart.rowIdx[cooWriteIdx] = row;
                hybridMatrix.cooPart.colIdx[cooWriteIdx] = elem.col;
                hybridMatrix.cooPart.values[cooWriteIdx] = elem.value;
                cooWriteIdx++;
            }
        }
    }

    cudaMalloc(&hybridMatrix.ellPart.colIdx, ellEntries * sizeof(int));
    cudaMalloc(&hybridMatrix.ellPart.values, ellEntries * sizeof(float));
    cudaMemcpy(hybridMatrix.ellPart.colIdx, ellColIdx, ellEntries * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(hybridMatrix.ellPart.values, ellValues, ellEntries * sizeof(float), cudaMemcpyHostToDevice);

    free(ellColIdx);
    free(ellValues);

    return hybridMatrix;
}

void spmv_hybrid(HybridMatrix hybridMatrix, float* d_x, float* h_y) {
    int blockSize = 256;
    int gridSize = (hybridMatrix.ellPart.numRows + blockSize - 1) / blockSize;
    int ySizeBytes = hybridMatrix.ellPart.numRows * (int)sizeof(float);

    float* d_y;
    float* h_x = (float*)malloc(hybridMatrix.cooPart.numCols * sizeof(float));
    cudaMalloc(&d_y, ySizeBytes);
    cudaMemset(d_y, 0, ySizeBytes);

    spmv_ell_kernel<<<gridSize, blockSize>>>(hybridMatrix.ellPart, d_x, d_y);
    cudaMemcpy(h_y, d_y, ySizeBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x, d_x, hybridMatrix.cooPart.numCols * sizeof(float), cudaMemcpyDeviceToHost);

    spmv_coo(hybridMatrix.cooPart, h_x, h_y);

    free(h_x);
    cudaFree(d_y);
}

int main() {
    SparseElement row0[] = {{0, 1.0f}, {1, 7.0f}};
    SparseElement row1[] = {{0, 5.0f}, {2, 3.0f}, {3, 9.0f}};
    SparseElement row2[] = {{1, 2.0f}, {2, 8.0f}};
    SparseElement row3[] = {{3, 6.0f}};
    SparseRow matrix[] = {
        {row0, 2},
        {row1, 3},
        {row2, 2},
        {row3, 1},
    };
    float h_x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int numRows = (int)(sizeof(matrix) / sizeof(matrix[0]));

    HybridMatrix hybrid = convertToHybrid(matrix, numRows, 2);

    float* d_x;
    cudaMalloc(&d_x, sizeof(h_x));
    cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice);

    spmv_hybrid(hybrid, d_x, h_y);

    printf("Result y: ");
    for (int i = 0; i < numRows; ++i) {
        printf("%f ", h_y[i]);
    }
    printf("\n");

    cudaFree(hybrid.ellPart.colIdx);
    cudaFree(hybrid.ellPart.values);
    free(hybrid.cooPart.rowIdx);
    free(hybrid.cooPart.colIdx);
    free(hybrid.cooPart.values);
    cudaFree(d_x);

    return 0;
}
