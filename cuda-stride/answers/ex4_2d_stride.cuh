// ============================================================================
// Answer: Exercise 4 — 2D Grid-Stride Transpose
// ============================================================================
#pragma once

__global__ void transpose(const float* A, float* B, int M, int N) {
  int startRow = blockIdx.x * blockDim.x + threadIdx.x;
  int startCol = blockIdx.y * blockDim.y + threadIdx.y;
  int strideRow = gridDim.x * blockDim.x;
  int strideCol = gridDim.y * blockDim.y;

  for (int row = startRow; row < M; row += strideRow) {
    for (int col = startCol; col < N; col += strideCol) {
      B[col * M + row] = A[row * N + col];
    }
  }
}
