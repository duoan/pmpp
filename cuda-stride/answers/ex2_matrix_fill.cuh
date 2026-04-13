// ============================================================================
// Answer: Exercise 2 — 2D Matrix Fill
// ============================================================================
#pragma once

__global__ void matrix_fill(float* C, int M, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    C[row * N + col] = (float)(row * 1000 + col);
  }
}
