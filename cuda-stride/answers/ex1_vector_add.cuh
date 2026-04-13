// ============================================================================
// Answer: Exercise 1 — 1D Vector Add
// ============================================================================
#pragma once

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    C[i] = A[i] + B[i];
  }
}
