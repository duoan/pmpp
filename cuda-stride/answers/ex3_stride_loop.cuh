// ============================================================================
// Answer: Exercise 3 — Grid-Stride Loop
// ============================================================================
#pragma once

__global__ void double_elements(const float* A, float* C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (; i < N; i += stride) {
    C[i] = A[i] * 2.0f;
  }
}
