// ============================================================================
// Exercise 3: Grid-Stride Loop (1D)
// ============================================================================
//
// GOAL: C[i] = A[i] * 2.0f, for i in [0, N)
//
// TWIST: N can be WAY bigger than total threads.
//   e.g. N = 1,000,000 but we only launch 256 * 32 = 8192 threads.
//   Each thread must process MULTIPLE elements using a stride loop.
//
// The pattern:
//   total_threads = gridDim.x * blockDim.x
//   for (i = my_id; i < N; i += total_threads) { ... }
//
// Visual (N=10, only 3 threads total for simplicity):
//
//   Thread 0: processes i=0, 3, 6, 9
//   Thread 1: processes i=1, 4, 7
//   Thread 2: processes i=2, 5, 8
//
//   i:    0  1  2  3  4  5  6  7  8  9
//   who: t0 t1 t2 t0 t1 t2 t0 t1 t2 t0
//                  ↑ stride = 3 (total_threads)
//
// WHY is this useful?
//   - You can launch a FIXED number of blocks (e.g. 32)
//   - Works for ANY N, no matter how large
//   - Often faster than launching millions of blocks
//
// Launch config:
//   blockDim = 256
//   gridDim  = 32   (only 8192 threads, but N could be millions)
//
// YOUR TASK: Fill in the for-loop.
// ============================================================================
#pragma once

__global__ void double_elements(const float* A, float* C, int N) {
  // TODO: write a grid-stride loop
  // Step 1: compute your starting index (same as ex1)
  // Step 2: compute the stride (total number of threads in the entire grid)
  // Step 3: loop from start to N with that stride

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; i < N; i += stride) {
    C[i] = A[i] * 2.0f;
  }
}
