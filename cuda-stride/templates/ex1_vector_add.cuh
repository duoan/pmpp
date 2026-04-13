// ============================================================================
// Exercise 1: 1D Vector Add
// ============================================================================
//
// GOAL: C[i] = A[i] + B[i], for i in [0, N)
//
// This is the simplest CUDA kernel — each thread handles one element.
// You need to figure out "I am thread #??" from the built-in variables.
//
// Launch config (given to you, don't worry about it):
//   blockDim = 256  (1D, 256 threads per block)
//   gridDim  = ceil(N / 256)
//
// Visual (N=10, blockDim=4 for simplicity):
//
//   gridDim = ceil(10/4) = 3 blocks
//
//   block 0          block 1          block 2
//   ┌──┬──┬──┬──┐   ┌──┬──┬──┬──┐   ┌──┬──┬──┬──┐
//   │t0│t1│t2│t3│   │t0│t1│t2│t3│   │t0│t1│t2│t3│
//   └──┴──┴──┴──┘   └──┴──┴──┴──┘   └──┴──┴──┴──┘
//    ↓  ↓  ↓  ↓      ↓  ↓  ↓  ↓      ↓  ↓  ↓  ↓
//   C[0] [1] [2][3] [4] [5] [6][7]  [8] [9]  ✗  ✗
//                                          ↑
//                                     needs bounds check!
//
// YOUR TASK: Fill in the two lines marked TODO.
//
// HINT: global_id = blockIdx.___ * blockDim.___ + threadIdx.___
//       (all 1D, so which component? x? y? z?)
// ============================================================================
#pragma once

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
  // TODO 1: compute your global thread index
  int i = 0;  // ← REPLACE THIS

  // TODO 2: bounds check — don't access out-of-range memory
  // if ( ??? ) {
  C[i] = A[i] + B[i];
  // }
}
