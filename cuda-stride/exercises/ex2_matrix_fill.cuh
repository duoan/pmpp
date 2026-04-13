// ============================================================================
// Exercise 2: 2D Matrix Fill
// ============================================================================
//
// GOAL: Fill matrix C so that C[row][col] = row * 1000 + col
//       e.g. C[2][3] = 2003, C[0][5] = 5, C[10][0] = 10000
//       This lets you visually verify your indexing is correct.
//
// C is stored in row-major 1D memory: C[row][col] => C[row * N + col]
//
// Launch config:
//   blockDim = (16, 16)   (2D block, 256 threads)
//   gridDim  = (ceil(M/16), ceil(N/16))
//
// Visual (M=4, N=4, blockDim=2×2):
//
//   Grid of blocks:              Expected output C:
//   ┌──────────┬──────────┐      ┌──────┬──────┬──────┬──────┐
//   │block(0,0)│block(0,1)│      │    0 │    1 │    2 │    3 │
//   │          │          │      │ 1000 │ 1001 │ 1002 │ 1003 │
//   ├──────────┼──────────┤      ├──────┼──────┼──────┼──────┤
//   │block(1,0)│block(1,1)│      │ 2000 │ 2001 │ 2002 │ 2003 │
//   │          │          │      │ 3000 │ 3001 │ 3002 │ 3003 │
//   └──────────┴──────────┘      └──────┴──────┴──────┴──────┘
//
//   Thread (tx=1, ty=0) in block(0,1):
//     row = blockIdx.x * 2 + threadIdx.x = 0*2+1 = 1
//     col = blockIdx.y * 2 + threadIdx.y = 1*2+0 = 2
//     => C[1*4+2] = C[6] = 1*1000+2 = 1002 ✓
//
// YOUR TASK: Fill in the three lines marked TODO.
// ============================================================================
#pragma once

__global__ void matrix_fill(float* C, int M, int N) {
  // TODO 1: compute row from blockIdx.x and threadIdx.x
  int row = blockIdx.x * blockDim.x + threadIdx.x;  // ← REPLACE THIS

  // TODO 2: compute col from blockIdx.y and threadIdx.y
  int col = blockIdx.y * blockDim.y + threadIdx.y;  // ← REPLACE THIS

  // TODO 3: bounds check, then write: C[???] = row * 1000 + col
  // Remember: row-major means address = row * N + col
  if (row < M && col < N) {
    C[row * N + col] = row * 1000 + col;
  }
}
