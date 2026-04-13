// ============================================================================
// Exercise 4: 2D Grid-Stride Loop (BOSS LEVEL)
// ============================================================================
//
// GOAL: Matrix transpose: B[col][row] = A[row][col]
//       A is M×N, B is N×M. Both row-major in 1D memory.
//
// TWIST: M and N can be much larger than the grid.
//   You need a 2D grid-stride loop (nested loops).
//
// Visual (M=3, N=4):
//
//   A (3×4):              B (4×3):
//   ┌──┬──┬──┬──┐        ┌──┬──┬──┐
//   │ 0│ 1│ 2│ 3│        │ 0│ 4│ 8│
//   │ 4│ 5│ 6│ 7│   =>   │ 1│ 5│ 9│
//   │ 8│ 9│10│11│        │ 2│ 6│10│
//   └──┴──┴──┴──┘        │ 3│ 7│11│
//                        └──┴──┴──┘
//   A[1][2] = 6  =>  B[2][1] = 6
//   A addr: 1*4+2 = 6     B addr: 2*3+1 = 7
//
// Memory formulas:
//   A[row][col] => A[row * N + col]     (A is M×N)
//   B[col][row] => B[col * M + row]     (B is N×M)
//
// 2D grid-stride pattern:
//   for (row = startRow; row < M; row += strideRow)
//     for (col = startCol; col < N; col += strideCol)
//       B[col * M + row] = A[row * N + col];
//
// Launch config:
//   blockDim = (16, 16)
//   gridDim  = (4, 4)     (small grid, but M,N can be 1000+)
//
// YOUR TASK: Fill in the complete 2D grid-stride loop.
// ============================================================================
#pragma once

__global__ void transpose(const float* A, float* B, int M, int N) {
  // TODO 1: compute starting row and col
  // int startRow = ???;
  // int startCol = ???;

  // TODO 2: compute strides (total threads in each dimension)
  // int strideRow = ???;
  // int strideCol = ???;

  // TODO 3: nested grid-stride loop
  // for (int row = startRow; row < M; row += strideRow) {
  //   for (int col = startCol; col < N; col += strideCol) {
  //     B[???] = A[???];
  //   }
  // }
}
