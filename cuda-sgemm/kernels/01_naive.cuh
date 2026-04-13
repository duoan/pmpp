// ============================================================================
// Kernel 01: Naive SGEMM
// ============================================================================
//
// C(M,N) = α * A(M,K) @ B(K,N) + β * C
// Each thread computes exactly ONE element of C.
//
// ============================================================================
// 1. GRID & THREAD MAPPING (example: M=N=4, blockDim=2×2)
// ============================================================================
//
//   gridDim = (ceil(M/2), ceil(N/2)) = (2, 2)
//
//   Grid of blocks:                   Matrix C:
//   ┌────────────┬────────────┐       ┌─────┬─────┬─────┬─────┐
//   │ block(0,0) │ block(0,1) │       │ C00 │ C01 │ C02 │ C03 │
//   │ tx: 0,1    │ tx: 0,1    │       │ C10 │ C11 │ C12 │ C13 │
//   │ ty: 0,1    │ ty: 0,1    │       ├─────┼─────┼─────┼─────┤
//   ├────────────┼────────────┤       │ C20 │ C21 │ C22 │ C23 │
//   │ block(1,0) │ block(1,1) │       │ C30 │ C31 │ C32 │ C33 │
//   │ tx: 0,1    │ tx: 0,1    │       └─────┴─────┴─────┴─────┘
//   │ ty: 0,1    │ ty: 0,1    │
//   └────────────┴────────────┘
//
//   Thread (tx=1, ty=0) in block(0,1):
//     x = blockIdx.x * 2 + threadIdx.x = 0 * 2 + 1 = 1  (row)
//     y = blockIdx.y * 2 + threadIdx.y = 1 * 2 + 0 = 2  (col)
//     => computes C[1][2]
//
// ============================================================================
// 2. HOW ONE THREAD COMPUTES C[x][y] (example: x=1, y=2, K=4)
// ============================================================================
//
//        Matrix A (M×K)              Matrix B (K×N)
//        col: 0   1   2   3         col: 0   1  [2]  3
//   row 0 │ .   .   .   . │    row 0 │ .   .   b0  . │
//   row 1 │ a0  a1  a2  a3│    row 1 │ .   .   b1  . │
//   row 2 │ .   .   .   . │    row 2 │ .   .   b2  . │
//   row 3 │ .   .   .   . │    row 3 │ .   .   b3  . │
//            ↑ row x=1                    ↑ col y=2
//
//   tmp = a0*b0 + a1*b1 + a2*b2 + a3*b3
//   C[1][2] = α * tmp + β * C[1][2]
//
// ============================================================================
// 3. ROW-MAJOR MEMORY LAYOUT & INDEX FORMULA
// ============================================================================
//
//   Matrix A in memory (row-major, M=4, K=4):
//
//   row 0:  A[0]  A[1]  A[2]  A[3]      address = A + row * K + col
//   row 1:  A[4]  A[5]  A[6]  A[7]
//   row 2:  A[8]  A[9]  A[10] A[11]     A[x][i] => A[x * K + i]
//   row 3:  A[12] A[13] A[14] A[15]     B[i][y] => B[i * N + y]
//                                       C[x][y] => C[x * N + y]
//   Thread (x=1, y=2) reads:
//     A[1*4+0]=A[4], A[1*4+1]=A[5], A[1*4+2]=A[6], A[1*4+3]=A[7]  (row 1)
//     B[0*4+2]=B[2], B[1*4+2]=B[6], B[2*4+2]=B[10], B[3*4+2]=B[14] (col 2)
//           ↑ consecutive (stride 1)      ↑ strided (stride N=4)
//
// ============================================================================
// 4. PERFORMANCE ANALYSIS
// ============================================================================
//
// FLOPs per thread:  2K + 3  (K mul+add in loop, 2 mul + 1 add in epilogue)
// Total FLOPs:       M * N * (2K + 3)
// M=N=K=4092:        ~137 GFLOPs
//
// Memory per thread: reads (2K+1) floats, writes 1 float = (2K+2)*4 bytes
// Total traffic:     M * N * (8K+8) bytes
// M=N=K=4092:        ~548 GB reads + ~64 MB writes
// Actual data size:  (MK + KN + MN) * 4 = ~192 MB => 2860x redundancy
//
// ---- Why this is slow ----
//
//   Problem 1: Redundant traffic (2860x)
//     Row 1 of A is read by every thread that computes C[1][*] (N threads).
//     Col 2 of B is read by every thread that computes C[*][2] (M threads).
//     => Tiled shared memory kernels fix this by loading tiles once per block.
//
//   Problem 2: Poor memory coalescing
//     A warp = 32 threads with consecutive threadIdx.x (same threadIdx.y).
//     They have consecutive rows (x) but the same column (y).
//
//     A[x*K+i]: consecutive x => stride K => 32 cache lines => BAD
//     B[i*N+y]: same y => same address => broadcast => GOOD
//     C[x*N+y]: consecutive x => stride N => 32 cache lines => BAD
//
//     => Kernel 02 fixes this by remapping so warp threads have
//        same row, consecutive columns => coalesced access.
//
// ============================================================================

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  // row in C: which block row * block size + thread's row within block
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  // col in C: which block col * block size + thread's col within block
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      //  A[x * K + i] = A[x][i] = row x of A, walking cols 0..K-1
      //  B[i * N + y] = B[i][y] = col y of B, walking rows 0..K-1
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
