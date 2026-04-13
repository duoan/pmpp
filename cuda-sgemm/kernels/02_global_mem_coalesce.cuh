// ============================================================================
// Kernel 02: Global Memory Coalescing
// ============================================================================
//
// C(M,N) = α * A(M,K) @ B(K,N) + β * C
// Same math as kernel 01, but thread-to-element mapping is changed so that
// warp threads access consecutive memory addresses (coalesced access).
//
// ============================================================================
// 1. THE PROBLEM: WARP LAYOUT IN KERNEL 01 (2D blockDim)
// ============================================================================
//
//   Kernel 01 uses blockDim(32, 32), 2D thread index:
//     row = blockIdx.x * 32 + threadIdx.x    (tx = 0..31)
//     col = blockIdx.y * 32 + threadIdx.y    (ty = 0..31)
//
//   A warp = 32 threads with consecutive flat index.
//   Flat index = threadIdx.y * blockDim.x + threadIdx.x
//   => Warp 0 = threads with ty=0, tx=0..31 (same col, consecutive rows)
//
//   Warp layout in the 32×32 tile of C:
//
//            col 0   col 1   col 2   ...  col 31
//   row  0 │ W0:t0 │ W1:t0 │ W2:t0 │ ... │ W31:t0 │
//   row  1 │ W0:t1 │ W1:t1 │ W2:t1 │ ... │ W31:t1 │
//   row  2 │ W0:t2 │ W1:t2 │ W2:t2 │ ... │ W31:t2 │
//     :    │  :    │  :    │  :    │     │  :     │
//   row 31 │ W0:t31│ W1:t31│ W2:t31│ ... │ W31:t31│
//                ↑
//         Warp 0 is a COLUMN (same col, consecutive rows)
//
//   Memory access pattern for Warp 0 (ty=0, tx=0..31):
//
//   A[row * K + i]:  row = 0,1,2,...,31  (consecutive rows!)
//     addr: A[0*K+i], A[1*K+i], A[2*K+i], ... A[31*K+i]
//           |<--K-->| |<--K-->| |<--K-->|
//     stride = K elements between each thread's access
//     => 32 different cache lines => 32 memory transactions!  ✗ BAD
//
//   B[i * N + col]:  col = 0 for all threads
//     addr: B[i*N+0], B[i*N+0], B[i*N+0], ... (all same!)
//     => broadcast, 1 transaction  ✓ OK
//
//   C[row * N + col]:  row = 0,1,...,31; col = 0
//     addr: C[0*N+0], C[1*N+0], C[2*N+0], ... C[31*N+0]
//     stride = N => 32 transactions!  ✗ BAD
//
//   Visual: addresses accessed by Warp 0 in one iteration
//
//   A in memory (row-major, each row = K floats):
//   ┌───────────────────────────────────────────┐
//   │ row 0: [a00 a01 a02 ... ]                 │ ← t0 reads A[0*K+i]
//   │ row 1: [a10 a11 a12 ... ]                 │ ← t1 reads A[1*K+i]
//   │ row 2: [a20 a21 a22 ... ]                 │ ← t2 reads A[2*K+i]
//   │  ...    ...                               │
//   │ row31: [a310 a311 ...]                    │ ← t31 reads A[31*K+i]
//   └───────────────────────────────────────────┘
//     32 threads hit 32 different rows = 32 cache lines = STRIDED = SLOW
//
// ============================================================================
// 2. THE FIX: 1D blockDim WITH ROW-MAJOR THREAD MAPPING
// ============================================================================
//
//   This kernel uses blockDim(BS * BS, 1, 1)  — a 1D block.
//   Thread-to-tile mapping uses integer division:
//     row = threadIdx.x / BS   (changes every BS threads)
//     col = threadIdx.x % BS   (changes every thread)
//
//   Example with BS=4 (actual BS=32):
//
//   threadIdx.x:  0  1  2  3 | 4  5  6  7 | 8  9 10 11 |12 13 14 15
//   row (tx/BS):  0  0  0  0 | 1  1  1  1 | 2  2  2  2 | 3  3  3  3
//   col (tx%BS):  0  1  2  3 | 0  1  2  3 | 0  1  2  3 | 0  1  2  3
//
//   Warp layout in the tile (BS=4, warp = first 4 threads here):
//
//         col 0   col 1   col 2   col 3
//   row 0 │ t0    │ t1    │ t2    │ t3    │  ← Warp 0 (tx 0..3)
//   row 1 │ t4    │ t5    │ t6    │ t7    │  ← Warp 1 (tx 4..7)
//   row 2 │ t8    │ t9    │ t10   │ t11   │  ← Warp 2 (tx 8..11)
//   row 3 │ t12   │ t13   │ t14   │ t15   │  ← Warp 3 (tx 12..15)
//               ↑
//         Warp 0 is now a ROW (same row, consecutive cols)
//
//   With actual BS=32: warp 0 = tx 0..31 => row=0, col=0..31
//                      warp 1 = tx 32..63 => row=1, col=0..31
//                      ... (each warp fills exactly one row of the tile)
//
// ============================================================================
// 3. WHY THIS FIXES COALESCING
// ============================================================================
//
//   Warp 0 threads: row=0, col=0,1,2,...,31
//
//   A[row * K + i]:  row = 0 for ALL threads
//     addr: A[0*K+i], A[0*K+i], A[0*K+i], ... (all same!)
//     => broadcast, 1 transaction  ✓ OK
//
//   B[i * N + col]:  col = 0,1,2,...,31 (consecutive)
//     addr: B[i*N+0], B[i*N+1], B[i*N+2], ... B[i*N+31]
//     => 32 consecutive floats = 128 bytes = 1 cache line = 1 transaction!  ✓✓
//
//   C[row * N + col]:  row=0, col=0,1,...,31
//     addr: C[0*N+0], C[0*N+1], C[0*N+2], ... C[0*N+31]
//     => 1 coalesced transaction!  ✓✓
//
//   Visual: addresses accessed by Warp 0 in one iteration
//
//   B in memory (row-major, each row = N floats):
//   ┌───────────────────────────────────────────┐
//   │ row i: [bi0  bi1  bi2  ... bi31 ...]      │
//   └───────────────────────────────────────────┘
//             ↑t0  ↑t1  ↑t2      ↑t31
//     32 threads hit 32 CONSECUTIVE addresses = 1 cache line = COALESCED = FAST
//
// ============================================================================
// 4. SIDE-BY-SIDE COMPARISON
// ============================================================================
//
//              │ Kernel 01 (naive)        │ Kernel 02 (this)
//   ───────────┼──────────────────────────┼──────────────────────────
//   blockDim   │ (32, 32)     2D          │ (1024, 1)     1D
//   warp shape │ column (same col)        │ row (same row)
//   ───────────┼──────────────────────────┼──────────────────────────
//   A access   │ stride-K  ✗ 32 txns      │ broadcast ✓ 1 txn
//   B access   │ broadcast ✓ 1 txn        │ coalesced ✓ 1 txn
//   C access   │ stride-N  ✗ 32 txns      │ coalesced ✓ 1 txn
//   ───────────┼──────────────────────────┼──────────────────────────
//   txns/warp  │ ~33 per K-loop iter      │ ~2 per K-loop iter
//   ───────────┼──────────────────────────┼──────────────────────────
//   FLOPs      │ identical                │ identical
//   bandwidth  │ ~1/32 utilization        │ ~full utilization
//
//   Note: same total bytes transferred, but each transaction now serves
//   32 threads instead of 1 => ~32x better HW bandwidth utilization.
//
// ============================================================================
// 5. MEMORY ACCESS DIAGRAM (one K-loop iteration, BS=4 example)
// ============================================================================
//
//   A (4×4):            B (4×4):            C (4×4):
//   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
//   │▓▓▓▓▓▓▓▓▓▓▓▓▓│r0  │ .  .  .  .  │    │▓▓▓▓▓▓▓▓▓▓▓▓▓│r0
//   │ .  .  .  .  │    │ .  .  .  .  │    │ .  .  .  .  │
//   │ .  .  .  .  │    │ .  .  .  .  │    │ .  .  .  .  │
//   │ .  .  .  .  │    │ .  .  .  .  │    │ .  .  .  .  │
//   └─────────────┘    └─────────────┘    └─────────────┘
//     ↑ Warp 0: all       ↑ Warp 0: all     ↑ Warp 0: all
//       read same row       read row i,        write same row,
//       (broadcast)         cols 0..3          cols 0..3
//                           (coalesced!)       (coalesced!)
//
//   ▓ = accessed by Warp 0 (threads t0..t3, row=0, col=0..3)
//
// ============================================================================

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float* A, const float* B,
                                          float beta, float* C) {
  // 1D thread index -> 2D tile position
  // threadIdx.x % BS gives column (consecutive threads = consecutive cols)
  // threadIdx.x / BS gives row
  const uint cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (cRow < M && cCol < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      // A[cRow * K + i]: all warp threads share cRow => same addr, broadcast
      // B[i * N + cCol]: warp threads have consecutive cCol => coalesced read
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    // C[cRow * N + cCol]: consecutive cCol => coalesced write
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}
