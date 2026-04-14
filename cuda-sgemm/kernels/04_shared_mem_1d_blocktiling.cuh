
// =============================================================================
// Kernel 4: Shared Memory 1D Block-Tiling SGEMM
// =============================================================================
//
// Key idea vs Kernel 3 (plain shared-mem blocking):
//   Kernel 3: each thread computes 1 element  of C → arithmetic intensity low
//   Kernel 4: each thread computes TM elements of C → reuse B from register
//
// With BM=64, BN=64, BK=8, TM=8:
//   blockDim = (BM*BN)/TM = 512 threads
//   Each thread outputs a TM×1 column-strip in the BM×BN output tile.
//
// ─────────────────────────────────────────────────────────────────────────────
//  GLOBAL MATRIX VIEW  (C = alpha * A @ B + beta * C)
// ─────────────────────────────────────────────────────────────────────────────
//
//          N                              K                 N
//    ┌───────────┐                  ┌──────────┐      ┌───────────┐
//    │           │                  │          │      │           │
//  M │     C     │  =  alpha *   M  │    A     │  @   │     B     │  + beta*C
//    │           │                  │          │    K │           │
//    └───────────┘                  └──────────┘      └───────────┘
//
// ─────────────────────────────────────────────────────────────────────────────
//  GRID / BLOCK TILING OVERVIEW
// ─────────────────────────────────────────────────────────────────────────────
//
//  The output C is partitioned into BM×BN tiles (64×64).
//  Each thread-block computes one such tile.
//
//  gridDim = ( ceil(N/BN), ceil(M/BM) )
//            (    x-dim  ,    y-dim   )
//
//  Why gridDim.x = N-direction?
//    → blocks with sequential blockIdx.x walk columns of B sequentially
//    → better L2 spatial locality since B is row-major
//
//        blockIdx.x →  0       1       2    ...
//                   ┌───────┬───────┬───────┬───
//  blockIdx.y = 0   │ 64×64 │ 64×64 │ 64×64 │
//                   ├───────┼───────┼───────┼───
//  blockIdx.y = 1   │ 64×64 │ 64×64 │       │
//                   ├───────┼───────┼       │
//  ...              │       │       │       │
//
// ─────────────────────────────────────────────────────────────────────────────
//  THREAD → OUTPUT MAPPING  (1D block-tiling)
// ─────────────────────────────────────────────────────────────────────────────
//
//  512 threads in a flat 1D block (threadIdx.x = 0..511).
//  Map to a 2D position in the BM×BN output tile:
//
//    threadCol = threadIdx.x % BN          (0..63)
//    threadRow = threadIdx.x / BN          (0..7)
//
//  Each thread owns TM=8 consecutive rows at one column:
//    rows  [threadRow*TM .. threadRow*TM+7],  col = threadCol
//
//                         BN = 64 columns
//            threadCol:  0  1  2  ...  63
//           ┌──────────────────────────────┐
//  tRow=0   │ T0 T1 T2 ...          T63    │  ← rows 0..7
//  (TM rows)│ T0 T1 T2 ...          T63    │
//           │ .. .. ..               ..    │
//           │ T0 T1 T2 ...          T63    │
//           ├──────────────────────────────┤
//  tRow=1   │T64 T65 T66 ...        T127   │  ← rows 8..15
//  (TM rows)│T64 T65 T66 ...        T127   │
//           │ .. .. ..               ..    │
//           │T64 T65 T66 ...        T127   │
//           ├──────────────────────────────┤
//           │            ...               │     ...
//           ├──────────────────────────────┤
//  tRow=7   │T448 T449 ...          T511   │  ← rows 56..63
//  (TM rows)│T448 T449 ...          T511   │
//           │ .. .. ..               ..    │
//           │T448 T449 ...          T511   │
//           └──────────────────────────────┘
//                       BM = 64 rows total
//
//  So 8 threadRows × TM=8 rows each = 64 = BM  ✓
//     64 threadCols × 1 col each     = 64 = BN  ✓
//     Total outputs = 64 × 64 = 4096,  512 threads × 8 each = 4096  ✓
//
// ─────────────────────────────────────────────────────────────────────────────
//  LOADING A & B TILES INTO SHARED MEMORY
// ─────────────────────────────────────────────────────────────────────────────
//
//  Each k-iteration loads a BM×BK slice of A and a BK×BN slice of B.
//
//  As[BM × BK] = 64×8 = 512 elements   → exactly 1 per thread
//  Bs[BK × BN] = 8×64 = 512 elements   → exactly 1 per thread
//
//  Loading A (BM=64, BK=8):              Loading B (BK=8, BN=64):
//    innerColA = tid % BK  (0..7)          innerColB = tid % BN  (0..63)
//    innerRowA = tid / BK  (0..63)         innerRowB = tid / BN  (0..7)
//
//  A is loaded with coalesced access along the K dimension:
//    threads 0-7   → row 0, cols 0-7      (one 32B cache line)
//    threads 8-15  → row 1, cols 0-7
//    ...
//    threads 504-511 → row 63, cols 0-7
//
//        BK=8 cols
//     ┌──────────┐
//     │ t0 .. t7 │  row 0     As[64 × 8]
//     │ t8 ..t15 │  row 1
//     │   ...    │
//     │t504..511 │  row 63
//     └──────────┘
//       64 rows
//
//  B is loaded with coalesced access along the N dimension:
//    threads 0-63   → row 0, cols 0-63   (two 128B cache lines)
//    threads 64-127 → row 1, cols 0-63
//    ...
//    threads 448-511 → row 7, cols 0-63
//
//        BN=64 cols
//     ┌──────────────────┐
//     │ t0 ......... t63 │  row 0    Bs[8 × 64]
//     │ t64 ....... t127 │  row 1
//     │       ...        │
//     │ t448 ...... t511 │  row 7
//     └──────────────────┘
//       8 rows
//
// ─────────────────────────────────────────────────────────────────────────────
//  OUTER LOOP: SLIDING WINDOW ALONG K
// ─────────────────────────────────────────────────────────────────────────────
//
//  A[M×K]                          B[K×N]
//  ┌───────────────────────┐        ┌───────────────────────────┐
//  │        K              │        │            N              │
//  │  ┌────┐               │        │  cCol*BN                  │
//  │  │As  │←BK            │      K │     ↓                     │
//  │  │    │  ←── slides →→│        │  ┌──────┐                 │
//  │  └────┘               │        │  │ Bs   │←BN              │
//  │  ↑                    │        │  │      │ slides          │
//  │  cRow*BM              │        │  └──────┘  ↓↓             │
//  └───────────────────────┘        └───────────────────────────┘
//
//  for (bkIdx = 0; bkIdx < K; bkIdx += BK):
//    1. Load As[BM×BK], Bs[BK×BN] from GMEM → SMEM
//    2. __syncthreads()
//    3. Compute partial dot-products
//    4. __syncthreads()
//    5. Advance A pointer right by BK, B pointer down by BK rows
//
// ─────────────────────────────────────────────────────────────────────────────
//  INNER COMPUTE: DOT-PRODUCT WITH REGISTER REUSE
// ─────────────────────────────────────────────────────────────────────────────
//
//  for dotIdx in 0..BK-1:
//    tmpB = Bs[dotIdx][threadCol]          ← 1 SMEM read, reused TM times
//    for resIdx in 0..TM-1:
//      threadResults[resIdx] += As[threadRow*TM+resIdx][dotIdx] * tmpB
//
//  Visually for one thread (threadRow=r, threadCol=c):
//
//    As (BM×BK)              Bs (BK×BN)              threadResults (TM×1)
//    ┌────────┐             ┌──────────────┐         ┌────┐
//    │        │             │      c       │         │res0│ ← row r*TM+0
//    │  r*TM →│■■■■■■■■│    │      ↓       │         │res1│ ← row r*TM+1
//    │        │■■■■■■■■│    │  ■ ■ ■ ■ ■ ■ │←dotIdx  │res2│
//    │        │■■■■■■■■│    │              │         │res3│
//    │        │■■■■■■■■│    │              │         │res4│
//    │        │■■■■■■■■│    │              │         │res5│
//    │        │■■■■■■■■│    └──────────────┘         │res6│
//    │        │■■■■■■■■│                             │res7│ ← row r*TM+7
//    │        │■■■■■■■■│◄─ TM=8 rows                 └────┘
//    │        │        │
//    └────────┘        │      ■ = elements accessed by this thread
//     BK=8 cols ───────┘
//
//  The outer dotIdx loop lets us cache tmpB in a register and reuse it
//  across all TM accumulations — key for arithmetic intensity.
//
//  Arithmetic intensity vs Kernel 3:
//    Kernel 3: 1 thread → 1 output  → 2*BK SMEM reads for BK FMAs → 0.5
//    FMA/read Kernel 4: 1 thread → TM output → (TM+1)*BK SMEM reads for TM*BK
//    FMAs
//              = TM/(TM+1) FMA/read ≈ 0.89 FMA/read (TM=8)
//
// =============================================================================

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha,
                                     const float* __restrict__ A,
                                     const float* __restrict__ B, float beta,
                                     float* __restrict__ C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // Each thread loads exactly one element of As and one of Bs.
  // BM*BK == BN*BK == blockDim.x == 512.
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  float threadResults[TM] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // ── Load GMEM → SMEM (bounds-checked for non-tile-aligned dims) ─────
    As[innerRowA * BK + innerColA] =
        (cRow * BM + innerRowA < M && bkIdx + innerColA < K)
            ? A[innerRowA * K + innerColA]
            : 0.0f;
    Bs[innerRowB * BN + innerColB] =
        (bkIdx + innerRowB < K && cCol * BN + innerColB < N)
            ? B[innerRowB * N + innerColB]
            : 0.0f;
    __syncthreads();

    A += BK;
    B += BK * N;

    // ── Compute: dotIdx outer, resIdx inner (reuse tmpB across TM rows) ─
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // ── Write results back to GMEM (bounds-checked) ────────────────────────
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    const uint globalRow = cRow * BM + threadRow * TM + resIdx;
    const uint globalCol = cCol * BN + threadCol;
    if (globalRow < M && globalCol < N) {
      C[(threadRow * TM + resIdx) * N + threadCol] =
          alpha * threadResults[resIdx] +
          beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
  }
}
