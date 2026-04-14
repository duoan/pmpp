
// ==========================================================================
// Shared-Memory Tiled SGEMM
// ==========================================================================
//
// C (M×N) = α · A (M×K) · B (K×N) + β · C (M×N)
//
// Key idea: instead of each thread fetching its own row/col from slow
// global memory (GMEM), a whole threadblock cooperatively loads a
// BLOCKSIZE×BLOCKSIZE tile into fast shared memory (SMEM), computes
// partial dot-products, then moves to the next tile along the K
// dimension.  This converts O(K) GMEM reads per thread into O(K/BS)
// GMEM reads, giving ~BLOCKSIZE× reuse.
//
// ──────────────────────────────────────────────────────────────────
//  Global matrix view  (row-major)
// ──────────────────────────────────────────────────────────────────
//
//       A (M×K)                    B (K×N)
//  ┌─────────────────┐       ┌─────────────────┐
//  │                 │       │    cCol         │
//  │    cRow ──►┌────┤       │      ▼          │
//  │            │tile│  ×    ├────┬────┬───────┤
//  │            │ As │       │    │tile│       │
//  │            └────┤       │    │ Bs │       │
//  │                 │       │    └────┘       │
//  └─────────────────┘       └─────────────────┘
//
//          ──── slides along K ────►
//
//       C (M×N)
//  ┌─────────────────┐
//  │        cCol     │
//  │    cRow ▼       │
//  │      ┌────┐     │   Each threadblock computes one BLOCKSIZE×BLOCKSIZE
//  │      │ Cs │     │   output tile of C by accumulating partial results
//  │      └────┘     │   from all tiles along K.
//  └─────────────────┘
//
// ──────────────────────────────────────────────────────────────────
//  Thread → element mapping inside one tile
// ──────────────────────────────────────────────────────────────────
//
//  blockDim = (BLOCKSIZE * BLOCKSIZE, 1, 1)   — 1-D thread block
//
//  threadIdx.x :  0  1  2 ... BS-1  BS BS+1 ... 2·BS-1  ...
//  threadRow   :  0  0  0 ...  0     1   1  ...   1     ...
//  threadCol   :  0  1  2 ... BS-1   0   1  ... BS-1    ...
//
//  So consecutive threadIdx.x → consecutive threadCol
//  → consecutive addresses in GMEM → coalesced loads!
//
// ──────────────────────────────────────────────────────────────────
//  Bounds checking  (when M, N, or K is not a multiple of BLOCKSIZE)
// ──────────────────────────────────────────────────────────────────
//
//  Example: M=4092, BLOCKSIZE=32.  4092/32 = 127.875 → 128 full blocks + 1
//  partial block covering only 28 valid rows.
//
//       K dim ──────────────────────────►
//       0             ...          4064  4092
//  M  ┌──────────────────────────────┬────┐
//  d  │                              │pad │ ← last bkIdx tile:
//  i  │        valid A region        │ 0s │   cols 4092..4095 are OOB,
//  m  │                              │    │   load 0.0f instead
//  │  ├──────────────────────────────┼────┤
//  ▼  │ pad rows (globalRow ≥ M)     │    │ ← bottom-right corner:
//     │ load 0s, don't write C       │ 0s │   both row & col OOB
//     └──────────────────────────────┴────┘
//
//  Without bounds checks, OOB threads load garbage into SMEM,
//  which poisons the dot-product for ALL threads sharing that tile —
//  including threads whose output position IS valid.
//  Fix: OOB loads → 0.0f (identity for addition), OOB stores → skip.

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float* __restrict__ A,
                                       const float* __restrict__ B, float beta,
                                       float* __restrict__ C) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // Allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  //
  //   threadIdx.x = threadRow * BLOCKSIZE + threadCol
  //
  //   ┌──────────── BLOCKSIZE ────────────┐
  //   │ (0,0) (0,1) (0,2) ... (0,BS-1)    │  threadRow = 0
  //   │ (1,0) (1,1) (1,2) ... (1,BS-1)    │  threadRow = 1
  //   │  ...                              │
  //   │(BS-1,0)          ... (BS-1,BS-1)  │  threadRow = BS-1
  //   └──────────────────────────────────-┘
  const uint threadRow = threadIdx.x / BLOCKSIZE;
  const uint threadCol = threadIdx.x % BLOCKSIZE;

  // global row/col this thread is responsible for
  const uint globalRow = cRow * BLOCKSIZE + threadRow;
  const uint globalCol = cCol * BLOCKSIZE + threadCol;

  // advance pointer to the starting positions
  A += cRow * BLOCKSIZE * K;                     // row = cRow, col = 0
  B += cCol * BLOCKSIZE;                         // row = 0, col = cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;  // row = cRow, col = cCol

  float tmp = 0.0f;
  // ── slide the tile window across the K dimension ──
  //
  //  bkIdx=0        bkIdx=BS      bkIdx=2·BS     ...
  //  ┌────┐         ┌────┐        ┌────┐
  //  │ A0 │ × B0    │ A1 │ × B1   │ A2 │ × B2   ...
  //  └────┘         └────┘        └────┘
  //  tmp  +=  As·Bs  +=  As·Bs    +=  As·Bs      ...
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    //
    // A tile load:  As[r][c] = A[globalRow][bkIdx + threadCol]
    //   OOB when globalRow ≥ M  (bottom partial block)
    //         or  bkIdx+threadCol ≥ K  (right partial tile)
    As[threadRow * BLOCKSIZE + threadCol] =
        (globalRow < M && bkIdx + threadCol < K) ? A[threadRow * K + threadCol]
                                                 : 0.0f;
    // B tile load:  Bs[r][c] = B[bkIdx + threadRow][globalCol]
    //   OOB when bkIdx+threadRow ≥ K  (bottom partial tile)
    //         or  globalCol ≥ N       (right partial block)
    Bs[threadRow * BLOCKSIZE + threadCol] =
        (bkIdx + threadRow < K && globalCol < N) ? B[threadRow * N + threadCol]
                                                 : 0.0f;

    // block threads in this block until cache is fully populated
    __syncthreads();

    // advance pointers onto next chunk
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the current cached block
    //
    //  tmp += Σ  As[threadRow][dotIdx] * Bs[dotIdx][threadCol]
    //         dotIdx=0..BS-1
    //
    //  As row = this thread's row of A tile  (reused BS times)
    //  Bs col = this thread's col of B tile  (reused BS times)
    //  → each element loaded from GMEM is reused BLOCKSIZE times
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done.
    __syncthreads();
  }

  // write back only if this thread maps to a valid C element
  if (globalRow < M && globalCol < N) {
    C[threadRow * N + threadCol] =
        alpha * tmp + beta * C[threadRow * N + threadCol];
  }
}
