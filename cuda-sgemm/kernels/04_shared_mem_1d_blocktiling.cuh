
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha,
                                     const float* __restrict__ A,
                                     const float* __restrict__ B, float beta,
                                     float* __restrict__ C) {
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK;  // warp-level GMEM coalescing
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;  // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.x / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches (with bounds checks for non-tile-aligned dims)
    As[innerRowA * BK + innerColA] =
        (cRow * BM + innerRowA < M && bkIdx + innerColA < K)
            ? A[innerRowA * K + innerColA]
            : 0.0f;
    Bs[innerRowB * BN + innerColB] =
        (bkIdx + innerRowB < K && cCol * BN + innerColB < N)
            ? B[innerRowB * N + innerColB]
            : 0.0f;
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
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
