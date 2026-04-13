// ============================================================================
// Kernel 02: Global Memory Coalescing
// ============================================================================
//
// Problem with naive kernel (01):
//   naive uses 2D blockDim(32, 32), thread mapping:
//     row = blockIdx.x * 32 + threadIdx.x   (threadIdx.x = 0..31)
//     col = blockIdx.y * 32 + threadIdx.y   (threadIdx.y = 0..31)
//
//   A warp = 32 consecutive threads by threadIdx.x (same threadIdx.y).
//   So warp threads have: same col, consecutive rows.
//
//   When reading B[i * N + col]:
//     same col => same address => perfect, all threads read 1 value (broadcast)
//   When reading A[row * K + i]:
//     consecutive rows => stride-K access => each thread hits a different
//     cache line => 32 memory transactions instead of 1!
//   When writing C[row * N + col]:
//     consecutive rows => stride-N access => same problem, 32 transactions.
//
//   This is the worst case: warp threads access memory with stride = K (or N),
//   which is thousands of elements apart. No coalescing at all.
//
// Fix in this kernel:
//   Use 1D blockDim(BLOCKSIZE * BLOCKSIZE) and map threads so that
//   consecutive threadIdx.x values go along columns (the contiguous dimension):
//     row = blockIdx.x * BS + (threadIdx.x / BS)   <- changes every BS threads
//     col = blockIdx.y * BS + (threadIdx.x % BS)   <- changes every thread
//
//   Now a warp (32 consecutive threadIdx.x) has: same row, consecutive cols.
//
//   When reading A[row * K + i]:
//     same row => same address => broadcast, 1 transaction
//   When reading B[i * N + col]:
//     consecutive cols => consecutive addresses => coalesced, 1 transaction!
//   When writing C[row * N + col]:
//     consecutive cols => consecutive addresses => coalesced, 1 transaction!
//
//   Result: memory transactions per warp drop from ~32 to ~1 for B and C.
//   The total global memory traffic is the same, but HW utilization is ~32x
//   better because each transaction serves 32 threads instead of 1.
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
