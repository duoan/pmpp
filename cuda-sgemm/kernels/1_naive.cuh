__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32
  if (x < M && y < N) {
    float tmp = 0.0f;
    // flops: K * 2
    for (int i = 0; i < K; ++i) {
      // read 4B (1 float32) from A, and 4B from B
      // total 2 * 4B
      tmp += A[x * K + i] * B[i * N + y];
    }
    // flops: 3
    // C = α*(A@B)+β*C

    // read 1 float32 from C -> 4B
    // store: 1 float 32 to C -> 4B
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
  // per thread flops: (2K + 3)
  // total flops = num_threads * per_thread_flops
  // (M, K) @ (K, N) => (M, N)
  // so we launched M * N threads,
  // finally, the total flops = M * N * (2K + 3)
  // when M = N = K = 4092
  // 137,086,926,768 flops => 137 GFLOPS

  // data to read
  // per thread: K * 2 * 4 + 4 => total: M * N * (8K + 4)
  // 4092 * 4092 * (8 * 4092 + 4) = 548,213,751,360 bytes => ~548 GB

  // NOTE:
  // the minimum memory to read is (M * K + K * N + M * N) * 4 bytes
  // (4092 * 4092) * 3 * 4 = 200,987,568 bytes => ~192 MB
  // naive kernel reads 548 GB vs optimal 192 MB => ~2860x redundant traffica
  // lot of memory read

  // data to store
  // per thread: 4B => total: M * N * 4
  // 4092 * 4092 * 4 = 66,977,856 bytes => ~64 MB
}
