// Basic CUTLASS GEMM example — demonstrates using CUTLASS 2.x device-level API
// to compute C = alpha * A * B + beta * C  (SGEMM, row-major)
//
// Build:  make cutlass_gemm_basic
// Run:    ./cutlass_gemm_basic

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                \
  do {                                                                  \
    cudaError_t err = (expr);                                           \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                 \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

// ── CUTLASS GEMM type definition ────────────────────────────────────────────
// This typedef wires together all the template parameters that CUTLASS needs.
// For a simple SGEMM the defaults are fine — CUTLASS picks tile sizes, warp
// arrangements, and pipeline stages automatically based on the arch.
using Gemm = cutlass::gemm::device::Gemm<
    float,                       // Element type of A
    cutlass::layout::RowMajor,   // Layout of A
    float,                       // Element type of B
    cutlass::layout::RowMajor,   // Layout of B
    float,                       // Element type of C
    cutlass::layout::RowMajor,   // Layout of C
    float,                       // Accumulator type
    cutlass::arch::OpClassSimt,  // Use SIMT (works on all archs)
    cutlass::arch::Sm120         // Target arch (safe baseline)
    >;

void fill_random(float* data, int n) {
  for (int i = 0; i < n; ++i) data[i] = static_cast<float>(rand()) / RAND_MAX;
}

void reference_gemm(const float* A, const float* B, float* C, int M, int N,
                    int K, float alpha, float beta) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += A[m * K + k] * B[k * N + n];
      C[m * N + n] = alpha * acc + beta * C[m * N + n];
    }
}

int main() {
  const int M = 1 << 11, N = 1 << 11, K = 1 << 11;
  const float alpha = 1.0f, beta = 0.0f;

  printf("CUTLASS SGEMM: %d × %d × %d\n", M, N, K);

  // ── Host allocations ──────────────────────────────────────────────────────
  size_t sizeA = M * K * sizeof(float);
  size_t sizeB = K * N * sizeof(float);
  size_t sizeC = M * N * sizeof(float);

  float* h_A = (float*)malloc(sizeA);
  float* h_B = (float*)malloc(sizeB);
  float* h_C = (float*)malloc(sizeC);
  float* h_ref = (float*)malloc(sizeC);

  srand(42);
  fill_random(h_A, M * K);
  fill_random(h_B, K * N);
  fill_random(h_C, M * N);
  memcpy(h_ref, h_C, sizeC);

  // ── Device allocations ────────────────────────────────────────────────────
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, sizeA));
  CUDA_CHECK(cudaMalloc(&d_B, sizeB));
  CUDA_CHECK(cudaMalloc(&d_C, sizeC));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice));

  // ── Run CUTLASS GEMM ─────────────────────────────────────────────────────
  Gemm gemm_op;

  Gemm::Arguments args({M, N, K},     // problem size
                       {d_A, K},      // A: ptr + leading dim
                       {d_B, N},      // B: ptr + leading dim
                       {d_C, N},      // C (source)
                       {d_C, N},      // D (destination) — in-place
                       {alpha, beta}  // epilogue scalars
  );

  cutlass::Status status = gemm_op(args);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "CUTLASS GEMM failed: %d\n", static_cast<int>(status));
    return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // ── Verify ────────────────────────────────────────────────────────────────
  float* h_result = (float*)malloc(sizeC);
  CUDA_CHECK(cudaMemcpy(h_result, d_C, sizeC, cudaMemcpyDeviceToHost));

  reference_gemm(h_A, h_B, h_ref, M, N, K, alpha, beta);

  double max_err = 0.0;
  for (int i = 0; i < M * N; ++i) {
    double diff = fabs(h_result[i] - h_ref[i]);
    if (diff > max_err) max_err = diff;
  }
  printf("Max absolute error vs CPU reference: %e\n", max_err);
  printf("%s\n", max_err < 1e-3 ? "PASSED" : "FAILED");

  // ── Cleanup ───────────────────────────────────────────────────────────────
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_ref);
  free(h_result);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return max_err < 1e-3 ? EXIT_SUCCESS : EXIT_FAILURE;
}
