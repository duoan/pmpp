#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "exercises/ex1_vector_add.cuh"
#include "exercises/ex2_matrix_fill.cuh"
#include "exercises/ex3_stride_loop.cuh"
#include "exercises/ex4_2d_stride.cuh"

#define CUDA_CHECK(err)                                             \
  do {                                                              \
    cudaError_t e = (err);                                          \
    if (e != cudaSuccess) {                                         \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                               \
      exit(1);                                                      \
    }                                                               \
  } while (0)

#define KERNEL_CHECK()                   \
  do {                                   \
    CUDA_CHECK(cudaGetLastError());      \
    CUDA_CHECK(cudaDeviceSynchronize()); \
  } while (0)

// ── helpers ──────────────────────────────────────────────────────────────────

void fill_random(float* h, int n) {
  for (int i = 0; i < n; i++) h[i] = (float)(rand() % 100) / 10.0f;
}

void print_pass_fail(const char* name, bool passed) {
  printf("  %-30s %s\n", name, passed ? "✅ PASS" : "❌ FAIL");
}

// ── Exercise 1 test ─────────────────────────────────────────────────────────

bool test_ex1() {
  const int N = 10007;
  float *hA, *hB, *hC;
  hA = (float*)malloc(N * sizeof(float));
  hB = (float*)malloc(N * sizeof(float));
  hC = (float*)malloc(N * sizeof(float));
  fill_random(hA, N);
  fill_random(hB, N);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((N + 255) / 256);
  vector_add<<<grid, block>>>(dA, dB, dC, N);
  KERNEL_CHECK();

  CUDA_CHECK(cudaMemcpy(hC, dC, N * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (fabsf(hC[i] - (hA[i] + hB[i])) > 1e-5f) {
      printf("    ex1 mismatch at i=%d: got %f, expected %f\n", i, hC[i],
             hA[i] + hB[i]);
      ok = false;
      break;
    }
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  free(hA);
  free(hB);
  free(hC);
  return ok;
}

// ── Exercise 2 test ─────────────────────────────────────────────────────────

bool test_ex2() {
  const int M = 100, N = 200;
  float* hC = (float*)malloc(M * N * sizeof(float));

  float* dC;
  CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
  CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));

  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  matrix_fill<<<grid, block>>>(dC, M, N);
  KERNEL_CHECK();

  CUDA_CHECK(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int r = 0; r < M && ok; r++) {
    for (int c = 0; c < N && ok; c++) {
      float expected = (float)(r * 1000 + c);
      if (fabsf(hC[r * N + c] - expected) > 0.5f) {
        printf("    ex2 mismatch at [%d][%d]: got %f, expected %f\n", r, c,
               hC[r * N + c], expected);
        ok = false;
      }
    }
  }

  cudaFree(dC);
  free(hC);
  return ok;
}

// ── Exercise 3 test ─────────────────────────────────────────────────────────

bool test_ex3() {
  const int N = 1000000;
  float* hA = (float*)malloc(N * sizeof(float));
  float* hC = (float*)malloc(N * sizeof(float));
  fill_random(hA, N);

  float *dA, *dC;
  CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid(32);
  double_elements<<<grid, block>>>(dA, dC, N);
  KERNEL_CHECK();

  CUDA_CHECK(cudaMemcpy(hC, dC, N * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (fabsf(hC[i] - hA[i] * 2.0f) > 1e-5f) {
      printf("    ex3 mismatch at i=%d: got %f, expected %f\n", i, hC[i],
             hA[i] * 2.0f);
      ok = false;
      break;
    }
  }

  cudaFree(dA);
  cudaFree(dC);
  free(hA);
  free(hC);
  return ok;
}

// ── Exercise 4 test ─────────────────────────────────────────────────────────

bool test_ex4() {
  const int M = 513, N = 1025;
  float* hA = (float*)malloc(M * N * sizeof(float));
  float* hB = (float*)malloc(N * M * sizeof(float));
  for (int i = 0; i < M * N; i++) hA[i] = (float)i;

  float *dA, *dB;
  CUDA_CHECK(cudaMalloc(&dA, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, N * M * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA, M * N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid(4, 4);
  transpose<<<grid, block>>>(dA, dB, M, N);
  KERNEL_CHECK();

  CUDA_CHECK(cudaMemcpy(hB, dB, N * M * sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int r = 0; r < M && ok; r++) {
    for (int c = 0; c < N && ok; c++) {
      float a_val = hA[r * N + c];
      float b_val = hB[c * M + r];
      if (fabsf(a_val - b_val) > 1e-5f) {
        printf("    ex4 mismatch: A[%d][%d]=%f but B[%d][%d]=%f\n", r, c, a_val,
               c, r, b_val);
        ok = false;
      }
    }
  }

  cudaFree(dA);
  cudaFree(dB);
  free(hA);
  free(hB);
  return ok;
}

// ── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  int ex = 0;
  if (argc > 1) ex = atoi(argv[1]);

  printf("\n🏋️ CUDA Indexing Exercises\n");
  printf("══════════════════════════════════════════\n");

  if (ex == 0 || ex == 1) print_pass_fail("Ex1: 1D Vector Add", test_ex1());
  if (ex == 0 || ex == 2) print_pass_fail("Ex2: 2D Matrix Fill", test_ex2());
  if (ex == 0 || ex == 3) print_pass_fail("Ex3: Grid-Stride Loop", test_ex3());
  if (ex == 0 || ex == 4)
    print_pass_fail("Ex4: 2D Stride Transpose", test_ex4());

  printf("══════════════════════════════════════════\n\n");
  return 0;
}
