#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "runner.cuh"

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const int WARMUP_RUNS = 5;
const int BENCH_RUNS = 10;

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("Usage: %s <kernel_num> [M N K]\n", argv[0]);
    printf("  kernel_num: 0 = cuBLAS, 1 = naive\n");
    printf("  M N K: matrix dimensions (default 4096)\n");
    return 1;
  }

  CudaDeviceInfo();

  int kernel_num = atoi(argv[1]);
  int M = (argc > 4) ? atoi(argv[2]) : 4092;
  int N = (argc > 4) ? atoi(argv[3]) : 4092;
  int K = (argc > 4) ? atoi(argv[4]) : 4092;

  printf("\nMatrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
  printf("Kernel: %d\n\n", kernel_num);

  float alpha = 1.0f, beta = 0.0f;

  // host memory
  float* hA = new float[M * K];
  float* hB = new float[K * N];
  float* hC = new float[M * N];
  float* hC_ref = new float[M * N];

  randomize_matrix(hA, M * K);
  randomize_matrix(hB, K * N);
  zero_init_matrix(hC, M * N);
  zero_init_matrix(hC_ref, M * N);

  // device memory
  float *dA, *dB, *dC, *dC_ref;
  cudaCheck(cudaMalloc(&dA, M * K * sizeof(float)));
  cudaCheck(cudaMalloc(&dB, K * N * sizeof(float)));
  cudaCheck(cudaMalloc(&dC, M * N * sizeof(float)));
  cudaCheck(cudaMalloc(&dC_ref, M * N * sizeof(float)));

  cudaCheck(cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, hC, M * N * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, hC_ref, M * N * sizeof(float),
                       cudaMemcpyHostToDevice));

  // cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // --- cuBLAS reference ---
  printf("Running cuBLAS reference...\n");
  run_kernel(0, M, N, K, alpha, dA, dB, beta, dC_ref, handle);
  cudaDeviceSynchronize();

  // --- verify target kernel against cuBLAS ---
  if (kernel_num > 0) {
    cudaCheck(
        cudaMemcpy(dC, hC, M * N * sizeof(float), cudaMemcpyHostToDevice));
    run_kernel(kernel_num, M, N, K, alpha, dA, dB, beta, dC, handle);
    cudaDeviceSynchronize();

    cudaCheck(
        cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(hC_ref, dC_ref, M * N * sizeof(float),
                         cudaMemcpyDeviceToHost));

    if (verify_matrix(hC_ref, hC, M * N)) {
      printf("Kernel %d: PASSED correctness check.\n\n", kernel_num);
    } else {
      printf("Kernel %d: FAILED correctness check!\n\n", kernel_num);
    }
  }

  // --- benchmark ---
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < WARMUP_RUNS; i++) {
    run_kernel(kernel_num, M, N, K, alpha, dA, dB, beta, dC, handle);
  }
  cudaDeviceSynchronize();

  float total_ms = 0.0f;
  for (int i = 0; i < BENCH_RUNS; i++) {
    cudaEventRecord(start);
    run_kernel(kernel_num, M, N, K, alpha, dA, dB, beta, dC, handle);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;
  }

  float avg_ms = total_ms / BENCH_RUNS;
  double gflops = 2.0 * M * N * K / (avg_ms * 1e6);
  printf("Average time: %.3f ms  |  Performance: %.1f GFLOPS\n", avg_ms,
         gflops);

  // cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  delete[] hA;
  delete[] hB;
  delete[] hC;
  delete[] hC_ref;

  return 0;
}
