#include <cublas_v2.h>

void run_sgemm_cublas(int M, int N, int K, float alpha, float* A, float* B,
                      float beta, float* C, cublasHandle_t handle) {
  // cuBLAS uses column-major, so compute C^T = B^T * A^T => C = A * B
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
              &beta, C, N);
}
