void run_sgemm_cublas(int M, int N, int K, float alpha, float* A, float* B,
                      float beta, float* C, cublasHandle_t handle) {
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           B, N, A, K, &beta, C, N));
}
