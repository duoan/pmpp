#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

#define M 16
#define N 16
#define K 16

__global__ void tensor_core_matmul(half* a, half* b, float* c, float* d) {
  // declare the fragments
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
  wmma::fragment<wmma::accumulator, M, N, K, float> d_frag;

  // Load the inputs
  wmma::load_matrix_sync(a_frag, a, K);
  wmma::load_matrix_sync(b_frag, b, K);
  wmma::load_matrix_sync(c_frag, c, N, wmma::mem_col_major);

  // Perform the matmul
  wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

  // Store the output
  wmma::store_matrix_sync(d, d_frag, N, wmma::mem_col_major);
}

int main() {
  half h_A[M * K], h_B[K * N];
  float h_C[M * N], h_D[M * N];

  for (int i = 0; i < M * K; ++i) {
    h_A[i] = static_cast<half>(1 % 3);
  }

  for (int i = 0; i < K * N; ++i) {
    h_B[i] = static_cast<half>(1 % 3);
  }

  for (int i = 0; i < K * N; ++i) {
    h_C[i] = static_cast<float>(1 % 3);
  }

  half *d_A, *d_B;
  float *d_C, *d_D;
  cudaMalloc((void**)&d_A, M * K * sizeof(half));
  cudaMalloc((void**)&d_B, K * N * sizeof(half));
  cudaMalloc((void**)&d_C, M * N * sizeof(float));
  cudaMalloc((void**)&d_D, M * N * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel
  dim3 grid(1);
  dim3 block(32, 1, 1);
  tensor_core_matmul<<<grid, block>>>(d_A, d_B, d_C, d_D);

  cudaDeviceSynchronize();

  cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  bool success = true;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);

  return 0;
}