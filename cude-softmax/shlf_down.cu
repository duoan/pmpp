#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

__global__ void test_shuf_down_sync(int* out, const int* inp) {
  int val = inp[threadIdx.x];
  val = __shfl_down_sync(0xFFFFFFFF, val, 2, 32);
  out[threadIdx.x] = val;
}

int main() {
  const int numThreads = 32;

  int h_inp[numThreads];
  int h_out[numThreads];

  for (int i = 0; i < numThreads; ++i) {
    h_inp[i] = i;
  }

  int *d_inp, *d_out;

  cudaMalloc(&d_inp, numThreads * sizeof(int));
  cudaMalloc(&d_out, numThreads * sizeof(int));

  cudaMemcpy(d_inp, h_inp, numThreads * sizeof(int), cudaMemcpyHostToDevice);

  test_shuf_down_sync<<<1, numThreads>>>(d_out, d_inp);

  cudaMemcpy(h_out, d_out, numThreads * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < numThreads; i++) {
    printf("%d = %d\n", i, h_out[i]);
  }

  return 0;
}