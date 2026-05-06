#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf(                                                                  \
          "CUDA Error:\nFile: %s\nLine: %d\nError code: %d\nError text: %s\n", \
          __FILE__, __LINE__, err, cudaGetErrorString(err));                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_LAST_CUDA_ERROR()                                              \
  do {                                                                       \
    cudaError_t err = cudaGetLastError();                                    \
    if (err != cudaSuccess) {                                                \
      printf(                                                                \
          "CUDA Kernel Launch Error:\nFile: %s\nLine: %d\nError text: %s\n", \
          __FILE__, __LINE__, cudaGetErrorString(err));                      \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

void softmax_fwd_cpu(float* out, const float* inp, int N, int C) {
  for (int i = 0; i < N; i++) {
    const float* inp_row = inp + i * C;
    float* out_row = out + i * C;

    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) {
      if (inp_row[j] > maxval) {
        maxval = inp_row[j];
      }
    }

    float sum = 0.f;
    for (int j = 0; j < C; j++) {
      out_row[j] = expf(inp_row[j] - maxval);
      sum += out_row[j];
    }

    for (int j = 0; j < C; j++) {
      out_row[j] /= (float)sum;
    }
  }
}

// naive implementation
__global__ void softmax_fwd_kernell_1(float* out, const float* inp, int N,
                                      int C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    const float* inp_row = inp + i * C;
    float* out_row = out + i * C;

    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) {
      if (inp_row[j] > maxval) {
        maxval = inp_row[j];
      }
    }

    float sum = 0.f;

    // only 1 thread calculate here
    for (int j = 0; j < C; j++) {
      out_row[j] = expf(inp_row[j] - maxval);
      sum += out_row[j];
    }

    for (int j = 0; j < C; j++) {
      out_row[j] /= (float)sum;
    }
  }
}

// use shared memory and reducation to speed up
__global__ void softmax_fwd_kernell_2(float* out, const float* inp, int N,
                                      int C) {
  extern __shared__ float shared[];
  int idx = blockIdx.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x;
  const float* x = inp + idx * C;  // idx-th row of inp

  // thread coarsening
  float maxval = -INFINITY;
  for (int i = tid; i < C; i++) {
    maxval = fmaxf(maxval, x[i]);
  }
  shared[tid] = maxval;
  __syncthreads();

  // reducations
  for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
  }

  __syncthreads();
  float offset = shared[0];
  // compute expf and write the result to global memory
  for (int i = tid; i < C; i += block_size) {
    out[idx * C + i] = expf(x[i] - offset);
  }
  __syncthreads();

  // thread coarsening again, for the sum
  x = out + idx * C;  // idx-th row of out
  float sumval = 0.0f;
  for (int i = tid; i < C; i += block_size) {
    sumval += x[i];
  }
  shared[tid] = sumval;
  __syncthreads();

  // reudcations
  for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
  }
  // broadcast the sum to all threads in the block
  __syncthreads();

  float sum = shared[0];
  // divide the input value by the sum
  for (int i = tid; i < C; i += block_size) {
    out[idx * C + i] = x[i] / sum;
  }
}

// Kernel 3
__device__ float warpReduceMax(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

__device__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

__global__ void softmax_fwd_kernell_3(float* out, const float* inp, int N,
                                      int C) {
  int idx = blockIdx.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x;
  const float* x = inp + idx * C;  // idx-th row of inp

  // thread coarsening
  float maxval = -INFINITY;
  for (int i = tid; i < C; i += blockDim.x) {
    maxval = fmaxf(maxval, x[i]);
  }
  maxval = warpReduceMax(maxval);

  float offset = __shfl_sync(0xFFFFFFFF, maxval, 0);
  // compute expf and write the result to global memory
  for (int i = tid; i < C; i += block_size) {
    out[idx * C + i] = expf(x[i] - offset);
  }
  __syncthreads();

  // thread coarsening again, for the sum
  x = out + idx * C;  // idx-th row of out
  float sumval = 0.0f;
  for (int i = tid; i < C; i += block_size) {
    sumval += x[i];
  }
  sumval = warpReduceSum(sumval);

  float sum = __shfl_sync(0xFFFFFFFF, sumval, 0);
  // divide the input value by the sum
  for (int i = tid; i < C; i += block_size) {
    out[idx * C + i] = x[i] / sum;
  }
}

// kernel 4
__global__ void softmax_fwd_kernell_4(float* out, const float* inp, int N,
                                      int C) {
  // shared[] must be allocated to have 2 * warpsPerBlock elements
  extern __shared__ float shared[];

  int idx = blockIdx.x;
  int tid = threadIdx.x;
  int warpId = threadIdx.x / 32;  // warp index within a block
  int laneId = threadIdx.x % 32;  // thread index within a warp

  // the number of warps per block. recall that blockDim.x is block_size
  int warpsPerBlock = blockDim.x / 32;

  // fist half for max values, the second half for sum values
  float* maxvals = shared;
  float* sumvals = &shared[warpsPerBlock];

  // one row of inp, i.e. inp[idx, :] of shape (C,)
  const float* x = inp + idx * C;

  // fist, thread coarsening by directly accessing global memory in series
  float maxval = -INFINITY;
  for (int i = tid; i < C; i += blockDim.x) {
    maxval = fmaxf(maxval, x[i]);
  }
  // now within-warp reductions for maxval
  maxval = warpReduceMax(maxval);

  // the 0-th thread of each warp writes the maxval of that warp to shared
  // memory
  if (laneId == 0) {
    maxvals[warpId] = maxval;
  }
  __syncthreads();

  // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
  if (tid == 0) {
    float val = maxvals[tid];
    for (int i = 1; i < warpsPerBlock; i++) {
      val = fmaxf(val, maxvals[i]);
    }
    // store the final max in the fist position
    maxvals[0] = val;
  }
  __syncthreads();

  // broadcast the max to all threads
  float offset = maxvals[0];

  // compute expf and write the result to global memory
  for (int i = tid; i < C; i += blockDim.x) {
    out[idx * C + i] = expf(x[i] - offset);
  }

  // okay now we calculated exp(x - max(x))
  // step 2: sum all the values and divied by the sum

  // thread coarsening for sum
  x = out + idx * C;
  float sumval = 0.0f;
  for (int i = tid; i < C; i += blockDim.x) {
    sumval += x[i];
  }
  // within-warp reduction for sumval
  sumval = warpReduceSum(sumval);

  // write sumval to shared memory
  if (laneId == 0) {
    sumvals[warpId] = sumval;
  }
  __syncthreads();

  if (tid == 0) {
    float val = sumvals[tid];
    for (int i = 1; i < warpsPerBlock; i++) {
      val += sumvals[i];
    }
    sumvals[0] = val;
  }
  __syncthreads();
  // broadcast the sum to all threads
  float sum = sumvals[0];

  // divide the input value by the sum
  for (int i = tid; i < C; i += blockDim.x) {
    out[idx * C + i] = x[i] / sum;
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

int main() {
  int N = 1024;  // Example array size
  int C = 4096;

  size_t numel = N * C;

  float* inp = (float*)malloc(numel * sizeof(float));
  float* out_cpu = (float*)malloc(numel * sizeof(float));
  float* out_gpu = (float*)malloc(numel * sizeof(float));

  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      inp[n * C + c] = float(C);
    }
  }

  clock_t start_cpu = clock();
  softmax_fwd_cpu(out_cpu, inp, N, C);
  clock_t end_cpu = clock();
  double time_cpu = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float *d_out, *d_inp;
  CHECK_CUDA(cudaMalloc((void**)&d_out, N * C * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&d_inp, N * C * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_inp, inp, numel, cudaMemcpyHostToDevice));

  cudaEventRecord(start);

  // Launch kernel
  // kernel 1
  //   int blockSize = 1;
  //   int numBlocks = N;
  //   softmax_fwd_kernell_1<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);

  // kernel 2
  //   int blockSize = 128;
  //   int numBlocks = N;
  //   softmax_fwd_kernell_2<<<numBlocks, blockSize, blockSize *
  //   sizeof(float)>>>(
  //       d_out, d_inp, N, C);

  // kernel 3
  // int blockSize = 32;  // must be 32
  // int numBlocks = N;
  // softmax_fwd_kernell_3<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);

  // kernel 4
  int blockSize = 512;
  int numBlocks = N;
  softmax_fwd_kernell_4<<<numBlocks, blockSize,
                          2 * blockSize / 32 * sizeof(float)>>>(d_out, d_inp, N,
                                                                C);

  CHECK_LAST_CUDA_ERROR();
  cudaEventRecord(stop);

  // Wait for the event to compute
  CHECK_CUDA(cudaEventSynchronize(stop));
  float gpu_time_ms = 0;
  cudaEventElapsedTime(&gpu_time_ms, start, stop);

  // Copy result back to host
  cudaMemcpy(out_gpu, d_out, numel * sizeof(float), cudaMemcpyDeviceToHost);

  // cleanup
  cudaFree(d_out);
  cudaFree(d_inp);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // compare the result
  bool correct = true;
  for (int i = 0; i < numel; i++) {
    if (fabs(out_cpu[i] - out_gpu[i]) > 1e-5) {
      printf("Result verification failed at element %d: CPU=%f, GPU=%f\n", i,
             out_cpu[i], out_gpu[i]);
      correct = false;
      break;
    }
  }
  if (correct) {
    printf("Result verification passed!\n");
  }

  // Print performance comparition
  printf("CPU time: %f ms\n", time_cpu * 1000.0);
  printf("GPU time: %f ms\n", gpu_time_ms);
  printf("Speedup: %fx\n", time_cpu / (gpu_time_ms / 1000.0f));

  free(inp);
  free(out_cpu);
  free(out_gpu);

  return 0;
}