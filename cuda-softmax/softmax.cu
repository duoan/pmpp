#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <math_constants.h>
#include <stdio.h>
#include <time.h>

#ifdef BUILD_PYTORCH_EXTENSION
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#endif

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

struct BenchmarkResult {
  float avg_ms;
  float min_ms;
  float max_ms;
  float bandwidth_gbps;  // optional, 0 if bytes == 0
};

template <typename LaunchFunc>
BenchmarkResult benchmark_cuda_kernel_batched(
    LaunchFunc launch, int warmup = 20, int iters = 100,
    int inner_repeats = 100, size_t bytes_moved_per_launch = 0) {
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) {
    for (int j = 0; j < inner_repeats; ++j) launch();
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  float total_ms = 0.f;
  float min_ms = FLT_MAX;
  float max_ms = 0.f;

  for (int i = 0; i < iters; ++i) {
    CHECK_CUDA(cudaEventRecord(start));
    for (int j = 0; j < inner_repeats; ++j) launch();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    float per_launch_ms = ms / inner_repeats;
    total_ms += per_launch_ms;
    if (per_launch_ms < min_ms) min_ms = per_launch_ms;
    if (per_launch_ms > max_ms) max_ms = per_launch_ms;
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  BenchmarkResult result;
  result.avg_ms = total_ms / iters;
  result.min_ms = min_ms;
  result.max_ms = max_ms;

  if (bytes_moved_per_launch > 0) {
    result.bandwidth_gbps =
        (bytes_moved_per_launch / 1e9) / (result.avg_ms / 1e3);
  } else {
    result.bandwidth_gbps = 0.f;
  }

  return result;
}

void print_benchmark(const char* name, const BenchmarkResult& r) {
  printf("[%s]\n", name);
  printf("  avg: %.3f ms\n", r.avg_ms);
  printf("  min: %.3f ms\n", r.min_ms);
  printf("  max: %.3f ms\n", r.max_ms);
  if (r.bandwidth_gbps > 0) {
    printf("  bw : %.2f GB/s\n", r.bandwidth_gbps);
  }
}

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
  for (int i = tid; i < C; i += block_size) {
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
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
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
  maxval = warp_reduce_max(maxval);

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
  sumval = warp_reduce_sum(sumval);

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
  maxval = warp_reduce_max(maxval);

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
  sumval = warp_reduce_sum(sumval);

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

template <int BLOCK_THREADS>
__device__ __forceinline__ float block_reduce_max(float v) {
  __shared__ float shared[32];  // one per warp
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  v = warp_reduce_max(v);
  if (lane == 0) shared[warp] = v;
  __syncthreads();

  float out = -CUDART_INF_F;
  if (warp == 0) {
    int num_warps = BLOCK_THREADS / 32;
    out = (lane < num_warps) ? shared[lane] : -CUDART_INF_F;
    out = warp_reduce_max(out);
    if (lane == 0) shared[0] = out;
  }
  __syncthreads();
  return shared[0];
}

template <int BLOCK_THREADS>
__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float shared[32];  // one per warp
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0) shared[warp] = v;
  __syncthreads();

  float out = 0.f;
  if (warp == 0) {
    int num_warps = BLOCK_THREADS / 32;
    out = (lane < num_warps) ? shared[lane] : 0.f;
    out = warp_reduce_sum(out);
    if (lane == 0) shared[0] = out;
  }
  __syncthreads();
  return shared[0];
}

template <int BLOCK_THREADS = 256, int PACKS_PER_THREAD = 4>
__global__ void softmax_fwd_kernel_5_fp32_c4096(float* __restrict__ out,
                                                const float* __restrict__ inp,
                                                int N) {
  constexpr int VEC = 4;  // float4
  constexpr int C = 4096;
  constexpr int PACKS_PER_ROW = C / VEC;  // 1024 float4

  int row = blockIdx.x;
  if (row >= N) return;

  const float4* row_in = reinterpret_cast<const float4*>(inp + row * C);
  float4* row_out = reinterpret_cast<float4*>(out + row * C);

  // each thread has 4 float4 -> 16 float
  float4 reg[PACKS_PER_THREAD];

  float thread_max = -CUDART_INF_F;
#pragma unroll
  for (int p = 0; p < PACKS_PER_THREAD; ++p) {
    int idx4 = threadIdx.x + p * BLOCK_THREADS;  // 0..1023
    float4 v = row_in[idx4];
    reg[p] = v;

    thread_max = fmaxf(thread_max, v.x);
    thread_max = fmaxf(thread_max, v.y);
    thread_max = fmaxf(thread_max, v.z);
    thread_max = fmaxf(thread_max, v.w);
  }

  float row_max = block_reduce_max<BLOCK_THREADS>(thread_max);

  float thread_sum = 0.f;
#pragma unroll
  for (int p = 0; p < PACKS_PER_THREAD; ++p) {
    float4 v = reg[p];
    v.x = __expf(v.x - row_max);
    v.y = __expf(v.y - row_max);
    v.z = __expf(v.z - row_max);
    v.w = __expf(v.w - row_max);

    reg[p] = v;

    thread_sum += v.x + v.y + v.z + v.w;
  }

  float row_sum = block_reduce_sum<BLOCK_THREADS>(thread_sum);
  float inv_sum = __fdividef(1.f, row_sum);

#pragma unroll
  for (int p = 0; p < PACKS_PER_THREAD; ++p) {
    float4 v = reg[p];
    v.x *= inv_sum;
    v.y *= inv_sum;
    v.z *= inv_sum;
    v.w *= inv_sum;

    int idx4 = threadIdx.x + p * BLOCK_THREADS;
    row_out[idx4] = v;
  }
}

// kernel 6: fixed-C template version of v5.
//
// v5 is excellent for C=4096 because every row stays in registers after the
// first global load. v6 keeps that idea but moves C into a template parameter,
// so common softmax widths can each get a fully unrolled, vectorized kernel.
template <int C, int BLOCK_THREADS>
__global__ void softmax_fwd_kernel_6_fp32_fixed_c(float* __restrict__ out,
                                                  const float* __restrict__ inp,
                                                  int N) {
  constexpr int VEC = 4;  // float4
  static_assert(C % (VEC * BLOCK_THREADS) == 0,
                "C must be covered exactly by float4 packs");
  constexpr int PACKS_PER_THREAD = C / (VEC * BLOCK_THREADS);

  int row = blockIdx.x;
  if (row >= N) return;

  const float4* row_in = reinterpret_cast<const float4*>(inp + row * C);
  float4* row_out = reinterpret_cast<float4*>(out + row * C);

  float4 reg[PACKS_PER_THREAD];

  float thread_max = -CUDART_INF_F;
#pragma unroll
  for (int p = 0; p < PACKS_PER_THREAD; ++p) {
    int idx4 = threadIdx.x + p * BLOCK_THREADS;
    float4 v = row_in[idx4];
    reg[p] = v;

    thread_max = fmaxf(thread_max, v.x);
    thread_max = fmaxf(thread_max, v.y);
    thread_max = fmaxf(thread_max, v.z);
    thread_max = fmaxf(thread_max, v.w);
  }

  float row_max = block_reduce_max<BLOCK_THREADS>(thread_max);

  float thread_sum = 0.f;
#pragma unroll
  for (int p = 0; p < PACKS_PER_THREAD; ++p) {
    float4 v = reg[p];
    v.x = __expf(v.x - row_max);
    v.y = __expf(v.y - row_max);
    v.z = __expf(v.z - row_max);
    v.w = __expf(v.w - row_max);

    reg[p] = v;
    thread_sum += v.x + v.y + v.z + v.w;
  }

  float row_sum = block_reduce_sum<BLOCK_THREADS>(thread_sum);
  float inv_sum = __fdividef(1.f, row_sum);

#pragma unroll
  for (int p = 0; p < PACKS_PER_THREAD; ++p) {
    float4 v = reg[p];
    v.x *= inv_sum;
    v.y *= inv_sum;
    v.z *= inv_sum;
    v.w *= inv_sum;

    int idx4 = threadIdx.x + p * BLOCK_THREADS;
    row_out[idx4] = v;
  }
}

// kernel 7a: generic fused softmax for arbitrary C that fits in shared memory.
//
// This replaces the old kernel_4 fallback. kernel_4 writes exp(x - max) to
// global memory, reads it back to sum, then writes normalized output. v7a keeps
// that intermediate in shared memory, so the fallback path no longer burns
// extra global bandwidth.
template <int BLOCK_THREADS>
__global__ void softmax_fwd_kernel_7_fp32_shared(float* __restrict__ out,
                                                 const float* __restrict__ inp,
                                                 int N, int C) {
  extern __shared__ float row_exp[];

  int row = blockIdx.x;
  if (row >= N) return;

  const float* x = inp + row * C;
  float* y = out + row * C;

  float thread_max = -CUDART_INF_F;
  for (int i = threadIdx.x; i < C; i += BLOCK_THREADS) {
    thread_max = fmaxf(thread_max, x[i]);
  }
  float row_max = block_reduce_max<BLOCK_THREADS>(thread_max);

  float thread_sum = 0.f;
  for (int i = threadIdx.x; i < C; i += BLOCK_THREADS) {
    float v = __expf(x[i] - row_max);
    row_exp[i] = v;
    thread_sum += v;
  }
  float row_sum = block_reduce_sum<BLOCK_THREADS>(thread_sum);
  float inv_sum = __fdividef(1.f, row_sum);

  for (int i = threadIdx.x; i < C; i += BLOCK_THREADS) {
    y[i] = row_exp[i] * inv_sum;
  }
}

// kernel 7b: arbitrary-C streaming fallback when one full row does not fit in
// shared memory. It reads input three times but still avoids global-memory
// intermediate writes.
template <int BLOCK_THREADS>
__global__ void softmax_fwd_kernel_7_fp32_streaming(
    float* __restrict__ out, const float* __restrict__ inp, int N, int C) {
  int row = blockIdx.x;
  if (row >= N) return;

  const float* x = inp + row * C;
  float* y = out + row * C;

  float thread_max = -CUDART_INF_F;
  for (int i = threadIdx.x; i < C; i += BLOCK_THREADS) {
    thread_max = fmaxf(thread_max, x[i]);
  }
  float row_max = block_reduce_max<BLOCK_THREADS>(thread_max);

  float thread_sum = 0.f;
  for (int i = threadIdx.x; i < C; i += BLOCK_THREADS) {
    thread_sum += __expf(x[i] - row_max);
  }
  float row_sum = block_reduce_sum<BLOCK_THREADS>(thread_sum);
  float inv_sum = __fdividef(1.f, row_sum);

  for (int i = threadIdx.x; i < C; i += BLOCK_THREADS) {
    y[i] = __expf(x[i] - row_max) * inv_sum;
  }
}

// kernel 8: arbitrary-C shared-memory fallback with one global input read.
//
// v7a still reads input twice from global memory: once for max and once for
// exp/sum. v8 first stages the whole row into shared memory, then does max,
// exp/sum, and output from that shared copy. For row widths that fit in shared
// memory, this matches the ideal global-memory traffic of read input + write
// output.
template <int BLOCK_THREADS>
__global__ void softmax_fwd_kernel_8_fp32_shared_input(
    float* __restrict__ out, const float* __restrict__ inp, int N, int C) {
  extern __shared__ float row_vals[];

  int row = blockIdx.x;
  if (row >= N) return;

  const float* x = inp + row * C;
  float* y = out + row * C;

  for (int i = threadIdx.x; i < C; i += BLOCK_THREADS) {
    row_vals[i] = x[i];
  }
  __syncthreads();

  float thread_max = -CUDART_INF_F;
  for (int i = threadIdx.x; i < C; i += BLOCK_THREADS) {
    thread_max = fmaxf(thread_max, row_vals[i]);
  }
  float row_max = block_reduce_max<BLOCK_THREADS>(thread_max);

  float thread_sum = 0.f;
  for (int i = threadIdx.x; i < C; i += BLOCK_THREADS) {
    float v = __expf(row_vals[i] - row_max);
    row_vals[i] = v;
    thread_sum += v;
  }
  float row_sum = block_reduce_sum<BLOCK_THREADS>(thread_sum);
  float inv_sum = __fdividef(1.f, row_sum);

  for (int i = threadIdx.x; i < C; i += BLOCK_THREADS) {
    y[i] = row_vals[i] * inv_sum;
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

#ifdef BUILD_PYTORCH_EXTENSION
template <int C, int BLOCK_THREADS>
void launch_softmax_v6(float* out, const float* x, int N, cudaStream_t stream) {
  softmax_fwd_kernel_6_fp32_fixed_c<C, BLOCK_THREADS>
      <<<N, BLOCK_THREADS, 0, stream>>>(out, x, N);
}

template <int BLOCK_THREADS>
void launch_softmax_v7(float* out, const float* x, int N, int C,
                       cudaStream_t stream) {
  constexpr int MAX_DYNAMIC_SHARED_BYTES = 96 * 1024;
  size_t shared_bytes = static_cast<size_t>(C) * sizeof(float);
  if (shared_bytes <= MAX_DYNAMIC_SHARED_BYTES) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        softmax_fwd_kernel_8_fp32_shared_input<BLOCK_THREADS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_DYNAMIC_SHARED_BYTES));
    softmax_fwd_kernel_8_fp32_shared_input<BLOCK_THREADS>
        <<<N, BLOCK_THREADS, shared_bytes, stream>>>(out, x, N, C);
  } else {
    softmax_fwd_kernel_7_fp32_streaming<BLOCK_THREADS>
        <<<N, BLOCK_THREADS, 0, stream>>>(out, x, N, C);
  }
}

at::Tensor softmax_cuda(at::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "softmax_cuda expects a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == at::kFloat,
              "softmax_cuda only supports float32");
  TORCH_CHECK(input.dim() == 2, "softmax_cuda expects a 2D tensor");
  TORCH_CHECK(input.size(0) <= INT_MAX, "row count exceeds int32 range");
  TORCH_CHECK(input.size(1) <= INT_MAX, "column count exceeds int32 range");

  c10::cuda::CUDAGuard device_guard(input.device());

  auto x = input.contiguous();
  auto out = at::empty_like(x);

  int N = static_cast<int>(x.size(0));
  int C = static_cast<int>(x.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  float* out_ptr = out.data_ptr<float>();
  const float* x_ptr = x.data_ptr<float>();

  switch (C) {
    case 256:
      launch_softmax_v6<256, 64>(out_ptr, x_ptr, N, stream);
      break;
    case 512:
      launch_softmax_v6<512, 128>(out_ptr, x_ptr, N, stream);
      break;
    case 1024:
      launch_softmax_v6<1024, 128>(out_ptr, x_ptr, N, stream);
      break;
    case 2048:
      launch_softmax_v6<2048, 128>(out_ptr, x_ptr, N, stream);
      break;
    case 4096:
      launch_softmax_v6<4096, 256>(out_ptr, x_ptr, N, stream);
      break;
    case 8192:
      launch_softmax_v6<8192, 256>(out_ptr, x_ptr, N, stream);
      break;
    default:
      if (C <= 1024) {
        launch_softmax_v7<128>(out_ptr, x_ptr, N, C, stream);
      } else {
        launch_softmax_v7<256>(out_ptr, x_ptr, N, C, stream);
      }
      break;
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
#endif

#ifndef BUILD_PYTORCH_EXTENSION
int main() {
  int N = 1024;  // Example array size
  int C = 4096;

  size_t numel = N * C;

  float* inp = (float*)malloc(numel * sizeof(float));
  float* out_cpu = (float*)malloc(numel * sizeof(float));
  float* out_gpu = (float*)malloc(numel * sizeof(float));

  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      inp[n * C + c] = ((float)rand() / RAND_MAX) * 20.f - 10.f;
    }
  }

  clock_t start_cpu = clock();
  softmax_fwd_cpu(out_cpu, inp, N, C);
  clock_t end_cpu = clock();
  double time_cpu = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

  float *d_out, *d_inp;
  CHECK_CUDA(cudaMalloc((void**)&d_out, N * C * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&d_inp, N * C * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(d_inp, inp, numel * sizeof(float), cudaMemcpyHostToDevice));

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
  // int blockSize = 512;
  // int numBlocks = N;
  // softmax_fwd_kernell_4<<<numBlocks, blockSize,
  //                         2 * blockSize / 32 * sizeof(float)>>>(d_out, d_inp,
  //                         N,
  //                                                               C);

  // kernel 5
  constexpr int BLOCK_THREADS = 256;
  constexpr int PACKS_PER_THREAD = 4;

  if (C != 4096) {
    printf("This v5 kernel only supports C=4096\n");
    return 1;
  }

  int numBlocks = N;

  // benchmark
  size_t bytes_moved = 2ull * N * C * sizeof(float);
  // read inp + write out

  auto result = benchmark_cuda_kernel_batched(
      [&] {
        softmax_fwd_kernel_5_fp32_c4096<BLOCK_THREADS, PACKS_PER_THREAD>
            <<<N, BLOCK_THREADS>>>(d_out, d_inp, N);
      },
      20,   // warmup
      100,  // iters
      100,  // inner_repeats
      bytes_moved);

  CHECK_LAST_CUDA_ERROR();
  // Copy result back to host
  CHECK_CUDA(cudaMemcpy(out_gpu, d_out, numel * sizeof(float),
                        cudaMemcpyDeviceToHost));

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

  print_benchmark("softmax_v5", result);

  // Print performance comparition
  printf("CPU time: %f ms\n", time_cpu * 1000.0);
  printf("GPU time: %f ms\n", result.avg_ms);
  printf("Speedup: %fx\n", time_cpu / (result.avg_ms / 1000.0f));

  // cleanup
  cudaFree(d_out);
  cudaFree(d_inp);

  free(inp);
  free(out_cpu);
  free(out_gpu);

  return 0;
}
#endif