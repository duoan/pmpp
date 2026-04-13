#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

#include "runner.cuh"
#include "kernels.cuh"

float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float& beg, float& end) { return 1.0e-6 * (end - beg); }

void cudaCheck(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

void cublasCheck(cublasStatus_t status, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[cuBLAS ERROR] at file %s:%d: status code %d\n", file, line,
           status);
    exit(EXIT_FAILURE);
  }
}

void CudaDeviceInfo() {
  int deviceId;
  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  int clockRateKHz = 0, memClockRateKHz = 0;
  cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, deviceId);
  cudaDeviceGetAttribute(&memClockRateKHz, cudaDevAttrMemoryClockRate,
                         deviceId);

  // FP32 cores per SM varies by arch; 128 is typical for Ampere/Blackwell
  double fp32_tflops =
      (double)props.multiProcessorCount * 128 * 2 * clockRateKHz * 1e-9;
  // memClockRate (KHz) * busWidth (bits) * 2 (DDR) / 8 (bits->bytes)
  double mem_bw_gb =
      (double)memClockRateKHz * 1e-6 * (props.memoryBusWidth / 8) * 2;

  printf(
      "Device ID: %d\n"
      "  Name:                          %s\n"
      "  Compute Capability:            %d.%d\n"
      "  Warp Size:                     %d\n"
      "  SM Count:                      %d\n"
      "  Clock Rate:                    %.0f MHz\n"
      "  FP32 Peak (est.):              %.1f TFLOPS\n"
      "\n"
      "  Global Memory:                 %zu MB\n"
      "  Memory Bus Width:              %d-bit\n"
      "  Memory Clock:                  %.0f MHz\n"
      "  Memory Bandwidth (est.):       %.1f GB/s\n"
      "  L2 Cache Size:                 %d KB\n"
      "\n"
      "  Shared Memory / Block:         %zu KB\n"
      "  Shared Memory / SM:            %zu KB\n"
      "  Registers / Block:             %d\n"
      "  Registers / SM:                %d\n"
      "\n"
      "  Max Threads / Block:           %d\n"
      "  Max Threads / SM:              %d\n"
      "  Max Blocks / SM:               %d\n"
      "  Const Memory:                  %zu KB\n",
      deviceId, props.name, props.major, props.minor, props.warpSize,
      props.multiProcessorCount, clockRateKHz / 1000.0, fp32_tflops,
      props.totalGlobalMem / 1024 / 1024, props.memoryBusWidth,
      memClockRateKHz / 1000.0, mem_bw_gb, props.l2CacheSize / 1024,
      props.sharedMemPerBlock / 1024, props.sharedMemPerMultiprocessor / 1024,
      props.regsPerBlock, props.regsPerMultiprocessor, props.maxThreadsPerBlock,
      props.maxThreadsPerMultiProcessor, props.maxBlocksPerMultiProcessor,
      props.totalConstMem / 1024);
};

void randomize_matrix(float* mat, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time{};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

void range_init_matrix(float* mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = i;
  }
}

void zero_init_matrix(float* mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0.0;
  }
}

void copy_matrix(const float* src, float* dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++) {
    *(dest + i) = *(src + i);
  }
  if (i != N) {
    printf("copy failed at %d while there are %d elements in total.\n", i, N);
  }
}

void print_matrix(const float* A, int M, int N, std::ofstream& fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed;  // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0) {
      fs << std::setw(5) << A[i];  // Set field width and write the value
    } else {
      fs << std::setw(5) << A[i] << ", ";
    }
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N) {
        fs << ";\n";
      }
    }
  }
  fs << "]\n";
}

bool verify_matrix(float* matRef, float* matOut, int N) {
  const double ABS_TOL = 0.05;
  const double REL_TOL = 1e-3;
  for (int i = 0; i < N; i++) {
    double diff = std::fabs(matRef[i] - matOut[i]);
    double absmax = std::fmax(std::fabs(matRef[i]), std::fabs(matOut[i]));
    if (isnan(diff) || (diff > ABS_TOL && diff > absmax * REL_TOL)) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}

void run_sgemm_naive(int M, int N, int K, float alpha, float* A, float* B,
                     float beta, float* C) {
  dim3 blockDim(32, 32);
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_KERNEL_CHECK();
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float* A,
                float* B, float beta, float* C, cublasHandle_t handle) {
  switch (kernel_num) {
    case 0:
      run_sgemm_cublas(M, N, K, alpha, A, B, beta, C, handle);
      break;
    case 1:
      run_sgemm_naive(M, N, K, alpha, A, B, beta, C);
      break;
    default:
      throw std::invalid_argument("Unknown kernel number");
  }
}
