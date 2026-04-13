Follow this blog to reproduce the high performance matmul
https://siboehm.com/articles/22/CUDA-MMM

---
## Naive
(pmpp) ➜  cuda-sgemm git:(main) ✗ ./sgemm 1
Device ID: 0
  Name:                          NVIDIA RTX PRO 6000 Blackwell Server Edition
  Compute Capability:            12.0
  Warp Size:                     32
  SM Count:                      188
  Clock Rate:                    2430 MHz
  FP32 Peak (est.):              117.0 TFLOPS

  Global Memory:                 97249 MB
  Memory Bus Width:              512-bit
  Memory Clock:                  12481 MHz
  Memory Bandwidth (est.):       1597.6 GB/s
  L2 Cache Size:                 131072 KB

  Shared Memory / Block:         48 KB
  Shared Memory / SM:            100 KB
  Registers / Block:             65536
  Registers / SM:                65536

  Max Threads / Block:           1024
  Max Threads / SM:              1536
  Max Blocks / SM:               24
  Const Memory:                  64 KB

Matrix dimensions: M=4092, N=4092, K=4092
Kernel: 1

Running cuBLAS reference...
Kernel 1: PASSED correctness check.

Average time: 55.514 ms  |  Performance: 2468.5 GFLOPS

----

## global memory coalesce

➜  cuda-sgemm git:(main) ✗ ./sgemm 2  
Device ID: 0
  Name:                          NVIDIA RTX PRO 6000 Blackwell Server Edition
  Compute Capability:            12.0
  Warp Size:                     32
  SM Count:                      188
  Clock Rate:                    2430 MHz
  FP32 Peak (est.):              117.0 TFLOPS

  Global Memory:                 97249 MB
  Memory Bus Width:              512-bit
  Memory Clock:                  12481 MHz
  Memory Bandwidth (est.):       1597.6 GB/s
  L2 Cache Size:                 131072 KB

  Shared Memory / Block:         48 KB
  Shared Memory / SM:            100 KB
  Registers / Block:             65536
  Registers / SM:                65536

  Max Threads / Block:           1024
  Max Threads / SM:              1536
  Max Blocks / SM:               24
  Const Memory:                  64 KB

Matrix dimensions: M=4092, N=4092, K=4092
Kernel: 2

Running cuBLAS reference...
Kernel 2: PASSED correctness check.

Average time: 28.978 ms  |  Performance: 4729.1 GFLOPS

-----
