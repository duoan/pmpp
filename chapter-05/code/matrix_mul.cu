#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>


#define TILE_WIDTH 32

// =====================================================================
// Tiled Matrix Multiplication Kernel: C = A * B   (square: width x width)
// =====================================================================
//
// ---- CUDA vocabulary used in this file (how they relate) ----------------
//
//   SM (Streaming Multiprocessor)
//     One of many identical "compute clusters" on the GPU chip. The hardware
//     schedules whole *blocks* onto SMs; many blocks from the grid run at
//     once, spread across all SMs, until the kernel finishes.
//
//   Block (thread block)
//     A fixed-size team of threads (here: TILE_WIDTH x TILE_WIDTH = 1024).
//     One block always runs on a single SM (not split across SMs). Threads
//     in the same block can use __shared__ memory and __syncthreads().
//
//   TILE_WIDTH (= block width/height in threads)
//     Side length of the square tile we stage in shared memory. We pick 32 so
//     that one *row* of the block (ty fixed, tx = 0..31) is exactly one
//     *warp* (see below), which makes global-memory loads coalesced.
//
//   Warp
//     The smallest hardware scheduling unit: 32 threads that execute the
//     same instruction together (SIMT). In this kernel, for a fixed ty,
//     threads (tx=0..31) form one warp; their A and B loads are contiguous
//     in memory -> few, wide transactions instead of many scattered ones.
//
// ---- Why this tiled kernel is much faster than a naive one --------------
//
//   Naive matmul re-reads A and B from global memory O(width) times per
//   output element -> huge traffic and latency (~hundreds of cycles per
//   global access).
//
//   Here, each phase loads a 32x32 chunk of A and B into *shared* memory once,
//   then all 1024 threads reuse those values for 32 multiply-adds each.
//   That cuts global-memory traffic by ~O(TILE_WIDTH) and shifts most work
//   to fast shared memory / registers — that is where the speedup comes from,
//   together with coalesced warps and many blocks keeping SMs busy.
//
// __restrict__ tells the compiler A, B, C never alias, enabling aggressive optimizations.
//
// -------------------- Memory Hierarchy (latency) --------------------
//
//   Registers (Cvalue)          ~1 cycle    <- fastest, per-thread private
//       |
//   Shared Memory (s_A, s_B)    ~5 cycles   <- shared within a block
//       |
//   Global Memory (A, B, C)   ~400 cycles   <- slowest, visible to all threads
//
// Core idea: move data from Global -> Shared once per phase, then reuse it many times.
//
__global__ void matmul_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    int width
) {
    // Each block allocates two shared memory tiles for caching A and B
    //
    // Resource limits (both affect whether the kernel can launch and how fast it runs):
    //
    //   Shared memory per block — hardware caps how much __shared__ one block may use
    //   (often 48 KiB default per SM partition; some GPUs allow 96 KiB with a config
    //   trade-off vs L1). Here: 2 * TILE_WIDTH^2 * sizeof(float) = 8 KiB for TILE_WIDTH=32,
    //   so we are safely below typical limits. If TILE_WIDTH were much larger, you could
    //   exceed the per-block limit and the launch would fail.
    //
    //   Registers per thread — the compiler allocates registers for Cvalue, indices, etc.
    //   Each SM has a fixed register file; if each thread uses many registers, fewer
    //   threads / blocks can run concurrently on that SM (lower occupancy), which can
    //   hurt performance by reducing latency hiding. nvcc --ptxas-options=-v shows usage.
    //
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;   // block's x index in the grid (column direction)
    int by = blockIdx.y;   // block's y index in the grid (row direction)
    int tx = threadIdx.x;  // thread's x index within the block
    int ty = threadIdx.y;  // thread's y index within the block

    // -------------------- Thread -> Matrix Element Mapping --------------------
    //
    //  Col = bx * TILE_WIDTH + tx    (global column index)
    //  Row = by * TILE_WIDTH + ty    (global row index)
    //
    //  Example: by=1, ty=5, TILE_WIDTH=32 -> Row = 32+5 = 37
    //           bx=2, tx=3, TILE_WIDTH=32 -> Col = 64+3 = 67
    //           This thread computes C[37][67]
    //
    //          Col=67
    //            |
    //  C: +------------------+
    //     |                  |
    //     |     block(2,1)   |
    //  37-|  .....[C37,67]   |  <- thread(3,5) owns this element
    //     |                  |
    //     +------------------+
    //
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Accumulate dot-product in a register (fastest storage level)
    float Cvalue = 0.0f;

    // -------------------- Tiled Computation (phased) --------------------
    //
    // C[Row][Col] = sum_k( A[Row][k] * B[k][Col] ),  k = 0..width-1
    //
    // Split the k dimension into width/TILE_WIDTH phases, each covering 32 k-values:
    //
    //  Matrix A (width=1024):               Matrix B (width=1024):
    //                                                Col
    //                                                 |
    //        +----+----+----+---+----+       +--------+----+--------+
    //        |    |    |    |   |    |       |        |    |        |
    //  Row-->|ph=0|ph=1|ph=2|...|ph31|       |  ph=0  |XXXX|        |  32 rows
    //        | 32 | 32 | 32 |   | 32 |       +--------+----+--------+
    //        |cols|cols|cols|   |cols|       |  ph=1  |XXXX|        |  32 rows
    //        +----+----+----+---+----+       +--------+----+--------+
    //        |____ width = 1024 _____|       |  ph=2  |XXXX|        |  32 rows
    //                                        +--------+----+--------+
    //  Each phase: 1024 threads              |  ...   | .. |        |
    //  cooperatively load one 32x32          +--------+----+--------+
    //  tile from A and one 32x32
    //  tile from B into shared memory.
    //
    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {

        // ============ Step 1: Cooperative tile loading into shared memory ============
        //
        // Each thread loads 1 element; 32x32 = 1024 threads fill one tile exactly.
        //
        // ---- Loading s_A: row "Row" of A, phase "ph" ----
        //
        //  A in row-major 1D memory:
        //
        //  addr:  Row*W+ph*32+0  Row*W+ph*32+1  ...  Row*W+ph*32+31
        //              |               |                    |
        //            tx=0            tx=1                 tx=31
        //
        //  Warp = 32 threads with same ty, tx=0..31
        //  -> they read contiguous addresses -> coalesced -> single 128B transaction
        //
        s_A[ty][tx] = A[Row * width + ph * TILE_WIDTH + tx];

        // ---- Loading s_B: column "Col" of B, phase "ph" ----
        //
        //  B[(ph*32 + ty) * width + Col],  where Col = bx*32 + tx
        //
        //  Memory access pattern (one warp, ty fixed):
        //
        //  addr:  (ph*32+ty)*W+bx*32+0   (ph*32+ty)*W+bx*32+1 ... (ph*32+ty)*W+bx*32+31
        //              |                        |                        |
        //            tx=0                     tx=1                     tx=31
        //         <------------- contiguous, one memory transaction ------------->
        //
        s_B[ty][tx] = B[(ph * TILE_WIDTH + ty) * width + Col];

        // Barrier: wait until all 1024 threads finish loading
        __syncthreads();

        // ============ Step 2: Partial dot product using shared memory ============
        //
        //  Cvalue += s_A[ty][0]*s_B[0][tx]
        //          + s_A[ty][1]*s_B[1][tx]
        //          + ...
        //          + s_A[ty][31]*s_B[31][tx]
        //
        //  From thread(tx, ty)'s perspective:
        //
        //       s_A                  s_B               result
        //    col 0 ... 31         col tx
        //   +------------+       +------+
        //   |            |  row 0|  *   |
        //   |            |  row 1|  *   |         C[Row][Col]
        //   | row ty --> |   ... |  *   |   ==>      +=
        //   | * * * * *  |  row31|  *   |       (partial sum)
        //   +------------+       +------+
        //    (read one row)   (read one col)
        //
        //  All 32 multiply-adds hit shared memory (~5 cycles), zero global access
        //
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += s_A[ty][k] * s_B[k][tx];
        }

        // Barrier: all threads done computing before next phase overwrites shared memory
        __syncthreads();
    }

    // Bounds check, then write accumulated result back to global memory
    if (Row < width && Col < width) {
        C[Row * width + Col] = Cvalue;
    }

}

int main() {
    int width = 1024; // Matrix width (Assuming square matrices)
    size_t size = width * width * sizeof(float);

    // Allocate and initialize host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < width * width; ++i) {
        h_A[i] = static_cast<float>(rand() % 100);
        h_B[i] = static_cast<float>(rand() % 100);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrix from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // =====================================================================
    //                          Launch Kernel
    // =====================================================================
    //
    // The grid has (width/TILE_WIDTH)^2 blocks. The GPU assigns blocks to SMs
    // dynamically: when a block finishes, that SM can take another block.
    // High occupancy (many warps per SM, many blocks in flight) hides memory
    // latency while others wait on global loads — another reason this stays fast
    // at scale (together with tiling + coalescing inside each block).
    //
    // CUDA Events for precise GPU-side timing (hardware timer, more accurate than CPU clock)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -------------------- Grid / Block Layout --------------------
    //
    // dimBlock = (32, 32) -> each block has 32x32 = 1024 threads (hardware max)
    //
    //  Inside one Block (32x32 threads):
    //
    //        tx=0    tx=1    tx=2    ...    tx=31
    //       +-------+-------+-------+---+---------+
    //  ty=0 | (0,0) | (1,0) | (2,0) |...| (31,0)  |  <- warp 0
    //       +-------+-------+-------+---+---------+
    //  ty=1 | (0,1) | (1,1) | (2,1) |...| (31,1)  |  <- warp 1
    //       +-------+-------+-------+---+---------+
    //  ty=2 | (0,2) | (1,2) | (2,2) |...| (31,2)  |  <- warp 2
    //       +-------+-------+-------+---+---------+
    //   ... |  ...  |  ...  |  ...  |...|  ...    |
    //       +-------+-------+-------+---+---------+
    // ty=31 |(0,31) |(1,31) |(2,31) |...| (31,31) |  <- warp 31
    //       +-------+-------+-------+---+---------+
    //
    //  32 warps x 32 threads/warp = 1024 threads total
    //
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    // dimGrid = (32, 32) -> 32x32 = 1024 blocks total
    //
    //  Grid layout (32x32 blocks -> covers the 1024x1024 output matrix C):
    //
    //          bx=0      bx=1      bx=2            bx=31
    //       +---------+---------+---------+-----+---------+
    // by=0  | blk(0,0)| blk(1,0)| blk(2,0)| ... |blk(31,0)|
    //       |  32x32  |  32x32  |  32x32  |     |  32x32  |
    //       +---------+---------+---------+-----+---------+
    // by=1  | blk(0,1)| blk(1,1)| blk(2,1)| ... |blk(31,1)|
    //       |  32x32  |  32x32  |  32x32  |     |  32x32  |
    //       +---------+---------+---------+-----+---------+
    //  ...  |   ...   |   ...   |   ...   | ... |   ...   |
    //       +---------+---------+---------+-----+---------+
    // by=31 |blk(0,31)|blk(1,31)|blk(2,31)| ... |blk(31,31)|
    //       +---------+---------+---------+-----+---------+
    //
    //  Each block's 1024 threads compute a 32x32 sub-block of output matrix C
    //  Total: 1024 blocks x 1024 threads/block = 1,048,576 threads (= 1024x1024)
    //
    dim3 dimGrid(width / TILE_WIDTH, width / TILE_WIDTH);

    cudaEventRecord(start);

    // <<<dimGrid, dimBlock>>> is CUDA's kernel launch syntax
    // CPU returns immediately (asynchronous), GPU executes in the background
    matmul_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);

    cudaEventRecord(stop);

    // Block CPU until GPU finishes the kernel
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms" << std::endl;

    // FLOPs for matrix multiply: 2*N^3 (N^3 multiplications + N^3 additions)
    double gflops = 2.0 * width * width * width / (ms * 1e6);
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify results with CPU multiplication for correctness
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0.0;
            for (int k = 0; k < width; ++k) {
                sum += h_A[i * width + k] * h_B[k * width + j];
            }
            // std::cout << i << " " << j <<  " " << h_C[i * width + j] << "  " << sum << std::endl;
            assert(fabs(h_C[i * width + j] - sum) < 1e-3);
        }
    }
    std::cout << "Matrix multiplication completed successfully." << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}



    

