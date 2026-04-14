#pragma once

#include "kernels/00_cublas.cuh"
#include "kernels/01_naive.cuh"
#include "kernels/02_global_mem_coalesce.cuh"
#include "kernels/03_shared_mem_blocking.cuh"
#include "kernels/04_shared_mem_1d_blocktiling.cuh"
#include "kernels/05_shared_mem_2d_blocktiling.cuh"