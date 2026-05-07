"""Triton 版 softmax 教程, 带零基础中文注释。

这个文件做的事情只有一个: 对二维矩阵 x 的每一行做 softmax。

假设 x.shape = (n_rows, n_cols), 可以把它想成一张表:

    columns:   0      1      2      3      ...   n_cols-1
    row 0   [ 1.0,   2.0,   3.0,   4.0,   ... ]
    row 1   [ 0.2,  -1.0,   5.0,   0.7,   ... ]
    row 2   [ ...                                  ]

softmax 是"按行"做的, 每一行互不影响:

    softmax(row)[j] = exp(row[j] - max(row)) / sum(exp(row[k] - max(row)))

为什么要减 max(row)?
    这是数值稳定技巧。exp(很大的数)容易溢出。减去最大值后,
    每个指数的输入都 <= 0, 结果更安全。

Triton kernel 的核心思想:

    1 个 Triton program 负责处理 1 行, 或者多行中的某一行。
    每个 program 一次性把这一整行加载成一个向量, 在向量上做:

        load row -> subtract max -> exp -> sum -> divide -> store row

一个 program 内部看到的列向量:

    col_offsets = [0, 1, 2, 3, ..., BLOCK_SIZE-1]

如果 n_cols = 5, BLOCK_SIZE = 8:

    有效列:       0    1    2    3    4
    padding列:                         5    6    7
    mask:       True True True True True False False False

mask 的作用是: BLOCK_SIZE 常常比真实列数大, 不能读写越界。
"""

import torch
import triton
import triton.language as tl
from pathlib import Path
from torch import Tensor
from torch.utils.cpp_extension import load
from triton.runtime import driver
from jaxtyping import Float

DEVICE = triton.runtime.driver.active.get_active_torch_device()
_cuda_softmax_module = None


def _load_cuda_softmax_module():
    """懒加载 softmax.cu 编译出来的 PyTorch extension。"""
    global _cuda_softmax_module
    if _cuda_softmax_module is None:
        source_dir = Path(__file__).resolve().parent
        _cuda_softmax_module = load(
            name="pmpp_cuda_softmax_ext",
            sources=[
                str(source_dir / "softmax_extension.cpp"),
                str(source_dir / "softmax.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-DBUILD_PYTORCH_EXTENSION",
            ],
            verbose=False,
        )
    return _cuda_softmax_module


def softmax_cuda(x: Float[Tensor, "b s"]) -> Float[Tensor, "b s"]:  # noqa: F722
    """调用 softmax.cu 中的 CUDA kernel, 通过 PyTorch extension 暴露。"""
    return _load_cuda_softmax_module().softmax_cuda(x)


def naive_softmax(x: Float[Tensor, "b s"]) -> Float[Tensor, "b s"]:  # noqa: F722
    """用普通 PyTorch 写一个容易理解但不一定最快的 softmax。

    这个函数不是 Triton kernel, 只是用来:
    1. 当作正确答案参考。
    2. 和 Triton 版本做性能对比。

    dim=-1 表示沿着最后一个维度做操作。对二维矩阵来说, 就是"每一行"。
    """
    # 每一行取最大值。
    #
    # x.max(dim=-1) 会返回两个东西:
    #   1. values: 每行最大值
    #   2. indices: 最大值所在列
    #
    # 这里只需要 values, 所以取 [0]。
    # 如果 x.shape = (3, 4), 那么 x_max.shape = (3,)。
    x_max = x.max(dim=-1)[0]

    # x_max[:, None] 把 shape 从 (3,) 变成 (3, 1)。
    #
    # 这样 PyTorch 才能把每行自己的最大值广播到整行:
    #
    #   x row:       [1, 2, 3, 4]
    #   row max:      4
    #   x - max:    [-3, -2, -1, 0]
    z = x - x_max[:, None]

    # 对每个元素做 exp。
    numerator = torch.exp(z)

    # 每一行把 exp 后的结果加起来, 得到分母。
    # denominator.shape = (n_rows,)
    denominator = numerator.sum(dim=-1)

    # denominator[:, None] 再次把 (n_rows,) 变成 (n_rows, 1),
    # 这样每一行的所有列都除以这一行自己的分母。
    ret = numerator / denominator[:, None]
    return ret


@triton.jit
def softmax_fwd_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Triton GPU kernel: 按行计算 softmax。

    这段代码运行在 GPU 上, 不是普通 Python 循环。

    关键概念:

    1. program
       Triton 里的 program 可以粗略理解成"一个小任务"。
       这里我们让很多 program 并行工作, 每个 program 处理若干行。

    2. tl.program_id(0)
       当前 program 在第 0 个 grid 维度上的编号。
       如果 launch 了 8 个 program, 它们的 id 是 0,1,2,...,7。

    3. pointer + stride
       input_ptr 是矩阵第一个元素的地址。
       input_row_stride 表示"走到下一行需要跨过多少个元素"。

       对连续的二维矩阵:

           x.shape = (n_rows, n_cols)
           x.stride(0) = n_cols

       第 row_idx 行起点:

           input_ptr + row_idx * input_row_stride

    4. BLOCK_SIZE
       Triton 向量长度必须是编译期常量, 通常取 n_cols 的 2 的幂上取整。
       例如 n_cols=781, BLOCK_SIZE=1024。
       多出来的 243 个位置用 mask 屏蔽掉。
    """
    # 当前 program 负责的第一行。
    #
    # 例子:
    #   如果一共 launch 4 个 program:
    #     program 0: row_start = 0
    #     program 1: row_start = 1
    #     program 2: row_start = 2
    #     program 3: row_start = 3
    row_start = tl.program_id(0)

    # 总共有多少个 program 在第 0 维并行。
    #
    # 如果 row_step=4, 那每个 program 会隔 4 行处理:
    #
    #   program 0: row 0, 4,  8, 12, ...
    #   program 1: row 1, 5,  9, 13, ...
    #   program 2: row 2, 6, 10, 14, ...
    #   program 3: row 3, 7, 11, 15, ...
    #
    # 这样做是为了当 n_rows 比 program 数量多时, 每个 program 可以循环处理多行。
    row_step = tl.num_programs(0)

    # tl.range 是 Triton 的循环。它会被 Triton 编译器优化。
    # row_idx 依次取 row_start, row_start + row_step, ...
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # 找到当前行的第一个元素地址。
        #
        # 如果 input_row_stride = n_cols = 5:
        #
        #   row 0 起点: input_ptr + 0 * 5
        #   row 1 起点: input_ptr + 1 * 5
        #   row 2 起点: input_ptr + 2 * 5
        row_start_ptr = input_ptr + row_idx * input_row_stride

        # 生成列偏移向量。
        #
        #   BLOCK_SIZE = 8
        #   col_offsets = [0, 1, 2, 3, 4, 5, 6, 7]
        #
        # 注意: 这是 GPU 上的向量, 不是 Python list。
        col_offsets = tl.arange(0, BLOCK_SIZE)

        # 当前行里每一列的地址。
        #
        #   input_ptrs = row_start_ptr + [0,1,2,3,...]
        #
        # 也就是:
        #
        #   [
        #     &x[row_idx, 0],
        #     &x[row_idx, 1],
        #     &x[row_idx, 2],
        #     ...
        #   ]
        input_ptrs = row_start_ptr + col_offsets

        # mask 标记哪些列是真实存在的。
        #
        # 例子: n_cols=5, BLOCK_SIZE=8
        #
        #   col_offsets = [0, 1, 2, 3, 4, 5, 6, 7]
        #   mask        = [T, T, T, T, T, F, F, F]
        #
        # False 的地方既不能读, 也不能写。
        mask = col_offsets < n_cols

        # 从 GPU 全局内存读取一整行。
        #
        # mask=True  的位置正常读取。
        # mask=False 的位置填 other=-inf。
        #
        # 为什么填 -inf?
        #   因为后面要做 max 和 exp。
        #   max(..., -inf) 不会影响真实最大值。
        #   exp(-inf) = 0, 不会影响 sum。
        #
        # 例子:
        #   真实行: [1, 2, 3, 4, 5]
        #   读取成: [1, 2, 3, 4, 5, -inf, -inf, -inf]
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

        # 先减去这一行的最大值, 做数值稳定。
        #
        # axis=0 的意思:
        #   row 是一个 BLOCK_SIZE 长度的一维向量,
        #   在这个向量上做归约(reduction)。
        #
        # 例子:
        #   row = [1, 2, 3, 4]
        #   max = 4
        #   row_minus_max = [-3, -2, -1, 0]
        row_minus_max = row - tl.max(row, axis=0)

        # 对每个元素取指数。tl.exp 是 GPU 上的向量化 exp。
        numerator = tl.exp(row_minus_max)

        # 把这一行所有 exp 后的值加起来, 得到 softmax 分母。
        denominator = tl.sum(numerator, axis=0)

        # 每个元素除以同一个 denominator。
        #
        # 得到的一整行满足:
        #   1. 每个值都在 0 到 1 之间
        #   2. 这一行所有值加起来约等于 1
        softmax_output = numerator / denominator

        # 输出矩阵当前行的起始地址。
        output_row_start_ptr = output_ptr + row_idx * output_row_stride

        # 输出矩阵当前行每一列的地址。
        output_ptrs = output_row_start_ptr + col_offsets

        # 把结果写回 output。
        # 仍然要用 mask, 避免写到 padding 列对应的越界地址。
        tl.store(output_ptrs, softmax_output, mask=mask)


# 下面这些属性来自当前 GPU, 用来估算应该同时启动多少个 program。
#
# SM 可以粗略理解成 GPU 上的"计算核心组"。NVIDIA 里叫 Streaming Multiprocessor。
# 一个 GPU 有很多 SM, 每个 SM 可以同时跑多个 program/warp。
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax_triton(x: Float[Tensor, "b s"]) -> Float[Tensor, "b s"]:  # noqa: F722
    """Python 端包装函数: 准备输出、选择配置、启动 Triton kernel。

    这个函数运行在 CPU/Python 端, 负责"调度" GPU kernel。
    真正的 softmax 计算发生在 softmax_fwd_kernel 里面。

    调用关系:

        user/Python
            |
            v
        softmax_triton(x)
            |
            |-- 计算 BLOCK_SIZE / num_warps / num_stages
            |-- 首次遇到某个配置时 warmup, 拿到资源占用信息并缓存
            |-- 根据 GPU 资源估算 num_programs
            |
            v
        softmax_fwd_kernel[...](
            y, x, strides, n_rows, n_cols, ...
        )
            |
            v
        返回 y
    """
    assert x.dim() == 2
    x = x.contiguous()

    # 只支持二维矩阵: n_rows 行, n_cols 列。
    n_rows, n_cols = x.shape

    # Triton block 的大小通常取 2 的幂。
    #
    # 为什么不是直接用 n_cols?
    #   Triton 编译器对 2 的幂长度的向量操作更友好。
    #
    # 例子:
    #   n_cols = 781
    #   BLOCK_SIZE = 1024
    #
    # 多出来的元素靠 mask 屏蔽。
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # 一个 program 里使用多少个 warp。
    #
    # warp 是 GPU 线程调度的基本单位。NVIDIA 上通常 1 warp = 32 个线程。
    #
    # RTX PRO 6000 Blackwell Server Edition 上实测:
    #   - BLOCK_SIZE <= 2048 时, 4 warps 通常更快。
    #   - 更大的行归约里, 8 warps 和 16 warps 很接近, 8 warps 寄存器压力更低。
    num_warps = 4 if BLOCK_SIZE <= 2048 else 8

    # num_stages 影响流水线深度和 shared memory 压力。
    #
    # 这张 Blackwell 卡的 Triton 可用 shared memory 是 101376 bytes。
    # BLOCK_SIZE >= 8192 时继续用 2/4 stages 容易在 16K block 上超 shared memory,
    # 所以大行归约用 1 stage; 小行归约适当增加流水线深度。
    if BLOCK_SIZE >= 8192:
        num_stages = 1
    elif BLOCK_SIZE >= 2048:
        num_stages = 2
    elif BLOCK_SIZE >= 512:
        num_stages = 3
    else:
        num_stages = 2

    # 创建输出张量, shape/dtype/device 都和 x 一样。
    y = torch.empty_like(x)

    # warmup 只需要对同一种编译配置做一次。
    #
    # n_rows / n_cols 是运行时参数; 真正影响这个 kernel 编译结果的主要是
    # dtype、设备、BLOCK_SIZE、num_warps 和 num_stages。
    cache_key = (
        target.backend,
        target.arch,
        x.device.type,
        x.device.index,
        x.dtype,
        BLOCK_SIZE,
        num_warps,
        num_stages,
    )

    if cache_key not in kernels:
        # warmup 会让 Triton 先编译 kernel, 但这里不是真的用最终 grid 跑完整任务。
        #
        # 为什么要 warmup?
        #   因为编译后才能知道这个 kernel 会占多少寄存器、共享内存。
        #   这些信息会影响后面 num_programs 的估算。
        kernel = softmax_fwd_kernel.warmup(
            y,
            x,
            x.stride(0),
            y.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
            grid=(1,),
        )

        # 初始化底层 kernel 句柄, 之后才能读到资源信息。
        kernel._init_handles()

        # 这个 kernel 每个线程大概需要多少寄存器。
        n_regs = kernel.n_regs

        # 这个 kernel 每个 program 需要多少 shared memory。
        size_smem = kernel.metadata.shared

        # occupancy 粗略表示一个 SM 上能同时放多少个这样的 program。
        #
        # 第一项受寄存器限制:
        #
        #   NUM_REGS / (每个线程寄存器数 * 每个 warp 线程数 * warp 数)
        #
        # 第二项受 shared memory 限制:
        #
        #   SIZE_SMEM / 每个 program 需要的 shared memory
        #
        # 两个限制里取更小的那个。
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)

        # 总 program 数 = 每个 SM 能放的 program 数 * SM 数。
        kernels[cache_key] = (kernel, NUM_SM * occupancy)

    kernel, max_num_programs = kernels[cache_key]

    # 实际 launch 的 program 数不能超过行数, 因为最多也就 n_rows 行需要处理。
    #
    # 对 2K+ 列的行归约, 这张 Blackwell 上实测一行一个 program 比 persistent
    # program 循环多行略快; 小列数保留 occupancy 估算, 避免启动过多很轻的 program。
    if BLOCK_SIZE >= 2048:
        num_programs = n_rows
    else:
        num_programs = min(max_num_programs, n_rows)

    # 正式启动 GPU kernel。
    #
    # kernel[num_programs, 1, 1] 表示 launch grid:
    #
    #   grid x 维 = num_programs
    #   grid y 维 = 1
    #   grid z 维 = 1
    #
    # 每个 program 会在 softmax_fwd_kernel 里通过 tl.program_id(0)
    # 拿到自己的编号。
    kernel[num_programs, 1, 1](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_stages,
    )

    return y


def test_softmax():
    """正确性测试: Triton 输出应该接近 torch.softmax 输出。"""
    torch.manual_seed(0)

    # 故意选一个不整齐的列数 781。
    # 这样 BLOCK_SIZE 会变成 1024, 可以测试 mask 是否正确处理越界 padding。
    x = torch.randn(1823, 781, device=DEVICE)
    y_triton = softmax_triton(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # 图的 x 轴变量名, 这里 N 表示列数 n_cols。
        x_vals=[128 * i for i in range(2, 100)],  # 依次测试不同列数。
        line_arg="provider",  # 不同 provider 画成不同曲线。
        line_vals=["triton", "torch", "cuda", "naive_softmax"],  # 要比较的实现。
        line_names=["Triton", "Torch", "CUDA C++", "Naive Softmax"],  # 图例显示名称。
        styles=[("blue", "-"), ("green", "-"), ("orange", "-"), ("red", "-")],  # 曲线样式。
        ylabel="GB/s",  # y 轴: 每秒处理多少 GB 数据。
        plot_name="softmax-performance",  # 输出图片/报告名称。
        args={"M": 4096},  # 固定行数 M=4096, 只变化列数 N。
    )
)
def benchmark(M, N, provider):
    """性能测试函数。

    M: 行数
    N: 列数
    provider: 用哪个实现来跑, 可以是 triton/torch/cuda/naive_softmax

    为什么返回 GB/s?
        softmax 大体上要读一次 x, 写一次 y。
        所以数据量近似是:

            2 * x.numel() * x.element_size()

        再除以运行时间, 得到吞吐量。
    """
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    # 创建并切换到一个新的 GPU stream, 避免和其他默认 stream 操作混在一起。
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax_triton(x))
    if provider == "cuda":
        ms = triton.testing.do_bench(lambda: softmax_cuda(x))
    if provider == "naive_softmax":
        ms = triton.testing.do_bench(lambda: naive_softmax(x))

    # 把毫秒转换成 GB/s:
    #
    #   bytes = 2 * 元素数量 * 每个元素字节数
    #   GB = bytes * 1e-9
    #   seconds = ms * 1e-3
    def gbps(ms):
        return 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    # 直接运行这个文件时, 会跑 benchmark 并展示性能图。
    script_dir = Path(__file__).resolve().parent
    benchmark.run(show_plots=True, print_data=True, save_path=script_dir)
