from pathlib import Path
import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime import driver
from jaxtyping import Float

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _torch_cuda_allocator(size: int, alignment: int, stream):
    return torch.empty((size,), device=DEVICE, dtype=torch.uint8)


triton.set_allocator(_torch_cuda_allocator)

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
_kernels = {}


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _triton_linear_forward_kernel(
    x_ptr,  # input X
    w_ptr,  # input W
    y_ptr,  # output Y
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    X: shape [M, K]
    W: shape [K, N]
    Y: shape [M, N]
    Every program handles output the block [M, N] in Y
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[M, K],
        strides=(stride_xm, stride_xk),
        block_shape=[BLOCK_M, BLOCK_K],
    )

    w_desc = tl.make_tensor_descriptor(
        w_ptr,
        shape=[K, N],
        strides=(stride_wk, stride_wn),
        block_shape=[BLOCK_K, BLOCK_N],
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        x = x_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
        w = w_desc.load([k * BLOCK_K, pid_n * BLOCK_N])
        acc = tl.dot(x, w, acc=acc)

    y_desc = tl.make_tensor_descriptor(
        y_ptr,
        shape=[M, N],
        strides=[stride_ym, stride_yn],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    y_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc)


def _triton_linear_forward(
    x: Float[Tensor, "... n"],
    w: Float[Tensor, "k n"],
    out: Tensor | None = None,
) -> Float[Tensor, "... n"]:
    assert w.dim() == 2
    assert w.is_cuda and x.is_cuda
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert x.dtype == w.dtype
    assert x.size(-1) == w.size(0)

    x_prefix_shape = tuple(x.shape[:-1])
    x = x.contiguous().view(-1, x.size(-1))

    M, K = x.shape
    K, N = w.shape

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    if out is None:
        y = torch.empty((x.size(0), w.size(1)), dtype=x.dtype, device=x.device)
    else:
        y = out.view(-1, N)
        assert y.shape == (M, N)
        assert y.is_cuda and y.dtype == x.dtype

    _triton_linear_forward_kernel[grid](
        x,
        w,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
    )

    return y.view(*x_prefix_shape, N)


class TritonLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: Float[Tensor, "... k"],  # noqa: F722
        w: Float[Tensor, "k n"],  # noqa: F722
        *args,
        **kwargs,
    ):
        y = _triton_linear_forward(x, w)
        ctx.save_for_backward(x, w)
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: Float[Tensor, "..."],
    ):
        pass


PROBLEM_SIZES = [
    ("256x256x256", 256, 256, 256),
    ("512x512x512", 512, 512, 512),
    ("1024x1024x1024", 1024, 1024, 1024),
    ("2048x1024x1024", 2048, 1024, 1024),
    ("4096x1024x1024", 4096, 1024, 1024),
    ("4096x2048x1024", 4096, 2048, 1024),
    ("4096x4096x4096", 4096, 4096, 4096),
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["matrix_size", "M", "K", "N"],
        x_vals=PROBLEM_SIZES,
        line_arg="provider",
        line_vals=[
            "triton",
            "torch",
        ],
        line_names=[
            "Triton",
            "Torch",
        ],
        styles=[("blue", "-"), ("green", "-")],
        xlabel="X[M,K] @ W[K,N]",
        ylabel="TFLOPS",
        plot_name="linear-performance",
        args={
            "dtype": torch.float16,
        },
    )
)
def benchmark(matrix_size, M, K, N, dtype, provider):
    x = torch.randn(M, K, device=DEVICE, dtype=dtype)
    w = torch.randn(K, N, device=DEVICE, dtype=dtype)
    y = torch.empty((M, N), device=DEVICE, dtype=dtype)

    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == "triton":
        _triton_linear_forward(x, w, y)
        ms = triton.testing.do_bench(lambda: _triton_linear_forward(x, w, y))
    else:
        torch.mm(x, w, out=y)
        ms = triton.testing.do_bench(lambda: torch.mm(x, w, out=y))

    return 2 * M * N * K * 1e-9 / ms


def _save_labeled_plot(df, save_path: Path):
    import matplotlib.pyplot as plt

    x_vals = list(range(len(df)))
    labels = df["matrix_size"].tolist()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        x_vals,
        df["Triton (TFLOPS)"],
        label="Triton",
        color="blue",
        linestyle="-",
        marker="o",
    )
    ax.plot(
        x_vals,
        df["Torch (TFLOPS)"],
        label="Torch",
        color="green",
        linestyle="-",
        marker="o",
    )

    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("X[M,K] @ W[K,N]")
    ax.set_ylabel("TFLOPS")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path / "linear-performance.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent / "linear"
    script_dir.mkdir(exist_ok=True)
    df = benchmark.run(
        show_plots=False, print_data=True, save_path=script_dir, return_df=True
    )
    _save_labeled_plot(df, script_dir)
