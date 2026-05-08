from pathlib import Path

import torch
import triton
import triton.language as tl


DEVICE = torch.device("cuda")


def _torch_cuda_allocator(size: int, alignment: int, stream):
    return torch.empty((size,), device=DEVICE, dtype=torch.uint8)


triton.set_allocator(_torch_cuda_allocator)


def require_cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device is visible to PyTorch/Triton in this process. "
            "Check nvidia-smi, CUDA_VISIBLE_DEVICES, and whether this terminal has GPU access."
        )


def get_autotune_configs():
    return [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=2,
        ),
    ]


@triton.autotune(configs=get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # Each program computes one rectangular C tile:
    #   C[pid_m * BLOCK_M : (pid_m + 1) * BLOCK_M,
    #     pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N]
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache ordering:
    #
    # A simple row-major launch would visit C tiles like:
    #   (m=0,n=0), (m=0,n=1), (m=0,n=2), ...
    # That keeps the same A tile for a while, but every neighboring program
    # touches a different B tile.
    #
    # This grouped ordering visits a small stack of M tiles for the same N:
    #   (m=0,n=0), (m=1,n=0), ... (m=GROUP_M-1,n=0),
    #   (m=0,n=1), (m=1,n=1), ...
    # Those neighboring programs all load the same B[:, n] tile for each K
    # block, so B is more likely to still be in L2 cache.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M

    # The last group may have fewer than GROUP_M tile rows.
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    # local_pid is the program id inside the current grouped region.
    # M changes fastest inside the group; N changes after group_size_m programs.
    local_pid = pid % num_pid_in_group
    pid_m = first_pid_m + (local_pid % group_size_m)
    pid_n = local_pid // group_size_m

    # Descriptors describe the logical 2D tensors. The kernel can then load
    # tiles by coordinates instead of doing manual pointer arithmetic.
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    # Accumulate in fp32 for numerical stability, then cast once before store.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load one A tile and one B tile along the K dimension:
        #   A tile: [BLOCK_M, BLOCK_K]
        #   B tile: [BLOCK_K, BLOCK_N]
        a = a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
        b = b_desc.load([k * BLOCK_K, pid_n * BLOCK_N])
        acc = tl.dot(a, b, acc)

    # Store this program's C tile. Out-of-bound elements in edge tiles are
    # ignored by tensor descriptor store.
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(OUT_DTYPE))


def _triton_out_dtype(dtype: torch.dtype):
    if dtype is torch.float16:
        return tl.float16
    if dtype is torch.bfloat16:
        return tl.bfloat16
    if dtype is torch.float32:
        return tl.float32
    raise TypeError(f"Unsupported dtype: {dtype}")


def _assert_descriptor_aligned(name: str, tensor: torch.Tensor):
    # Tensor descriptor block offsets must land on 16-byte aligned addresses.
    assert tensor.stride(0) * tensor.element_size() % 16 == 0, (
        f"{name} row stride must be 16-byte aligned for tl.make_tensor_descriptor"
    )


def matmul(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2
    assert a.is_cuda and b.is_cuda
    assert a.dtype == b.dtype
    assert a.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert a.shape[1] == b.shape[0]
    assert a.stride(1) == 1, "A must be row-major contiguous in K"
    assert b.stride(1) == 1, "B must be row-major contiguous in N"
    _assert_descriptor_aligned("A", a)
    _assert_descriptor_aligned("B", b)

    M, K = a.shape
    K, N = b.shape
    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out
        assert c.shape == (M, N)
        assert c.is_cuda and c.dtype == a.dtype
        assert c.stride(1) == 1, "C must be row-major contiguous in N"
        _assert_descriptor_aligned("C", c)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        OUT_DTYPE=_triton_out_dtype(a.dtype),
    )
    return c


def check_correctness():
    torch.manual_seed(0)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        a = torch.randn((512, 512), device=DEVICE, dtype=dtype)
        b = torch.randn((512, 512), device=DEVICE, dtype=dtype)
        actual = matmul(a, b)
        expected = torch.matmul(a, b)

        if dtype is torch.float32:
            atol, rtol = 1e-1, 1e-2
        else:
            atol, rtol = 1e-2, 1e-2

        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        print(f"{dtype}: Triton and Torch match")


PROBLEM_SIZES = [
    ("256x256x256", 256, 256, 256),
    ("512x512x512", 512, 512, 512),
    ("1024x1024x1024", 1024, 1024, 1024),
    ("2048x2048x2048", 2048, 2048, 2048),
    ("4096x4096x4096", 4096, 4096, 4096),
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["matrix_size", "M", "N", "K"],
        x_vals=PROBLEM_SIZES,
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        xlabel="A[M,K] @ B[K,N]",
        ylabel="TFLOPS",
        plot_name="matmul-fp16",
        args={"dtype": torch.float16},
    )
)
def benchmark(matrix_size, M, N, K, provider, dtype):
    a = torch.randn((M, K), device=DEVICE, dtype=dtype)
    b = torch.randn((K, N), device=DEVICE, dtype=dtype)
    c = torch.empty((M, N), device=DEVICE, dtype=dtype)

    quantiles = (0.5, 0.2, 0.8)
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.mm(a, b, out=c),
            quantiles=quantiles,
        )
    else:
        matmul(a, b, out=c)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b, out=c),
            quantiles=quantiles,
        )

    def tflops(runtime_ms):
        return 2 * M * N * K * 1e-12 / (runtime_ms * 1e-3)

    return tflops(ms), tflops(max_ms), tflops(min_ms)


def _save_labeled_plot(df, save_path: Path):
    import matplotlib.pyplot as plt

    x_vals = list(range(len(df)))
    labels = df["matrix_size"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_vals, df["Torch (TFLOPS)"], label="Torch", color="green", marker="o")
    ax.plot(x_vals, df["Triton (TFLOPS)"], label="Triton", color="blue", marker="o")
    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("A[M,K] @ B[K,N]")
    ax.set_ylabel("TFLOPS")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path / "matmul-fp16.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    require_cuda_device()
    check_correctness()

    save_path = Path(__file__).resolve().parent / "matmul"
    save_path.mkdir(exist_ok=True)
    df = benchmark.run(
        show_plots=False, print_data=True, save_path=save_path, return_df=True
    )
    _save_labeled_plot(df, save_path)
