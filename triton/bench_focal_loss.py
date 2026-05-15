"""Benchmark Triton sigmoid focal loss vs torch.compile(torchvision.ops.sigmoid_focal_loss).

用法:
    uv run python triton/bench_focal_loss.py
    uv run python triton/bench_focal_loss.py --dtype fp16 --reduction mean
    uv run python triton/bench_focal_loss.py --sizes "1024,1048576,16777216"
"""
import argparse

import torch
import torchvision
import triton

import torch._dynamo as dynamo
# 默认 recompile_limit=8 会让多 size benchmark 回退到 eager, 这里提到足够大.
dynamo.config.recompile_limit = 128

from focal_loss import triton_sigmoid_focal_loss


def _parse_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _parse_sizes(arg: str) -> list[int]:
    return [int(s) for s in arg.split(",") if s.strip()]


def _make_inputs(n: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(0)
    inputs = (torch.randn(n, generator=g, device="cuda", dtype=torch.float32) * 3.0).to(dtype)
    targets = (torch.rand(n, generator=g, device="cuda", dtype=torch.float32) > 0.5).to(dtype)
    return inputs, targets


def _gbps(n_elements: int, dtype: torch.dtype, ms: float, reduction: str) -> float:
    """估算有效带宽: 读 inputs + targets, 写 outputs (reduction=none) 或 标量 (mean/sum).

    对于 reduction != 'none', output 字节数可以忽略.
    """
    bytes_per_elem = torch.finfo(dtype).bits // 8
    in_bytes = 2 * n_elements * bytes_per_elem
    out_bytes = n_elements * bytes_per_elem if reduction == "none" else 0
    total_bytes = in_bytes + out_bytes
    return total_bytes / (ms * 1e-3) / 1e9


def _bench_one(fn, warmup_iters: int = 25, rep_iters: int = 100) -> float:
    """Return median ms over rep_iters."""
    return triton.testing.do_bench(fn, warmup=warmup_iters, rep=rep_iters)


def _build_providers(
    alpha: float, gamma: float, reduction: str
) -> dict[str, callable]:
    """返回 name -> 接受 (inputs, targets) 的 callable.

    NOTE: 为了避免 `torch.compile` 在多个 N 之间触发 recompile_limit 后回退到 eager,
    我们对每个 size 单独 compile (按输入 shape cache).
    """
    eager_ref = torchvision.ops.sigmoid_focal_loss
    compiled_default_cache: dict[tuple, callable] = {}
    compiled_reduce_cache: dict[tuple, callable] = {}

    def _get_compiled(cache: dict, mode: str, x: torch.Tensor) -> callable:
        key = (mode, tuple(x.shape), x.dtype)
        if key not in cache:
            cache[key] = torch.compile(
                torchvision.ops.sigmoid_focal_loss, mode=mode, dynamic=False
            )
        return cache[key]

    return {
        "torchvision_eager": lambda x, y: eager_ref(
            x, y, alpha=alpha, gamma=gamma, reduction=reduction
        ),
        "torch_compile_default": lambda x, y: _get_compiled(
            compiled_default_cache, "default", x
        )(x, y, alpha=alpha, gamma=gamma, reduction=reduction),
        "torch_compile_reduce_overhead": lambda x, y: _get_compiled(
            compiled_reduce_cache, "reduce-overhead", x
        )(x, y, alpha=alpha, gamma=gamma, reduction=reduction),
        "triton": lambda x, y: triton_sigmoid_focal_loss(
            x, y, alpha=alpha, gamma=gamma, reduction=reduction
        ),
    }


def benchmark(
    sizes: list[int],
    dtype: torch.dtype,
    alpha: float,
    gamma: float,
    reduction: str,
    check: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    providers = _build_providers(alpha, gamma, reduction)

    print(
        f"device={torch.cuda.get_device_name()} "
        f"dtype={dtype} alpha={alpha} gamma={gamma} reduction={reduction}"
    )
    header = (
        f"{'N':>12} | "
        f"{'eager ms':>10} {'compile ms':>11} {'compile-ro ms':>14} {'triton ms':>11} | "
        f"{'speedup(vs eager)':>18} {'speedup(vs compile)':>20} | "
        f"{'triton GB/s':>12}"
    )
    print(header)
    print("-" * len(header))

    for n in sizes:
        inputs, targets = _make_inputs(n, dtype)

        if check:
            ref = providers["torchvision_eager"](inputs.float(), targets.float()).to(dtype)
            out = providers["triton"](inputs, targets)
            rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (2e-2, 2e-2)
            torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)

        ms = {name: _bench_one(lambda fn=fn: fn(inputs, targets)) for name, fn in providers.items()}

        triton_ms = ms["triton"]
        speedup_eager = ms["torchvision_eager"] / triton_ms
        speedup_compile = ms["torch_compile_default"] / triton_ms
        bw = _gbps(n, dtype, triton_ms, reduction)

        print(
            f"{n:>12} | "
            f"{ms['torchvision_eager']:>10.4f} "
            f"{ms['torch_compile_default']:>11.4f} "
            f"{ms['torch_compile_reduce_overhead']:>14.4f} "
            f"{triton_ms:>11.4f} | "
            f"{speedup_eager:>17.2f}x "
            f"{speedup_compile:>19.2f}x | "
            f"{bw:>12.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton focal_loss vs torch.compile(torchvision.ops.sigmoid_focal_loss)"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="4096,16384,65536,262144,1048576,4194304,16777216,67108864",
        help="逗号分隔的 N (元素个数) 列表",
    )
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument(
        "--reduction", choices=("none", "mean", "sum"), default="none"
    )
    parser.add_argument(
        "--check", action="store_true", help="benchmark 前先对每个 size 做一次正确性校验"
    )
    args = parser.parse_args()

    benchmark(
        sizes=_parse_sizes(args.sizes),
        dtype=_parse_dtype(args.dtype),
        alpha=args.alpha,
        gamma=args.gamma,
        reduction=args.reduction,
        check=args.check,
    )


if __name__ == "__main__":
    main()
