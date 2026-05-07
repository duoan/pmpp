import argparse
import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime import driver
from jaxtyping import Float


# input is a 2D tensor
# each of program handle a row
@triton.jit
def _softmax_forward_kernel(
    input_ptr,  # input pointer
    output_ptr,  # output pointer
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        input_row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offset = tl.arange(0, BLOCK_SIZE)  # every column has n_cols elements

        input_ptrs = input_row_start_ptr + col_offset

        mask = col_offset < n_cols

        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

        row_minus_max = row - tl.max(row, axis=0)

        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)

        probs = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offset
        tl.store(output_ptrs, probs, mask=mask)


@triton.jit
def _softmax_backward_kernel(
    probs_ptr,
    grad_output_ptr,
    grad_input_ptr,
    probs_row_stride,
    grad_output_row_stride,
    grad_input_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        col_offset = tl.arange(0, BLOCK_SIZE)
        mask = col_offset < n_cols

        probs_ptrs = probs_ptr + row_idx * probs_row_stride + col_offset
        grad_output_ptrs = grad_output_ptr + row_idx * grad_output_row_stride + col_offset

        probs = tl.load(probs_ptrs, mask=mask, other=0.0)
        grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0)

        grad_input = grad_output * probs
        dot = tl.sum(grad_input, axis=0)
        grad_input = grad_input - probs * dot

        grad_input_ptrs = grad_input_ptr + row_idx * grad_input_row_stride + col_offset
        tl.store(grad_input_ptrs, grad_input, mask=mask)


DEVICE = triton.runtime.driver.active.get_active_torch_device()
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def _max_programs_from_kernel(kernel, num_warps: int) -> int:
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    if size_smem > 0:
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
    return max(1, NUM_SM * occupancy)


def _backward_launch_config(block_size: int) -> tuple[int, int]:
    if block_size < 512:
        return 4, 2
    if block_size == 512:
        return 4, 3
    if block_size <= 2048:
        return 2, 1
    else:
        return 4, 1


def _backward_num_programs(
    n_rows: int,
    block_size: int,
    max_num_programs: int,
) -> int:
    if block_size == 512:
        return min(NUM_SM * 10, max_num_programs, n_rows)
    if block_size == 1024:
        return min(NUM_SM * 12, max_num_programs, n_rows)

    return min(max_num_programs, n_rows)


class TritonSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: Float[Tensor, "..."],
        *args,
        **kwargs,
    ):
        assert x.is_cuda
        assert x.dim() == 2
        assert x.is_contiguous()

        BLOCK_SIZE = triton.next_power_of_2(x.size(1))
        num_warps = 4 if BLOCK_SIZE <= 2048 else 8

        if BLOCK_SIZE >= 8192:
            num_stages = 1
        elif BLOCK_SIZE >= 2048:
            num_stages = 2
        elif BLOCK_SIZE >= 512:
            num_stages = 3
        else:
            num_stages = 2

        n_rows, n_cols = x.size(0), x.size(1)

        cache_key = (
            "forward",
            target.backend,
            target.arch,
            x.device.type,
            x.device.index,
            x.dtype,
            BLOCK_SIZE,
            num_warps,
            num_stages,
        )

        probs = torch.empty_like(x)

        if cache_key not in kernels:
            kernel = _softmax_forward_kernel.warmup(
                x,
                probs,
                x.stride(0),
                probs.stride(0),
                n_rows,
                n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
                num_stages=num_stages,
                num_warps=num_warps,
                grid=(1,),
            )
            kernels[cache_key] = (kernel, _max_programs_from_kernel(kernel, num_warps))

        kernel, max_num_programs = kernels[cache_key]

        if BLOCK_SIZE >= 2048:
            num_programs = n_rows
        else:
            num_programs = min(max_num_programs, n_rows)

        kernel[num_programs, 1, 1](
            x,
            probs,
            x.stride(0),
            probs.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE,
            num_stages,
        )

        ctx.save_for_backward(probs)

        return probs

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output,
    ):
        (probs,) = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(probs)

        n_rows, n_cols = probs.size(0), probs.size(1)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps, num_stages = _backward_launch_config(BLOCK_SIZE)

        cache_key = (
            "backward",
            target.backend,
            target.arch,
            probs.device.type,
            probs.device.index,
            probs.dtype,
            BLOCK_SIZE,
            num_warps,
            num_stages,
        )

        if cache_key not in kernels:
            kernel = _softmax_backward_kernel.warmup(
                probs,
                grad_output,
                grad_input,
                probs.stride(0),
                grad_output.stride(0),
                grad_input.stride(0),
                n_rows,
                n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
                num_stages=num_stages,
                num_warps=num_warps,
                grid=(1,),
            )
            kernels[cache_key] = (kernel, _max_programs_from_kernel(kernel, num_warps))

        kernel, max_num_programs = kernels[cache_key]

        num_programs = _backward_num_programs(n_rows, BLOCK_SIZE, max_num_programs)

        kernel[num_programs, 1, 1](
            probs,
            grad_output,
            grad_input,
            probs.stride(0),
            grad_output.stride(0),
            grad_input.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE,
            num_stages,
        )

        return grad_input


def _parse_cols(cols: str) -> list[int]:
    return [int(col) for col in cols.split(",") if col.strip()]


def _parse_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _benchmark_forward(
    provider: str,
    x: Tensor,
) -> tuple[float, float, float]:
    if provider == "torch":
        fn = lambda: torch.softmax(x, dim=1)
    elif provider == "triton":
        fn = lambda: TritonSoftmax.apply(x)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])


def _benchmark_backward(
    provider: str,
    x: Tensor,
    grad_output: Tensor,
) -> tuple[float, float, float]:
    if provider == "torch":
        y = torch.softmax(x, dim=1)
    elif provider == "triton":
        y = TritonSoftmax.apply(x)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    fn = lambda: torch.autograd.grad(y, x, grad_output, retain_graph=True)
    return triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])


def _validate_softmax(n_rows: int, n_cols: int, dtype: torch.dtype) -> None:
    x = torch.randn((n_rows, n_cols), device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    y = TritonSoftmax.apply(x)
    y_ref = torch.softmax(x_ref, dim=1)
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    grad_output = torch.randn_like(y)
    y.backward(grad_output)
    y_ref.backward(grad_output)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-2, atol=1e-2)


def benchmark(
    n_rows: int,
    cols: list[int],
    dtype: torch.dtype,
    mode: str,
    check: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    print(f"device={torch.cuda.get_device_name()} rows={n_rows} dtype={dtype}")
    print(f"{'mode':<10} {'cols':>8} {'torch ms':>12} {'triton ms':>12} {'speedup':>10}")
    print("-" * 58)

    for n_cols in cols:
        if check:
            _validate_softmax(min(n_rows, 64), n_cols, dtype)

        x = torch.randn((n_rows, n_cols), device="cuda", dtype=dtype, requires_grad=True)
        grad_output = torch.randn_like(x)

        if mode in ("forward", "all"):
            torch_ms, _, _ = _benchmark_forward("torch", x)
            triton_ms, _, _ = _benchmark_forward("triton", x)
            speedup = torch_ms / triton_ms
            print(f"{'forward':<10} {n_cols:>8} {torch_ms:>12.4f} {triton_ms:>12.4f} {speedup:>10.2f}x")

        if mode in ("backward", "all"):
            torch_ms, _, _ = _benchmark_backward("torch", x, grad_output)
            triton_ms, _, _ = _benchmark_backward("triton", x, grad_output)
            speedup = torch_ms / triton_ms
            print(f"{'backward':<10} {n_cols:>8} {torch_ms:>12.4f} {triton_ms:>12.4f} {speedup:>10.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark row-wise softmax: PyTorch vs Triton")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=str, default="128,256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp32")
    parser.add_argument("--mode", choices=("forward", "backward", "all"), default="all")
    parser.add_argument("--check", action="store_true", help="Validate correctness before each column benchmark")
    args = parser.parse_args()

    benchmark(
        n_rows=args.rows,
        cols=_parse_cols(args.cols),
        dtype=_parse_dtype(args.dtype),
        mode=args.mode,
        check=args.check,
    )


if __name__ == "__main__":
    main()
