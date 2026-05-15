"""Triton sigmoid focal loss, API 对齐 torchvision.ops.sigmoid_focal_loss.

实现要点:
1. 两个 kernel:
   - `_focal_loss_kernel`: reduction='none' 时使用, 每元素一次写
   - `_focal_loss_reduce_kernel`: reduction='mean'/'sum' 时使用, 把 block 内 sum
      atomic_add 到一个标量, 省掉第二次 pass
2. 用 `triton.autotune` 调 BLOCK_SIZE / num_warps / num_stages, 不同 N 自动选最优.
3. 数值稳定 sigmoid + softplus 形式的 log(p), 用 |x| 分支避免 exp(big positive) 溢出.
4. 计算在 fp32 中进行, store 时 cast 回输入 dtype.
"""
import torch
import triton
import triton.language as tl
from torch import Tensor


_TORCH_TO_TL: dict[torch.dtype, tl.dtype] = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _torch_dtype_to_tl(dtype: torch.dtype) -> tl.dtype:
    return _TORCH_TO_TL[dtype]


def _autotune_configs() -> list:
    """常见 BLOCK_SIZE / num_warps / num_stages 组合, 同时覆盖小尺寸和大尺寸."""
    configs = []
    for bs in (1024, 2048, 4096, 8192):
        for nw, ns in [(2, 1), (2, 2), (4, 1), (4, 2), (4, 3), (8, 2), (8, 3)]:
            configs.append(
                triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
            )
    return configs


# Persistent partial-sum kernel 用; XBLOCK 是 inner-loop tile size,
# 总 program 数 = NUM_PROGRAMS (固定, 不随 N 变化), 每个 program 内部循环吃完
# 自己负责的 [pid::NUM_PROGRAMS] 切片. 这样大 N 时 launch overhead 几乎为 0,
# 同时 inner loop 让寄存器复用 + Triton 自动 software-pipeline.
# 固定 NUM_PROGRAMS 为 constexpr (不让 autotune 选), 这样 launcher 不需要
# 在每次调用时查 best_config; partial buffer 大小也是定死的, 不用每次重算.
# 1024 经验上对 N=64K-128M 都接近最优 (per autotune sweep), 而且让 partial buffer
# 永远只有 4KB.
_FIXED_NUM_PROGRAMS = 1024


def _partial_sum_configs() -> list:
    # Persistent reduce: NUM_PROGRAMS 固定为 _FIXED_NUM_PROGRAMS, autotune 只调
    # R0_BLOCK / num_warps / num_stages 这些影响 pipelining + occupancy 的参数.
    # R0_BLOCK 是 inner loop tile, 一个 program 连续读 chunk_size 个元素时, 每次
    # 处理 R0_BLOCK 个 fp32. Smaller R0_BLOCK -> 更多 iter, 更好的 pipelining;
    # bigger R0_BLOCK -> 更少 iter, 更高 BW utilization.
    configs = []
    for r0 in (1024, 2048, 4096, 8192):
        for nw, ns in [(4, 2), (4, 3), (4, 4), (8, 2), (8, 3), (8, 4)]:
            configs.append(
                triton.Config({"R0_BLOCK": r0}, num_warps=nw, num_stages=ns)
            )
    return configs


@triton.jit
def _focal_loss_compute(x, y, alpha, gamma, USE_ALPHA: tl.constexpr):
    """Pure compute: x (logits) + y (targets) -> per-element focal loss, all in fp32.

    被两个 kernel 共享, 避免代码漂移.
    """
    # 数值稳定 sigmoid: 用 |x| 分支避免 exp(big positive) 溢出
    abs_x = tl.abs(x)
    neg_abs_exp = tl.exp(-abs_x)
    # log(1 + exp(-|x|)) — exp(-|x|) ∈ (0, 1], 永远不会溢出
    softplus_neg_abs = tl.log(1.0 + neg_abs_exp)
    p = tl.where(
        x >= 0, 1.0 / (1.0 + neg_abs_exp), neg_abs_exp / (1.0 + neg_abs_exp)
    )

    # log-sigmoid (数值稳定)
    # log(p)     = -softplus(-x) = -[max(-x, 0) + log(1 + exp(-|x|))]
    # log(1 - p) = -softplus(x)  = -[max(x, 0)  + log(1 + exp(-|x|))]
    zero = tl.zeros_like(x)
    log_p = -(tl.maximum(-x, zero) + softplus_neg_abs)
    log_1_p = -(tl.maximum(x, zero) + softplus_neg_abs)

    # p_t & log(p_t)
    p_t = y * p + (1 - y) * (1 - p)
    log_pt = y * log_p + (1 - y) * log_1_p

    # Focal modulator: (1 - p_t) ** gamma
    # 用 exp(gamma * log(base)) 代替 pow; clamp base 避免 log(0).
    # base==0 时 log_pt 也接近 0, 所以 clamp 不引入误差.
    one_minus_pt = 1.0 - p_t
    log_one_minus_pt = tl.log(tl.maximum(one_minus_pt, 1e-20))
    modulator = tl.exp(gamma * log_one_minus_pt)
    loss = -modulator * log_pt

    # 可选 alpha 加权 (USE_ALPHA=False 时整段消失)
    if USE_ALPHA:
        alpha_t = y * alpha + (1 - y) * (1 - alpha)
        loss = alpha_t * loss

    return loss


@triton.autotune(configs=_autotune_configs(), key=["n_elements"])
@triton.jit
def _focal_loss_kernel(
    inputs_ptr, targets_ptr, out_ptr,
    alpha, gamma,
    n_elements,
    USE_ALPHA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(inputs_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(targets_ptr + offsets, mask=mask).to(tl.float32)

    loss = _focal_loss_compute(x, y, alpha, gamma, USE_ALPHA)

    tl.store(out_ptr + offsets, loss, mask=mask)


@triton.autotune(configs=_partial_sum_configs(), key=["n_elements"])
@triton.jit
def _focal_loss_partial_sum_kernel(
    inputs_ptr, targets_ptr, partial_ptr,
    alpha, gamma,
    n_elements,
    USE_ALPHA: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    """Stage 1 (persistent + contiguous): 每个 program 负责 [pid*chunk, (pid+1)*chunk)
    一个连续大块, 内部 inner loop 按 R0_BLOCK 步长往前扫.

    Contiguous (而非 strided) 让 L2 / DRAM 行预取更友好; 同时 program 数固定 (=
    NUM_PROGRAMS), 大 N 时 launch overhead 几乎为 0.
    """
    pid = tl.program_id(0)
    # 每个 program 拿到 [pid*chunk, (pid+1)*chunk) 范围, chunk = ceil(N / NUM_PROGRAMS)
    chunk = (n_elements + NUM_PROGRAMS - 1) // NUM_PROGRAMS
    start_pid = pid * chunk
    end_pid = tl.minimum(start_pid + chunk, n_elements)

    acc = tl.zeros((R0_BLOCK,), dtype=tl.float32)
    for r0_offset in tl.range(start_pid, end_pid, R0_BLOCK):
        offsets = r0_offset + tl.arange(0, R0_BLOCK)
        mask = offsets < end_pid
        x = tl.load(inputs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        loss = _focal_loss_compute(x, y, alpha, gamma, USE_ALPHA)
        loss = tl.where(mask, loss, 0.0)
        acc += loss

    block_sum = tl.sum(acc, axis=0)
    tl.store(partial_ptr + pid, block_sum)


@triton.jit
def _final_reduce_kernel(
    partial_ptr, out_scalar_ptr,
    n_partials,
    scale,
    OUT_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 2: 单 block 把所有 partial sum 加起来, 乘以 scale, cast 到 OUT_DTYPE,
    写一个标量到 out_scalar_ptr.

    用 loop 处理 n_partials > BLOCK_SIZE 的情况; BLOCK_SIZE 通常选 1024 或 2048.
    """
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in tl.range(0, n_partials, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_partials
        vals = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
        acc += vals
    total = tl.sum(acc, axis=0) * scale
    tl.store(out_scalar_ptr, total.to(OUT_DTYPE))


def triton_sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> Tensor:
    """Triton 版本的 sigmoid focal loss, API 对齐 torchvision.ops.sigmoid_focal_loss.

    Args:
        inputs: 任意 shape 的浮点 tensor, 表示每个样本的 logits.
        targets: 与 inputs 同 shape 的浮点 tensor, 二分类标签 (0 或 1).
        alpha: [0, 1] 范围内的权重系数; -1 表示不使用 alpha 加权.
        gamma: 调制因子 (1 - p_t) 的指数.
        reduction: 'none' | 'mean' | 'sum'.

    Returns:
        与 inputs 同 shape (reduction='none') 或标量 tensor (reduction='mean'/'sum') 的 loss.
    """
    if not (0 <= alpha <= 1) and alpha != -1:
        raise ValueError(
            f"Invalid alpha value: {alpha}. alpha must be in the range [0,1] or -1 for ignore."
        )
    if inputs.shape != targets.shape:
        raise ValueError(
            f"inputs and targets must have the same shape, got {inputs.shape} vs {targets.shape}"
        )
    if not inputs.is_cuda:
        raise ValueError("inputs must be on a CUDA device")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction}'. Supported: 'none', 'mean', 'sum'."
        )

    inputs_c = inputs.contiguous()
    targets_c = targets.to(inputs_c.dtype).contiguous()
    n_elements = inputs_c.numel()

    if n_elements == 0:
        if reduction == "none":
            return torch.empty_like(inputs_c)
        return torch.zeros((), device=inputs_c.device, dtype=inputs_c.dtype)

    use_alpha = alpha >= 0

    if reduction == "none":
        out = torch.empty_like(inputs_c)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _focal_loss_kernel[grid](
            inputs_c, targets_c, out,
            alpha, gamma,
            n_elements,
            USE_ALPHA=use_alpha,
        )
        return out

    # reduction='mean' / 'sum': 两阶段 persistent reduce
    # Stage 1: _FIXED_NUM_PROGRAMS (=1024) 个 program 各自吃自己的 strided slice,
    #          partial sum -> partial[NUM_PROGRAMS]
    # Stage 2: grid=1, 单 block + inner loop 把所有 partial 加起来, 乘以 scale
    #          并 cast 到目标 dtype.
    #
    # NUM_PROGRAMS 是 constexpr (=1024), partial buffer 永远 4KB, launcher 不需要
    # 查 autotune best_config.
    partial = torch.empty(_FIXED_NUM_PROGRAMS, device=inputs_c.device, dtype=torch.float32)
    out_scalar = torch.empty((), device=inputs_c.device, dtype=inputs_c.dtype)

    _focal_loss_partial_sum_kernel[(_FIXED_NUM_PROGRAMS,)](
        inputs_c, targets_c, partial,
        alpha, gamma,
        n_elements,
        USE_ALPHA=use_alpha,
        NUM_PROGRAMS=_FIXED_NUM_PROGRAMS,
    )

    scale = (1.0 / n_elements) if reduction == "mean" else 1.0
    _final_reduce_kernel[(1,)](
        partial, out_scalar,
        _FIXED_NUM_PROGRAMS,
        scale,
        OUT_DTYPE=_torch_dtype_to_tl(inputs_c.dtype),
        BLOCK_SIZE=_FIXED_NUM_PROGRAMS,  # 整除, mask 永远全 True
        num_warps=4,
    )

    return out_scalar
