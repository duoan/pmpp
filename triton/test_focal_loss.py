"""Test Triton sigmoid focal loss against torch.compile(torchvision.ops.sigmoid_focal_loss).

Reference:
    https://docs.pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html
"""
import pytest
import torch
import torchvision

from focal_loss import triton_sigmoid_focal_loss


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Triton focal_loss tests"
)


_compiled_cache: dict[str, torch.nn.Module] = {}


def _get_compiled_reference(mode: str = "default"):
    """Lazily compile torchvision's sigmoid_focal_loss; cache per-mode to avoid recompiles."""
    if mode not in _compiled_cache:
        _compiled_cache[mode] = torch.compile(
            torchvision.ops.sigmoid_focal_loss, mode=mode, dynamic=False
        )
    return _compiled_cache[mode]


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-5, 1e-5


def _make_inputs(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    # logits 大致落在 [-5, 5], 覆盖 sigmoid 的饱和与线性两段
    inputs = (torch.randn(shape, generator=g, device="cuda", dtype=torch.float32) * 3.0).to(dtype)
    targets = (torch.rand(shape, generator=g, device="cuda", dtype=torch.float32) > 0.5).to(dtype)
    return inputs, targets


@pytest.mark.parametrize(
    "shape",
    [
        (1024,),
        (256, 80),
        (4, 16, 32),
        (1023,),  # 非 BLOCK_SIZE 整数倍, 检查 mask
        (2, 3, 5, 7),
    ],
    ids=["1d", "2d", "3d", "1d_unaligned", "4d"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75, -1.0])
@pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0, 3.5])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_matches_torchvision_compiled(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    alpha: float,
    gamma: float,
    reduction: str,
) -> None:
    inputs, targets = _make_inputs(shape, dtype)

    triton_out = triton_sigmoid_focal_loss(
        inputs, targets, alpha=alpha, gamma=gamma, reduction=reduction
    )

    ref_fn = _get_compiled_reference()
    # torchvision 在 fp16/bf16 下精度更敏感, 在 fp32 上算 reference 再 cast 回来更可靠
    ref_out = ref_fn(
        inputs.float(), targets.float(), alpha=alpha, gamma=gamma, reduction=reduction
    ).to(dtype)

    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(triton_out, ref_out, rtol=rtol, atol=atol)


def test_invalid_alpha_raises() -> None:
    inputs, targets = _make_inputs((16,), torch.float32)
    with pytest.raises(ValueError, match="Invalid alpha"):
        triton_sigmoid_focal_loss(inputs, targets, alpha=1.5)


def test_invalid_reduction_raises() -> None:
    inputs, targets = _make_inputs((16,), torch.float32)
    with pytest.raises(ValueError, match="Invalid Value for arg 'reduction'"):
        triton_sigmoid_focal_loss(inputs, targets, reduction="bogus")


def test_shape_mismatch_raises() -> None:
    inputs = torch.randn(16, device="cuda")
    targets = torch.randn(8, device="cuda")
    with pytest.raises(ValueError, match="same shape"):
        triton_sigmoid_focal_loss(inputs, targets)


def test_non_contiguous_inputs() -> None:
    """Triton 启动器应当能正确处理非连续输入 (内部 .contiguous())."""
    base = torch.randn(64, 32, device="cuda", dtype=torch.float32)
    inputs = base.t()  # transpose 后变为非连续
    targets = (torch.rand_like(inputs) > 0.5).float()
    assert not inputs.is_contiguous()

    triton_out = triton_sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0)
    ref_out = torchvision.ops.sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0)
    torch.testing.assert_close(triton_out, ref_out, rtol=1e-5, atol=1e-5)


def test_large_logits_numerical_stability() -> None:
    """logits 取极端值时 (|x| >> 1), 数值不应当溢出或产生 NaN."""
    inputs = torch.tensor(
        [100.0, -100.0, 50.0, -50.0, 0.0, 1e3, -1e3],
        device="cuda",
        dtype=torch.float32,
    )
    targets = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], device="cuda")

    triton_out = triton_sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0)
    ref_out = torchvision.ops.sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0)

    assert torch.isfinite(triton_out).all(), f"triton produced non-finite: {triton_out}"
    torch.testing.assert_close(triton_out, ref_out, rtol=1e-5, atol=1e-5)
