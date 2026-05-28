import sys

import pytest
import torch

from sglang.jit_kernel.diffusion.pixel_norm import apply_pixel_norm
from sglang.jit_kernel.diffusion.triton.pixel_norm import (
    _pixel_norm_native,
    triton_pixel_norm,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]

# Representative shapes covering all production call sites.
# All use channel_dim=1, matching every LTX-2 family call.
TEST_CASES = [
    # (B, C, F, H, W) — LTX-2 video VAE residual block @ 256 channels, mid-decoder
    pytest.param((1, 256, 7, 22, 40), 1, id="ltx2_video_mid_dec"),
    # (B, C, F, H, W) — LTX-2 video VAE final upsampled latent
    pytest.param((1, 128, 28, 90, 160), 1, id="ltx2_video_decoder_out_720p"),
    # (B, C, H, W) — LTX-2 audio attention block input
    pytest.param((1, 128, 256, 256), 1, id="ltx2_audio_attn_4d"),
    # (B, C, H, W) — LTX-2 audio ResBlock norm
    pytest.param((1, 64, 64, 64), 1, id="ltx2_audio_resblock_4d"),
    # Small / odd-channel sanity case
    pytest.param((2, 7, 5, 9), 1, id="small_odd_channels"),
    # 3D tensor (channel_dim=1)
    pytest.param((4, 32, 128), 1, id="bcm_3d"),
    # Negative channel_dim
    pytest.param((2, 16, 8, 8), -3, id="neg_channel_dim"),
]


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    # The kernel accumulates in fp32 and rounds once on store; eager does
    #     x**2 -> mean -> sqrt -> x / sqrt
    # with intermediate fp16/bf16 roundings, so the two paths can disagree by a
    # handful of ULPs. The tolerance below tracks dtype ULP at magnitude ~1.0:
    # fp16 ULP ~= 9.8e-4, bf16 ULP ~= 7.8e-3. ``test_kernel_matches_fp32_reference_bf16``
    # below pins the *sharp* parity (fused vs fp32 reference) at ~1 ULP.
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.bfloat16:
        return 1e-2, 1e-2
    return 5e-3, 5e-3


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


def _eager_reference(x: torch.Tensor, channel_dim: int, eps: float) -> torch.Tensor:
    # Mirrors the original (production) eager implementation exactly:
    #     y = x / sqrt(mean(x**2, dim=channel_dim, keepdim=True) + eps)
    # in the input dtype. Used to validate that the kernel matches the
    # observable behavior of the eager code we are replacing.
    mean_sq = torch.mean(x**2, dim=channel_dim, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    return x / rms


@torch.no_grad()
@pytest.mark.parametrize("shape,channel_dim", TEST_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_pixel_norm_matches_eager(
    shape: tuple[int, ...], channel_dim: int, dtype: torch.dtype
) -> None:
    eps = 1e-8
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    actual = triton_pixel_norm(x, channel_dim=channel_dim, eps=eps)
    expected = _eager_reference(x, channel_dim, eps)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.no_grad()
@pytest.mark.parametrize("shape,channel_dim", TEST_CASES[:3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_apply_pixel_norm_dispatch(
    shape: tuple[int, ...], channel_dim: int, dtype: torch.dtype
) -> None:
    """``apply_pixel_norm`` (the public helper) and the raw kernel should agree."""
    eps = 1e-8
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    actual = apply_pixel_norm(x, channel_dim=channel_dim, eps=eps)
    expected = triton_pixel_norm(x, channel_dim=channel_dim, eps=eps)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@torch.no_grad()
def test_kernel_matches_fp32_reference_bf16() -> None:
    """Fused kernel must agree with the fp32-accumulating native fallback.

    This is a sharper check than ``_eager_reference``: both the fused kernel
    and ``_pixel_norm_native`` accumulate in fp32, so the only difference is
    the final cast back to bf16. They should match to within ~1 ulp of bf16.
    """
    shape = (1, 128, 7, 22, 40)
    x = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)

    actual = triton_pixel_norm(x, channel_dim=1, eps=1e-8)
    fp32_ref = _pixel_norm_native(x, channel_dim=1, eps=1e-8)

    torch.testing.assert_close(actual, fp32_ref, atol=4e-3, rtol=4e-3)


@torch.no_grad()
def test_non_contiguous_input_bf16() -> None:
    """Non-contiguous inputs should be handled (kernel calls ``.contiguous()``)."""
    shape = (2, 64, 8, 16, 16)
    x_full = torch.randn(shape + (2,), device=DEVICE, dtype=torch.bfloat16)
    x = x_full[..., 0]  # non-contiguous strided slice

    assert not x.is_contiguous()

    actual = triton_pixel_norm(x, channel_dim=1, eps=1e-8)
    expected = _eager_reference(x, 1, 1e-8)

    atol, rtol = _tol(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.no_grad()
def test_large_inner_dim_bf16() -> None:
    """720p decoder output shape — the largest per-call workload."""
    shape = (1, 128, 28, 90, 160)
    x = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)

    actual = triton_pixel_norm(x, channel_dim=1, eps=1e-8)
    expected = _eager_reference(x, 1, 1e-8)

    atol, rtol = _tol(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.no_grad()
def test_dtype_preserved_bf16_via_spy() -> None:
    """End-to-end check that ``apply_pixel_norm`` actually dispatches to the
    fused kernel (rather than silently falling back) on bf16 CUDA inputs."""
    from unittest.mock import patch

    import sglang.jit_kernel.diffusion.triton.pixel_norm as pn_mod

    shape = (1, 128, 4, 32, 32)
    x = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)

    real = pn_mod._triton_pixel_norm_cuda
    with patch.object(
        pn_mod, "_triton_pixel_norm_cuda", wraps=real
    ) as mock_cuda:
        out = apply_pixel_norm(x, channel_dim=1, eps=1e-8)

    assert mock_cuda.call_count >= 1, (
        "apply_pixel_norm should call the fused CUDA path on bf16 GPU inputs; "
        "the spy recorded no calls, which means the helper silently fell back."
    )
    assert out.dtype == x.dtype
    assert out.shape == x.shape


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
