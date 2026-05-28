"""CPU-side unit tests for the LTX-2 PixelNorm fused-helper wiring.

The CUDA / Triton kernel itself is covered by
``python/sglang/jit_kernel/tests/diffusion/test_pixel_norm.py``. The tests in
this file pin the *module-level wiring* contract that:

1. Each of the three LTX-2 family PixelNorm modules
   (:class:`PerChannelRMSNorm`, :class:`LTX2AudioPixelNorm`,
   :class:`LTX23VideoPixelNorm`) routes through
   :func:`sglang.jit_kernel.diffusion.pixel_norm.apply_pixel_norm`.
2. The native (CPU) fallback inside ``apply_pixel_norm`` is numerically
   equivalent to the eager formula it replaces.

These tests run on CPU and never hit the Triton kernel — they catch
wiring-level regressions before CUDA CI has a chance to run.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from sglang.jit_kernel.diffusion.pixel_norm import apply_pixel_norm
from sglang.jit_kernel.diffusion.triton.pixel_norm import _pixel_norm_native


def _eager_reference(x: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
    """The original (pre-fusion) PyTorch implementation, byte-for-byte."""
    mean_sq = torch.mean(x**2, dim=dim, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    return x / rms


@pytest.mark.parametrize(
    "shape,dim",
    [
        ((1, 128, 7, 22, 40), 1),  # ltx-2 video VAE residual block
        ((1, 64, 32, 32), 1),  # ltx-2 audio attn block
        ((2, 16, 8, 8), 1),  # small case
        ((4, 8), 1),  # 2D
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_apply_pixel_norm_cpu_matches_eager(
    shape: tuple[int, ...], dim: int, dtype: torch.dtype
) -> None:
    eps = 1e-8
    x = torch.randn(shape, dtype=dtype)

    actual = apply_pixel_norm(x, channel_dim=dim, eps=eps)
    expected = _eager_reference(x, dim, eps)

    # The native fallback accumulates in fp32, so it is *more* accurate than
    # the eager formula. Tolerance is bounded by the final-store rounding.
    atol = 1e-5 if dtype == torch.float32 else 1e-2
    rtol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_native_fallback_dtype_preserved() -> None:
    x = torch.randn((2, 8, 4, 4), dtype=torch.bfloat16)
    out = _pixel_norm_native(x, channel_dim=1, eps=1e-8)
    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape


def test_per_channel_rms_norm_dispatches_through_apply_pixel_norm() -> None:
    """The eager class wrapper must funnel through ``apply_pixel_norm``."""
    from sglang.multimodal_gen.runtime.models.vaes import ltx_2_vae

    module = ltx_2_vae.PerChannelRMSNorm(channel_dim=1, eps=1e-8)
    x = torch.randn((1, 32, 4, 8, 8))

    with patch.object(
        ltx_2_vae, "apply_pixel_norm", wraps=apply_pixel_norm
    ) as spy:
        out = module(x)

    assert spy.call_count == 1, (
        "PerChannelRMSNorm.forward should call apply_pixel_norm exactly once"
    )
    _, kwargs = spy.call_args
    assert kwargs == {"channel_dim": 1, "eps": 1e-8}
    assert out.shape == x.shape


def test_per_channel_rms_norm_preserves_channel_dim_override_bug() -> None:
    """The original implementation accepted ``channel_dim`` but never honored
    it. We preserve that behavior to avoid silently changing observable
    semantics; future fixes should be opt-in."""
    from sglang.multimodal_gen.runtime.models.vaes import ltx_2_vae

    module = ltx_2_vae.PerChannelRMSNorm(channel_dim=1, eps=1e-8)
    x = torch.randn((1, 32, 4, 8, 8))

    # Passing a different override must NOT change the result (matches eager).
    out_default = module(x)
    out_override = module(x, channel_dim=2)
    torch.testing.assert_close(out_default, out_override, atol=0.0, rtol=0.0)


def test_ltx2_audio_pixel_norm_dispatches_through_apply_pixel_norm() -> None:
    from sglang.multimodal_gen.runtime.models.vaes import ltx_2_audio

    module = ltx_2_audio.LTX2AudioPixelNorm(dim=1, eps=1e-6)
    x = torch.randn((1, 32, 16, 16))

    with patch.object(
        ltx_2_audio, "apply_pixel_norm", wraps=apply_pixel_norm
    ) as spy:
        out = module(x)

    assert spy.call_count == 1
    _, kwargs = spy.call_args
    assert kwargs == {"channel_dim": 1, "eps": 1e-6}
    assert out.shape == x.shape


def test_ltx23_video_pixel_norm_dispatches_through_apply_pixel_norm() -> None:
    from sglang.multimodal_gen.runtime.models.vaes import ltx_2_3_condition_encoder

    module = ltx_2_3_condition_encoder.LTX23VideoPixelNorm(dim=1, eps=1e-8)
    x = torch.randn((1, 32, 4, 8, 8))

    with patch.object(
        ltx_2_3_condition_encoder,
        "apply_pixel_norm",
        wraps=apply_pixel_norm,
    ) as spy:
        out = module(x)

    assert spy.call_count == 1
    _, kwargs = spy.call_args
    assert kwargs == {"channel_dim": 1, "eps": 1e-8}
    assert out.shape == x.shape


def test_class_wrappers_numerical_equivalence_against_eager() -> None:
    """All three module classes must give the same answer as the eager formula
    they replace, for the canonical channel_dim=1 case."""
    from sglang.multimodal_gen.runtime.models.vaes.ltx_2_3_condition_encoder import (
        LTX23VideoPixelNorm,
    )
    from sglang.multimodal_gen.runtime.models.vaes.ltx_2_audio import (
        LTX2AudioPixelNorm,
    )
    from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import PerChannelRMSNorm

    shape = (1, 32, 4, 8, 8)
    eps = 1e-8
    x = torch.randn(shape, dtype=torch.float32)
    expected = _eager_reference(x, 1, eps)

    for cls in (PerChannelRMSNorm, LTX2AudioPixelNorm, LTX23VideoPixelNorm):
        if cls is PerChannelRMSNorm:
            module = cls(channel_dim=1, eps=eps)
            shape_for_cls = shape
        elif cls is LTX2AudioPixelNorm:
            module = cls(dim=1, eps=eps)
            shape_for_cls = (1, 32, 16, 16)
        else:
            module = cls(dim=1, eps=eps)
            shape_for_cls = shape

        local_x = torch.randn(shape_for_cls, dtype=torch.float32)
        torch.testing.assert_close(
            module(local_x),
            _eager_reference(local_x, 1, eps),
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
