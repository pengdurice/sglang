"""Public wrapper for fused per-pixel RMS normalization.

Mirrors the ``apply_group_norm_silu`` pattern: a thin dispatch that calls the
Triton fast path on CUDA inference inputs and otherwise falls back to the
original eager formula::

    y = x / torch.sqrt(torch.mean(x ** 2, dim=channel_dim, keepdim=True) + eps)

The CUDA fast path is registered as a ``torch.library`` custom op whose
backend table contains only CUDA + Meta; calling it with a CPU tensor would
raise ``NotImplementedError`` from the dispatcher *before* the Python body
gets a chance to fall back. So the ``x.is_cuda`` gate has to live here, in
the public wrapper, instead of inside the registered op body.

The fallback is inlined (no import from the ``triton`` subpackage) so this
module stays importable on systems that don't ship Triton.
"""

from __future__ import annotations

import torch


def apply_pixel_norm(
    x: torch.Tensor, *, channel_dim: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    """Per-pixel RMS normalization along ``channel_dim``.

    Computes (in math)::

        y = x * rsqrt(mean(x ** 2, dim=channel_dim, keepdim=True) + eps)

    The Triton fast path is taken on CUDA inference inputs (bf16/fp16/fp32)
    when the channel dimension fits in a single Triton block. All other paths
    (CPU tensors, training / grad-enabled inputs, MPS, NPU, ...) fall back to
    a byte-identical eager implementation.
    """
    if (
        x.is_cuda
        and not torch.is_grad_enabled()
        and not x.requires_grad
    ):
        from sglang.jit_kernel.diffusion.triton.pixel_norm import triton_pixel_norm

        return triton_pixel_norm(x, channel_dim=channel_dim, eps=eps)

    # Eager fallback. Matches the pre-fusion production formula in the input
    # dtype (no fp32 promotion), so this wiring is a no-op on CPU.
    mean_sq = torch.mean(x**2, dim=channel_dim, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    return x / rms


__all__ = ["apply_pixel_norm"]
