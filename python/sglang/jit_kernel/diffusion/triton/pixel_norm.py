"""Fused per-pixel RMS normalization (a.k.a. PixelNorm) for the LTX-2 family.

The eager reference computes, for each position along the non-channel dims, the
RMS of the channel vector and divides the input by it::

    mean_sq = (x * x).mean(dim=channel_dim, keepdim=True)
    y = x / torch.sqrt(mean_sq + eps)

In PyTorch this is three separate kernel launches (``pow``, ``mean``, ``sqrt+div``)
and materializes a full-sized ``x * x`` temporary. The fused kernel below does
the reduction and the normalization in a single pass, with fp32 accumulation,
and writes back in the input dtype.

Channel layouts handled by the kernel:

* ``channel_dim`` selects the reduction axis (e.g. ``1`` for ``[B, C, ...]``).
* All other dims are flattened into an ``(outer, inner)`` triple (``B, C, M``)
  where ``M = product(shape after channel_dim)``. The kernel uses a 2D tile
  ``[BLOCK_C, BLOCK_HW]``: ``BLOCK_HW`` lanes load contiguous memory
  (innermost dim, fully coalesced) and each lane independently reduces over
  ``BLOCK_C`` channels along the strided axis. This is the same access pattern
  PyTorch's ``mean(dim=channel_dim)`` uses for non-innermost reductions, and
  is what avoids the catastrophic strided-gather of a naive
  one-program-per-position layout (which was 4x slower than eager on a
  ``(1, 128, 256, 256)`` audio attn block).

Backwards (training) is not supported; this is an inference-only kernel.
"""

from __future__ import annotations

import math

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.srt.utils.custom_op import register_custom_op

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_MAX_BLOCK_C = 4096


@triton.jit
def _pixel_norm_kernel(
    x_ptr,
    out_ptr,
    eps,
    n_inner,
    n_channels,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """2D-tiled per-pixel RMS norm.

    Grid: ``(n_outer, cdiv(n_inner, BLOCK_HW))``. Each program owns a tile
    ``[BLOCK_C, BLOCK_HW]`` of the logical ``x[outer, channel, inner]`` view
    (with memory strides ``(C * M, M, 1)``, ``M = n_inner``).

    The innermost (``BLOCK_HW``) dim is contiguous in memory, which Triton
    maps to consecutive lanes within a warp -> fully coalesced loads/stores.
    The strided ``BLOCK_C`` dim is the reduction axis; each ``BLOCK_HW``
    column accumulates its own sum-of-squares in fp32.
    """
    outer_id = tl.program_id(0).to(tl.int64)
    hw_block_id = tl.program_id(1).to(tl.int64)

    hw_idx = hw_block_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_idx < n_inner

    c_idx = tl.arange(0, BLOCK_C)
    c_mask = c_idx < n_channels

    base = outer_id * n_channels * n_inner
    # offs has shape [BLOCK_C, BLOCK_HW]; innermost dim (BLOCK_HW) is contiguous.
    offs = base + c_idx[:, None] * n_inner + hw_idx[None, :]
    mask = c_mask[:, None] & hw_mask[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Reduce over the strided channel axis (axis=0) -> one RMS per (outer, hw).
    sum_sq = tl.sum(x * x, axis=0)
    mean_sq = sum_sq / n_channels
    rstd = tl.rsqrt(mean_sq + eps)  # shape [BLOCK_HW]

    y = x * rstd[None, :]
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def _choose_block_hw(block_c: int) -> int:
    """Pick a BLOCK_HW that keeps the tile near 4-8K elements (register-friendly)
    while still giving each warp a full lane width (32) of contiguous loads."""
    if block_c <= 64:
        return 128
    if block_c <= 128:
        return 64
    if block_c <= 256:
        return 32
    if block_c <= 512:
        return 32
    return 16


def _can_use_triton_pixel_norm(x: torch.Tensor, channel_dim: int) -> bool:
    if not x.is_cuda:
        return False
    if torch.is_grad_enabled() or x.requires_grad:
        return False
    if x.dtype not in _SUPPORTED_DTYPES:
        return False
    # Normalize channel_dim to a non-negative index in [0, ndim).
    if x.ndim < 2:
        return False
    cd = channel_dim if channel_dim >= 0 else channel_dim + x.ndim
    if cd < 0 or cd >= x.ndim:
        return False
    n_channels = x.shape[cd]
    if n_channels <= 0 or n_channels > _MAX_BLOCK_C:
        return False
    return True


def _pixel_norm_native(
    x: torch.Tensor, channel_dim: int, eps: float
) -> torch.Tensor:
    # Match the eager implementation exactly: x / sqrt(mean(x**2) + eps),
    # all in fp32 for numerical parity with the kernel's fp32 accumulator.
    orig_dtype = x.dtype
    x32 = x.float()
    mean_sq = torch.mean(x32 * x32, dim=channel_dim, keepdim=True)
    rstd = torch.rsqrt(mean_sq + eps)
    return (x32 * rstd).to(orig_dtype)


@register_custom_op(op_name="triton_pixel_norm_cuda", out_shape="x")
def _triton_pixel_norm_cuda(
    x: torch.Tensor, channel_dim: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    if not _can_use_triton_pixel_norm(x, channel_dim):
        return _pixel_norm_native(x, channel_dim, eps)

    cd = channel_dim if channel_dim >= 0 else channel_dim + x.ndim
    n_channels = x.shape[cd]
    n_outer = math.prod(x.shape[:cd]) if cd > 0 else 1
    n_inner = math.prod(x.shape[cd + 1 :]) if cd + 1 < x.ndim else 1

    # Force the contiguous (outer, C, inner) layout so the stride math in the
    # kernel matches what the launcher assumes.
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    block_c = max(16, triton.next_power_of_2(n_channels))
    block_hw = _choose_block_hw(block_c)
    # When n_inner is small, cap BLOCK_HW so we don't waste lanes on masked tail.
    if n_inner < block_hw:
        block_hw = max(16, triton.next_power_of_2(n_inner))

    grid = (n_outer, triton.cdiv(n_inner, block_hw))
    num_warps = 4 if block_c * block_hw <= 4096 else 8

    with torch.cuda.device(x.device):
        _pixel_norm_kernel[grid](
            x_contig,
            y,
            eps,
            n_inner,
            n_channels,
            BLOCK_C=block_c,
            BLOCK_HW=block_hw,
            num_warps=num_warps,
            num_stages=2,
        )
    return y


def triton_pixel_norm(
    x: torch.Tensor, channel_dim: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    """Thin alias for the registered custom op.

    Non-CUDA callers should go through ``apply_pixel_norm`` in
    :mod:`sglang.jit_kernel.diffusion.pixel_norm`; that wrapper avoids the
    custom-op dispatch entirely for CPU / MPS / grad-enabled inputs (the op
    is only registered for CUDA + Meta backends, so a CPU call here would
    raise ``NotImplementedError``).
    """
    return _triton_pixel_norm_cuda(x, channel_dim, eps)


__all__ = ["triton_pixel_norm"]
