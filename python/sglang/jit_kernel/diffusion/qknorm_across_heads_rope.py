"""Python JIT wrapper for the fused across-heads QK-Norm + RoPE kernel.

Pairs with ``python/sglang/jit_kernel/csrc/diffusion/qknorm_across_heads_rope.cuh``.

This is the across-heads sibling of
``sglang.jit_kernel.diffusion.qknorm_rope.fused_inplace_qknorm_rope``:

* The per-head fused kernel takes ``q_weight: [head_dim]`` and applies a
  separate RMSNorm reduction per (token, head). Used by FLUX / Qwen-Image /
  Z-Image / HunyuanVideo.
* The across-heads variant takes ``q_weight: [num_heads * head_dim]`` and
  reduces a single RMSNorm across all heads per token. Used by Wan / Sana /
  Helios (configs set ``qk_norm = "rms_norm_across_heads"``).

Both kernels accept the same FlashInfer-style ``cos_sin_cache``
``[max_pos, rope_dim]`` (float32, ``[cos | sin]`` along the last dim) and
``positions`` ``[num_tokens]`` (int32 or int64). RoPE style is interleave
(GPT-J / ``is_neox=False``); NeoX rotation is out of scope here — diffusion
across-heads configs all use interleave.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


logger = logging.getLogger(__name__)


@cache_once
def _jit_qknorm_across_heads_rope_module(
    head_dim: int,
    rope_dim: int,
    dtype: torch.dtype,
) -> "Module":
    args = make_cpp_args(head_dim, rope_dim, dtype)
    return load_jit(
        "qknorm_across_heads_rope",
        *args,
        cuda_files=["diffusion/qknorm_across_heads_rope.cuh"],
        cuda_wrappers=[
            (
                "qknorm_across_heads_rope",
                f"QKNormAcrossHeadsRopeKernel<{args}>::run",
            )
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_fused_inplace_qknorm_across_heads_rope(
    head_dim: int,
    rope_dim: int,
    dtype: torch.dtype,
) -> bool:
    """Return True if the across-heads QK-Norm + RoPE fused kernel supports the shape.

    Mirrors the gates in :func:`sglang.jit_kernel.diffusion.qknorm_rope.can_use_fused_inplace_qknorm_rope`.
    """
    if head_dim not in (64, 128, 256):
        logger.warning(
            f"Unsupported head_dim={head_dim} for JIT fused QKNorm-across-heads+RoPE"
        )
        return False
    if rope_dim <= 0 or rope_dim > head_dim:
        logger.warning(
            f"Unsupported rope_dim={rope_dim} for head_dim={head_dim} "
            "in fused QKNorm-across-heads+RoPE"
        )
        return False
    if rope_dim % 2 != 0:
        logger.warning(
            f"rope_dim={rope_dim} must be even for fused QKNorm-across-heads+RoPE"
        )
        return False
    # The per-thread vec width determines minimum alignment for head_dim
    # and rope_dim. bf16/fp16 with the 32-byte vec path needs 16-element
    # alignment; the 16-byte path needs 8. ``head_dim`` >= 64 with the
    # required power-of-two layout satisfies both, so we check the 16-byte
    # case explicitly and let the kernel reject anything narrower at launch.
    if head_dim % 8 != 0 or rope_dim % 8 != 0:
        logger.warning(
            "head_dim and rope_dim must be multiples of 8 for fused "
            "QKNorm-across-heads+RoPE"
        )
        return False
    try:
        _jit_qknorm_across_heads_rope_module(head_dim, rope_dim, dtype)
        return True
    except Exception as e:
        logger.warning(
            f"Failed to load JIT fused QKNorm-across-heads+RoPE kernel: {e}"
        )
        return False


@register_custom_op(mutates_args=["q", "k"])
def fused_inplace_qknorm_across_heads_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    eps: float = 1e-6,
    head_dim: int,
    rope_dim: int = 0,
) -> None:
    """Fused in-place across-heads QK-Norm + interleave (GPT-J) RoPE.

    Args:
        q: ``[num_tokens, num_heads * head_dim]``, fp16 or bf16, contiguous.
        k: ``[num_tokens, num_heads * head_dim]``, fp16 or bf16, contiguous.
        q_weight: ``[num_heads * head_dim]`` matching dtype.
        k_weight: ``[num_heads * head_dim]`` matching dtype.
        cos_sin_cache: ``[max_pos, rope_dim]`` float32, layout
            ``[cos[0..rope_dim/2-1] | sin[0..rope_dim/2-1]]`` along the last
            dim (FlashInfer convention).
        positions: ``[num_tokens]`` int32 or int64.
        eps: RMSNorm epsilon.
        head_dim: per-head dimension (required; the kernel applies RoPE on a
            head-dim window of ``q``/``k``).
        rope_dim: RoPE dimension, defaults to ``cos_sin_cache.size(-1)``.

    Both ``q`` and ``k`` are written in-place.
    """
    rope_dim = rope_dim or cos_sin_cache.size(-1)
    module = _jit_qknorm_across_heads_rope_module(head_dim, rope_dim, q.dtype)
    module.qknorm_across_heads_rope(
        q, k, q_weight, k_weight, cos_sin_cache, positions, eps
    )
