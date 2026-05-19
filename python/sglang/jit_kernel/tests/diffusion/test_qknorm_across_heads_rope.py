"""CUDA parity test for the fused across-heads QK-Norm + RoPE kernel.

Mirrors ``test_qknorm_rope.py`` (the per-head sibling). Compares:

  baseline (split): fused_inplace_qknorm_across_heads + flashinfer
                    apply_rope_with_cos_sin_cache_inplace
  fused          : fused_inplace_qknorm_across_heads_rope

The baseline reproduces exactly what Wan / Sana / Helios do in
``WanTransformerBlock.forward`` today — RMSNorm across the full hidden dim
on each of q,k followed by FlashInfer RoPE on the unflattened
``[B*L, num_heads, head_dim]`` view. The fused kernel collapses both into a
single in-place pass.

Note RoPE in this fused path is interleave / GPT-J style (``is_neox=False``)
only; across-heads diffusion configs (Wan / Sana / Helios) all use
interleave. The fallback path inside
``apply_qk_norm_across_heads_with_optional_rope`` handles NeoX correctly via
the existing FlashInfer wrapper, so the fused kernel never has to.
"""

import itertools
import sys

import pytest
import torch
import triton

from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=80, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_SEQ_LEN = 131072
ROPE_BASE = 10000.0
ATOL = 8e-2
RTOL = 1e-2


def create_cos_sin_cache(
    rotary_dim: int,
    max_position: int = MAX_SEQ_LEN,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=DEVICE)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, dtype=torch.float32, device=DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def split_qknorm_across_heads_rope(
    q: torch.Tensor,  # [N, num_heads * head_dim]
    k: torch.Tensor,
    q_weight: torch.Tensor,  # [num_heads * head_dim]
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> None:
    """Reference path: across-heads RMSNorm + FlashInfer RoPE on [N, num_heads, head_dim]."""
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    from sglang.jit_kernel.norm import fused_inplace_qknorm_across_heads

    fused_inplace_qknorm_across_heads(q, k, q_weight, k_weight)
    apply_rope_with_cos_sin_cache_inplace(
        positions=positions.long(),
        query=q.view(q.shape[0], -1),
        key=k.view(k.shape[0], -1),
        head_size=head_dim,
        cos_sin_cache=cos_sin_cache,
        is_neox=False,
    )


def fused_qknorm_across_heads_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> None:
    from sglang.jit_kernel.diffusion.qknorm_across_heads_rope import (
        fused_inplace_qknorm_across_heads_rope,
    )

    fused_inplace_qknorm_across_heads_rope(
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        head_dim=head_dim,
        rope_dim=cos_sin_cache.shape[-1],
    )


# Wan / Sana / Helios production configurations share head_dim=128, but Wan
# uses 24 heads (dim=3072) while Sana ~14 heads at varying widths. Test a
# representative spread plus the CI-prioritized 24/128 path.
BS_LIST = [1, 16, 256, 4096]
BS_LIST = get_ci_test_range(BS_LIST, [1, 16, 4096])
NUM_HEADS_LIST = get_ci_test_range([8, 16, 24, 32], [16, 24])
HEAD_DIM_LIST = get_ci_test_range([64, 128, 256], [64, 128])
POSITION_DTYPES = [torch.int32, torch.int64]
ROPE_DIM_CHOICES = {
    64: [64],
    128: [64, 128],
    256: [128, 256],
}


@pytest.mark.parametrize(
    "batch_size,num_heads,head_dim,position_dtype",
    list(
        itertools.product(
            BS_LIST,
            NUM_HEADS_LIST,
            HEAD_DIM_LIST,
            POSITION_DTYPES,
        )
    ),
)
def test_qknorm_across_heads_rope(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    position_dtype: torch.dtype,
) -> None:
    rope_dims = ROPE_DIM_CHOICES[head_dim]
    hidden_size = num_heads * head_dim

    # Cap hidden_size to the kernel's supported range (see qknorm_across_heads.cuh
    # max bound). 8192 for 16-byte vec, 12288 for 32-byte vec; this kernel uses
    # device::kMaxVecBytes so the upper bound floats with the arch.
    if hidden_size > 8192:
        pytest.skip(f"hidden_size={hidden_size} exceeds across-heads kernel limit")

    for rope_dim in rope_dims:
        q = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=DTYPE)
        k = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=DTYPE)
        q_weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE)
        k_weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE)
        positions = torch.randint(
            0, MAX_SEQ_LEN, (batch_size,), device=DEVICE, dtype=position_dtype
        )
        cos_sin_cache = create_cos_sin_cache(rope_dim)

        q_ref, k_ref = q.clone(), k.clone()
        q_fused, k_fused = q.clone(), k.clone()

        split_qknorm_across_heads_rope(
            q_ref,
            k_ref,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            num_heads,
            head_dim,
        )
        fused_qknorm_across_heads_rope(
            q_fused,
            k_fused,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            num_heads,
            head_dim,
        )

        # Same BF16-rounding-step tolerance as the per-head sibling test —
        # split baseline = separate BF16 norm + separate FlashInfer rope,
        # fused path keeps elements in fp32 registers across both phases.
        triton.testing.assert_close(q_ref, q_fused, atol=ATOL, rtol=RTOL)
        triton.testing.assert_close(k_ref, k_fused, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
