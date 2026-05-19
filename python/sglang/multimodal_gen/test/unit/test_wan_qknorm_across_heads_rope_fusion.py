"""CPU parity tests for the across-heads QK-Norm + RoPE Python helper.

The kernel itself is CUDA-only (see
``python/sglang/jit_kernel/tests/diffusion/test_qknorm_across_heads_rope.py``
for the GPU side). On the CPU collection harness, the fused fast path inside
``apply_qk_norm_across_heads_with_optional_rope`` is gated off and the helper
falls back to the same recipe Wan / Sana / Helios run today:

    1. RMSNorm across the full hidden_size (per-token, weight ``[hidden]``).
    2. Reshape to ``[B, L, num_heads, head_dim]``.
    3. FlashInfer-style RoPE (interleave / GPT-J).

These tests pin that fallback against an explicit eager reference, so:

  * the helper does the right thing on CPU (no CUDA-only branches fire);
  * Wan's wired call-site preserves pre-/post-refactor math.

The Wan / Sana / Helios fast path on real GPUs is covered separately by the
CUDA test referenced above.
"""

from __future__ import annotations

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    apply_qk_norm_across_heads_with_optional_rope,
)


# -- Local copy of the GPT-J rotary kernel ---------------------------------


def _apply_rotary_emb_eager(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Interleave / GPT-J style rotary.

    ``x``  has shape ``[..., head_dim]`` and we apply rotation to the first
    ``cos.shape[-1] * 2`` channels (so ``cos`` is ``[seq, head_dim // 2]``).
    Same arithmetic as ``apply_rotary_embedding_native`` in
    ``sglang/jit_kernel/diffusion/triton/torch_fallback.py`` --
    keeping a local copy here to avoid pulling in the GPU-only
    ``rotary_embedding/utils.py`` import chain at collection time.
    """
    cos_b = cos.unsqueeze(-2).to(x.dtype)
    sin_b = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos_b - x2 * sin_b
    o2 = x2 * cos_b + x1 * sin_b
    return torch.stack((o1, o2), dim=-1).flatten(-2)


# -- Fixtures -------------------------------------------------------------


def _make_qk_norms(
    hidden_size: int, dtype: torch.dtype
) -> tuple[RMSNorm, RMSNorm]:
    """Make a (q_norm, k_norm) pair with non-trivial weights.

    Across-heads variant: weight shape is ``[hidden_size]``, not ``[head_dim]``.
    """
    torch.manual_seed(0)
    q_norm = RMSNorm(hidden_size, eps=1e-6, dtype=dtype)
    k_norm = RMSNorm(hidden_size, eps=1e-6, dtype=dtype)
    with torch.no_grad():
        q_norm.weight.copy_(torch.empty(hidden_size).uniform_(0.9, 1.1))
        k_norm.weight.copy_(torch.empty(hidden_size).uniform_(0.85, 1.15))
    return q_norm, k_norm


def _make_cos_sin(
    num_tokens: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (cos, sin, cos_sin_cache) matching the FlashInfer layout."""
    torch.manual_seed(1)
    half = head_dim // 2
    cos = torch.randn(num_tokens, half) * 0.1 + 0.9
    sin = torch.randn(num_tokens, half) * 0.1
    cos_sin_cache = torch.cat(
        [cos.to(torch.float32).contiguous(), sin.to(torch.float32).contiguous()],
        dim=-1,
    )
    return cos, sin, cos_sin_cache


# -- Parity: RMSNorm-then-rope == fused helper ----------------------------


@pytest.mark.parametrize(
    "num_tokens,num_heads,head_dim",
    [
        # Wan-ish: 24 heads * 128 = 3072 hidden
        (64, 4, 64),
        # Bigger token count
        (128, 6, 32),
        # Multi-head head_dim=128 (production Wan)
        (32, 8, 128),
    ],
)
def test_across_heads_norm_rope_parity(num_tokens, num_heads, head_dim):
    """eager (norm + rope) == fused helper (cos_sin_cache path)."""
    dtype = torch.float32  # CPU eager parity = exact in fp32
    hidden_size = num_heads * head_dim
    q_norm, k_norm = _make_qk_norms(hidden_size, dtype)
    cos, sin, cos_sin_cache = _make_cos_sin(num_tokens, head_dim)

    torch.manual_seed(2)
    q = torch.randn(num_tokens, hidden_size, dtype=dtype)
    k = torch.randn(num_tokens, hidden_size, dtype=dtype)

    # -- Eager reference ---------------------------------------------------
    ref_q = q_norm(q.contiguous())
    ref_k = k_norm(k.contiguous())
    ref_q4d = ref_q.view(1, num_tokens, num_heads, head_dim)
    ref_k4d = ref_k.view(1, num_tokens, num_heads, head_dim)
    ref_q4d = _apply_rotary_emb_eager(ref_q4d, cos, sin)
    ref_k4d = _apply_rotary_emb_eager(ref_k4d, cos, sin)
    ref_q_out = ref_q4d.reshape(num_tokens, hidden_size)
    ref_k_out = ref_k4d.reshape(num_tokens, hidden_size)

    # -- Fused-helper path (falls back to eager on CPU) --------------------
    out_q = q.clone().contiguous()
    out_k = k.clone().contiguous()
    out_q, out_k = apply_qk_norm_across_heads_with_optional_rope(
        q=out_q,
        k=out_k,
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=head_dim,
        cos_sin_cache=cos_sin_cache,
        allow_inplace=True,
    )

    assert out_q.shape == ref_q_out.shape
    assert out_q.dtype == ref_q_out.dtype
    torch.testing.assert_close(out_q, ref_q_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_k, ref_k_out, atol=1e-5, rtol=1e-5)


# -- No-RoPE path: fused helper falls back to fused_inplace_qknorm_across_heads or eager
@pytest.mark.parametrize(
    "num_tokens,hidden_size",
    [
        (16, 256),
        (64, 512),
    ],
)
def test_no_rope_path_matches_rmsnorm(num_tokens, hidden_size):
    """cos_sin_cache=None routes to plain across-heads RMSNorm (no rope)."""
    dtype = torch.float32
    q_norm, k_norm = _make_qk_norms(hidden_size, dtype)
    torch.manual_seed(5)
    q = torch.randn(num_tokens, hidden_size, dtype=dtype)
    k = torch.randn(num_tokens, hidden_size, dtype=dtype)

    ref_q = q_norm(q.contiguous())
    ref_k = k_norm(k.contiguous())

    out_q, out_k = apply_qk_norm_across_heads_with_optional_rope(
        q=q.clone().contiguous(),
        k=k.clone().contiguous(),
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=32,
        cos_sin_cache=None,
        allow_inplace=True,
    )

    torch.testing.assert_close(out_q, ref_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_k, ref_k, atol=1e-5, rtol=1e-5)


# -- Wan wiring smoke test -------------------------------------------------


def test_wanvideo_module_wires_fused_helper():
    """After the refactor, wanvideo.py imports the new helper at module scope."""
    import sglang.multimodal_gen.runtime.models.dits.wanvideo as wan

    assert hasattr(wan, "apply_qk_norm_across_heads_with_optional_rope")
    # The wiring keeps the legacy code paths around for the TP-rmsnorm and
    # per-head config branches -- those continue to exist as a fallback.
    assert hasattr(wan, "tensor_parallel_rms_norm")
    assert hasattr(wan, "apply_flashinfer_rope_qk_inplace")
