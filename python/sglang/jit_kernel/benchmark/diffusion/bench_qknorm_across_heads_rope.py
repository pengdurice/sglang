"""Speed benchmark for the fused across-heads QK-Norm + RoPE kernel.

Counterpart to ``bench_qknorm_rope.py`` (per-head variant). Compares:

  split : fused_inplace_qknorm_across_heads + flashinfer
          apply_rope_with_cos_sin_cache_inplace   (Wan / Sana / Helios
          today's path)
  fused : fused_inplace_qknorm_across_heads_rope (this PR)

The split provider mimics what Wan's ``WanTransformerBlock.forward`` runs
in production: one BF16 RMSNorm kernel over the full hidden dim per token,
followed by one FlashInfer RoPE pass on the unflattened
``[N, num_heads, head_dim]`` view. The fused provider collapses both into a
single in-place pass.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    get_benchmark_range,
    run_benchmark_no_cudagraph,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-kernel-benchmark-1-gpu-large")

MAX_SEQ_LEN = 131072
ROPE_BASE = 10000.0


@dataclass(frozen=True)
class CaseSpec:
    name: str
    num_tokens: int
    num_heads: int
    head_dim: int
    rope_dim: int


# Wan2.1 14B / Wan2.2 dim=3072 (num_heads=24, head_dim=128).
# Sana-1.6B-1024 dim=2240 (num_heads=14, head_dim=160)
#   -> head_dim=160 is outside the JIT support set (64/128/256), so we
#      pick the closest Sana-style 1024 config that lands on 128 in the
#      bench (full Sana support depends on a follow-up to relax the
#      head_dim gate or compile a 160-wide template).
# Helios dim=2048 (num_heads=16, head_dim=128).
#
# Token counts cover a small image, a medium frame stack, and a longer
# video latent.
BENCH_CASES = (
    # Wan-shape, image-scale token count
    CaseSpec("wan_4096", 4096, 24, 128, 128),
    # Wan-shape, ~5s clip
    CaseSpec("wan_8192", 8192, 24, 128, 128),
    # Wan-shape, longer clip
    CaseSpec("wan_24576", 24576, 24, 128, 128),
    # Helios-shape, image-scale
    CaseSpec("helios_4096", 4096, 16, 128, 128),
    # Partial rope (rope_dim < head_dim) -- exercises the non-rotated tail
    CaseSpec("wan_partial_rope", 4096, 24, 128, 64),
)
CASE_BY_NAME = {case.name: case for case in BENCH_CASES}
CASE_NAMES = get_benchmark_range(
    full_range=[case.name for case in BENCH_CASES],
    ci_range=[case.name for case in BENCH_CASES],
)
LINE_VALS = ["split", "fused"]
LINE_NAMES = [
    "AcrossHeads QKNorm + FlashInfer RoPE",
    "SGL JIT Fused AcrossHeads QKNorm+RoPE",
]
STYLES = [("red", "-"), ("blue", "--")]


def create_cos_sin_cache(
    rotary_dim: int,
    max_position: int = MAX_SEQ_LEN,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=DEFAULT_DEVICE)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, dtype=torch.float32, device=DEFAULT_DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def make_inputs(case: CaseSpec) -> dict[str, torch.Tensor | int]:
    hidden_size = case.num_heads * case.head_dim
    seed = (
        case.num_tokens * 1_000_003
        + case.num_heads * 8191
        + case.head_dim * 127
        + case.rope_dim
    )
    generator = torch.Generator(device=DEFAULT_DEVICE)
    generator.manual_seed(seed)
    return {
        "q": torch.randn(
            case.num_tokens,
            hidden_size,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            generator=generator,
        ),
        "k": torch.randn(
            case.num_tokens,
            hidden_size,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            generator=generator,
        ),
        "q_weight": torch.randn(
            hidden_size,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            generator=generator,
        ),
        "k_weight": torch.randn(
            hidden_size,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            generator=generator,
        ),
        "positions": torch.randint(
            0,
            MAX_SEQ_LEN,
            (case.num_tokens,),
            device=DEFAULT_DEVICE,
            dtype=torch.int64,
            generator=generator,
        ),
        "cos_sin_cache": create_cos_sin_cache(case.rope_dim),
        "head_dim": case.head_dim,
        "num_heads": case.num_heads,
    }


def clone_inputs(
    inputs: dict[str, torch.Tensor | int],
) -> dict[str, torch.Tensor | int]:
    out: dict[str, torch.Tensor | int] = {}
    for key, value in inputs.items():
        out[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return out


def split_across_heads_qknorm_rope(inputs: dict[str, torch.Tensor | int]) -> None:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    from sglang.jit_kernel.norm import fused_inplace_qknorm_across_heads

    q = inputs["q"]
    k = inputs["k"]
    fused_inplace_qknorm_across_heads(
        q, k, inputs["q_weight"], inputs["k_weight"]
    )
    apply_rope_with_cos_sin_cache_inplace(
        positions=inputs["positions"].long(),
        query=q.view(q.shape[0], -1),
        key=k.view(k.shape[0], -1),
        head_size=int(inputs["head_dim"]),
        cos_sin_cache=inputs["cos_sin_cache"],
        is_neox=False,
    )


def fused_across_heads_qknorm_rope(inputs: dict[str, torch.Tensor | int]) -> None:
    from sglang.jit_kernel.diffusion.qknorm_across_heads_rope import (
        fused_inplace_qknorm_across_heads_rope,
    )

    fused_inplace_qknorm_across_heads_rope(
        inputs["q"],
        inputs["k"],
        inputs["q_weight"],
        inputs["k_weight"],
        inputs["cos_sin_cache"],
        inputs["positions"],
        head_dim=int(inputs["head_dim"]),
        rope_dim=inputs["cos_sin_cache"].shape[-1],
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["case_name"],
        x_vals=CASE_NAMES,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="diffusion-qknorm-across-heads-rope-performance",
        args={},
    )
)
def benchmark(case_name: str, provider: str) -> Tuple[float, float, float]:
    case = CASE_BY_NAME[case_name]
    inputs = make_inputs(case)
    fn = (
        split_across_heads_qknorm_rope
        if provider == "split"
        else fused_across_heads_qknorm_rope
    )
    return run_benchmark_no_cudagraph(lambda: fn(inputs))


if __name__ == "__main__":
    print(
        "Running diffusion across-heads qknorm + rope performance benchmark..."
    )
    benchmark.run(print_data=True)
