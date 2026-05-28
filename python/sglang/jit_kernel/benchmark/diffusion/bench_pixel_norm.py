"""Benchmark fused PixelNorm against the eager PyTorch implementation.

Cases are sized after the LTX-2 family production call sites:

* Video VAE residual blocks (5D ``[B, C, F, H, W]``) at multiple resolutions.
* Audio VAE / attention blocks (4D ``[B, C, H, W]``).
* LTX-2.3 condition encoder output (small 5D).

Run::

    python python/sglang/jit_kernel/benchmark/diffusion/bench_pixel_norm.py

Use ``--cases all-large`` to additionally cover the 720p decoder-output case
(``ltx2_video_720p``) which allocates ~3 GB and is not included in the default
CI sweep.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import triton.testing

from sglang.jit_kernel.diffusion.triton.pixel_norm import (
    _pixel_norm_native,
    triton_pixel_norm,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(
    est_time=30,
    suite="base-b-kernel-benchmark-1-gpu-large",
    disabled="standalone benchmark",
)

DEVICE = "cuda"
EPS = 1e-8
QUANTILES = [0.5, 0.2, 0.8]


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    channel_dim: int


# Default CI-friendly cases. These cover all production call sites except the
# 720p decoder-output (kept in LARGE_CASES to bound memory).
CASES = [
    Case("ltx2_video_mid_dec", (1, 256, 7, 22, 40), 1),
    Case("ltx2_video_mid_enc", (1, 512, 7, 11, 20), 1),
    Case("ltx2_audio_attn", (1, 128, 256, 256), 1),
    Case("ltx2_audio_resblock", (1, 64, 64, 64), 1),
    Case("ltx23_cond_encoder", (1, 128, 7, 11, 20), 1),
]

# Memory-heavy cases — opt in via ``--cases all-large``.
LARGE_CASES = [
    Case("ltx2_video_720p", (1, 128, 28, 90, 160), 1),
]

CASE_BY_NAME = {case.name: case for case in CASES + LARGE_CASES}


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[name]


def dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }[dtype]


def parse_dtypes(text: str) -> list[torch.dtype]:
    return [dtype_from_name(item.strip()) for item in text.split(",") if item.strip()]


def parse_cases(text: str) -> list[Case]:
    if text == "all":
        return CASES
    if text == "large":
        return LARGE_CASES
    if text == "all-large":
        return CASES + LARGE_CASES
    names = [item.strip() for item in text.split(",") if item.strip()]
    missing = sorted(set(names) - CASE_BY_NAME.keys())
    if missing:
        raise ValueError(f"Unknown cases: {missing}")
    return [CASE_BY_NAME[name] for name in names]


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.bfloat16:
        return 1e-2, 1e-2
    return 1e-3, 1e-3


def native_pixel_norm(x: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """The (pre-fusion) production eager implementation. Three kernel launches
    in the input dtype: ``x**2``, ``mean``, ``sqrt + div``."""
    mean_sq = torch.mean(x**2, dim=channel_dim, keepdim=True)
    rms = torch.sqrt(mean_sq + EPS)
    return x / rms


def make_input(case: Case, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(case.channel_dim * 1009 + sum(case.shape))
    return torch.randn(case.shape, device=DEVICE, dtype=dtype, generator=generator)


def do_bench_us(fn: Callable[[], object], warmup: int, rep: int) -> tuple[float, ...]:
    median_ms, p20_ms, p80_ms = triton.testing.do_bench(
        fn,
        quantiles=QUANTILES,
        warmup=warmup,
        rep=rep,
    )
    return median_ms * 1000.0, p20_ms * 1000.0, p80_ms * 1000.0


def summarize(values: list[float]) -> float:
    return statistics.median(values)


def run_case(
    case: Case,
    dtype: torch.dtype,
    rounds: int,
    warmup: int,
    rep: int,
) -> dict[str, object]:
    x = make_input(case, dtype)

    with torch.inference_mode():
        # Sanity: kernel must agree with the fp32-accumulating native reference.
        actual = triton_pixel_norm(x, channel_dim=case.channel_dim, eps=EPS)
        expected = _pixel_norm_native(x, channel_dim=case.channel_dim, eps=EPS)
        atol, rtol = tolerance(dtype)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

        native_stats, fused_stats = [], []
        for _ in range(rounds):
            native_stats.append(
                do_bench_us(
                    lambda: native_pixel_norm(x, case.channel_dim),
                    warmup=warmup,
                    rep=rep,
                )
            )
            fused_stats.append(
                do_bench_us(
                    lambda: triton_pixel_norm(
                        x, channel_dim=case.channel_dim, eps=EPS
                    ),
                    warmup=warmup,
                    rep=rep,
                )
            )

    native_median_us = summarize([s[0] for s in native_stats])
    fused_median_us = summarize([s[0] for s in fused_stats])
    torch.cuda.empty_cache()
    return {
        "case": case.name,
        "shape": "x".join(str(d) for d in case.shape),
        "channel_dim": case.channel_dim,
        "dtype": dtype_name(dtype),
        "native_median_us": native_median_us,
        "native_p20_us": summarize([s[1] for s in native_stats]),
        "native_p80_us": summarize([s[2] for s in native_stats]),
        "fused_median_us": fused_median_us,
        "fused_p20_us": summarize([s[1] for s in fused_stats]),
        "fused_p80_us": summarize([s[2] for s in fused_stats]),
        "speedup": native_median_us / fused_median_us,
        "rounds": rounds,
        "warmup": warmup,
        "rep": rep,
    }


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict[str, object]]) -> None:
    header = ("case", "dtype", "shape", "native_us", "fused_us", "speedup")
    print("| " + " | ".join(header) + " |")
    print("|---|---|---|---:|---:|---:|")
    for row in rows:
        print(
            "| {case} | {dtype} | {shape} | {native:.2f} | {fused:.2f} | {speedup:.3f}x |".format(
                case=row["case"],
                dtype=row["dtype"],
                shape=row["shape"],
                native=row["native_median_us"],
                fused=row["fused_median_us"],
                speedup=row["speedup"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fused PixelNorm against eager PyTorch PixelNorm. Use "
            "--cases all-large to additionally include the 720p decoder-output "
            "shape (memory-heavy)."
        )
    )
    parser.add_argument("--cases", default="all")
    parser.add_argument("--dtypes", default="bf16,fp16")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    cases = parse_cases(args.cases)
    dtypes = parse_dtypes(args.dtypes)

    rows = []
    for case in cases:
        for dtype in dtypes:
            rows.append(run_case(case, dtype, args.rounds, args.warmup, args.rep))

    print_rows(rows)
    if args.output_csv:
        write_csv(rows, Path(args.output_csv))
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    if is_in_ci():
        print("Skipping bench_pixel_norm.py in CI")
        sys.exit(0)
    main()
