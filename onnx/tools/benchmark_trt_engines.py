#!/usr/bin/env python3
"""
Benchmark all TensorRT engines under onnx_exports/<quant_mode>.

For each quantisation directory the script discovers *.engine files,
measures inference latency using the optimized runner from
`onnx/trt_inference.py`, and summarises FPS rankings. This is intended
to help compare precision/quantisation trade-offs after running the
conversion pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse the TensorRT runner + helpers from the existing inference script.
from trt_inference import (  # type: ignore[import]
    SimpleTrtRunner,
    ensure_c_contig,
    load_images_nchw,
)


PRECISION_SUFFIX = {
    "fp32": "",
    "fp16": "_fp16",
    "bf16": "_bf16",
    "fp8": "_fp8",
    "int8": "_int8",
}


def detect_precision(engine_path: Path) -> str:
    """Best-effort precision classification from filename suffix."""
    name = engine_path.name
    # Check longer suffixes first to avoid matching the empty fp32 suffix prematurely.
    sorted_suffixes = sorted(
        PRECISION_SUFFIX.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    for prec, suffix in sorted_suffixes:
        if suffix:
            token = f"{suffix}.engine"
            if name.endswith(token):
                return prec
    # Fallback to fp32 (no suffix match)
    return "fp32"


def prepare_batch(
    runner: SimpleTrtRunner,
    *,
    images_dir: Optional[Path],
    use_random: bool,
    norm: str,
    engine_path: Path,
) -> np.ndarray:
    """Prepare an input batch matching the engine's expectations."""
    shape = tuple(runner.input["shape"])
    dtype = runner.input["np_dtype"]
    expect_n = shape[0]

    if not use_random and images_dir is not None:
        try:
            batch = load_images_nchw(
                str(images_dir),
                expect_n,
                size_hw=(shape[2], shape[3]),
                norm=norm,
                dtype=np.float32,
            )
        except Exception as exc:
            print(f"[WARN] Falling back to random input for {engine_path.name}: {exc}")
            batch = np.random.randn(*shape).astype(np.float32)
    else:
        batch = np.random.randn(*shape).astype(np.float32)

    if batch.dtype != dtype:
        batch = batch.astype(dtype, copy=False)

    return ensure_c_contig(batch, dtype)


def benchmark_engine(
    engine_path: Path,
    *,
    iters: int,
    warmup: int,
    images_dir: Optional[Path],
    use_random: bool,
    norm: str,
    cuda_events: bool,
    verbose_runner: bool,
) -> Dict[str, float]:
    """Run the latency benchmark for a single engine."""
    with SimpleTrtRunner(
        str(engine_path),
        verbose=verbose_runner,
    ) as runner:
        batch = prepare_batch(
            runner,
            images_dir=images_dir,
            use_random=use_random,
            norm=norm,
            engine_path=engine_path,
        )
        stats = runner.benchmark(
            batch,
            iters=iters,
            warmup=warmup,
            cuda_events=cuda_events,
        )

    # Normalise stat keys between CUDA-event and host timing paths.
    if cuda_events:
        latency_ms = float(stats["total_mean_ms"])
        jitter_ms = float(stats["total_std_ms"])
    else:
        latency_ms = float(stats["mean_ms"])
        jitter_ms = float(stats["std_ms"])

    fps = 1000.0 / latency_ms if latency_ms > 0 else float("inf")
    return {
        "mean_ms": latency_ms,
        "std_ms": jitter_ms,
        "fps": fps,
    }


def summarize(results: List[Dict[str, object]]) -> None:
    """Pretty-print the collected results ordered by FPS."""
    if not results:
        print("[INFO] No engines benchmarked.")
        return

    results_sorted = sorted(results, key=lambda r: r["fps"], reverse=True)

    print("\n=== TensorRT Engine Benchmark Summary ===")
    header = f"{'Rank':>4}  {'FPS':>8}  {'Latency(ms)':>14}  {'Quant':>18}  {'Precision':>9}  Engine"
    print(header)
    print("-" * len(header))

    for idx, row in enumerate(results_sorted, start=1):
        fps = row["fps"]
        mean_ms = row["mean_ms"]
        quant = row["quant_mode"]
        prec = row["precision"]
        engine = row["engine"]
        print(
            f"{idx:4d}  {fps:8.2f}  {mean_ms:14.2f}  {quant:>18}  {prec:>9}  {engine}"
        )

    best = results_sorted[0]
    print("\nFastest configuration:")
    print(
        f"  {best['quant_mode']} / {best['precision']} "
        f"→ {best['fps']:.2f} FPS ({best['mean_ms']:.2f} ms)"
    )

    over_30 = [r for r in results_sorted if r["fps"] >= 30.0]
    over_15 = [r for r in results_sorted if r["fps"] >= 15.0]
    print(f"\n≥30 FPS configurations: {len(over_30)}")
    print(f"≥15 FPS configurations: {len(over_15)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TensorRT engines under onnx_exports/<quant_mode>.",
    )
    parser.add_argument(
        "--root",
        default="onnx_exports",
        help="Root directory containing quantisation subdirectories.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Timed iterations per engine (default: 100).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations per engine (default: 20).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Optional image directory with exactly N frames (fallbacks to random).",
    )
    parser.add_argument(
        "--use-random",
        action="store_true",
        help="Force random input even if --images-dir is supplied.",
    )
    parser.add_argument(
        "--norm",
        choices=["none", "unit", "imagenet"],
        default="unit",
        help="Normalization to apply when loading images (default: unit).",
    )
    parser.add_argument(
        "--cuda-events",
        action="store_true",
        help="Use CUDA events for timing breakdown (default: host timing).",
    )
    parser.add_argument(
        "--verbose-runner",
        action="store_true",
        help="Enable verbose TensorRT runner logs.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to save raw results as JSON.",
    )

    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"[ERROR] Root directory does not exist: {root}")

    quant_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    results: List[Dict[str, object]] = []

    for quant_dir in quant_dirs:
        quant_mode = quant_dir.name
        engine_paths = sorted(quant_dir.glob("*.engine"))
        if not engine_paths:
            continue

        print(f"\n[INFO] Benchmarking quantisation mode '{quant_mode}'")
        for engine_path in engine_paths:
            precision = detect_precision(engine_path)
            try:
                stats = benchmark_engine(
                    engine_path,
                    iters=args.iters,
                    warmup=args.warmup,
                    images_dir=args.images_dir,
                    use_random=args.use_random,
                    norm=args.norm,
                    cuda_events=args.cuda_events,
                    verbose_runner=args.verbose_runner,
                )
            except Exception as exc:
                print(f"[ERROR] Benchmark failed for {engine_path}: {exc}")
                continue

            record: Dict[str, object] = {
                "quant_mode": quant_mode,
                "precision": precision,
                "engine": str(engine_path),
                "fps": stats["fps"],
                "mean_ms": stats["mean_ms"],
                "std_ms": stats["std_ms"],
            }
            results.append(record)
            print(
                f"  {precision:>9} → {stats['fps']:.2f} FPS "
                f"({stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms)  [{engine_path.name}]"
            )

    summarize(results)

    if args.json:
        try:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with args.json.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\n[INFO] Saved benchmark results to {args.json}")
        except Exception as exc:
            print(f"[WARN] Failed to save JSON: {exc}")


if __name__ == "__main__":
    main()
