#!/usr/bin/env python3
"""E214: benchmark the post-hoc SafeConf routing layer."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/实验结果/E214_runtime_scaling_20260918"
SIZES = (1_000, 10_000, 100_000, 1_000_000, 5_000_000)
WORKERS = (1, 2, 4, 8, 16, 32)
SEED = 20_260_918


class BenchmarkFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--parallel-batches", type=int, default=32)
    parser.add_argument("--parallel-batch-size", type=int, default=250_000)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False, float_format="%.17g"))


def feature_arrays(size: int, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=size)
    magnitude = np.abs(0.65 * latent + rng.normal(scale=0.75, size=size))
    safeconf = 0.35 * latent + rng.normal(scale=0.95, size=size)
    disagreement = np.abs(0.45 * latent + rng.normal(scale=0.85, size=size))
    history_scarcity = rng.gamma(shape=1.7, scale=0.8, size=size)
    context_novelty = np.abs(rng.normal(size=size))
    return magnitude, safeconf, disagreement, history_scarcity, context_novelty


def zscore(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 0:
        raise BenchmarkFailure("invalid feature scale")
    return (values - float(np.mean(values))) / scale


def routing_kernel(features: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    magnitude, safeconf, disagreement, scarcity, novelty = features
    base = (
        zscore(safeconf)
        + zscore(disagreement)
        + zscore(scarcity)
        + zscore(novelty)
    ) / 4.0
    magnitude_rank = rankdata(magnitude, method="average") / len(magnitude)
    safeconf_rank = rankdata(base, method="average") / len(base)
    fixed = 0.8 * magnitude_rank + 0.2 * safeconf_rank
    one_sided = magnitude_rank + 0.25 * np.maximum(
        safeconf_rank - magnitude_rank, 0.0
    )
    if not np.isfinite(fixed).all() or not np.isfinite(one_sided).all():
        raise BenchmarkFailure("routing kernel produced non-finite output")
    return fixed, one_sided


def one_timing(size: int, seed: int) -> tuple[float, float]:
    features = feature_arrays(size, seed)
    started = time.perf_counter()
    fixed, one_sided = routing_kernel(features)
    elapsed = time.perf_counter() - started
    checksum = float(fixed.sum() + one_sided.sum())
    return elapsed, checksum


def single_batch_benchmark(sizes: tuple[int, ...], repeats: int) -> pd.DataFrame:
    rows = []
    for size in sizes:
        values = []
        checksum = None
        for repeat in range(repeats):
            elapsed, checksum = one_timing(size, SEED + size + repeat)
            values.append(elapsed)
        rows.append(
            {
                "n_tasks": size,
                "repeats": repeats,
                "median_seconds": statistics.median(values),
                "q25_seconds": float(np.quantile(values, 0.25)),
                "q75_seconds": float(np.quantile(values, 0.75)),
                "tasks_per_second": size / statistics.median(values),
                "input_bytes": size * 5 * 8,
                "output_bytes": size * 2 * 8,
                "checksum_last": checksum,
            }
        )
    return pd.DataFrame(rows)


def parallel_job(arguments: tuple[int, int]) -> tuple[float, float]:
    size, seed = arguments
    return one_timing(size, seed)


def parallel_benchmark(
    batch_size: int, n_batches: int, max_workers: int
) -> pd.DataFrame:
    rows = []
    candidates = tuple(value for value in WORKERS if value <= max_workers)
    for workers in candidates:
        arguments = [
            (batch_size, SEED + workers * 100_003 + index)
            for index in range(n_batches)
        ]
        started = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(parallel_job, arguments))
        wall = time.perf_counter() - started
        if not all(np.isfinite(value) for result in results for value in result):
            raise BenchmarkFailure("parallel result contains non-finite value")
        rows.append(
            {
                "workers": workers,
                "n_batches": n_batches,
                "batch_size": batch_size,
                "total_tasks": n_batches * batch_size,
                "wall_seconds": wall,
                "tasks_per_second": n_batches * batch_size / wall,
                "median_worker_kernel_seconds": statistics.median(
                    result[0] for result in results
                ),
            }
        )
    frame = pd.DataFrame(rows)
    baseline = float(frame.loc[frame.workers.eq(1), "tasks_per_second"].iloc[0])
    frame["speedup_vs_one_worker"] = frame.tasks_per_second / baseline
    frame["parallel_efficiency"] = frame.speedup_vs_one_worker / frame.workers
    return frame


def render(single: pd.DataFrame, parallel: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), facecolor="white")
    axes[0].plot(
        single.n_tasks,
        single.median_seconds,
        marker="o",
        color="#118A7E",
        linewidth=1.8,
    )
    axes[0].fill_between(
        single.n_tasks,
        single.q25_seconds,
        single.q75_seconds,
        color="#118A7E",
        alpha=0.18,
        linewidth=0,
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Tasks per batch")
    axes[0].set_ylabel("Routing latency (s)")
    axes[0].text(-0.15, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=13)

    axes[1].plot(
        parallel.workers,
        parallel.tasks_per_second,
        marker="o",
        color="#54769A",
        linewidth=1.8,
        label="Observed throughput",
    )
    axes[1].set_xlabel("CPU workers")
    axes[1].set_ylabel("Tasks per second")
    axes[1].set_xticks(parallel.workers)
    axes[1].text(-0.15, 1.04, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=13)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(color="#E6E8EB", linewidth=0.6)
        axis.set_axisbelow(True)
    fig.tight_layout(w_pad=2.5)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(
            figures / f"E214_RUNTIME_SCALING.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return platform.processor()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def main() -> None:
    args = parse_args()
    if args.self_test:
        fixed, one = routing_kernel(feature_arrays(100, SEED))
        assert fixed.shape == (100,) and one.shape == (100,)
        assert np.isfinite(fixed).all() and np.isfinite(one).all()
        assert np.all(one >= rankdata(feature_arrays(100, SEED)[0]) / 100)
        print("PASS")
        return
    if args.output.exists() and any(args.output.iterdir()):
        allowed = {"BENCHMARK_CONTRACT.md"}
        unexpected = {path.name for path in args.output.iterdir()} - allowed
        if unexpected:
            raise BenchmarkFailure(f"refusing non-empty output: {args.output}")
    if args.repeats < 2 or args.max_workers < 1:
        raise BenchmarkFailure("invalid repeats or workers")
    sizes = (1_000, 10_000, 100_000) if args.quick else SIZES
    batches = 4 if args.quick else args.parallel_batches
    batch_size = 10_000 if args.quick else args.parallel_batch_size
    max_workers = min(4, args.max_workers) if args.quick else args.max_workers
    single = single_batch_benchmark(sizes, args.repeats)
    parallel = parallel_benchmark(batch_size, batches, max_workers)
    tables = args.output / "tables"
    atomic_csv(tables / "E214_SINGLE_BATCH_SCALING.csv", single)
    atomic_csv(tables / "E214_PARALLEL_SCALING.csv", parallel)
    render(single, parallel, args.output)

    million = single.loc[single.n_tasks.eq(max(single.n_tasks))].iloc[0]
    best = parallel.loc[parallel.tasks_per_second.idxmax()]
    report = f"""# E214｜风险路由计算开销与规模扩展

本实验只测预测完成后的 SafeConf 风险路由层，不包含上游模型训练和推断。

- 最大单批规模：{int(million.n_tasks):,} 个任务；中位延迟 {million.median_seconds:.3f} 秒；吞吐 {million.tasks_per_second:,.0f} 任务/秒。
- 并行测试最高吞吐：{best.tasks_per_second:,.0f} 任务/秒（{int(best.workers)} 个 CPU worker）。
- 单批输入数组理论占用为 {int(million.input_bytes) / 2**20:.1f} MiB，两个输出分数为 {int(million.output_bytes) / 2**20:.1f} MiB；排序过程还会产生临时数组，因此这不是进程峰值内存上限。

完整数据见 `tables/`，图见 `figures/E214_RUNTIME_SCALING.*`。数字只适用于状态文件记录的当前服务器和软件环境。
"""
    atomic_text(args.output / "E214_REPORT.md", report)
    status = {
        "experiment": "E214_runtime_scaling",
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "sizes": list(sizes),
        "repeats": args.repeats,
        "parallel_batches": batches,
        "parallel_batch_size": batch_size,
        "max_workers": max_workers,
        "upstream_model_time_included": False,
    }
    atomic_text(
        args.output / "RUN_STATUS.json",
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
    )


if __name__ == "__main__":
    main()

