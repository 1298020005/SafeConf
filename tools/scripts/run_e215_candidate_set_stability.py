#!/usr/bin/env python3
"""E215: audit rank-fusion stability under candidate-set subsampling."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "docs/实验结果/E153_eight_study_formal_meta_20260714/tables"
    / "E153_ABSOLUTE_TASK_INPUT.csv"
)
OUTPUT = ROOT / "docs/实验结果/E215_candidate_set_stability_20260918"
SCRATCH = Path("/home/yyf/data/safeconf_cpu/e215_20260918")
EXPECTED_SHA256 = "b75f5edae0bb585ba5ff18aecafcc2389b0f05fd5cc86b36960afb4b62e4a15a"
FRACTIONS = (0.25, 0.50, 0.75)
FORMULAS = ("fixed_80_20", "one_sided_025")
SEED = 20_260_918


class StabilityFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    parser.add_argument("--replicates", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False, float_format="%.17g"))


def ranks(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) < 4 or not np.isfinite(array).all():
        raise StabilityFailure("rank input is invalid")
    return rankdata(array, method="average") / len(array)


def formulas(magnitude: np.ndarray, safeconf: np.ndarray) -> dict[str, np.ndarray]:
    magnitude_rank, safeconf_rank = ranks(magnitude), ranks(safeconf)
    return {
        "fixed_80_20": 0.8 * magnitude_rank + 0.2 * safeconf_rank,
        "one_sided_025": magnitude_rank
        + 0.25 * np.maximum(safeconf_rank - magnitude_rank, 0.0),
    }


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = ranks(left), ranks(right)
    if np.std(left_rank) <= 0 or np.std(right_rank) <= 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def top_indices(score: np.ndarray, fraction: float = 0.20) -> set[int]:
    k = max(1, int(math.ceil(len(score) * fraction)))
    return set(np.argsort(score, kind="mergesort")[-k:].tolist())


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def tie_mean(score: np.ndarray, outcome: np.ndarray, fraction: float) -> float:
    k = max(1, int(math.ceil(len(score) * fraction)))
    threshold = np.sort(score)[-k]
    above, tied = score > threshold, score == threshold
    remaining = k - int(above.sum())
    return float((outcome[above].sum() + remaining * outcome[tied].mean()) / k)


def utility(score: np.ndarray, outcome: np.ndarray) -> float:
    selected = tie_mean(score, outcome, 0.20)
    oracle = tie_mean(outcome, outcome, 0.20)
    random = float(np.mean(outcome))
    return float((selected - random) / (oracle - random)) if oracle > random else float("nan")


def validate(path: Path) -> pd.DataFrame:
    if sha256_file(path) != EXPECTED_SHA256:
        raise StabilityFailure("E153 input hash changed")
    frame = pd.read_csv(path)
    required = {
        "dataset", "fold_id", "task_id", "perturbation",
        "baseline_predicted_magnitude", "safeconf_calibrated_pair_risk",
        "error_two_predictor_mean_rmse",
    }
    if len(frame) != 3_465 or frame.dataset.nunique() != 8:
        raise StabilityFailure("E153 dimensions changed")
    if not required.issubset(frame.columns):
        raise StabilityFailure("E153 columns changed")
    return frame


def fold_simulation(
    block: pd.DataFrame,
    dataset: str,
    fold_id: str,
    n_replicates: int,
) -> pd.DataFrame:
    block = block.reset_index(drop=True)
    full_scores = formulas(
        block.baseline_predicted_magnitude.to_numpy(float),
        block.safeconf_calibrated_pair_risk.to_numpy(float),
    )
    clusters = sorted(block.perturbation.astype(str).unique())
    cluster_members = {
        name: np.flatnonzero(block.perturbation.astype(str).to_numpy() == name)
        for name in clusters
    }
    label_seed = int(hashlib.sha256(f"{dataset}|{fold_id}".encode()).hexdigest()[:12], 16)
    rows = []
    for replicate in range(n_replicates):
        rng = np.random.default_rng(SEED + label_seed + 1_000_003 * replicate)
        for fraction in FRACTIONS:
            n_clusters = max(4, int(math.ceil(len(clusters) * fraction)))
            chosen = rng.choice(clusters, size=n_clusters, replace=False)
            positions = np.sort(
                np.concatenate([cluster_members[str(name)] for name in chosen])
            )
            sample = block.iloc[positions]
            recomputed = formulas(
                sample.baseline_predicted_magnitude.to_numpy(float),
                sample.safeconf_calibrated_pair_risk.to_numpy(float),
            )
            outcome = sample.error_two_predictor_mean_rmse.to_numpy(float)
            for formula in FORMULAS:
                reference = full_scores[formula][positions]
                current = recomputed[formula]
                rows.append(
                    {
                        "dataset": dataset,
                        "fold_id": fold_id,
                        "replicate": replicate,
                        "fraction": fraction,
                        "formula": formula,
                        "n_tasks": len(sample),
                        "n_clusters": n_clusters,
                        "score_spearman": spearman(current, reference),
                        "top20_jaccard": jaccard(
                            top_indices(current), top_indices(reference)
                        ),
                        "delta_outcome_spearman": spearman(current, outcome)
                        - spearman(reference, outcome),
                        "delta_utility20": utility(current, outcome)
                        - utility(reference, outcome),
                    }
                )
    return pd.DataFrame(rows)


def run_parallel(frame: pd.DataFrame, replicates: int, workers: int) -> pd.DataFrame:
    tasks = [
        (block.copy(), str(dataset), str(fold_id), replicates)
        for (dataset, fold_id), block in frame.groupby(["dataset", "fold_id"], sort=True)
    ]
    outputs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fold_simulation, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            outputs.append(future.result())
    result = pd.concat(outputs, ignore_index=True)
    expected = len(tasks) * replicates * len(FRACTIONS) * len(FORMULAS)
    if len(result) != expected:
        raise StabilityFailure(f"simulation row count {len(result)} != {expected}")
    return result.sort_values(
        ["dataset", "fold_id", "replicate", "fraction", "formula"]
    ).reset_index(drop=True)


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_summary = (
        raw.groupby(["dataset", "fold_id", "fraction", "formula"], as_index=False)
        .agg(
            n_tasks_median=("n_tasks", "median"),
            score_spearman_median=("score_spearman", "median"),
            score_spearman_p05=("score_spearman", lambda x: np.quantile(x, 0.05)),
            top20_jaccard_median=("top20_jaccard", "median"),
            top20_jaccard_p05=("top20_jaccard", lambda x: np.quantile(x, 0.05)),
            delta_outcome_spearman_mean=("delta_outcome_spearman", "mean"),
            delta_utility20_mean=("delta_utility20", "mean"),
        )
    )
    study = (
        fold_summary.groupby(["dataset", "fraction", "formula"], as_index=False)
        .agg(
            n_folds=("fold_id", "size"),
            score_spearman_median=("score_spearman_median", "mean"),
            score_spearman_p05=("score_spearman_p05", "mean"),
            top20_jaccard_median=("top20_jaccard_median", "mean"),
            top20_jaccard_p05=("top20_jaccard_p05", "mean"),
            delta_outcome_spearman_mean=("delta_outcome_spearman_mean", "mean"),
            delta_utility20_mean=("delta_utility20_mean", "mean"),
        )
    )
    overall = (
        study.groupby(["fraction", "formula"], as_index=False)
        .agg(
            n_studies=("dataset", "size"),
            score_spearman_median=("score_spearman_median", "mean"),
            score_spearman_p05=("score_spearman_p05", "mean"),
            top20_jaccard_median=("top20_jaccard_median", "mean"),
            top20_jaccard_p05=("top20_jaccard_p05", "mean"),
            delta_outcome_spearman_mean=("delta_outcome_spearman_mean", "mean"),
            delta_utility20_mean=("delta_utility20_mean", "mean"),
        )
    )
    return study, overall


def render(overall: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.8})
    labels = {"fixed_80_20": "SafeConf-M (4:1)", "one_sided_025": "One-sided correction"}
    colors = {"fixed_80_20": "#118A7E", "one_sided_025": "#7E6AAD"}
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), facecolor="white")
    for formula in FORMULAS:
        block = overall.loc[overall.formula.eq(formula)].sort_values("fraction")
        x = 100 * block.fraction.to_numpy(float)
        axes[0].plot(x, block.score_spearman_median, marker="o", color=colors[formula], label=labels[formula])
        axes[0].plot(x, block.score_spearman_p05, linestyle="--", color=colors[formula], alpha=0.7)
        axes[1].plot(x, block.top20_jaccard_median, marker="o", color=colors[formula], label=labels[formula])
        axes[1].plot(x, block.top20_jaccard_p05, linestyle="--", color=colors[formula], alpha=0.7)
    axes[0].set_xlabel("Candidate perturbations retained (%)")
    axes[0].set_ylabel("Rank stability (Spearman)")
    axes[1].set_xlabel("Candidate perturbations retained (%)")
    axes[1].set_ylabel("Top-20% overlap (Jaccard)")
    for letter, axis in zip(("a", "b"), axes):
        axis.text(-0.16, 1.04, letter, transform=axis.transAxes, fontweight="bold", fontsize=13)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(color="#E6E8EB", linewidth=0.6)
        axis.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(w_pad=2.4)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(
            figures / f"E215_CANDIDATE_SET_STABILITY.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.self_test:
        m = np.array([0.1, 0.4, 0.3, 0.2])
        s = np.array([0.4, 0.1, 0.2, 0.3])
        values = formulas(m, s)
        assert set(values) == set(FORMULAS)
        assert all(np.isfinite(value).all() for value in values.values())
        assert math.isclose(jaccard({1, 2}, {2, 3}), 1 / 3)
        print("PASS")
        return
    if args.output.exists() and any(args.output.iterdir()):
        unexpected = {path.name for path in args.output.iterdir()} - {"ANALYSIS_CONTRACT.md"}
        if unexpected:
            raise StabilityFailure(f"refusing non-empty output: {args.output}")
    if args.replicates < 100 or not 1 <= args.workers <= 32:
        raise StabilityFailure("invalid replicate or worker count")
    frame = validate(args.input.resolve())
    raw = run_parallel(frame, args.replicates, args.workers)
    study, overall = summarize(raw)
    args.scratch.mkdir(parents=True, exist_ok=True)
    raw_path = args.scratch / "E215_SUBSAMPLE_DRAWS.csv"
    atomic_csv(raw_path, raw)
    tables = args.output / "tables"
    atomic_csv(tables / "E215_STUDY_SUMMARY.csv", study)
    atomic_csv(tables / "E215_OVERALL_SUMMARY.csv", overall)
    atomic_csv(
        tables / "E215_SCRATCH_MANIFEST.csv",
        pd.DataFrame([{"path": str(raw_path), "rows": len(raw), "bytes": raw_path.stat().st_size, "sha256": sha256_file(raw_path)}]),
    )
    render(overall, args.output)
    rows = []
    for _, row in overall.iterrows():
        rows.append(
            f"| {row.formula} | {int(100 * row.fraction)}% | {row.score_spearman_median:.4f} | "
            f"{row.score_spearman_p05:.4f} | {row.top20_jaccard_median:.4f} | "
            f"{row.top20_jaccard_p05:.4f} | {row.delta_outcome_spearman_mean:+.4f} | "
            f"{row.delta_utility20_mean:+.4f} |"
        )
    report = f"""# E215｜候选任务集合变化下的路由稳定性

证据身份：已解封八研究数据上的部署稳定性审计。

| 公式 | 保留扰动 | 排名稳定中位数 | 排名稳定P05 | top-20% Jaccard中位数 | Jaccard P05 | Δ误差相关 | Δ20%效用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

实线图示中位稳定性，虚线图示各 fold 的 5% 分位稳定性。后两列比较“在抽样集合内重新计算百分位”与“完整集合计算后限制到同一任务”；接近 0 表示没有系统性评价漂移。

原始 {len(raw):,} 行抽样结果保存在数据盘，仓库中的 manifest 记录路径、字节数和 SHA-256。
"""
    atomic_text(args.output / "E215_REPORT.md", report)
    atomic_text(
        args.output / "RUN_STATUS.json",
        json.dumps(
            {
                "experiment": "E215_candidate_set_stability",
                "evidence_identity": "released_data_deployment_stability_audit",
                "input_sha256": EXPECTED_SHA256,
                "n_tasks": len(frame),
                "n_studies": int(frame.dataset.nunique()),
                "n_folds": int(frame.groupby(["dataset", "fold_id"]).ngroups),
                "replicates_per_fraction_per_fold": args.replicates,
                "fractions": list(FRACTIONS),
                "workers": args.workers,
                "raw_rows": len(raw),
                "target_truth_modified": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()

