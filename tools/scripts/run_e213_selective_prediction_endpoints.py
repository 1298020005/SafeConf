#!/usr/bin/env python3
"""E213: audit fixed SafeConf routes with selective-prediction endpoints."""

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
INPUT = (
    ROOT
    / "docs/实验结果/E153_eight_study_formal_meta_20260714/tables"
    / "E153_ABSOLUTE_TASK_INPUT.csv"
)
OUTPUT = ROOT / "docs/实验结果/E213_selective_prediction_endpoints_20260918"
SCRATCH = Path("/home/yyf/data/safeconf_cpu/e213_20260918")
EXPECTED_SHA256 = "b75f5edae0bb585ba5ff18aecafcc2389b0f05fd5cc86b36960afb4b62e4a15a"
EXPECTED_ROWS = 3_465
SEED = 20_260_918
BUDGETS = (0.05, 0.10, 0.20, 0.30)
COVERAGES = tuple(float(value) for value in np.linspace(0.50, 1.00, 11))
FORMULAS = ("magnitude", "safeconf", "fixed_80_20", "one_sided_025")
LABELS = {
    "magnitude": "Predicted magnitude",
    "safeconf": "Original SafeConf",
    "fixed_80_20": "SafeConf-M (4:1)",
    "one_sided_025": "One-sided correction",
}
COLORS = {
    "magnitude": "#D55E5E",
    "safeconf": "#54769A",
    "fixed_80_20": "#118A7E",
    "one_sided_025": "#7E6AAD",
}


class AnalysisFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
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


def percentile(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) < 4 or not np.isfinite(array).all():
        raise AnalysisFailure("rank input must contain at least four finite values")
    return rankdata(array, method="average") / len(array)


def materialize_scores(frame: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    for _, block in frame.groupby(["dataset", "fold_id"], sort=True):
        block = block.copy()
        block["magnitude"] = percentile(block.baseline_predicted_magnitude)
        block["safeconf"] = percentile(block.safeconf_calibrated_pair_risk)
        block["fixed_80_20"] = 0.8 * block.magnitude + 0.2 * block.safeconf
        block["one_sided_025"] = block.magnitude + 0.25 * np.maximum(
            block.safeconf - block.magnitude, 0.0
        )
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def validate_input(path: Path) -> pd.DataFrame:
    if sha256_file(path) != EXPECTED_SHA256:
        raise AnalysisFailure("E153 input hash changed")
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "fold_id",
        "perturbation",
        "error_two_predictor_mean_rmse",
        "baseline_predicted_magnitude",
        "safeconf_calibrated_pair_risk",
    }
    if len(frame) != EXPECTED_ROWS or frame.dataset.nunique() != 8:
        raise AnalysisFailure("E153 dimensions changed")
    if not required.issubset(frame.columns):
        raise AnalysisFailure("E153 columns changed")
    if not np.isfinite(
        frame[
            [
                "error_two_predictor_mean_rmse",
                "baseline_predicted_magnitude",
                "safeconf_calibrated_pair_risk",
            ]
        ].to_numpy(float)
    ).all():
        raise AnalysisFailure("E153 contains non-finite values")
    return materialize_scores(frame)


def tie_aware_mean(
    score: np.ndarray, outcome: np.ndarray, fraction: float, largest: bool
) -> float:
    score = np.asarray(score, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    k = min(len(score), max(1, int(math.ceil(fraction * len(score)))))
    ordered = np.sort(score)
    threshold = ordered[-k] if largest else ordered[k - 1]
    strict = score > threshold if largest else score < threshold
    tied = score == threshold
    remaining = k - int(strict.sum())
    return float(
        (outcome[strict].sum() + remaining * outcome[tied].mean()) / k
    )


def binary_top_label(outcome: np.ndarray, fraction: float = 0.20) -> np.ndarray:
    outcome = np.asarray(outcome, dtype=float)
    k = min(len(outcome) - 1, max(1, int(math.ceil(fraction * len(outcome)))))
    order = np.argsort(outcome, kind="mergesort")
    label = np.zeros(len(outcome), dtype=bool)
    label[order[-k:]] = True
    return label


def binary_auroc(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    positive, negative = int(label.sum()), int((~label).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    value = (ranks[label].sum() - positive * (positive + 1) / 2) / (
        positive * negative
    )
    return float(value)


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    if label.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ranked = label[order].astype(float)
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked.astype(bool)].mean())


def review_utility(score: np.ndarray, outcome: np.ndarray, budget: float) -> float:
    selected = tie_aware_mean(score, outcome, budget, largest=True)
    oracle = tie_aware_mean(outcome, outcome, budget, largest=True)
    random = float(np.mean(outcome))
    denominator = oracle - random
    return float((selected - random) / denominator) if denominator > 1e-15 else float("nan")


def selective_risk_auc(score: np.ndarray, outcome: np.ndarray) -> float:
    full = float(np.mean(outcome))
    if full <= 0:
        return float("nan")
    risks = np.asarray(
        [tie_aware_mean(score, outcome, coverage, largest=False) / full for coverage in COVERAGES]
    )
    return float(np.trapz(risks, np.asarray(COVERAGES)) / (COVERAGES[-1] - COVERAGES[0]))


def fold_metrics(block: pd.DataFrame, formula: str) -> dict[str, float]:
    score = block[formula].to_numpy(float)
    outcome = block.error_two_predictor_mean_rmse.to_numpy(float)
    label = binary_top_label(outcome)
    result = {
        "auroc_top20": binary_auroc(score, label),
        "average_precision_top20": average_precision(score, label),
        "selective_risk_auc": selective_risk_auc(score, outcome),
    }
    for budget in BUDGETS:
        result[f"utility_{int(100 * budget):02d}"] = review_utility(
            score, outcome, budget
        )
    return result


METRICS = (
    "auroc_top20",
    "average_precision_top20",
    "selective_risk_auc",
    "utility_05",
    "utility_10",
    "utility_20",
    "utility_30",
)


def study_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, study in frame.groupby("dataset", sort=True):
        values = {formula: {metric: [] for metric in METRICS} for formula in FORMULAS}
        for _, block in study.groupby("fold_id", sort=True):
            for formula in FORMULAS:
                measured = fold_metrics(block, formula)
                for metric in METRICS:
                    values[formula][metric].append(measured[metric])
        for formula in FORMULAS:
            row = {
                "dataset": dataset,
                "formula": formula,
                "n_tasks": len(study),
                "n_folds": study.fold_id.nunique(),
            }
            row.update(
                {
                    metric: float(np.nanmean(values[formula][metric]))
                    for metric in METRICS
                }
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    for metric in METRICS:
        baseline = result.loc[
            result.formula.eq("magnitude"), ["dataset", metric]
        ].rename(columns={metric: f"magnitude_{metric}"})
        result = result.merge(baseline, on="dataset", validate="many_to_one")
        if metric == "selective_risk_auc":
            result[f"gain_{metric}_vs_magnitude"] = (
                result[f"magnitude_{metric}"] - result[metric]
            )
        else:
            result[f"gain_{metric}_vs_magnitude"] = (
                result[metric] - result[f"magnitude_{metric}"]
            )
    return result


def resample_clusters(study: pd.DataFrame, dataset: str, replicate: int) -> pd.DataFrame:
    label_seed = int(hashlib.sha256(dataset.encode()).hexdigest()[:12], 16)
    rng = np.random.default_rng(SEED + label_seed + 1_000_003 * replicate)
    clusters = sorted(study.perturbation.astype(str).unique())
    members = {
        cluster: study.loc[study.perturbation.astype(str).eq(cluster)]
        for cluster in clusters
    }
    chosen = rng.choice(clusters, size=len(clusters), replace=True)
    return pd.concat([members[str(cluster)] for cluster in chosen], ignore_index=True)


def bootstrap_chunk(
    study: pd.DataFrame, dataset: str, replicates: list[int]
) -> pd.DataFrame:
    rows = []
    for replicate in replicates:
        sampled = resample_clusters(study, dataset, replicate)
        measured = study_metrics(sampled)
        measured = measured.loc[
            measured.formula.isin(("fixed_80_20", "one_sided_025"))
        ]
        for _, row in measured.iterrows():
            output = {
                "dataset": dataset,
                "replicate": replicate,
                "formula": row.formula,
            }
            output.update(
                {
                    f"gain_{metric}": float(
                        row[f"gain_{metric}_vs_magnitude"]
                    )
                    for metric in METRICS
                }
            )
            rows.append(output)
    return pd.DataFrame(rows)


def parallel_bootstrap(
    frame: pd.DataFrame, n_bootstrap: int, workers: int
) -> pd.DataFrame:
    workers = max(1, min(workers, 32))
    chunk_size = max(10, int(math.ceil(n_bootstrap / max(1, workers // 2))))
    tasks = []
    for dataset, study in frame.groupby("dataset", sort=True):
        for start in range(0, n_bootstrap, chunk_size):
            tasks.append(
                (study.copy(), str(dataset), list(range(start, min(start + chunk_size, n_bootstrap))))
            )
    frames = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(bootstrap_chunk, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            frames.append(future.result())
    result = pd.concat(frames, ignore_index=True)
    expected = 8 * n_bootstrap * 2
    if len(result) != expected:
        raise AnalysisFailure(f"bootstrap row count {len(result)} != {expected}")
    return result.sort_values(["dataset", "replicate", "formula"]).reset_index(drop=True)


def hierarchical_summary(
    points: pd.DataFrame, draws: pd.DataFrame, n_bootstrap: int
) -> pd.DataFrame:
    datasets = sorted(points.dataset.unique())
    rng = np.random.default_rng(SEED + 7)
    rows = []
    for formula in FORMULAS:
        block = points.loc[points.formula.eq(formula)]
        for metric in METRICS:
            gain_column = f"gain_{metric}_vs_magnitude"
            gains = block[gain_column].to_numpy(float)
            row = {
                "formula": formula,
                "metric": metric,
                "mean_value": float(block[metric].mean()),
                "mean_gain_vs_magnitude": float(gains.mean()),
                "positive_studies": int((gains > 0).sum()),
                "n_studies": len(block),
                "ci95_lower": float("nan"),
                "ci95_upper": float("nan"),
            }
            if formula in {"fixed_80_20", "one_sided_025"}:
                values = []
                for _ in range(n_bootstrap):
                    selected = rng.choice(datasets, size=len(datasets), replace=True)
                    sample = []
                    for dataset in selected:
                        candidates = draws.loc[
                            draws.dataset.eq(dataset) & draws.formula.eq(formula),
                            f"gain_{metric}",
                        ].to_numpy(float)
                        sample.append(float(rng.choice(candidates)))
                    values.append(float(np.mean(sample)))
                row["ci95_lower"] = float(np.quantile(values, 0.025))
                row["ci95_upper"] = float(np.quantile(values, 0.975))
            rows.append(row)
    return pd.DataFrame(rows)


def render_figures(points: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.8})
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    selected_metrics = (
        "auroc_top20",
        "average_precision_top20",
        "selective_risk_auc",
        "utility_20",
    )
    selected_formulas = ("fixed_80_20", "one_sided_025")
    datasets = sorted(points.dataset.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), facecolor="white")
    for axis, formula, letter in zip(axes, selected_formulas, ("a", "b")):
        block = points.loc[points.formula.eq(formula)].set_index("dataset")
        matrix = np.asarray(
            [
                [block.loc[dataset, f"gain_{metric}_vs_magnitude"] for metric in selected_metrics]
                for dataset in datasets
            ]
        )
        bound = max(0.02, float(np.nanmax(np.abs(matrix))))
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_xticks(
            range(4), ["AUROC", "Average\nprecision", "Selective\nrisk", "20% utility"]
        )
        axis.set_yticks(range(len(datasets)), [name.replace("_", " ") for name in datasets])
        axis.set_xlabel(LABELS[formula])
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix[row, column]:+.2f}", ha="center", va="center", fontsize=7.5)
        axis.text(-0.13, 1.04, letter, transform=axis.transAxes, fontweight="bold", fontsize=13)
        for spine in axis.spines.values():
            spine.set_visible(False)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout(w_pad=2.4)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(
            figures / f"E213_ENDPOINT_ROBUSTNESS.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 4.2), facecolor="white")
    budget_metrics = ("utility_05", "utility_10", "utility_20", "utility_30")
    for formula in FORMULAS:
        block = summary.loc[
            summary.formula.eq(formula) & summary.metric.isin(budget_metrics)
        ].set_index("metric").reindex(budget_metrics)
        axis.plot(
            [5, 10, 20, 30],
            block.mean_value.to_numpy(float),
            marker="o",
            linewidth=1.8,
            color=COLORS[formula],
            label=LABELS[formula],
        )
    axis.set_xlabel("Review budget (%)")
    axis.set_ylabel("Oracle-normalized review utility")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#E6E8EB", linewidth=0.6)
    axis.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(
            figures / f"E213_BUDGET_CURVES.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.self_test:
        outcome = np.arange(1.0, 11.0)
        perfect = outcome.copy()
        label = binary_top_label(outcome)
        assert math.isclose(binary_auroc(perfect, label), 1.0)
        assert math.isclose(average_precision(perfect, label), 1.0)
        assert review_utility(perfect, outcome, 0.20) > 0.999
        assert selective_risk_auc(perfect, outcome) < selective_risk_auc(-perfect, outcome)
        print("PASS")
        return
    if args.output.exists() and any(args.output.iterdir()):
        allowed = {"ANALYSIS_CONTRACT.md"}
        unexpected = {path.name for path in args.output.iterdir()} - allowed
        if unexpected:
            raise AnalysisFailure(f"refusing non-empty output: {args.output}")
    if args.bootstrap < 100 or not 1 <= args.workers <= 32:
        raise AnalysisFailure("invalid bootstrap or worker count")
    frame = validate_input(args.input.resolve())
    points = study_metrics(frame)
    draws = parallel_bootstrap(frame, args.bootstrap, args.workers)
    summary = hierarchical_summary(points, draws, args.bootstrap)

    args.scratch.mkdir(parents=True, exist_ok=True)
    raw = args.scratch / "E213_CLUSTER_BOOTSTRAP_DRAWS.csv"
    atomic_csv(raw, draws)
    tables = args.output / "tables"
    atomic_csv(tables / "E213_STUDY_RESULTS.csv", points)
    atomic_csv(tables / "E213_OVERALL_SUMMARY.csv", summary)
    manifest = pd.DataFrame(
        [
            {
                "path": str(raw),
                "rows": len(draws),
                "bytes": raw.stat().st_size,
                "sha256": sha256_file(raw),
            }
        ]
    )
    atomic_csv(tables / "E213_SCRATCH_MANIFEST.csv", manifest)
    render_figures(points, summary, args.output)

    fixed = summary.loc[summary.formula.eq("fixed_80_20")]
    one = summary.loc[summary.formula.eq("one_sided_025")]
    report_rows = []
    for formula, block in (("SafeConf-M 4:1", fixed), ("单向 0.25 修正", one)):
        indexed = block.set_index("metric")
        report_rows.append(
            f"| {formula} | {indexed.loc['auroc_top20', 'mean_gain_vs_magnitude']:+.4f} "
            f"[{indexed.loc['auroc_top20', 'ci95_lower']:+.4f}, {indexed.loc['auroc_top20', 'ci95_upper']:+.4f}] | "
            f"{indexed.loc['average_precision_top20', 'mean_gain_vs_magnitude']:+.4f} "
            f"[{indexed.loc['average_precision_top20', 'ci95_lower']:+.4f}, {indexed.loc['average_precision_top20', 'ci95_upper']:+.4f}] | "
            f"{indexed.loc['selective_risk_auc', 'mean_gain_vs_magnitude']:+.4f} "
            f"[{indexed.loc['selective_risk_auc', 'ci95_lower']:+.4f}, {indexed.loc['selective_risk_auc', 'ci95_upper']:+.4f}] | "
            f"{indexed.loc['utility_20', 'mean_gain_vs_magnitude']:+.4f} "
            f"[{indexed.loc['utility_20', 'ci95_lower']:+.4f}, {indexed.loc['utility_20', 'ci95_upper']:+.4f}] |"
        )
    report = f"""# E213｜多预算选择性预测端点结果

证据身份：已解封八研究数据上的回顾性端点扩展，不是新的外部确认。

| 固定公式 | ΔAUROC | Δ平均精确率 | 选择性风险改善 | Δ20%复核效用 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(report_rows)}

所有差值均相对预测幅度；选择性风险一列已统一成“幅度曲线面积减候选曲线面积”，因此四列均为正值表示改善。区间为研究与扰动条件两层配对自助 95% 区间。

本实验不选择新公式。完整八研究结果见 `tables/E213_STUDY_RESULTS.csv`，全部汇总见 `tables/E213_OVERALL_SUMMARY.csv`。原始 {len(draws):,} 行簇自助结果留在数据盘，路径和 SHA-256 见 `tables/E213_SCRATCH_MANIFEST.csv`。

图件：`figures/E213_ENDPOINT_ROBUSTNESS.*` 与 `figures/E213_BUDGET_CURVES.*`。
"""
    atomic_text(args.output / "E213_REPORT.md", report)
    status = {
        "experiment": "E213_selective_prediction_endpoints",
        "evidence_identity": "released_data_retrospective_endpoint_extension",
        "input_sha256": EXPECTED_SHA256,
        "n_tasks": len(frame),
        "n_studies": int(frame.dataset.nunique()),
        "n_folds": int(frame.groupby(["dataset", "fold_id"]).ngroups),
        "bootstrap_replicates": args.bootstrap,
        "workers": args.workers,
        "raw_bootstrap_rows": len(draws),
        "e205_or_e208_truth_used": False,
        "target_truth_modified": False,
    }
    atomic_text(
        args.output / "RUN_STATUS.json",
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
    )


if __name__ == "__main__":
    main()

