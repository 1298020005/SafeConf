#!/usr/bin/env python3
"""E166: leave-one-study-out rank stacking over the frozen E153 task table.

This is post-hoc method development.  Each held-out study is excluded from
weight fitting; its truth is used only after the convex score is fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E166_cross_study_rank_stacking_20260716"
CONTRACT = OUT / "ANALYSIS_CONTRACT.md"
STAGING = OUT / ".release.staging"
RELEASE = OUT / "release"
SOURCE = ROOT / "docs/实验结果/E153_eight_study_formal_meta_20260714/tables/E153_ABSOLUTE_TASK_INPUT.csv"
SOURCE_STATUS = ROOT / "docs/实验结果/E153_eight_study_formal_meta_20260714/RUN_STATUS.json"

EXPECTED_SOURCE_SHA256 = "b75f5edae0bb585ba5ff18aecafcc2389b0f05fd5cc86b36960afb4b62e4a15a"
EXPECTED_STATUS_SHA256 = "d2a431bb26fd1bf29c1e088f403bdffef527696685beae61cac4cce3a87c52d0"
EXPECTED_ROWS = 3465
EXPECTED_STUDIES = 8
EXPECTED_FOLDS = 34
SEED = 202607166
N_CLUSTER_BOOT = 3000
N_OVERALL_BOOT = 10000

TARGET = "error_two_predictor_mean_rmse"
FEATURES = {
    "magnitude": "baseline_predicted_magnitude",
    "disagreement": "risk_model_disagreement",
    "safeconf": "safeconf_calibrated_pair_risk",
}
SCORES = ["rank_stack_lodo", "magnitude", "disagreement", "safeconf"]
REQUIRED = [
    "dataset", "fold_id", "task_id", "split", "setting", "context",
    "perturbation", TARGET, *FEATURES.values(),
]
ALLOWLIST = {
    ".E166_TRANSACTION.json",
    "RUN_STATUS.json",
    "RESULTS_SHA256.csv",
    "README_先看这个.md",
    "reports/E166_REPORT.md",
    "tables/E166_INPUT_HASHES.csv",
    "tables/E166_LODO_WEIGHTS.csv",
    "tables/E166_TASK_SCORES.csv",
    "tables/E166_FOLD_RESULTS.csv",
    "tables/E166_STUDY_RESULTS.csv",
    "tables/E166_CLUSTER_BOOTSTRAP_DRAWS.csv",
    "tables/E166_CLUSTER_BOOTSTRAP_SUMMARY.csv",
    "tables/E166_OVERALL_SUMMARY.csv",
    "tables/E166_LEAKAGE_AUDIT.csv",
    "tables/E166_TOP20_ENRICHMENT.csv",
    "figures/F1_delta_vs_magnitude.svg",
}


class IntegrityFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def committed_and_matching(path: Path, head: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    try:
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"Required file is not committed: {relative}") from exc
    observed = sha256_file(path)
    committed_sha = hashlib.sha256(committed).hexdigest()
    if observed != committed_sha:
        raise IntegrityFailure(f"Working file differs from HEAD: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": observed}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 4 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    value = np.corrcoef(rankdata(a, method="average"), rankdata(b, method="average"))[0, 1]
    return float(value) if math.isfinite(value) else float("nan")


def percentile(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def load_and_validate(head: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    own = [committed_and_matching(RUNNER, head), committed_and_matching(CONTRACT, head)]
    for path, expected in ((SOURCE, EXPECTED_SOURCE_SHA256), (SOURCE_STATUS, EXPECTED_STATUS_SHA256)):
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise IntegrityFailure(f"Frozen E153 input changed: {path}")
        own.append(committed_and_matching(path, head))
    frame = pd.read_csv(SOURCE)
    missing = set(REQUIRED).difference(frame.columns)
    if missing:
        raise IntegrityFailure(f"Missing required columns: {sorted(missing)}")
    if len(frame) != EXPECTED_ROWS or frame.dataset.nunique() != EXPECTED_STUDIES or frame.fold_id.nunique() != EXPECTED_FOLDS:
        raise IntegrityFailure("Frozen E153 dimensions differ from contract")
    if not frame["split"].eq("test").all():
        raise IntegrityFailure("E153 snapshot contains a non-test row")
    numeric = [TARGET, *FEATURES.values()]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise IntegrityFailure("Non-finite endpoint or score")
    if frame.duplicated(["dataset", "fold_id", "task_id"]).any():
        raise IntegrityFailure("Duplicate dataset/fold/task row")
    status = json.loads(SOURCE_STATUS.read_text())
    if status.get("status") != "complete" and not str(status.get("phase", "")).startswith("complete"):
        raise IntegrityFailure("E153 status is not complete")
    return frame, own


def add_fold_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    groups = ranked.groupby(["dataset", "fold_id"], sort=False)
    ranked["target_rank_within_fold"] = groups[TARGET].transform(percentile)
    for short, column in FEATURES.items():
        ranked[f"score_rank_{short}"] = groups[column].transform(percentile)
    return ranked


def balanced_training_weights(train: pd.DataFrame) -> np.ndarray:
    n_datasets = train.dataset.nunique()
    values = np.zeros(len(train), dtype=float)
    for dataset, dataset_block in train.groupby("dataset", sort=True):
        folds = dataset_block.fold_id.unique()
        for fold in folds:
            positions = np.flatnonzero((train.dataset.to_numpy() == dataset) & (train.fold_id.to_numpy() == fold))
            values[positions] = 1.0 / (n_datasets * len(folds) * len(positions))
    if not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-12) or np.any(values <= 0):
        raise IntegrityFailure("Training weights do not sum to one")
    return values


def fit_convex_weights(train: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    columns = ["score_rank_magnitude", "score_rank_disagreement", "score_rank_safeconf"]
    x = train[columns].to_numpy(float)
    y = train["target_rank_within_fold"].to_numpy(float)
    sample_weight = balanced_training_weights(train.reset_index(drop=True))

    def objective(weights: np.ndarray) -> float:
        residual = x @ weights - y
        return float(np.sum(sample_weight * residual * residual))

    result = minimize(
        objective,
        x0=np.full(3, 1.0 / 3.0),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints=[{"type": "eq", "fun": lambda value: float(value.sum() - 1.0)}],
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    if not result.success:
        raise IntegrityFailure(f"Convex optimizer failed: {result.message}")
    weights = np.clip(np.asarray(result.x, float), 0.0, 1.0)
    weights /= weights.sum()
    if not np.isclose(weights.sum(), 1.0, atol=1e-12) or np.any(weights < -1e-12):
        raise IntegrityFailure("Invalid convex weights")
    return weights, {"objective": objective(weights), "iterations": int(result.nit), "optimizer_message": str(result.message)}


def fold_macro(block: pd.DataFrame, score: str) -> float:
    values = [spearman(part[score], part[TARGET]) for _, part in block.groupby("fold_id", sort=True)]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def top20_enrichment(block: pd.DataFrame, score: str) -> float:
    values = []
    for _, part in block.groupby("fold_id", sort=True):
        n = len(part)
        k = max(1, int(math.ceil(0.20 * n)))
        predicted = set(part.nlargest(k, score).index)
        observed = set(part.nlargest(k, TARGET).index)
        values.append((len(predicted & observed) / k) / (k / n))
    return float(np.mean(values))


def cluster_bootstrap(block: pd.DataFrame, dataset: str) -> pd.DataFrame:
    clusters = sorted(block.perturbation.astype(str).unique())
    indices = {cluster: np.flatnonzero(block.perturbation.astype(str).to_numpy() == cluster) for cluster in clusters}
    rng = np.random.default_rng(int(hashlib.sha256(f"{SEED}|{dataset}".encode()).hexdigest()[:16], 16))
    rows = []
    for draw in range(N_CLUSTER_BOOT):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        take = np.concatenate([indices[value] for value in sampled])
        resampled = block.iloc[take]
        estimates = {score: fold_macro(resampled, score) for score in SCORES}
        rows.append({
            "dataset": dataset,
            "draw": draw,
            **{f"rho_{key}": value for key, value in estimates.items()},
            "delta_stack_minus_magnitude": estimates["rank_stack_lodo"] - estimates["magnitude"],
            "delta_stack_minus_safeconf": estimates["rank_stack_lodo"] - estimates["safeconf"],
            "delta_stack_minus_disagreement": estimates["rank_stack_lodo"] - estimates["disagreement"],
        })
    return pd.DataFrame(rows)


def quantile_interval(values: pd.Series | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def two_level_bootstrap(draws: pd.DataFrame, metric: str) -> np.ndarray:
    datasets = sorted(draws.dataset.unique())
    arrays = {dataset: draws.loc[draws.dataset.eq(dataset), metric].dropna().to_numpy(float) for dataset in datasets}
    rng = np.random.default_rng(SEED + 100000)
    output = np.empty(N_OVERALL_BOOT, dtype=float)
    for index in range(N_OVERALL_BOOT):
        selected = rng.choice(datasets, size=len(datasets), replace=True)
        output[index] = np.mean([rng.choice(arrays[dataset]) for dataset in selected])
    return output


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("NA" if not math.isfinite(value) else f"{value:.4f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_figure(studies: pd.DataFrame, overall: dict[str, float], path: Path) -> None:
    ordered = studies.sort_values("delta_stack_minus_magnitude").reset_index(drop=True)
    labels = ordered.dataset.str.replace("_two_cellline", "", regex=False).tolist() + ["Equal-study mean"]
    points = ordered.delta_stack_minus_magnitude.tolist() + [overall["estimate"]]
    lows = ordered.delta_vs_magnitude_ci95_low.tolist() + [overall["ci95_low"]]
    highs = ordered.delta_vs_magnitude_ci95_high.tolist() + [overall["ci95_high"]]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 5.2), facecolor="white")
    ax.set_facecolor("white")
    colors = ["#4C78A8"] * len(ordered) + ["#D55E00"]
    ax.hlines(y, lows, highs, color=colors, linewidth=1.8)
    ax.scatter(points, y, c=colors, s=[36] * len(ordered) + [58], zorder=3)
    ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Delta fold-macro Spearman: LODO rank stack minus magnitude")
    ax.set_title("Cross-study transfer of the rank-stacked risk score", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def analyze(frame: pd.DataFrame) -> dict[str, Any]:
    ranked = add_fold_ranks(frame)
    task_blocks = []
    weight_rows = []
    leakage_rows = []
    for heldout in sorted(ranked.dataset.unique()):
        train = ranked.loc[~ranked.dataset.eq(heldout)].copy().reset_index(drop=True)
        test = ranked.loc[ranked.dataset.eq(heldout)].copy()
        weights, diagnostics = fit_convex_weights(train)
        test["rank_stack_lodo"] = (
            weights[0] * test["score_rank_magnitude"]
            + weights[1] * test["score_rank_disagreement"]
            + weights[2] * test["score_rank_safeconf"]
        )
        test["magnitude"] = test["score_rank_magnitude"]
        test["disagreement"] = test["score_rank_disagreement"]
        test["safeconf"] = test["score_rank_safeconf"]
        test["heldout_dataset"] = heldout
        test["heldout_truth_used_for_weight_fit"] = False
        task_blocks.append(test)
        weight_rows.append({
            "heldout_dataset": heldout,
            "n_train_datasets": train.dataset.nunique(),
            "n_train_rows": len(train),
            "n_test_rows": len(test),
            "weight_magnitude": weights[0],
            "weight_disagreement": weights[1],
            "weight_safeconf": weights[2],
            **diagnostics,
        })
        leakage_rows.append({
            "heldout_dataset": heldout,
            "train_datasets": ";".join(sorted(train.dataset.unique())),
            "heldout_absent_from_train": heldout not in set(train.dataset),
            "test_truth_rows_used_for_weight_fit": 0,
            "test_score_transform_uses_truth": False,
            "test_truth_first_used_for_evaluation": True,
        })
    tasks = pd.concat(task_blocks, ignore_index=True)
    weights = pd.DataFrame(weight_rows)
    leakage = pd.DataFrame(leakage_rows)
    if len(tasks) != len(frame) or tasks.heldout_dataset.nunique() != EXPECTED_STUDIES:
        raise IntegrityFailure("LODO task assembly failed")
    if not leakage.heldout_absent_from_train.all() or leakage.test_truth_rows_used_for_weight_fit.sum() != 0:
        raise IntegrityFailure("Held-out truth leakage audit failed")

    fold_rows = []
    enrichment_rows = []
    study_rows = []
    draw_blocks = []
    for dataset, block in tasks.groupby("dataset", sort=True):
        estimates = {}
        for fold, part in block.groupby("fold_id", sort=True):
            for score in SCORES:
                value = spearman(part[score], part[TARGET])
                fold_rows.append({"dataset": dataset, "fold_id": fold, "score": score, "n_tasks": len(part), "spearman": value})
                enrichment_rows.append({"dataset": dataset, "fold_id": fold, "score": score, "top20_error_enrichment": top20_enrichment(part, score)})
        for score in SCORES:
            estimates[score] = fold_macro(block, score)
        draws = cluster_bootstrap(block, dataset)
        draw_blocks.append(draws)
        ci_mag = quantile_interval(draws.delta_stack_minus_magnitude)
        ci_safe = quantile_interval(draws.delta_stack_minus_safeconf)
        ci_dis = quantile_interval(draws.delta_stack_minus_disagreement)
        study_rows.append({
            "dataset": dataset,
            "n_rows": len(block),
            "n_folds": block.fold_id.nunique(),
            "n_perturbation_clusters": block.perturbation.nunique(),
            "rho_rank_stack_lodo": estimates["rank_stack_lodo"],
            "rho_magnitude": estimates["magnitude"],
            "rho_safeconf": estimates["safeconf"],
            "rho_disagreement": estimates["disagreement"],
            "delta_stack_minus_magnitude": estimates["rank_stack_lodo"] - estimates["magnitude"],
            "delta_vs_magnitude_ci95_low": ci_mag[0],
            "delta_vs_magnitude_ci95_high": ci_mag[1],
            "delta_stack_minus_safeconf": estimates["rank_stack_lodo"] - estimates["safeconf"],
            "delta_vs_safeconf_ci95_low": ci_safe[0],
            "delta_vs_safeconf_ci95_high": ci_safe[1],
            "delta_stack_minus_disagreement": estimates["rank_stack_lodo"] - estimates["disagreement"],
            "delta_vs_disagreement_ci95_low": ci_dis[0],
            "delta_vs_disagreement_ci95_high": ci_dis[1],
        })

    fold_results = pd.DataFrame(fold_rows)
    enrichment = pd.DataFrame(enrichment_rows)
    studies = pd.DataFrame(study_rows)
    draws = pd.concat(draw_blocks, ignore_index=True)
    draw_summary_rows = []
    for dataset, block in draws.groupby("dataset", sort=True):
        for metric in ("delta_stack_minus_magnitude", "delta_stack_minus_safeconf", "delta_stack_minus_disagreement"):
            low, high = quantile_interval(block[metric])
            draw_summary_rows.append({
                "dataset": dataset,
                "metric": metric,
                "n_draws": len(block),
                "ci95_low": low,
                "ci95_high": high,
                "p_gt_zero": float(np.mean(block[metric] > 0)),
            })
    draw_summary = pd.DataFrame(draw_summary_rows)

    overall_rows = []
    overall_plot = None
    for metric in ("delta_stack_minus_magnitude", "delta_stack_minus_safeconf", "delta_stack_minus_disagreement"):
        observed = float(studies[metric].mean())
        hierarchical = two_level_bootstrap(draws, metric)
        low, high = quantile_interval(hierarchical)
        row = {
            "metric": metric,
            "equal_study_mean": observed,
            "two_level_ci95_low": low,
            "two_level_ci95_high": high,
            "positive_studies": int((studies[metric] > 0).sum()),
            "total_studies": len(studies),
            "p_gt_zero": float(np.mean(hierarchical > 0)),
        }
        overall_rows.append(row)
        if metric == "delta_stack_minus_magnitude":
            overall_plot = {"estimate": observed, "ci95_low": low, "ci95_high": high}
    overall = pd.DataFrame(overall_rows)
    primary = overall.loc[overall.metric.eq("delta_stack_minus_magnitude")].iloc[0]
    gate = bool(primary.equal_study_mean > 0 and primary.two_level_ci95_low > 0 and primary.positive_studies >= 6)
    return {
        "tasks": tasks,
        "weights": weights,
        "leakage": leakage,
        "fold_results": fold_results,
        "enrichment": enrichment,
        "studies": studies,
        "draws": draws,
        "draw_summary": draw_summary,
        "overall": overall,
        "overall_plot": overall_plot,
        "strict_gate_passed": gate,
    }


def write_release(result: dict[str, Any], inputs: list[dict[str, Any]], head: str) -> dict[str, Any]:
    if RELEASE.exists() or STAGING.exists():
        raise IntegrityFailure("E166 release is append-only and already exists")
    (STAGING / "reports").mkdir(parents=True)
    (STAGING / "tables").mkdir()
    (STAGING / "figures").mkdir()
    transaction = {"schema": "safeconf_e166_transaction_v1", "transaction_id": uuid.uuid4().hex, "created_at": now()}
    atomic_json(STAGING / ".E166_TRANSACTION.json", transaction)
    atomic_csv(STAGING / "tables/E166_INPUT_HASHES.csv", pd.DataFrame(inputs))
    atomic_csv(STAGING / "tables/E166_LODO_WEIGHTS.csv", result["weights"])
    task_columns = [
        "dataset", "fold_id", "task_id", "setting", "context", "perturbation", TARGET,
        "score_rank_magnitude", "score_rank_disagreement", "score_rank_safeconf", "rank_stack_lodo",
        "heldout_truth_used_for_weight_fit",
    ]
    atomic_csv(STAGING / "tables/E166_TASK_SCORES.csv", result["tasks"][task_columns])
    atomic_csv(STAGING / "tables/E166_FOLD_RESULTS.csv", result["fold_results"])
    atomic_csv(STAGING / "tables/E166_STUDY_RESULTS.csv", result["studies"])
    atomic_csv(STAGING / "tables/E166_CLUSTER_BOOTSTRAP_DRAWS.csv", result["draws"])
    atomic_csv(STAGING / "tables/E166_CLUSTER_BOOTSTRAP_SUMMARY.csv", result["draw_summary"])
    atomic_csv(STAGING / "tables/E166_OVERALL_SUMMARY.csv", result["overall"])
    atomic_csv(STAGING / "tables/E166_LEAKAGE_AUDIT.csv", result["leakage"])
    atomic_csv(STAGING / "tables/E166_TOP20_ENRICHMENT.csv", result["enrichment"])
    make_figure(result["studies"], result["overall_plot"], STAGING / "figures/F1_delta_vs_magnitude.svg")

    display_studies = result["studies"][[
        "dataset", "rho_rank_stack_lodo", "rho_magnitude", "delta_stack_minus_magnitude",
        "delta_vs_magnitude_ci95_low", "delta_vs_magnitude_ci95_high",
    ]].copy()
    display_weights = result["weights"][["heldout_dataset", "weight_magnitude", "weight_disagreement", "weight_safeconf"]].copy()
    overall = result["overall"]
    primary = overall.loc[overall.metric.eq("delta_stack_minus_magnitude")].iloc[0]
    report = (
        "# E166｜跨研究折内秩组合结果\n\n"
        "## 主结果\n\n"
        f"八研究等权平均 `Δrho(stack−magnitude)={primary.equal_study_mean:.4f}`，"
        f"两层 bootstrap 95% CI `[{primary.two_level_ci95_low:.4f}, {primary.two_level_ci95_high:.4f}]`；"
        f"点估计为正的研究为 `{int(primary.positive_studies)}/8`。"
        f"预设严格 gate：`{'PASS' if result['strict_gate_passed'] else 'FAIL'}`。\n\n"
        + markdown_table(display_studies) + "\n\n"
        "## 每个留出研究对应的训练权重\n\n"
        + markdown_table(display_weights) + "\n\n"
        "## 审计与边界\n\n"
        "每一轮权重只由另外七个研究的折内风险秩和误差秩拟合；留出研究真值用于评价的行数不进入权重拟合。"
        "输入为 E153 已公开任务快照，因此 E166 是 post-hoc 方法开发和跨研究交叉验证，不是新的独立确认。"
        "折内秩分数适合一批任务的相对质检，不能直接解释为单任务绝对失败概率。\n"
    )
    atomic_bytes(STAGING / "reports/E166_REPORT.md", report.encode())
    atomic_bytes(STAGING / "README_先看这个.md", b"# E166\n\nRead `reports/E166_REPORT.md` first.\n")

    manifest_rows = []
    for path in sorted(STAGING.rglob("*")):
        if path.is_symlink():
            raise IntegrityFailure("Symlink found in E166 staging")
        if path.is_file() and path.name not in {"RUN_STATUS.json", "RESULTS_SHA256.csv"}:
            manifest_rows.append({
                "relative_path": path.relative_to(STAGING).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    atomic_csv(STAGING / "RESULTS_SHA256.csv", pd.DataFrame(manifest_rows))
    primary_status = {
        "metric": str(primary.metric),
        "equal_study_mean": float(primary.equal_study_mean),
        "two_level_ci95_low": float(primary.two_level_ci95_low),
        "two_level_ci95_high": float(primary.two_level_ci95_high),
        "positive_studies": int(primary.positive_studies),
        "total_studies": int(primary.total_studies),
        "p_gt_zero": float(primary.p_gt_zero),
    }
    status = {
        "schema": "safeconf_e166_cross_study_rank_stacking_v1",
        "phase": "complete_posthoc_lodo_rank_stacking",
        "completed_at": now(),
        "git_head_at_formal_run": head,
        "transaction_id": transaction["transaction_id"],
        "source_expression_opened": False,
        "predictors_retrained": False,
        "test_truth_rows_used_for_weight_fit": 0,
        "n_rows": len(result["tasks"]),
        "n_studies": result["tasks"].dataset.nunique(),
        "n_folds": result["tasks"].fold_id.nunique(),
        "n_cluster_bootstrap_per_study": N_CLUSTER_BOOT,
        "n_two_level_bootstrap": N_OVERALL_BOOT,
        "strict_gate_passed": result["strict_gate_passed"],
        "primary": primary_status,
        "results_manifest_sha256": sha256_file(STAGING / "RESULTS_SHA256.csv"),
    }
    atomic_json(STAGING / "RUN_STATUS.json", status)
    observed = {path.relative_to(STAGING).as_posix() for path in STAGING.rglob("*") if path.is_file()}
    if observed != ALLOWLIST:
        raise IntegrityFailure(f"E166 allowlist mismatch: {sorted(observed ^ ALLOWLIST)}")
    os.replace(STAGING, RELEASE)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preflight", "formal"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    head = git_head()
    frame, inputs = load_and_validate(head)
    if args.mode == "preflight":
        print(json.dumps({
            "phase": "preflight_passed",
            "git_head": head,
            "rows": len(frame),
            "studies": frame.dataset.nunique(),
            "folds": frame.fold_id.nunique(),
            "source_sha256": sha256_file(SOURCE),
        }, ensure_ascii=False, indent=2))
        return
    result = analyze(frame)
    status = write_release(result, inputs, head)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
