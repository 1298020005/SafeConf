#!/usr/bin/env python3
"""Evaluate all frozen E189 pretruth predictions against isolated truth."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E189_primary_cd4_formal_cartesian_20260729"
DATA = Path("/home/yyf/data/safeconf_e189_primary_cd4_cartesian")
PANELS = ("H01", "H02", "H03", "H04")
SUPPORT_LEVELS = (1, 2, 3, 5)
SETTINGS = (
    "random_missing_pair",
    "unseen_context_row",
    "unseen_perturbation_column",
    "double_unseen",
)
SEEDS = (3407, 3408, 3409)
MODEL_KEYS = tuple(
    f"{architecture}_seed{seed}"
    for seed in SEEDS
    for architecture in ("scGPT", "GEARS")
)
N_GENES = 512
N_BOOT = 2000
TOL = 1e-10


class EvaluationFailure(RuntimeError):
    """Fail-closed E189 evaluation contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rmse_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2, axis=-1))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 4 or np.unique(a[keep]).size < 2 or np.unique(b[keep]).size < 2:
        return float("nan")
    return float(
        np.corrcoef(
            rankdata(a[keep], method="average"),
            rankdata(b[keep], method="average"),
        )[0, 1]
    )


def locked_truth(panel: str) -> tuple[pd.DataFrame, np.ndarray, list[dict[str, Any]]]:
    locks = pd.read_csv(OUT / "EVALUATION_ASSET_LOCKS.csv", keep_default_na=False)
    selected = locks.loc[locks.panel_id.astype(str).eq(panel)]
    if len(selected) != 2:
        raise EvaluationFailure(f"{panel}: evaluation lock count failed")
    hashes: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        path = DATA / str(row.path)
        expected_parent = (DATA / "evaluation_truth" / panel).resolve()
        if path.resolve().parent != expected_parent:
            raise EvaluationFailure(f"{panel}: unsafe evaluation path")
        observed = sha256_file(path)
        if observed != str(row.sha256) or path.stat().st_size != int(row.bytes):
            raise EvaluationFailure(f"{panel}: evaluation truth hash/size mismatch")
        hashes.append(
            {"path": str(row.path), "bytes": int(row.bytes), "sha256": observed}
        )
    query = pd.read_csv(
        DATA / "evaluation_truth" / panel / "QUERY_TASKS.csv",
        keep_default_na=False,
    )
    truth_path = DATA / "evaluation_truth" / panel / "EVALUATION_TRUTH.npz"
    with np.load(truth_path, allow_pickle=False) as archive:
        if set(archive.files) != set(query.task_id.astype(str)):
            raise EvaluationFailure(f"{panel}: truth IDs differ from query IDs")
        truth = np.stack(
            [np.asarray(archive[task_id], np.float32) for task_id in query.task_id.astype(str)]
        )
    if truth.shape != (840, N_GENES) or not np.isfinite(truth).all():
        raise EvaluationFailure(f"{panel}: invalid truth matrix")
    return query, truth, hashes


def load_release(panel: str, support: int, query: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, Any]]]:
    release = OUT / "pretruth_release" / panel / f"support_{support}"
    status = json.loads((release / "PRETRUTH_STATUS.json").read_text())
    required = {
        "experiment": "E189",
        "stage": "PRETRUTH_PREDICTION",
        "status": "PASS",
        "panel_id": panel,
        "support_contexts_per_seen_perturbation": support,
        "n_query_tasks": 840,
        "evaluation_truth_files_read": 0,
        "query_graphs_containing_y": 0,
    }
    for key, expected in required.items():
        if status.get(key) != expected:
            raise EvaluationFailure(f"{panel}/support_{support}: status mismatch {key}")
    if tuple(sorted(status.get("model_family_members", []))) != tuple(sorted(MODEL_KEYS)):
        raise EvaluationFailure(f"{panel}/support_{support}: family members changed")

    locks = pd.read_csv(release / "RELEASE_LOCKS.csv", keep_default_na=False)
    input_hashes: list[dict[str, Any]] = []
    for row in locks.itertuples(index=False):
        relative = Path(str(row.path))
        if relative.is_absolute() or ".." in relative.parts:
            raise EvaluationFailure("unsafe release lock path")
        path = (release / relative).resolve()
        try:
            path.relative_to(release.resolve())
        except ValueError as exc:
            raise EvaluationFailure("release lock escapes its root") from exc
        observed = sha256_file(path)
        if observed != str(row.sha256) or path.stat().st_size != int(row.bytes):
            raise EvaluationFailure(f"pretruth release lock mismatch: {path}")
        input_hashes.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": int(row.bytes),
                "sha256": observed,
            }
        )

    order = pd.read_csv(release / "tables/QUERY_ORDER.csv", keep_default_na=False)
    expected_order = query.task_id.astype(str).tolist()
    if (
        order.query_index.astype(int).tolist() != list(range(840))
        or order.task_id.astype(str).tolist() != expected_order
    ):
        raise EvaluationFailure(f"{panel}/support_{support}: query order changed")
    with np.load(release / "arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(MODEL_KEYS):
            raise EvaluationFailure(f"{panel}/support_{support}: model arrays changed")
        predictions = np.stack(
            [np.asarray(archive[key], np.float32) for key in MODEL_KEYS], axis=0
        )
    if predictions.shape != (6, 840, N_GENES) or not np.isfinite(predictions).all():
        raise EvaluationFailure(f"{panel}/support_{support}: invalid predictions")
    return predictions, input_hashes


def task_metrics(
    panel: str,
    support: int,
    query: pd.DataFrame,
    truth: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    member_errors = rmse_rows(predictions, truth[None, :, :])
    centroid = predictions.mean(axis=0)
    centroid_error = rmse_rows(centroid, truth)
    family_rms = np.sqrt(np.mean(member_errors**2, axis=0))
    diversity = np.sqrt(np.mean((predictions - centroid[None, :, :]) ** 2, axis=(0, 2)))
    worst = np.max(member_errors, axis=0)
    diameter = np.zeros(len(query), float)
    for left in range(len(MODEL_KEYS)):
        for right in range(left + 1, len(MODEL_KEYS)):
            diameter = np.maximum(
                diameter, rmse_rows(predictions[left], predictions[right])
            )
    diameter_lower = diameter / 2.0
    zero_error = rmse_rows(truth, np.zeros_like(truth))
    magnitude = rmse_rows(centroid, np.zeros_like(centroid))
    architecture_errors: dict[str, np.ndarray] = {}
    for architecture in ("scGPT", "GEARS"):
        take = [index for index, key in enumerate(MODEL_KEYS) if key.startswith(architecture)]
        architecture_errors[architecture] = np.sqrt(
            np.mean(member_errors[take] ** 2, axis=0)
        )
    frame = query.copy()
    if "panel_id" in frame.columns:
        if not frame.panel_id.astype(str).eq(panel).all():
            raise EvaluationFailure(f"{panel}: query table panel_id mismatch")
    else:
        frame.insert(0, "panel_id", panel)
    frame.insert(0, "support", support)
    frame["family_rms_error"] = family_rms
    frame["family_worst_error"] = worst
    frame["centroid_error"] = centroid_error
    frame["diversity_lower_bound"] = diversity
    frame["diameter_half_lower_bound"] = diameter_lower
    frame["centroid_predicted_magnitude"] = magnitude
    frame["zero_effect_error"] = zero_error
    frame["scGPT_family_rms_error"] = architecture_errors["scGPT"]
    frame["GEARS_family_rms_error"] = architecture_errors["GEARS"]
    frame["fraction_members_beating_zero"] = np.mean(
        member_errors < zero_error[None, :], axis=0
    )
    frame["family_rms_lower_violation"] = diversity > family_rms + TOL
    frame["family_worst_lower_violation"] = diameter_lower > worst + TOL
    frame["rms_identity_residual"] = np.abs(
        family_rms**2 - (centroid_error**2 + diversity**2)
    )
    for index, key in enumerate(MODEL_KEYS):
        frame[f"{key}_rmse"] = member_errors[index]
    return frame


def summarize(tasks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in tasks.groupby(
        ["panel_id", "support", "e189_setting"], sort=True
    ):
        panel, support, setting = keys
        rows.append(
            {
                "panel_id": panel,
                "support": support,
                "e189_setting": setting,
                "n_tasks": len(group),
                "family_rms_error_mean": group.family_rms_error.mean(),
                "family_worst_error_mean": group.family_worst_error.mean(),
                "centroid_error_mean": group.centroid_error.mean(),
                "diversity_lower_bound_mean": group.diversity_lower_bound.mean(),
                "diameter_half_lower_bound_mean": group.diameter_half_lower_bound.mean(),
                "zero_effect_error_mean": group.zero_effect_error.mean(),
                "fraction_members_beating_zero_mean": group.fraction_members_beating_zero.mean(),
                "diversity_to_family_rms_spearman": spearman(
                    group.diversity_lower_bound, group.family_rms_error
                ),
                "diameter_half_to_worst_spearman": spearman(
                    group.diameter_half_lower_bound, group.family_worst_error
                ),
                "magnitude_to_family_rms_spearman": spearman(
                    group.centroid_predicted_magnitude, group.family_rms_error
                ),
                "rms_lower_violations": int(group.family_rms_lower_violation.sum()),
                "worst_lower_violations": int(group.family_worst_lower_violation.sum()),
                "max_rms_identity_residual": group.rms_identity_residual.max(),
            }
        )
    return pd.DataFrame(rows)


def pooled_correlations(tasks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for support in SUPPORT_LEVELS:
        for setting in SETTINGS:
            group = tasks.loc[
                tasks.support.eq(support) & tasks.e189_setting.eq(setting)
            ].copy()
            groups = (
                group.panel_id.astype(str)
                + "::"
                + group.perturbed_gene_id.astype(str)
            ).to_numpy()
            unique_groups = np.unique(groups)
            rng = np.random.default_rng(
                int(hashlib.sha256(f"E189::{support}::{setting}".encode()).hexdigest()[:8], 16)
            )
            statistics = {
                "diversity_to_family_rms": (
                    group.diversity_lower_bound.to_numpy(float),
                    group.family_rms_error.to_numpy(float),
                ),
                "diameter_half_to_worst": (
                    group.diameter_half_lower_bound.to_numpy(float),
                    group.family_worst_error.to_numpy(float),
                ),
                "magnitude_to_family_rms": (
                    group.centroid_predicted_magnitude.to_numpy(float),
                    group.family_rms_error.to_numpy(float),
                ),
            }
            for name, (predictor, target) in statistics.items():
                observed = spearman(predictor, target)
                boot: list[float] = []
                for _ in range(N_BOOT):
                    sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                    take = np.concatenate([np.flatnonzero(groups == item) for item in sampled])
                    value = spearman(predictor[take], target[take])
                    if math.isfinite(value):
                        boot.append(value)
                rows.append(
                    {
                        "support": support,
                        "e189_setting": setting,
                        "association": name,
                        "n_tasks": len(group),
                        "n_clusters": len(unique_groups),
                        "spearman": observed,
                        "cluster_bootstrap_valid": len(boot),
                        "ci95_lower": np.quantile(boot, 0.025) if boot else np.nan,
                        "ci95_upper": np.quantile(boot, 0.975) if boot else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, path: Path) -> None:
    labels = {
        "random_missing_pair": "Random missing pair",
        "unseen_context_row": "Unseen context (row)",
        "unseen_perturbation_column": "Unseen perturbation (column)",
        "double_unseen": "Both unseen",
    }
    colors = {
        "random_missing_pair": "#3C5488",
        "unseen_context_row": "#00A087",
        "unseen_perturbation_column": "#E64B35",
        "double_unseen": "#7E6148",
    }
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.2), sharey=True)
    for axis, panel in zip(axes, PANELS):
        panel_data = summary.loc[summary.panel_id.eq(panel)]
        for setting in SETTINGS:
            data = panel_data.loc[panel_data.e189_setting.eq(setting)].sort_values("support")
            axis.plot(
                data.support,
                data.family_rms_error_mean,
                marker="o",
                linewidth=1.7,
                markersize=4,
                color=colors[setting],
                label=labels[setting],
            )
        axis.set_title(panel, fontsize=10)
        axis.set_xlabel("Observed contexts per seen perturbation", fontsize=8)
        axis.set_xticks(SUPPORT_LEVELS)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
    axes[0].set_ylabel("Family RMS error", fontsize=9)
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small report table without the optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    tables = OUT / "tables"
    reports = OUT / "reports"
    figures = OUT / "figures"
    for directory in (tables, reports, figures):
        directory.mkdir(parents=True, exist_ok=True)
    all_tasks: list[pd.DataFrame] = []
    all_hashes: list[dict[str, Any]] = []
    for panel in PANELS:
        query, truth, truth_hashes = locked_truth(panel)
        all_hashes.extend(truth_hashes)
        for support in SUPPORT_LEVELS:
            predictions, release_hashes = load_release(panel, support, query)
            all_hashes.extend(release_hashes)
            all_tasks.append(
                task_metrics(panel, support, query, truth, predictions)
            )
            print(f"[E189 evaluation] {panel}/support_{support}: loaded", flush=True)
    tasks = pd.concat(all_tasks, ignore_index=True)
    summary = summarize(tasks)
    correlations = pooled_correlations(tasks)
    tasks.to_csv(tables / "E189_TASK_METRICS.csv", index=False)
    summary.to_csv(tables / "E189_STRATIFIED_SUMMARY.csv", index=False)
    correlations.to_csv(tables / "E189_POOLED_CORRELATIONS.csv", index=False)
    pd.DataFrame(all_hashes).to_csv(tables / "E189_EVALUATION_INPUT_HASHES.csv", index=False)
    make_figure(summary, figures / "E189_cartesian_error_by_support.png")

    n_rms_violations = int(tasks.family_rms_lower_violation.sum())
    n_worst_violations = int(tasks.family_worst_lower_violation.sum())
    max_identity_residual = float(tasks.rms_identity_residual.max())
    status = {
        "experiment": "E189",
        "stage": "POSTTRUTH_EVALUATION",
        "status": "PASS"
        if n_rms_violations == 0
        and n_worst_violations == 0
        and max_identity_residual <= 1e-8
        else "FAIL",
        "n_task_instances": len(tasks),
        "n_unique_biological_tasks": tasks.task_id.nunique(),
        "n_panels": tasks.panel_id.nunique(),
        "support_levels": list(SUPPORT_LEVELS),
        "settings": list(SETTINGS),
        "family_rms_lower_violations": n_rms_violations,
        "family_worst_lower_violations": n_worst_violations,
        "max_rms_identity_residual": max_identity_residual,
        "performance_is_not_a_pass_gate": True,
    }
    (OUT / "EVALUATION_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pooled = (
        tasks.groupby(["support", "e189_setting"], as_index=False)
        .agg(
            n_tasks=("task_id", "size"),
            family_rms_error_mean=("family_rms_error", "mean"),
            family_worst_error_mean=("family_worst_error", "mean"),
            fraction_members_beating_zero_mean=("fraction_members_beating_zero", "mean"),
        )
    )
    report_lines = [
        "# E189 正式笛卡尔缺失实验",
        "",
        f"状态：**{status['status']}**。共评估 {len(tasks):,} 个任务实例；"
        f"family-RMS 下界违例 {n_rms_violations}，family-worst 下界违例 "
        f"{n_worst_violations}。",
        "",
        "同一 scGPT–GEARS 六成员模型族同时覆盖随机缺一格、未见细胞背景整行、"
        "未见扰动整列和二者同时未见。训练支持量固定为每个已见扰动 1、2、3、5 "
        "个背景。",
        "",
        "## 汇总",
        "",
        markdown_table(pooled),
        "",
        "PASS 只表示数据隔离、任务合同和确定性下界成立。预测性能、相关性以及是否"
        "优于 magnitude 均按原值报告，不作为事后改写的通过条件。",
        "",
    ]
    (reports / "E189_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if status["status"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
