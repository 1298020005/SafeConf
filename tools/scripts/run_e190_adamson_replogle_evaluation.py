#!/usr/bin/env python3
"""Evaluate frozen Adamson-trained predictions on Replogle target truth."""

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
OUT = ROOT / "docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729"
DATA = Path("/home/yyf/data/safeconf_e190_adamson_replogle")
ASSETS = DATA / "model_assets"
PREDICTIONS = OUT / "pretruth_release"
TRUTH = OUT / "evaluation_truth"
SEEDS = (3407, 3408, 3409)
MODEL_KEYS = tuple(
    f"{architecture}_seed{seed}"
    for seed in SEEDS
    for architecture in ("scGPT", "GEARS")
)
N_GENES = 512
N_TASKS = 692
TOL = 1e-10


class EvaluationFailure(RuntimeError):
    """Fail-closed E190 evaluation contract error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {str(key): np.asarray(archive[key], np.float32) for key in archive.files}


def verify_locks(root: Path, lock_file: Path) -> list[dict[str, Any]]:
    locks = pd.read_csv(lock_file, keep_default_na=False)
    hashes = []
    for row in locks.itertuples(index=False):
        relative = Path(str(row.path))
        if relative.is_absolute() or ".." in relative.parts:
            raise EvaluationFailure("unsafe relative lock path")
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise EvaluationFailure("locked file escapes release root") from exc
        observed = sha256_file(path)
        if observed != str(row.sha256) or path.stat().st_size != int(row.bytes):
            raise EvaluationFailure(f"lock mismatch: {path}")
        hashes.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": int(row.bytes),
                "sha256": observed,
            }
        )
    return hashes


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


def cluster_bootstrap_delta(
    frame: pd.DataFrame,
    estimator: str,
    baseline: str,
    seed: int,
    n_boot: int = 5000,
) -> dict[str, float]:
    by_gene = (
        frame.assign(delta=frame[estimator] - frame[baseline])
        .groupby("gene", observed=True)
        .delta.mean()
        .to_numpy(float)
    )
    rng = np.random.default_rng(seed)
    boot = np.mean(
        by_gene[rng.integers(0, len(by_gene), size=(n_boot, len(by_gene)))],
        axis=1,
    )
    return {
        "gene_cluster_mean_delta": float(by_gene.mean()),
        "ci95_lower": float(np.quantile(boot, 0.025)),
        "ci95_upper": float(np.quantile(boot, 0.975)),
        "n_boot": n_boot,
    }


def cluster_bootstrap_spearman(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
    seed: int,
    n_boot: int = 2000,
) -> dict[str, float]:
    groups = {
        str(gene): group.index.to_numpy(int)
        for gene, group in frame.groupby("gene", observed=True)
    }
    genes = np.asarray(sorted(groups))
    x = frame[predictor].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sampled = rng.choice(genes, size=len(genes), replace=True)
        take = np.concatenate([groups[str(gene)] for gene in sampled])
        value = spearman(x[take], y[take])
        if math.isfinite(value):
            boot.append(value)
    return {
        "spearman": spearman(x, y),
        "ci95_lower": float(np.quantile(boot, 0.025)),
        "ci95_upper": float(np.quantile(boot, 0.975)),
        "bootstrap_valid": len(boot),
    }


def markdown_table(frame: pd.DataFrame) -> str:
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


def make_figure(
    tasks: pd.DataFrame, comparisons: pd.DataFrame, path: Path
) -> None:
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#7F7F7F"]
    labels = ["scGPT", "GEARS", "Six-model family", "Source effect", "Zero effect"]
    columns = [
        "scGPT_family_rms_error",
        "GEARS_family_rms_error",
        "family_rms_error",
        "source_effect_error",
        "zero_effect_error",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.45))
    means = [tasks[column].mean() for column in columns]
    axes[0].bar(range(len(labels)), means, color=colors, width=0.72)
    axes[0].set_xticks(range(len(labels)), labels, rotation=32, ha="right")
    axes[0].set_ylabel("Mean RMSE")
    axes[0].set_title("A  Cross-study prediction error")

    axes[1].scatter(
        tasks.diversity_lower_bound,
        tasks.family_rms_error,
        s=10,
        alpha=0.45,
        color="#4C72B0",
        edgecolors="none",
    )
    lower = min(tasks.diversity_lower_bound.min(), tasks.family_rms_error.min())
    upper = max(tasks.diversity_lower_bound.max(), tasks.family_rms_error.max())
    axes[1].plot([lower, upper], [lower, upper], "--", color="#777777", linewidth=1)
    axes[1].set_xlabel("Diversity lower bound")
    axes[1].set_ylabel("Family RMS error")
    axes[1].set_title("B  Certificate tightness")

    gene_delta = (
        tasks.assign(delta=tasks.family_rms_error - tasks.zero_effect_error)
        .groupby("gene", observed=True)
        .delta.mean()
        .sort_values()
    )
    axes[2].bar(
        np.arange(len(gene_delta)),
        gene_delta.to_numpy(),
        color=np.where(gene_delta.to_numpy() <= 0, "#55A868", "#C44E52"),
        width=0.85,
    )
    axes[2].axhline(0, color="#333333", linewidth=0.8)
    axes[2].set_xlabel("Transfer gene (sorted)")
    axes[2].set_ylabel("Family RMSE − zero RMSE")
    axes[2].set_title("C  Per-gene gain or loss")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E1E1E1", linewidth=0.6)
        axis.tick_params(labelsize=8)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    truth_status = json.loads((TRUTH / "TARGET_TRUTH_BUILD_STATUS.json").read_text())
    prediction_status = json.loads((PREDICTIONS / "PRETRUTH_STATUS.json").read_text())
    if (
        truth_status.get("status") != "PASS"
        or truth_status.get("n_target_tasks") != N_TASKS
        or prediction_status.get("status") != "PASS"
        or prediction_status.get("target_perturbation_x_rows_read") != 0
    ):
        raise EvaluationFailure("E190 truth/prediction status contract failed")
    input_hashes = verify_locks(TRUTH, TRUTH / "TRUTH_LOCKS.csv")
    input_hashes.extend(
        verify_locks(PREDICTIONS, PREDICTIONS / "RELEASE_LOCKS.csv")
    )

    query = pd.read_csv(ASSETS / "QUERY_TASKS.csv", keep_default_na=False)
    order = pd.read_csv(PREDICTIONS / "tables/QUERY_ORDER.csv", keep_default_na=False)
    if (
        len(query) != N_TASKS
        or order.task_id.astype(str).tolist() != query.task_id.astype(str).tolist()
    ):
        raise EvaluationFailure("E190 evaluation query order mismatch")
    true_effects = load_npz(TRUTH / "arrays/TARGET_TRUE_EFFECTS.npz")
    source_effects = load_npz(ASSETS / "SOURCE_GENE_EFFECTS.npz")
    with np.load(
        PREDICTIONS / "arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False
    ) as archive:
        if set(archive.files) != set(MODEL_KEYS):
            raise EvaluationFailure("E190 model family arrays changed")
        predictions = np.stack(
            [np.asarray(archive[key], np.float32) for key in MODEL_KEYS], axis=0
        )
    truth = np.stack(
        [true_effects[task_id] for task_id in query.task_id.astype(str)]
    )
    source = np.stack(
        [source_effects[gene] for gene in query.gene.astype(str)]
    )
    if (
        predictions.shape != (6, N_TASKS, N_GENES)
        or truth.shape != (N_TASKS, N_GENES)
        or source.shape != (N_TASKS, N_GENES)
        or not np.isfinite(predictions).all()
        or not np.isfinite(truth).all()
        or not np.isfinite(source).all()
    ):
        raise EvaluationFailure("E190 aligned matrices invalid")

    member_errors = rmse_rows(predictions, truth[None, :, :])
    centroid = predictions.mean(axis=0)
    centroid_error = rmse_rows(centroid, truth)
    family_rms = np.sqrt(np.mean(member_errors**2, axis=0))
    family_worst = member_errors.max(axis=0)
    diversity = np.sqrt(
        np.mean((predictions - centroid[None, :, :]) ** 2, axis=(0, 2))
    )
    diameter = np.zeros(N_TASKS)
    for left in range(6):
        for right in range(left + 1, 6):
            diameter = np.maximum(
                diameter, rmse_rows(predictions[left], predictions[right])
            )
    diameter_half = diameter / 2.0
    zero_error = rmse_rows(truth, np.zeros_like(truth))
    source_error = rmse_rows(source, truth)
    frame = query.copy()
    frame["family_rms_error"] = family_rms
    frame["family_worst_error"] = family_worst
    frame["centroid_error"] = centroid_error
    frame["diversity_lower_bound"] = diversity
    frame["diameter_half_lower_bound"] = diameter_half
    frame["predicted_magnitude"] = rmse_rows(centroid, np.zeros_like(centroid))
    frame["source_effect_magnitude"] = rmse_rows(source, np.zeros_like(source))
    frame["zero_effect_error"] = zero_error
    frame["source_effect_error"] = source_error
    for architecture in ("scGPT", "GEARS"):
        take = [
            index
            for index, key in enumerate(MODEL_KEYS)
            if key.startswith(architecture)
        ]
        frame[f"{architecture}_family_rms_error"] = np.sqrt(
            np.mean(member_errors[take] ** 2, axis=0)
        )
    frame["fraction_members_beating_zero"] = np.mean(
        member_errors < zero_error[None, :], axis=0
    )
    frame["fraction_members_beating_source"] = np.mean(
        member_errors < source_error[None, :], axis=0
    )
    frame["family_rms_lower_violation"] = diversity > family_rms + TOL
    frame["family_worst_lower_violation"] = diameter_half > family_worst + TOL
    frame["rms_identity_residual"] = np.abs(
        family_rms**2 - (centroid_error**2 + diversity**2)
    )
    for index, key in enumerate(MODEL_KEYS):
        frame[f"{key}_rmse"] = member_errors[index]

    estimator_columns = {
        "scGPT": "scGPT_family_rms_error",
        "GEARS": "GEARS_family_rms_error",
        "six_model_family": "family_rms_error",
        "source_effect": "source_effect_error",
    }
    comparison_rows = []
    for label, column in estimator_columns.items():
        seed = int(hashlib.sha256(f"E190::{label}".encode()).hexdigest()[:8], 16)
        bootstrap = cluster_bootstrap_delta(
            frame, column, "zero_effect_error", seed
        )
        comparison_rows.append(
            {
                "estimator": label,
                "n_tasks": len(frame),
                "n_gene_clusters": frame.gene.nunique(),
                "mean_rmse": frame[column].mean(),
                "zero_mean_rmse": frame.zero_effect_error.mean(),
                "task_win_rate_vs_zero": float(np.mean(frame[column] < frame.zero_effect_error)),
                **bootstrap,
            }
        )
    comparisons = pd.DataFrame(comparison_rows)

    associations = []
    for predictor, outcome in (
        ("diversity_lower_bound", "family_rms_error"),
        ("diameter_half_lower_bound", "family_worst_error"),
        ("predicted_magnitude", "family_rms_error"),
        ("source_effect_magnitude", "family_rms_error"),
    ):
        seed = int(
            hashlib.sha256(f"E190::{predictor}::{outcome}".encode()).hexdigest()[:8],
            16,
        )
        associations.append(
            {
                "predictor": predictor,
                "outcome": outcome,
                "n_tasks": len(frame),
                "n_gene_clusters": frame.gene.nunique(),
                **cluster_bootstrap_spearman(frame, predictor, outcome, seed),
            }
        )
    associations = pd.DataFrame(associations)

    tables = OUT / "final_evaluation/tables"
    reports = OUT / "final_evaluation/reports"
    figures = OUT / "final_evaluation/figures"
    for directory in (tables, reports, figures):
        directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / "E190_TASK_METRICS.csv", index=False)
    comparisons.to_csv(tables / "E190_BASELINE_COMPARISONS.csv", index=False)
    associations.to_csv(tables / "E190_RISK_ASSOCIATIONS.csv", index=False)
    pd.DataFrame(input_hashes).to_csv(
        tables / "E190_EVALUATION_INPUT_HASHES.csv", index=False
    )
    make_figure(
        frame,
        comparisons,
        figures / "E190_crossstudy_transfer_summary.png",
    )

    rms_violations = int(frame.family_rms_lower_violation.sum())
    worst_violations = int(frame.family_worst_lower_violation.sum())
    max_residual = float(frame.rms_identity_residual.max())
    status = {
        "experiment": "E190",
        "stage": "FINAL_EVALUATION",
        "status": "PASS"
        if rms_violations == 0
        and worst_violations == 0
        and max_residual <= 1e-8
        else "FAIL",
        "n_tasks": len(frame),
        "n_gene_clusters": frame.gene.nunique(),
        "n_target_batches": frame.batch.nunique(),
        "family_rms_lower_violations": rms_violations,
        "family_worst_lower_violations": worst_violations,
        "max_rms_identity_residual": max_residual,
        "performance_is_not_a_pass_gate": True,
        "target_truth_used_for_model_training": False,
    }
    (OUT / "final_evaluation/E190_FINAL_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# E190 Adamson→Replogle 直接迁移",
        "",
        f"状态：**{status['status']}**。{len(frame)} 个任务、"
        f"{frame.gene.nunique()} 个共同扰动基因、{frame.batch.nunique()} 个目标 batch。",
        "",
        "## 与 zero-effect 比较",
        "",
        markdown_table(
            comparisons[
                [
                    "estimator",
                    "mean_rmse",
                    "zero_mean_rmse",
                    "task_win_rate_vs_zero",
                    "gene_cluster_mean_delta",
                    "ci95_lower",
                    "ci95_upper",
                ]
            ]
        ),
        "",
        "负的 delta 表示优于 zero-effect。置信区间按目标基因整簇抽样。",
        "",
        "## 预测前风险量与真实误差",
        "",
        markdown_table(
            associations[
                ["predictor", "outcome", "spearman", "ci95_lower", "ci95_upper"]
            ]
        ),
        "",
        f"family RMS 下界违例 {rms_violations}；worst-member 下界违例 "
        f"{worst_violations}。PASS 只表示冻结合同和确定性下界成立，不表示跨研究"
        "预测一定优于简单基线。",
        "",
    ]
    (reports / "E190_FINAL_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if status["status"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
