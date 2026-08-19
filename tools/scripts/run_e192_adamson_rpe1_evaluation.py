#!/usr/bin/env python3
"""Evaluate frozen E192 predictions and apply the preregistered ranking gate."""

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
OUT = ROOT / "docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729"
DATA = Path("/home/yyf/data/safeconf_e192_adamson_rpe1")
ASSETS = DATA / "model_assets"
PREDICTIONS = OUT / "pretruth_release"
TRUTH = OUT / "evaluation_truth"
FINAL = OUT / "final_evaluation"
SEEDS = (3407, 3408, 3409)
MODEL_KEYS = tuple(
    f"{architecture}_seed{seed}"
    for seed in SEEDS
    for architecture in ("scGPT", "GEARS")
)
N_GENES = 512
N_TASKS = 175
N_GENE_CLUSTERS = 21
BUDGETS = (0.10, 0.20, 0.30)
TOL = 1e-10


class EvaluationFailure(RuntimeError):
    """Fail-closed E192 evaluation error."""


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
    hashes: list[dict[str, Any]] = []
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
            {"path": str(path), "bytes": int(row.bytes), "sha256": observed}
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


def stable_key(value: str) -> str:
    return hashlib.sha256(f"E192\\0{value}".encode()).hexdigest()


def choose_top(frame: pd.DataFrame, column: str, n_select: int) -> set[str]:
    ordered = frame.assign(
        tie_key=[stable_key(task_id) for task_id in frame.bootstrap_task_id.astype(str)]
    ).sort_values([column, "tie_key"], ascending=[False, True])
    return set(ordered.head(n_select).bootstrap_task_id.astype(str))


def utility(frame: pd.DataFrame, predictor: str, outcome: str, budget: float) -> dict[str, float]:
    n_select = int(math.ceil(len(frame) * budget))
    oracle_ids = choose_top(frame, outcome, n_select)
    selected_ids = choose_top(frame, predictor, n_select)
    oracle_mean = frame.loc[
        frame.bootstrap_task_id.astype(str).isin(oracle_ids), outcome
    ].mean()
    selected_mean = frame.loc[
        frame.bootstrap_task_id.astype(str).isin(selected_ids), outcome
    ].mean()
    overall = frame[outcome].mean()
    denominator = oracle_mean - overall
    return {
        "budget": budget,
        "n_selected": n_select,
        "high_error_capture": len(oracle_ids & selected_ids) / n_select,
        "random_expected_capture": n_select / len(frame),
        "selected_mean_error": float(selected_mean),
        "overall_mean_error": float(overall),
        "error_lift": float(selected_mean / overall),
        "oracle_mean_error": float(oracle_mean),
        "oracle_normalized_utility": (
            float((selected_mean - overall) / denominator)
            if denominator > 1e-15
            else float("nan")
        ),
    }


def cluster_bootstrap_spearman(
    frame: pd.DataFrame, predictor: str, outcome: str, seed: int, n_boot: int = 5000
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


def cluster_bootstrap_utility(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
    budget: float,
    seed: int,
    n_boot: int = 3000,
) -> dict[str, float]:
    groups = {
        str(gene): group.copy()
        for gene, group in frame.groupby("gene", observed=True)
    }
    genes = np.asarray(sorted(groups))
    rng = np.random.default_rng(seed)
    boot = []
    for iteration in range(n_boot):
        sampled = rng.choice(genes, size=len(genes), replace=True)
        blocks = []
        for occurrence, gene in enumerate(sampled):
            block = groups[str(gene)].copy()
            block["bootstrap_task_id"] = (
                f"{iteration}::{occurrence}::" + block.task_id.astype(str)
            )
            blocks.append(block)
        value = utility(pd.concat(blocks, ignore_index=True), predictor, outcome, budget)[
            "oracle_normalized_utility"
        ]
        if math.isfinite(value):
            boot.append(value)
    return {
        "utility_ci95_lower": float(np.quantile(boot, 0.025)),
        "utility_ci95_upper": float(np.quantile(boot, 0.975)),
        "utility_bootstrap_valid": len(boot),
    }


def cluster_bootstrap_delta(
    frame: pd.DataFrame, estimator: str, baseline: str, seed: int, n_boot: int = 5000
) -> dict[str, float]:
    values = (
        frame.assign(delta=frame[estimator] - frame[baseline])
        .groupby("gene", observed=True)
        .delta.mean()
        .to_numpy(float)
    )
    rng = np.random.default_rng(seed)
    boot = np.mean(
        values[rng.integers(0, len(values), size=(n_boot, len(values)))], axis=1
    )
    return {
        "gene_cluster_mean_delta": float(values.mean()),
        "ci95_lower": float(np.quantile(boot, 0.025)),
        "ci95_upper": float(np.quantile(boot, 0.975)),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.5f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def make_figure(
    frame: pd.DataFrame, comparisons: pd.DataFrame, budgets: pd.DataFrame, path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.55))
    labels = ["scGPT", "GEARS", "Six-model", "Source", "Zero"]
    columns = [
        "scGPT_family_rms_error",
        "GEARS_family_rms_error",
        "family_rms_error",
        "source_effect_error",
        "zero_effect_error",
    ]
    colors = ["#3C5488", "#00A087", "#E64B35", "#7E6148", "#7F7F7F"]
    axes[0].bar(range(5), [frame[c].mean() for c in columns], color=colors)
    axes[0].set_xticks(range(5), labels, rotation=28, ha="right")
    axes[0].set_ylabel("Mean RMSE")
    axes[0].set_title("A  RPE1 transfer error")
    axes[1].scatter(
        frame.diversity_lower_bound,
        frame.family_rms_error,
        s=13,
        alpha=0.55,
        color="#3C5488",
        edgecolors="none",
    )
    axes[1].set_xlabel("Diversity lower certificate")
    axes[1].set_ylabel("Family RMS error")
    axes[1].set_title("B  Fresh-setting association")
    for predictor, color, marker in (
        ("diversity_lower_bound", "#3C5488", "o"),
        ("predicted_magnitude", "#E64B35", "s"),
        ("source_effect_magnitude", "#7E6148", "^"),
    ):
        take = budgets.loc[budgets.predictor.eq(predictor)]
        axes[2].plot(
            take.budget * 100,
            take.oracle_normalized_utility,
            color=color,
            marker=marker,
            label=predictor,
        )
    axes[2].axhline(0, color="#444444", linewidth=0.8)
    axes[2].set_xlabel("Review budget (%)")
    axes[2].set_ylabel("Oracle-normalized utility")
    axes[2].set_title("C  Review-budget utility")
    axes[2].legend(frameon=False, fontsize=7)
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
    if FINAL.exists():
        raise EvaluationFailure("E192 final evaluation is append-only and already exists")
    truth_status = json.loads((TRUTH / "TARGET_TRUTH_BUILD_STATUS.json").read_text())
    prediction_status = json.loads((PREDICTIONS / "PRETRUTH_STATUS.json").read_text())
    if (
        truth_status.get("status") != "PASS"
        or truth_status.get("n_target_tasks") != N_TASKS
        or prediction_status.get("status") != "PASS"
        or prediction_status.get("target_perturbation_x_rows_read") != 0
    ):
        raise EvaluationFailure("E192 truth/prediction status contract failed")
    input_hashes = verify_locks(TRUTH, TRUTH / "TRUTH_LOCKS.csv")
    input_hashes.extend(
        verify_locks(PREDICTIONS, PREDICTIONS / "RELEASE_LOCKS.csv")
    )
    query = pd.read_csv(ASSETS / "QUERY_TASKS.csv", keep_default_na=False)
    order = pd.read_csv(PREDICTIONS / "tables/QUERY_ORDER.csv", keep_default_na=False)
    if (
        len(query) != N_TASKS
        or query.gene.nunique() != N_GENE_CLUSTERS
        or order.task_id.astype(str).tolist() != query.task_id.astype(str).tolist()
    ):
        raise EvaluationFailure("E192 evaluation query order/count mismatch")
    true_effects = load_npz(TRUTH / "arrays/TARGET_TRUE_EFFECTS.npz")
    source_effects = load_npz(ASSETS / "SOURCE_GENE_EFFECTS.npz")
    with np.load(
        PREDICTIONS / "arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False
    ) as archive:
        if set(archive.files) != set(MODEL_KEYS):
            raise EvaluationFailure("E192 model family arrays changed")
        predictions = np.stack(
            [np.asarray(archive[key], np.float32) for key in MODEL_KEYS], axis=0
        )
    truth = np.stack([true_effects[t] for t in query.task_id.astype(str)])
    source = np.stack([source_effects[g] for g in query.gene.astype(str)])
    if (
        predictions.shape != (6, N_TASKS, N_GENES)
        or truth.shape != (N_TASKS, N_GENES)
        or source.shape != (N_TASKS, N_GENES)
        or not np.isfinite(predictions).all()
        or not np.isfinite(truth).all()
        or not np.isfinite(source).all()
    ):
        raise EvaluationFailure("E192 aligned matrices invalid")

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
    frame["bootstrap_task_id"] = frame.task_id.astype(str)
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
            index for index, key in enumerate(MODEL_KEYS) if key.startswith(architecture)
        ]
        frame[f"{architecture}_family_rms_error"] = np.sqrt(
            np.mean(member_errors[take] ** 2, axis=0)
        )
    frame["fraction_members_beating_zero"] = np.mean(
        member_errors < zero_error[None, :], axis=0
    )
    frame["family_rms_lower_violation"] = diversity > family_rms + TOL
    frame["family_worst_lower_violation"] = diameter_half > family_worst + TOL
    frame["rms_identity_residual"] = np.abs(
        family_rms**2 - (centroid_error**2 + diversity**2)
    )
    for index, key in enumerate(MODEL_KEYS):
        frame[f"{key}_rmse"] = member_errors[index]

    comparisons = []
    for label, column in (
        ("scGPT", "scGPT_family_rms_error"),
        ("GEARS", "GEARS_family_rms_error"),
        ("six_model_family", "family_rms_error"),
        ("source_effect", "source_effect_error"),
    ):
        seed = int(hashlib.sha256(f"E192::{label}".encode()).hexdigest()[:8], 16)
        comparisons.append(
            {
                "estimator": label,
                "mean_rmse": float(frame[column].mean()),
                "zero_mean_rmse": float(frame.zero_effect_error.mean()),
                "task_win_rate_vs_zero": float(
                    np.mean(frame[column] < frame.zero_effect_error)
                ),
                **cluster_bootstrap_delta(
                    frame, column, "zero_effect_error", seed
                ),
            }
        )
    comparisons = pd.DataFrame(comparisons)
    associations = []
    for predictor, outcome in (
        ("diversity_lower_bound", "family_rms_error"),
        ("diameter_half_lower_bound", "family_worst_error"),
        ("predicted_magnitude", "family_rms_error"),
        ("source_effect_magnitude", "family_rms_error"),
    ):
        seed = int(
            hashlib.sha256(f"E192::{predictor}::{outcome}".encode()).hexdigest()[:8],
            16,
        )
        associations.append(
            {
                "predictor": predictor,
                "outcome": outcome,
                **cluster_bootstrap_spearman(frame, predictor, outcome, seed),
            }
        )
    associations = pd.DataFrame(associations)
    budget_rows = []
    for predictor in (
        "diversity_lower_bound",
        "predicted_magnitude",
        "source_effect_magnitude",
    ):
        for budget in BUDGETS:
            values = utility(frame, predictor, "family_rms_error", budget)
            values.update(
                {
                    "predictor": predictor,
                    **cluster_bootstrap_utility(
                        frame,
                        predictor,
                        "family_rms_error",
                        budget,
                        int(
                            hashlib.sha256(
                                f"E192::{predictor}::{budget}".encode()
                            ).hexdigest()[:8],
                            16,
                        ),
                    ),
                }
            )
            budget_rows.append(values)
    budgets = pd.DataFrame(budget_rows)

    diversity_association = associations.loc[
        associations.predictor.eq("diversity_lower_bound")
        & associations.outcome.eq("family_rms_error")
    ].iloc[0]
    diversity_budgets = budgets.loc[
        budgets.predictor.eq("diversity_lower_bound")
    ].set_index("budget")
    certificate_gate = bool(
        not frame.family_rms_lower_violation.any()
        and not frame.family_worst_lower_violation.any()
        and frame.rms_identity_residual.max() <= 1e-7
    )
    ranking_gate = bool(
        diversity_association.ci95_lower > 0
        and diversity_budgets.loc[0.20, "utility_ci95_lower"] > 0
        and (diversity_budgets.oracle_normalized_utility > 0).all()
    )
    status = {
        "experiment": "E192",
        "stage": "FINAL_EVALUATION",
        "status": "PASS" if certificate_gate else "FAIL",
        "n_tasks": len(frame),
        "n_gene_clusters": frame.gene.nunique(),
        "n_target_batches": frame.batch.nunique(),
        "family_rms_lower_violations": int(frame.family_rms_lower_violation.sum()),
        "family_worst_lower_violations": int(
            frame.family_worst_lower_violation.sum()
        ),
        "max_rms_identity_residual": float(frame.rms_identity_residual.max()),
        "certificate_gate_pass": certificate_gate,
        "ranking_activation_gate_pass": ranking_gate,
        "ranking_setting_status": "ENABLED" if ranking_gate else "ABSTAIN",
        "performance_is_not_a_certificate_pass_gate": True,
        "target_truth_used_for_model_training": False,
    }
    tables = FINAL / "tables"
    reports = FINAL / "reports"
    figures = FINAL / "figures"
    for directory in (tables, reports, figures):
        directory.mkdir(parents=True, exist_ok=True)
    frame.drop(columns="bootstrap_task_id").to_csv(
        tables / "E192_TASK_METRICS.csv", index=False
    )
    comparisons.to_csv(tables / "E192_BASELINE_COMPARISONS.csv", index=False)
    associations.to_csv(tables / "E192_RISK_ASSOCIATIONS.csv", index=False)
    budgets.to_csv(tables / "E192_BUDGET_UTILITY.csv", index=False)
    pd.DataFrame(input_hashes).to_csv(tables / "E192_INPUT_HASHES.csv", index=False)
    make_figure(
        frame,
        comparisons,
        budgets,
        figures / "E192_rpe1_locked_confirmation.png",
    )
    (FINAL / "E192_FINAL_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# E192 Adamson→Replogle RPE1 锁定确认结果",
        "",
        f"确定性证书 gate：**{'PASS' if certificate_gate else 'FAIL'}**。",
        f"经验排序激活 gate：**{'PASS / ENABLED' if ranking_gate else 'FAIL / ABSTAIN'}**。",
        "",
        "## 预测器与 zero-effect",
        "",
        markdown_table(comparisons),
        "",
        "## 风险量与真实误差",
        "",
        markdown_table(associations),
        "",
        "## 固定复核预算",
        "",
        markdown_table(budgets),
        "",
        "证书 gate 与排序 gate 分开裁决。确定性下界成立，不能自动把外部 RPE1 "
        "setting 的经验排序改为可用；只有预注册的三个排序条件全部通过才启用。",
        "",
    ]
    (reports / "E192_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    interpretation = [
        "# E192 结果解释",
        "",
        f"- 目标：21 个基因、175 个任务、{frame.batch.nunique()} 个 RPE1 批次；",
        f"- family RMS / worst lower violation："
        f"{status['family_rms_lower_violations']} / "
        f"{status['family_worst_lower_violations']}；",
        f"- diversity–family RMS ρ={diversity_association.spearman:.3f}，"
        f"95% CI [{diversity_association.ci95_lower:.3f}, "
        f"{diversity_association.ci95_upper:.3f}]；",
        f"- 20% 预算 diversity utility="
        f"{diversity_budgets.loc[0.20, 'oracle_normalized_utility']:.3f}，"
        f"95% CI [{diversity_budgets.loc[0.20, 'utility_ci95_lower']:.3f}, "
        f"{diversity_budgets.loc[0.20, 'utility_ci95_upper']:.3f}]；",
        f"- 最终经验排序状态：**{status['ranking_setting_status']}**。",
        "",
        "该结果只允许按冻结 gate 解释，不用开真值后的表现修改公式或阈值。",
        "",
    ]
    (reports / "E192_INTERPRETATION.md").write_text(
        "\n".join(interpretation), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
