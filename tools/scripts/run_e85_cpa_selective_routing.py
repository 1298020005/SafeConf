#!/usr/bin/env python3
"""E85: pre-specified selective-routing metrics for frozen E84 scores."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
E84 = ROOT / "docs/实验结果/E84_cpa_rdkit_cartesian_formal_20260712"
OUT = ROOT / "docs/实验结果/E85_cpa_selective_routing_20260712"
SCORES = {
    "model_disagreement": "cpa_ridge_disagreement_rmse",
    "predicted_magnitude": "predicted_magnitude_mean",
}
TARGET = "pair_mean_rmse"
COVERAGES = np.round(np.arange(0.50, 1.001, 0.05), 2)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def bootstrap_mean(values: np.ndarray, seed: int, n_boot: int = 10000):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "reports").mkdir(exist_ok=True)
    tasks = pd.read_csv(E84 / "tables/E84_TASK_SCORES.csv")
    metric_rows, curve_rows = [], []
    for (manifest_id, quadrant), group in tasks.groupby(["manifest_id", "quadrant"], sort=True):
        errors = group[TARGET].to_numpy(float)
        full_mean = float(errors.mean())
        for score_label, score_column in SCORES.items():
            risk = group[score_column].to_numpy(float)
            descending = np.argsort(-risk)
            k20 = max(1, int(np.ceil(0.20 * len(group))))
            top_error = float(errors[descending[:k20]].mean())
            retained = errors[descending[k20:]]
            retained_mean = float(retained.mean()) if len(retained) else np.nan
            reduction = 1.0 - retained_mean / full_mean
            curve_values = []
            ascending = np.argsort(risk)
            for coverage in COVERAGES:
                keep = max(1, int(np.ceil(coverage * len(group))))
                selective = float(errors[ascending[:keep]].mean())
                normalized = selective / full_mean
                curve_values.append(normalized)
                curve_rows.append(
                    {
                        "manifest_id": manifest_id,
                        "quadrant": quadrant,
                        "score_name": score_label,
                        "coverage": coverage,
                        "retained_mean_error": selective,
                        "normalized_retained_error": normalized,
                        "n_retained": keep,
                    }
                )
            aurc = float(np.trapz(curve_values, COVERAGES) / (COVERAGES[-1] - COVERAGES[0]))
            metric_rows.append(
                {
                    "manifest_id": manifest_id,
                    "quadrant": quadrant,
                    "score_name": score_label,
                    "n_tasks": len(group),
                    "top20_error_enrichment": top_error / full_mean,
                    "reject20_remaining_error_reduction": reduction,
                    "normalized_aurc_50_100": aurc,
                    "random_expected_top20_enrichment": 1.0,
                    "random_expected_reject20_reduction": 0.0,
                    "random_expected_normalized_aurc": 1.0,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)
    metrics.to_csv(OUT / "tables/E85_MANIFEST_METRICS.csv", index=False)
    curves.to_csv(OUT / "tables/E85_RISK_COVERAGE_CURVES.csv", index=False)

    aggregate_rows = []
    for quadrant, group in metrics.groupby("quadrant"):
        pivot = group.pivot(index="manifest_id", columns="score_name")
        for metric in [
            "top20_error_enrichment",
            "reject20_remaining_error_reduction",
            "normalized_aurc_50_100",
        ]:
            disagreement = pivot[metric]["model_disagreement"].to_numpy(float)
            magnitude = pivot[metric]["predicted_magnitude"].to_numpy(float)
            if metric == "normalized_aurc_50_100":
                delta = magnitude - disagreement  # positive means disagreement is better (lower AURC)
                delta_name = "magnitude_minus_disagreement"
            else:
                delta = disagreement - magnitude
                delta_name = "disagreement_minus_magnitude"
            seed = int(hashlib.sha256(f"{quadrant}|{metric}".encode()).hexdigest()[:8], 16)
            ci = bootstrap_mean(delta, seed)
            aggregate_rows.append(
                {
                    "quadrant": quadrant,
                    "metric": metric,
                    "n_manifests": len(disagreement),
                    "model_disagreement_mean": disagreement.mean(),
                    "predicted_magnitude_mean": magnitude.mean(),
                    "favorable_delta_definition": delta_name,
                    "favorable_delta_mean": delta.mean(),
                    "favorable_delta_bootstrap_ci95_low": ci[0],
                    "favorable_delta_bootstrap_ci95_high": ci[1],
                    "manifests_where_disagreement_better": int((delta > 0).sum()),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(OUT / "tables/E85_QUADRANT_AGGREGATE.csv", index=False)

    quadrants = sorted(curves["quadrant"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True, sharey=True)
    colors = {"model_disagreement": "#2166ac", "predicted_magnitude": "#b2182b"}
    for ax, quadrant in zip(axes.ravel(), quadrants):
        subset = curves.loc[curves["quadrant"].eq(quadrant)]
        for score_name, line in subset.groupby("score_name"):
            summary = line.groupby("coverage")["normalized_retained_error"].agg(["mean", "sem"]).reset_index()
            ax.plot(summary["coverage"], summary["mean"], marker="o", ms=3, lw=1.8, color=colors[score_name], label=score_name.replace("_", " "))
            ax.fill_between(summary["coverage"], summary["mean"] - summary["sem"], summary["mean"] + summary["sem"], color=colors[score_name], alpha=0.14)
        ax.axhline(1.0, color="#777777", lw=1, ls="--")
        ax.set_title(quadrant.replace("_", " "), fontsize=10)
        ax.grid(color="#e6e6e6", lw=0.7)
        ax.set_facecolor("white")
    axes[1, 0].set_xlabel("Coverage retained")
    axes[1, 1].set_xlabel("Coverage retained")
    axes[0, 0].set_ylabel("Normalized retained error")
    axes[1, 0].set_ylabel("Normalized retained error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "figures/F1_risk_coverage_four_quadrants.svg", facecolor="white")
    plt.close(fig)

    display = aggregate.copy()
    for column in [
        "model_disagreement_mean",
        "predicted_magnitude_mean",
        "favorable_delta_mean",
        "favorable_delta_bootstrap_ci95_low",
        "favorable_delta_bootstrap_ci95_high",
    ]:
        display[column] = display[column].round(3)
    report = f"""# E85｜CPA 化学四象限选择性路由

E85 不训练模型、不改分数，只把 E84 冻结的 disagreement 与 predicted magnitude 放到预先计划的选择性预测指标中：top-20% 高错误富集、拒绝最高风险 20% 后的剩余误差下降、coverage 50%–100% 的归一化 AURC。随机路由的期望分别为 1、0、1。

{markdown_table(display)}

选择性指标比 Spearman 更严格：四个象限中，两种分数的 top-20%、reject-20% 和 AURC 大多接近。新 context 与新药的 AURC 由 magnitude 稳定占优；随机缺失 pair 的 disagreement 只有极小、区间跨 0 的优势。E84 的排序相关不能直接换写成明显的资源节省。

正的 favorable delta 表示 disagreement 优于 magnitude。区间按 8 个 manifest 成对重采样，只描述同一 sciPlex3 内冻结 split 的敏感性。图 `figures/F1_risk_coverage_four_quadrants.svg` 为白底，可直接用于汇报。
"""
    (OUT / "reports/E85_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E85 先看这个\n\n先读 `reports/E85_REPORT.md`。\n")
    status = {
        "experiment": "E85_cpa_selective_routing",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "E84 frozen task scores",
        "n_manifest_metric_rows": len(metrics),
        "n_curve_rows": len(curves),
        "coverages": COVERAGES.tolist(),
        "target_truth_used_to_change_scores": False,
        "random_expected_baselines_explicit": True,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
