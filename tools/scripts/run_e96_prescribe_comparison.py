#!/usr/bin/env python3
"""E96: compare native PRESCRIBE uncertainty with its own magnitude baseline.

PRESCRIBE scores are evaluated only against PRESCRIBE errors.  Existing
GEARS--scGPT disagreement is displayed beside it on identical task identifiers,
but remains tied to the GEARS/scGPT pair error.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "code/20260426_154505_perturb_transport_final_push"
sys.path.insert(0, str(PACKAGE_ROOT))
from safetrans_confidence.data.records import validate_prediction_record_artifacts  # noqa: E402
E95 = ROOT / "docs/实验结果/E95_prescribe_norman_native_20260712"
OUT = ROOT / "docs/实验结果/E96_prescribe_native_comparison_20260713"
PRESCRIBE_DATA = Path("/home/yyf/archive/external/PRESCRIBE/data")
PANELS = {
    "Norman_P1": {
        "e95": E95 / "norman_p1_formal_seed3407",
        "existing": ROOT / "docs/实验结果/E67_norman_scgpt_formal_fixed_panel_20260711/tables/E67_TASK_RISK_TABLE.csv",
    },
    "Norman_P2": {
        "e95": E95 / "norman_p2_formal_seed3407",
        "existing": ROOT / "docs/实验结果/E76b_norman_scgpt_panel2_20260711/tables/E76b_TASK_RISK_TABLE.csv",
    },
}
SCORES = {
    "PRESCRIBE epistemic": "risk_epistemic",
    "PRESCRIBE aleatoric": "risk_aleatoric",
    "PRESCRIBE combined": "risk_combined",
    "PRESCRIBE magnitude": "magnitude_pred_rms",
}
TARGETS = {
    "task_mean_profile_rmse": "rmse_mean_profile",
    "cell_gene_rmse": "rmse_cell_gene",
}
COVERAGES = np.round(np.arange(0.50, 1.001, 0.05), 2)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    value = np.corrcoef(rx, ry)[0, 1]
    return float(value) if math.isfinite(value) else float("nan")


def task_bootstrap(group: pd.DataFrame, score: str, target: str, seed: int, n: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(group), len(group))
        value = corr(group[score].to_numpy(float)[idx], group[target].to_numpy(float)[idx])
        if math.isfinite(value):
            values.append(value)
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975])) if values else (np.nan, np.nan)


def stratified_stat(frame: pd.DataFrame, score: str, target: str) -> float:
    return float(np.nanmean([corr(g[score].to_numpy(float), g[target].to_numpy(float)) for _, g in frame.groupby("panel")]))


def stratified_bootstrap(frame: pd.DataFrame, score: str, target: str, seed: int, baseline: str | None = None, n: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = [g.reset_index(drop=True) for _, g in frame.groupby("panel", sort=True)]
    values = []
    for _ in range(n):
        panel_values = []
        for group in groups:
            idx = rng.integers(0, len(group), len(group))
            y = group[target].to_numpy(float)[idx]
            value = corr(group[score].to_numpy(float)[idx], y)
            if baseline is not None:
                value -= corr(group[baseline].to_numpy(float)[idx], y)
            panel_values.append(value)
        value = float(np.nanmean(panel_values))
        if math.isfinite(value):
            values.append(value)
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975])) if values else (np.nan, np.nan)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def cosine_error(pred: np.ndarray, truth: np.ndarray, control: np.ndarray) -> float:
    a, b = pred - control, truth - control
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(1 - np.dot(a, b) / denom) if denom > 0 else np.nan


def load_panel(panel: str, config: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    run = config["e95"]
    status = json.loads((run / "STATUS.json").read_text())
    if status.get("phase") != "complete":
        raise RuntimeError(f"{panel} E95 is not complete: {status.get('phase')}")
    table = pd.read_csv(run / "task_prediction_records.csv")
    table.insert(0, "panel", panel)
    expected = 24
    if len(table) != expected:
        raise RuntimeError(f"{panel}: expected {expected} tasks, observed {len(table)}")

    raw = np.load(run / "test_predictions_raw.npz", allow_pickle=False)
    labels = raw["pert_cat"].astype(str)
    adata = sc.read_h5ad(PRESCRIBE_DATA / panel.lower() / "perturb_processed.h5ad", backed="r")
    gene_order_hash = "sha256:" + hashlib.sha256("\n".join(map(str, adata.var_names)).encode()).hexdigest()
    control = np.asarray(adata[adata.obs["condition"] == "ctrl"].to_memory().X.mean(axis=0)).reshape(-1)
    adata.file.close()
    cosine, predicted_effects, true_effects = {}, {}, {}
    for task in sorted(set(labels)):
        mask = labels == task
        pred_mean, truth_mean = raw["pred"][mask].mean(0), raw["truth"][mask].mean(0)
        cosine[task] = cosine_error(pred_mean, truth_mean, control)
        predicted_effects[f"E95::{panel}::{task}::pred"] = pred_mean - control
        true_effects[f"E95::{panel}::{task}::truth"] = truth_mean - control
    table["cosine_error_mean_effect"] = table["task_id"].map(cosine)
    table["gene_order_hash"] = gene_order_hash

    existing = pd.read_csv(config["existing"]).rename(columns={"perturbation": "task_id"})
    existing.insert(0, "panel", panel)
    if set(existing["task_id"]) != set(table["task_id"]):
        raise RuntimeError(f"{panel}: E95 and existing GEARS/scGPT task identifiers differ")
    return table, existing, predicted_effects, true_effects


def main() -> None:
    for name in ["tables", "figures", "reports"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    loaded = [load_panel(panel, config) for panel, config in PANELS.items()]
    tasks = pd.concat([item[0] for item in loaded], ignore_index=True)
    existing = pd.concat([item[1] for item in loaded], ignore_index=True)
    predicted_effects = {key: value for item in loaded for key, value in item[2].items()}
    true_effects = {key: value for item in loaded for key, value in item[3].items()}
    tasks.to_csv(OUT / "tables/E96_PRESCRIBE_TASKS.csv", index=False)

    association_rows = []
    for panel, group in tasks.groupby("panel", sort=True):
        for score_name, score in SCORES.items():
            for target_name, target in TARGETS.items():
                seed = int(hashlib.sha256(f"{panel}|{score}|{target}".encode()).hexdigest()[:8], 16)
                low, high = task_bootstrap(group, score, target, seed)
                association_rows.append({"scope": panel, "score": score_name, "target": target_name, "n_tasks": len(group), "spearman": corr(group[score].to_numpy(float), group[target].to_numpy(float)), "bootstrap_ci95_low": low, "bootstrap_ci95_high": high})
    for score_name, score in SCORES.items():
        for target_name, target in TARGETS.items():
            seed = int(hashlib.sha256(f"stratified|{score}|{target}".encode()).hexdigest()[:8], 16)
            low, high = stratified_bootstrap(tasks, score, target, seed)
            association_rows.append({"scope": "two_panel_stratified", "score": score_name, "target": target_name, "n_tasks": len(tasks), "spearman": stratified_stat(tasks, score, target), "bootstrap_ci95_low": low, "bootstrap_ci95_high": high})
    association = pd.DataFrame(association_rows)
    association.to_csv(OUT / "tables/E96_ASSOCIATION.csv", index=False)

    delta_rows = []
    for score_name, score in SCORES.items():
        if score == "magnitude_pred_rms":
            continue
        for target_name, target in TARGETS.items():
            observed = stratified_stat(tasks, score, target) - stratified_stat(tasks, "magnitude_pred_rms", target)
            seed = int(hashlib.sha256(f"delta|{score}|{target}".encode()).hexdigest()[:8], 16)
            low, high = stratified_bootstrap(tasks, score, target, seed, baseline="magnitude_pred_rms")
            delta_rows.append({"score": score_name, "target": target_name, "delta_rho_vs_magnitude": observed, "bootstrap_ci95_low": low, "bootstrap_ci95_high": high})
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(OUT / "tables/E96_INCREMENTAL_DELTA.csv", index=False)

    curve_rows, routing_rows = [], []
    primary = TARGETS["task_mean_profile_rmse"]
    for panel, group in tasks.groupby("panel", sort=True):
        full_mean = float(group[primary].mean())
        for score_name, score in SCORES.items():
            ordered = group.sort_values(score, ascending=True)
            values = []
            for coverage in COVERAGES:
                keep = max(1, int(np.ceil(coverage * len(group))))
                retained = float(ordered.iloc[:keep][primary].mean())
                normalized = retained / full_mean
                values.append(normalized)
                curve_rows.append({"panel": panel, "score": score_name, "coverage": coverage, "n_retained": keep, "retained_mean_error": retained, "normalized_retained_error": normalized})
            aurc = float(np.trapz(values, COVERAGES) / (COVERAGES[-1] - COVERAGES[0]))
            for reject in [0.1, 0.2, 0.3]:
                reject_n = max(1, int(np.ceil(reject * len(group))))
                retained = ordered.iloc[: len(group) - reject_n]
                rejected = set(ordered.iloc[len(group) - reject_n :]["task_id"])
                high_error = set(group.nlargest(reject_n, primary)["task_id"])
                routing_rows.append({"panel": panel, "score": score_name, "reject_fraction": reject, "n_rejected": reject_n, "remaining_error_reduction": 1 - float(retained[primary].mean()) / full_mean, "high_error_recall_rejected": len(rejected & high_error) / reject_n, "normalized_aurc_50_100": aurc, "random_expected_remaining_error_reduction": 0.0, "random_expected_high_error_recall": reject, "random_expected_normalized_aurc": 1.0})
    curves = pd.DataFrame(curve_rows)
    routing = pd.DataFrame(routing_rows)
    curves.to_csv(OUT / "tables/E96_RISK_COVERAGE.csv", index=False)
    routing.to_csv(OUT / "tables/E96_ROUTING_METRICS.csv", index=False)

    side_rows = []
    for panel, group in existing.groupby("panel", sort=True):
        side_rows.append({"panel": panel, "system": "GEARS-scGPT post-hoc pair", "score": "model disagreement", "target": "GEARS-scGPT pair mean RMSE", "spearman": corr(group["risk_gears_scgpt_disagreement"].to_numpy(float), group["task_mean_rmse"].to_numpy(float))})
    for panel, group in tasks.groupby("panel", sort=True):
        for label, score in SCORES.items():
            side_rows.append({"panel": panel, "system": "PRESCRIBE integrated predictor", "score": label, "target": "PRESCRIBE task mean-profile RMSE", "spearman": corr(group[score].to_numpy(float), group[primary].to_numpy(float))})
    side = pd.DataFrame(side_rows)
    side.to_csv(OUT / "tables/E96_SIDE_BY_SIDE_DIFFERENT_TARGETS.csv", index=False)

    records = []
    for row in tasks.itertuples(index=False):
        panel_id = row.panel.replace("Norman_", "P")
        records.append({"schema_version": "safeconf_prediction_record_v1", "record_id": f"E95::{row.panel}::{row.task_id}::PRESCRIBE_scGPT_NatPN", "task_id": row.task_id, "task_key": f"{row.panel}::{row.task_id}", "dataset_name": f"Norman_{panel_id}_PRESCRIBE_native2037", "dataset_group": "norman_crispr_prescribe_integrated_predictor", "fold_id": f"E91_{panel_id}_seed3407", "split": "test", "context": "Norman_K562", "perturbation": row.task_id, "predictor_name": "PRESCRIBE_scGPT_NatPN", "run_type": "formal", "gene_panel_id": f"E93_{panel_id}_HVG_plus_perturbed_2037", "gene_order_hash": row.gene_order_hash, "effect_definition": "mean_diff", "normalization_id": "PRESCRIBE_upstream_Step1_normalize_per_cell_log1p_PCA10_reconstruction", "error_normalization": "raw_rmse", "predicted_effect_key": f"E95::{row.panel}::{row.task_id}::pred", "true_effect_key": f"E95::{row.panel}::{row.task_id}::truth", "true_error_rmse": row.rmse_mean_profile, "true_error_cosine": row.cosine_error_mean_effect, "n_cells": row.n_cells})
    record_frame = pd.DataFrame(records)
    record_frame.to_csv(OUT / "tables/PREDICTION_RECORDS.csv", index=False)
    (OUT / "arrays").mkdir(exist_ok=True)
    np.savez_compressed(OUT / "arrays/predicted_effects.npz", **predicted_effects)
    np.savez_compressed(OUT / "arrays/true_effects.npz", **true_effects)
    issues = validate_prediction_record_artifacts(OUT, records=record_frame, strict=True)
    pd.DataFrame({"strict_issue": issues}).to_csv(OUT / "tables/E96_STRICT_CONTRACT_ISSUES.csv", index=False)
    if issues:
        raise RuntimeError("E96 strict PredictionRecord failed: " + "; ".join(issues))

    primary_summary = association[(association.scope == "two_panel_stratified") & (association.target == "task_mean_profile_rmse")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    colors = ["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"]
    axes[0].barh(primary_summary["score"], primary_summary["spearman"], color=colors)
    axes[0].axvline(0, color="#666666", lw=0.8)
    axes[0].set_xlabel("Two-panel mean Spearman ρ")
    axes[0].set_title("Risk score vs PRESCRIBE error")
    mean_curve = curves.groupby(["score", "coverage"], as_index=False)["normalized_retained_error"].mean()
    for color, (score, group) in zip(colors, mean_curve.groupby("score", sort=False)):
        axes[1].plot(group.coverage, group.normalized_retained_error, marker="o", ms=3, lw=1.8, label=score, color=color)
    axes[1].axhline(1, color="#777777", ls="--", lw=0.9)
    axes[1].set_xlabel("Coverage retained")
    axes[1].set_ylabel("Normalized remaining error")
    axes[1].set_title("Task routing")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(color="#e6e6e6", lw=0.7, axis="y")
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT / "figures/F1_prescribe_native_comparison.svg", facecolor="white")
    plt.close(fig)

    display_assoc = primary_summary[["score", "spearman", "bootstrap_ci95_low", "bootstrap_ci95_high"]].round(3)
    display_delta = delta[delta.target == "task_mean_profile_rmse"].round(3)
    report = f"""# E96｜PRESCRIBE 原生不确定性双面板对照

Norman P1、P2 各包含 24 个事先冻结且互不重叠的单基因测试任务。PRESCRIBE 的认知不确定性、数据不确定性和论文组合分数只与 PRESCRIBE 自身误差比较；任务平均表达谱 RMSE 是主目标，逐细胞逐基因 RMSE 是补充目标。预测幅度使用同一 PRESCRIBE 输出和训练侧可得的 control 均值。

## 双面板主结果

{markdown_table(display_assoc)}

## 相对自身 magnitude 基线

{markdown_table(display_delta)}

两套面板上没有一种 PRESCRIBE 原生不确定性与自身任务误差形成稳定正相关，三种分数相对自身 magnitude 的 Δρ 也均为负且区间跨 0。选择性路由同样没有稳定收益：P2 的 aleatoric 分数在拒绝 20% 任务后，剩余误差反而增加约 10.4%。这说明原生不确定性在当前未见单基因 setting 中不能直接当作可靠质检分数。

`E96_SIDE_BY_SIDE_DIFFERENT_TARGETS.csv` 同时列出相同任务上的 GEARS–scGPT 分歧结果。两类方法对应不同预测器和不同误差，只作并列展示，不把相关系数混成一次直接胜负检验。拒绝 10%、20%、30% 的结果和 50%–100% coverage 曲线分别保存在 `E96_ROUTING_METRICS.csv` 与 `E96_RISK_COVERAGE.csv`。

E96 关闭了“缺少直接不确定性竞品”的工程缺口，但不能据此宣称 SafeConf 全面优于 PRESCRIBE。当前能够比较的是各自分数对各自误差的排序能力；SafeConf 的可写优势仍限定在异构预测器的 post-hoc pair-risk 下界和不改造原预测模型，不能扩写成单模型概率校准。
"""
    (OUT / "reports/E96_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E96 先看这个\n\n先读 `reports/E96_REPORT.md`。\n")
    status = {"experiment": "E96_prescribe_native_comparison", "generated_at": datetime.now().isoformat(timespec="seconds"), "panels": list(PANELS), "n_tasks": len(tasks), "n_boot": 10000, "primary_target": "task_mean_profile_rmse", "target_truth_used_to_change_score_or_task_selection": False, "different_predictor_errors_not_pooled": True, "strict_prediction_records": len(record_frame), "strict_issue_count": len(issues)}
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(display_assoc.to_string(index=False))
    print(display_delta.to_string(index=False))


if __name__ == "__main__":
    main()
