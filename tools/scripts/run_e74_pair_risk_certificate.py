#!/usr/bin/env python3
"""E74: exact pair-risk certificate and three-dataset stratified audit.

For predictions p1, p2 and hidden truth y under RMSE distance d:

  max(d(p1,y), d(p2,y)) >= d(p1,p2)/2
  mean(d(p1,y), d(p2,y)) >= d(p1,p2)/2

For squared Euclidean error the following identity is exact:

  (MSE(p1,y)+MSE(p2,y))/2
    = MSE((p1+p2)/2,y) + MSE(p1,p2)/4.

The disagreement term therefore certifies pair-level risk without target
truth.  It does not identify which predictor is wrong and is not claimed to
upper-bound either individual error.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "实验结果" / "E74_pair_risk_certificate_20260711"
TABLES, REPORTS = OUT / "tables", OUT / "reports"
SOURCES = {
    "Adamson": ROOT / "docs" / "实验结果" / "E65_scgpt_formal_fixed_panel_20260711",
    "Norman": ROOT / "docs" / "实验结果" / "E67_norman_scgpt_formal_fixed_panel_20260711",
    "Frangieh": ROOT / "docs" / "实验结果" / "E72_frangieh_scgpt_formal_fixed_panel_20260711",
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def rank01(a: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(a, float), method="average") / len(a)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = np.asarray(a, float), np.asarray(b, float)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denominator = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
    if len(aa) < 3 or denominator <= 1e-12:
        return float("nan")
    return float(np.dot(aa, bb) / denominator)


def load_dataset(dataset: str, root: Path) -> pd.DataFrame:
    records = pd.read_csv(root / "tables" / "PREDICTION_RECORDS.csv")
    with np.load(root / "arrays" / "predicted_effects.npz") as store:
        predictions = {key: np.asarray(store[key], dtype=float) for key in store.files}
    with np.load(root / "arrays" / "true_effects.npz") as store:
        truths = {key: np.asarray(store[key], dtype=float) for key in store.files}
    rows = []
    for perturbation, group in records.groupby("perturbation", sort=True):
        if len(group) != 2:
            raise ValueError(f"{dataset}/{perturbation}: expected two predictor records, got {len(group)}")
        gears_row = group[group.predictor_name.str.contains("GEARS")]
        scgpt_row = group[group.predictor_name.str.contains("scGPT")]
        if len(gears_row) != 1 or len(scgpt_row) != 1:
            raise ValueError(f"{dataset}/{perturbation}: predictor identity is ambiguous")
        gears_row, scgpt_row = gears_row.iloc[0], scgpt_row.iloc[0]
        gears = predictions[gears_row.predicted_effect_key]
        scgpt = predictions[scgpt_row.predicted_effect_key]
        truth = truths[gears_row.true_effect_key]
        other_truth = truths[scgpt_row.true_effect_key]
        if not np.allclose(truth, other_truth, atol=1e-7, rtol=1e-6):
            raise ValueError(f"{dataset}/{perturbation}: predictors do not share truth")
        e_gears = rmse(gears, truth)
        e_scgpt = rmse(scgpt, truth)
        disagreement = rmse(gears, scgpt)
        mean_mse = 0.5 * (e_gears ** 2 + e_scgpt ** 2)
        ensemble_mse = rmse(0.5 * (gears + scgpt), truth) ** 2
        disagreement_mse_component = 0.25 * disagreement ** 2
        rows.append({
            "dataset": dataset, "perturbation": perturbation,
            "gears_rmse": e_gears, "scgpt_rmse": e_scgpt,
            "pair_mean_rmse": 0.5 * (e_gears + e_scgpt),
            "pair_max_rmse": max(e_gears, e_scgpt),
            "model_disagreement_rmse": disagreement,
            "triangle_certificate_rmse": 0.5 * disagreement,
            "triangle_mean_bound_holds": 0.5 * (e_gears + e_scgpt) + 1e-10 >= 0.5 * disagreement,
            "triangle_max_bound_holds": max(e_gears, e_scgpt) + 1e-10 >= 0.5 * disagreement,
            "pair_mean_mse": mean_mse,
            "ensemble_mean_prediction_mse": ensemble_mse,
            "disagreement_mse_component": disagreement_mse_component,
            "mse_identity_abs_error": abs(mean_mse - ensemble_mse - disagreement_mse_component),
            "certificate_fraction_of_pair_mean_rmse": (0.5 * disagreement) / (0.5 * (e_gears + e_scgpt)),
            "disagreement_component_fraction_of_pair_mean_mse": disagreement_mse_component / mean_mse,
            "gears_predicted_magnitude": float(np.linalg.norm(gears)),
            "scgpt_predicted_magnitude": float(np.linalg.norm(scgpt)),
        })
    frame = pd.DataFrame(rows)
    for column in ["model_disagreement_rmse", "gears_predicted_magnitude", "scgpt_predicted_magnitude"]:
        frame[f"rank__{column}"] = rank01(frame[column].to_numpy(float))
    frame["rank__magnitude_mean"] = 0.5 * (
        frame["rank__gears_predicted_magnitude"] + frame["rank__scgpt_predicted_magnitude"]
    )
    frame["rank__magnitude_max"] = np.maximum(
        frame["rank__gears_predicted_magnitude"], frame["rank__scgpt_predicted_magnitude"]
    )
    for target in ["gears_rmse", "scgpt_rmse", "pair_mean_rmse", "pair_max_rmse"]:
        frame[f"rank__{target}"] = rank01(frame[target].to_numpy(float))
    return frame


def stratified_stat(frame: pd.DataFrame, score: str, target: str) -> float:
    values = []
    for _, group in frame.groupby("dataset", sort=True):
        values.append(corr(group[score].to_numpy(float), group[target].to_numpy(float)))
    return float(np.mean(values))


def task_stratified_bootstrap(frame: pd.DataFrame, score: str, target: str, rng: np.random.Generator, n: int) -> tuple[float, float]:
    groups = [group.reset_index(drop=True) for _, group in frame.groupby("dataset", sort=True)]
    values = []
    for _ in range(n):
        per_dataset = []
        for group in groups:
            index = rng.integers(0, len(group), len(group))
            per_dataset.append(corr(group[score].to_numpy(float)[index], group[target].to_numpy(float)[index]))
        value = np.nanmean(per_dataset)
        if math.isfinite(value):
            values.append(value)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def within_dataset_permutation_p(frame: pd.DataFrame, score: str, target: str, observed: float, rng: np.random.Generator, n: int) -> float:
    groups = [group.reset_index(drop=True) for _, group in frame.groupby("dataset", sort=True)]
    exceed = 0
    for _ in range(n):
        value = np.mean([
            corr(group[score].to_numpy(float), rng.permutation(group[target].to_numpy(float)))
            for group in groups
        ])
        exceed += value >= observed
    return float((exceed + 1) / (n + 1))


def delta_bootstrap(frame: pd.DataFrame, candidate: str, baseline: str, target: str, rng: np.random.Generator, n: int) -> tuple[float, float]:
    groups = [group.reset_index(drop=True) for _, group in frame.groupby("dataset", sort=True)]
    values = []
    for _ in range(n):
        deltas = []
        for group in groups:
            index = rng.integers(0, len(group), len(group))
            y = group[target].to_numpy(float)[index]
            deltas.append(corr(group[candidate].to_numpy(float)[index], y) - corr(group[baseline].to_numpy(float)[index], y))
        value = np.nanmean(deltas)
        if math.isfinite(value):
            values.append(value)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def write_report(tasks: pd.DataFrame, summary: pd.DataFrame, delta: pd.DataFrame) -> None:
    lines = [
        "# E74｜模型家族分歧的 pair-risk 理论证书", "",
        "设两个模型在同一任务上的预测效应为 $p_1,p_2$，真实效应为 $y$，$d$ 为 RMSE 距离。三角不等式直接给出：", "",
        r"$$\max\{d(p_1,y),d(p_2,y)\}\geq \frac{1}{2}d(p_1,p_2),$$", "",
        r"$$\frac{d(p_1,y)+d(p_2,y)}{2}\geq \frac{1}{2}d(p_1,p_2).$$", "",
        "这个下界只用两个预测向量。它证明高分歧时至少一个模型存在相应规模的错误，也明确了它对应的是 pair mean/max risk，不能据此判断 GEARS 或 scGPT 谁错。", "",
        "平方误差下还有精确恒等式：", "",
        r"$$\frac{\mathrm{MSE}(p_1,y)+\mathrm{MSE}(p_2,y)}{2}=\mathrm{MSE}\left(\frac{p_1+p_2}{2},y\right)+\frac{1}{4}\mathrm{MSE}(p_1,p_2).$$", "",
        f"72 个真实任务中，三角下界违反数为 {int((~tasks.triangle_mean_bound_holds).sum() + (~tasks.triangle_max_bound_holds).sum())}；平方误差恒等式最大绝对数值误差为 {tasks.mse_identity_abs_error.max():.3e}。", "",
        "## 三数据集分层关联", "", "| score | target | 平均分层 ρ | task-bootstrap 95% CI | 分层置换 p |", "|---|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(f"| {row.score} | {row.target} | {row.stratified_mean_spearman:.3f} | [{row.bootstrap_ci95_low:.3f}, {row.bootstrap_ci95_high:.3f}] | {row.permutation_p_one_sided:.4f} |")
    lines += ["", "## 分歧相对固定 magnitude 聚合", "", "| target | baseline | Δρ | bootstrap 95% CI |", "|---|---|---:|---:|"]
    for _, row in delta.iterrows():
        lines.append(f"| {row.target} | {row.baseline} | {row.delta_stratified_spearman:.3f} | [{row.bootstrap_ci95_low:.3f}, {row.bootstrap_ci95_high:.3f}] |")
    lines += [
        "", "## 使用边界", "",
        "分歧证书是下界，不是误差上界，也不是概率校准。低分歧时两个模型仍可能一起犯错；高分歧时能够确定 pair-level 风险，但不能仅凭分歧定位错误模型。SafeConf 的学习部分只负责把这个证书与任务新颖性、支持度校准到具体筛选预算。",
    ]
    (REPORTS / "E74_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "README_先看这个.md").write_text("# E74 pair-risk certificate\n\n先读 `reports/E74_REPORT.md`。\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--n-permutation", type=int, default=10000)
    args = parser.parse_args()
    for directory in (TABLES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    tasks = pd.concat([load_dataset(name, path) for name, path in SOURCES.items()], ignore_index=True)
    score_columns = {
        "model_disagreement": "rank__model_disagreement_rmse",
        "magnitude_mean_rank": "rank__magnitude_mean",
        "magnitude_max_rank": "rank__magnitude_max",
    }
    target_columns = {
        "GEARS_RMSE": "rank__gears_rmse", "scGPT_RMSE": "rank__scgpt_rmse",
        "pair_mean_RMSE": "rank__pair_mean_rmse", "pair_max_RMSE": "rank__pair_max_rmse",
    }
    rng = np.random.default_rng(20260774)
    summary_rows = []
    for score_name, score in score_columns.items():
        for target_name, target in target_columns.items():
            observed = stratified_stat(tasks, score, target)
            low, high = task_stratified_bootstrap(tasks, score, target, rng, args.n_boot)
            pvalue = within_dataset_permutation_p(tasks, score, target, observed, rng, args.n_permutation)
            summary_rows.append({
                "score": score_name, "target": target_name, "n_datasets": len(SOURCES), "n_tasks": len(tasks),
                "stratified_mean_spearman": observed, "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                "permutation_p_one_sided": pvalue,
            })
    summary = pd.DataFrame(summary_rows)
    delta_rows = []
    candidate = score_columns["model_disagreement"]
    for target_name, target in {"pair_mean_RMSE": target_columns["pair_mean_RMSE"], "pair_max_RMSE": target_columns["pair_max_RMSE"]}.items():
        for baseline_name in ("magnitude_mean_rank", "magnitude_max_rank"):
            baseline = score_columns[baseline_name]
            observed = stratified_stat(tasks, candidate, target) - stratified_stat(tasks, baseline, target)
            low, high = delta_bootstrap(tasks, candidate, baseline, target, rng, args.n_boot)
            delta_rows.append({"target": target_name, "baseline": baseline_name, "delta_stratified_spearman": observed, "bootstrap_ci95_low": low, "bootstrap_ci95_high": high})
    delta = pd.DataFrame(delta_rows)
    tasks.to_csv(TABLES / "E74_TASK_CERTIFICATES.csv", index=False)
    summary.to_csv(TABLES / "E74_STRATIFIED_ASSOCIATION.csv", index=False)
    delta.to_csv(TABLES / "E74_INCREMENTAL_DELTA.csv", index=False)
    status = {
        "experiment": "E74_pair_risk_certificate", "generated_at": now(), "git_head_before_run": git_head(),
        "datasets": list(SOURCES), "n_tasks": len(tasks),
        "all_triangle_mean_bounds_hold": bool(tasks.triangle_mean_bound_holds.all()),
        "all_triangle_max_bounds_hold": bool(tasks.triangle_max_bound_holds.all()),
        "max_mse_identity_abs_error": float(tasks.mse_identity_abs_error.max()),
        "target_truth_used_in_certificate": False, "target_truth_used_for_evaluation_only": True,
        "n_boot": args.n_boot, "n_permutation": args.n_permutation,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(tasks, summary, delta)
    print(summary.to_string(index=False)); print(delta.to_string(index=False))


if __name__ == "__main__":
    main()
