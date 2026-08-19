#!/usr/bin/env python3
"""E69: source-trained risk calibration transfer between E65 and E67.

The perturbation predictors were trained within each dataset.  E69 transfers
only the task-risk calibrator: fit on source task errors, freeze every
standardization/feature/alpha/coefficient, then evaluate target task errors.
Target truth is never used for fitting or preprocessing.
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut


ROOT = Path(__file__).resolve().parents[2]
E65 = ROOT / "docs" / "实验结果" / "E65_scgpt_formal_fixed_panel_20260711"
E67 = ROOT / "docs" / "实验结果" / "E67_norman_scgpt_formal_fixed_panel_20260711"
OUT = ROOT / "docs" / "实验结果" / "E69_real_model_cross_dataset_risk_transfer_20260711"
TABLES, REPORTS, FIGURES = OUT / "tables", OUT / "reports", OUT / "figures"
ALPHAS = (0.0, 0.1, 1.0, 10.0, 100.0)
TARGET_FEATURES = {
    "gears_ensemble_rmse": {
        "magnitude": ["risk_gears_predicted_magnitude"],
        "magnitude_plus_disagreement": ["risk_gears_predicted_magnitude", "risk_gears_scgpt_disagreement"],
    },
    "scgpt_finetuned_rmse": {
        "magnitude": ["risk_scgpt_predicted_magnitude"],
        "magnitude_plus_disagreement": ["risk_scgpt_predicted_magnitude", "risk_gears_scgpt_disagreement"],
    },
    "task_mean_rmse": {
        "magnitude": ["risk_gears_predicted_magnitude", "risk_scgpt_predicted_magnitude"],
        "magnitude_plus_disagreement": ["risk_gears_predicted_magnitude", "risk_scgpt_predicted_magnitude", "risk_gears_scgpt_disagreement"],
    },
    "task_max_rmse": {
        "magnitude": ["risk_gears_predicted_magnitude", "risk_scgpt_predicted_magnitude"],
        "magnitude_plus_disagreement": ["risk_gears_predicted_magnitude", "risk_scgpt_predicted_magnitude", "risk_gears_scgpt_disagreement"],
    },
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan")
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def top20(score: np.ndarray, error: np.ndarray) -> tuple[int, float]:
    k = max(1, int(math.ceil(0.2 * len(score))))
    selected = error[np.argsort(-score, kind="stable")[:k]]
    return k, float(selected.mean() / error.mean())


def load_tasks(path: Path, dataset: str) -> pd.DataFrame:
    table = path / "tables" / "E65_TASK_RISK_TABLE.csv"
    frame = pd.read_csv(table)
    required = set(TARGET_FEATURES) | {
        "perturbation", "risk_gears_scgpt_disagreement",
        "risk_gears_predicted_magnitude", "risk_scgpt_predicted_magnitude",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{dataset} task table missing: {sorted(missing)}")
    frame = frame.copy()
    frame["dataset"] = dataset
    return frame


def source_standardize(x_source: np.ndarray, x_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_source.mean(axis=0)
    std = x_source.std(axis=0, ddof=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (x_source - mean) / std, (x_target - mean) / std, mean, std


def select_alpha_loo(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rows: list[tuple[float, float]] = []
    loo = LeaveOneOut()
    for alpha in ALPHAS:
        predictions = np.zeros(len(y), dtype=float)
        for train, val in loo.split(x):
            model = Ridge(alpha=alpha)
            model.fit(x[train], y[train])
            predictions[val] = model.predict(x[val])
        rows.append((alpha, float(np.mean(np.abs(predictions - y)))))
    return min(rows, key=lambda item: (item[1], item[0]))


def bootstrap_delta(combined: np.ndarray, baseline: np.ndarray, error: np.ndarray, seed: int, n_boot: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        index = rng.integers(0, len(error), len(error))
        delta = spearman(combined[index], error[index]) - spearman(baseline[index], error[index])
        if math.isfinite(delta):
            values.append(delta)
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def evaluate_direction(source: pd.DataFrame, target: pd.DataFrame, source_name: str, target_name: str, n_boot: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    summaries, predictions, coverages, deltas = [], [], [], []
    stored_predictions: dict[str, dict[str, np.ndarray]] = {}
    for target_column, model_specs in TARGET_FEATURES.items():
        y_source = source[target_column].to_numpy(float)
        y_target = target[target_column].to_numpy(float)
        stored_predictions[target_column] = {}
        for model_name, features in model_specs.items():
            x_source_raw = source[features].to_numpy(float)
            x_target_raw = target[features].to_numpy(float)
            x_source, x_target, mean, std = source_standardize(x_source_raw, x_target_raw)
            alpha, loo_mae = select_alpha_loo(x_source, y_source)
            model = Ridge(alpha=alpha).fit(x_source, y_source)
            target_risk = model.predict(x_target)
            stored_predictions[target_column][model_name] = target_risk
            k, enrichment = top20(target_risk, y_target)
            summaries.append({
                "source_dataset": source_name,
                "target_dataset": target_name,
                "target_error": target_column,
                "risk_model": model_name,
                "features": ";".join(features),
                "selected_alpha_source_loo": alpha,
                "source_loo_mae": loo_mae,
                "target_spearman": spearman(target_risk, y_target),
                "target_mae": float(np.mean(np.abs(target_risk - y_target))),
                "top20_k": k,
                "top20_error_enrichment": enrichment,
                "target_truth_used_for_fit_or_standardization": False,
                "source_feature_mean": json.dumps(mean.tolist()),
                "source_feature_std": json.dumps(std.tolist()),
                "coefficients": json.dumps(model.coef_.tolist()),
                "intercept": float(model.intercept_),
            })
            for task, actual, risk in zip(target["perturbation"], y_target, target_risk):
                predictions.append({
                    "source_dataset": source_name, "target_dataset": target_name,
                    "target_error": target_column, "risk_model": model_name,
                    "perturbation": task, "predicted_risk": float(risk),
                    "observed_target_error_evaluation_only": float(actual),
                })
            order = np.argsort(target_risk, kind="stable")
            for fraction in (0.0, 0.1, 0.2, 0.3):
                reject = int(math.ceil(fraction * len(order)))
                kept = order[: len(order) - reject] if reject else order
                coverages.append({
                    "source_dataset": source_name, "target_dataset": target_name,
                    "target_error": target_column, "risk_model": model_name,
                    "reject_fraction": fraction, "n_kept": len(kept),
                    "kept_mean_error": float(y_target[kept].mean()),
                    "kept_vs_all_error_ratio": float(y_target[kept].mean() / y_target.mean()),
                })
        baseline = stored_predictions[target_column]["magnitude"]
        combined = stored_predictions[target_column]["magnitude_plus_disagreement"]
        mean_delta, low, high = bootstrap_delta(combined, baseline, y_target, 20260769, n_boot)
        deltas.append({
            "source_dataset": source_name, "target_dataset": target_name,
            "target_error": target_column,
            "observed_spearman_delta_combined_minus_magnitude": spearman(combined, y_target) - spearman(baseline, y_target),
            "bootstrap_delta_mean": mean_delta,
            "bootstrap_delta_ci95_low": low,
            "bootstrap_delta_ci95_high": high,
            "combined_reliably_better_than_magnitude": bool(low > 0),
        })
    return summaries, predictions, coverages, deltas


def write_report(summary: pd.DataFrame, delta: pd.DataFrame) -> None:
    lines = [
        "# E69｜真实双模型风险校准器跨数据集迁移", "",
        "Adamson→Norman 与 Norman→Adamson 两个方向都只在 source tasks 上选择 Ridge alpha、计算特征均值/方差和拟合系数。target truth 只在冻结预测之后计算 Spearman、MAE、top20 enrichment 与 risk–coverage。这里迁移的是风险校准器，不声称 perturbation predictor 本身已经 A→B 迁移。", "",
        "## 目标域排序结果", "",
        "| source→target | error | risk model | target ρ | target MAE | top20 enrichment |", "|---|---|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(f"| {row.source_dataset}→{row.target_dataset} | {row.target_error} | {row.risk_model} | {row.target_spearman:.3f} | {row.target_mae:.4f} | {row.top20_error_enrichment:.3f} |")
    lines += ["", "## 加入模型分歧相对幅度基线的增量", "", "| source→target | error | Δρ | bootstrap 95% CI | 稳定超过幅度 |", "|---|---|---:|---:|---|"]
    for _, row in delta.iterrows():
        lines.append(f"| {row.source_dataset}→{row.target_dataset} | {row.target_error} | {row.observed_spearman_delta_combined_minus_magnitude:.3f} | [{row.bootstrap_delta_ci95_low:.3f}, {row.bootstrap_delta_ci95_high:.3f}] | {'是' if row.combined_reliably_better_than_magnitude else '否'} |")
    lines += [
        "", "## 解释边界", "",
        "1. 只有两个 source→target 方向、每个目标域 24 个任务，区间必须优先于点估计。",
        "2. 若 combined 没有稳定超过 magnitude，不能把 E65/E67 的同域相关性写成跨域独立增益。",
        "3. predictor 本身仍是各自在本数据集训练；完整回答老师的跨数据集问题还要让 predictor A 训练后直接在 B 输出。",
    ]
    (REPORTS / "E69_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "README_先看这个.md").write_text("# E69 跨数据集风险迁移\n\n先读 `reports/E69_REPORT.md`。\n", encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    for directory in (TABLES, REPORTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    adamson = load_tasks(E65, "Adamson")
    norman = load_tasks(E67, "Norman")
    all_summary, all_predictions, all_coverage, all_delta = [], [], [], []
    for source, target, source_name, target_name in [
        (adamson, norman, "Adamson", "Norman"),
        (norman, adamson, "Norman", "Adamson"),
    ]:
        summary, predictions, coverage, delta = evaluate_direction(source, target, source_name, target_name, args.n_boot)
        all_summary.extend(summary); all_predictions.extend(predictions); all_coverage.extend(coverage); all_delta.extend(delta)
    summary_df = pd.DataFrame(all_summary)
    predictions_df = pd.DataFrame(all_predictions)
    coverage_df = pd.DataFrame(all_coverage)
    delta_df = pd.DataFrame(all_delta)
    summary_df.to_csv(TABLES / "E69_TRANSFER_SUMMARY.csv", index=False)
    predictions_df.to_csv(TABLES / "E69_TARGET_PREDICTIONS.csv", index=False)
    coverage_df.to_csv(TABLES / "E69_RISK_COVERAGE.csv", index=False)
    delta_df.to_csv(TABLES / "E69_INCREMENTAL_DELTA.csv", index=False)
    status = {
        "experiment": "E69_real_model_cross_dataset_risk_transfer",
        "generated_at": now(), "git_head_before_run": git_head(),
        "source_experiments": ["E65", "E67"],
        "directions": ["Adamson_to_Norman", "Norman_to_Adamson"],
        "n_tasks_per_dataset": {"Adamson": len(adamson), "Norman": len(norman)},
        "target_truth_used_for_fit_standardization_or_alpha_selection": False,
        "transferred_object": "task-risk calibrator; not perturbation predictor",
        "n_boot": args.n_boot,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(summary_df, delta_df)
    print(delta_df.to_string(index=False))


if __name__ == "__main__":
    main()
