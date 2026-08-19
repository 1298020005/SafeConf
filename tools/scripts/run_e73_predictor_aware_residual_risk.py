#!/usr/bin/env python3
"""E73: leave-one-dataset-out predictor-aware residual risk routing.

The method is deliberately small-data and model-agnostic.  A magnitude-only
Ridge model first predicts the within-dataset error rank.  A second Ridge
model is trained on *cross-dataset out-of-fold residuals* using only
GEARS--scGPT disagreement features.  For each held-out dataset, every alpha,
coefficient and source transformation is fixed without target error labels.

Feature ranks are computed within each screening batch.  This is transductive
with respect to deployable prediction features, but never with respect to
target expression truth or target error.
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
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "实验结果" / "E73_predictor_aware_residual_risk_20260711"
TABLES, REPORTS = OUT / "tables", OUT / "reports"
DATASETS = {
    "Adamson": ROOT / "docs" / "实验结果" / "E65_scgpt_formal_fixed_panel_20260711" / "tables" / "E65_TASK_RISK_TABLE.csv",
    "Norman": ROOT / "docs" / "实验结果" / "E67_norman_scgpt_formal_fixed_panel_20260711" / "tables" / "E65_TASK_RISK_TABLE.csv",
    "Frangieh": ROOT / "docs" / "实验结果" / "E72_frangieh_scgpt_formal_fixed_panel_20260711" / "tables" / "E65_TASK_RISK_TABLE.csv",
}
ERRORS = ("gears_ensemble_rmse", "scgpt_finetuned_rmse", "task_mean_rmse", "task_max_rmse")
BASE_FEATURES = ("rank_gears_magnitude", "rank_scgpt_magnitude")
RESIDUAL_FEATURES = ("rank_model_disagreement", "rank_relative_disagreement", "rank_magnitude_gap")
ALPHAS = (0.0, 0.1, 1.0, 10.0, 100.0)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def rank01(value: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(value, dtype=float)).rank(method="average", pct=True).to_numpy(float)


def enrich_features(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    required = {
        "perturbation", "risk_gears_scgpt_disagreement",
        "risk_gears_predicted_magnitude", "risk_scgpt_predicted_magnitude",
        *ERRORS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{dataset} missing columns: {sorted(missing)}")
    result = frame.copy()
    result["dataset"] = dataset
    gears = result["risk_gears_predicted_magnitude"].to_numpy(float)
    scgpt = result["risk_scgpt_predicted_magnitude"].to_numpy(float)
    disagree = result["risk_gears_scgpt_disagreement"].to_numpy(float)
    relative = disagree / (np.sqrt(np.maximum(gears, 1e-12) * np.maximum(scgpt, 1e-12)) + 1e-12)
    result["rank_gears_magnitude"] = rank01(gears)
    result["rank_scgpt_magnitude"] = rank01(scgpt)
    result["rank_model_disagreement"] = rank01(disagree)
    result["rank_relative_disagreement"] = rank01(relative)
    result["rank_magnitude_gap"] = rank01(np.abs(gears - scgpt))
    for error in ERRORS:
        result[f"rank_error__{error}"] = rank01(result[error])
    return result


def spearman(score: np.ndarray, error: np.ndarray) -> float:
    if len(score) < 3 or len(np.unique(score)) < 2 or len(np.unique(error)) < 2:
        return float("nan")
    return float(pd.Series(score).corr(pd.Series(error), method="spearman"))


def grouped_cv_alpha(frame: pd.DataFrame, features: tuple[str, ...], response: str) -> tuple[float, pd.DataFrame]:
    groups = sorted(frame["dataset"].unique())
    rows = []
    for alpha in ALPHAS:
        errors = []
        for held in groups:
            train = frame[frame.dataset != held]
            valid = frame[frame.dataset == held]
            model = Ridge(alpha=alpha).fit(train[list(features)], train[response])
            prediction = model.predict(valid[list(features)])
            errors.extend(np.abs(prediction - valid[response].to_numpy(float)))
        rows.append({"alpha": alpha, "grouped_source_cv_mae": float(np.mean(errors))})
    table = pd.DataFrame(rows)
    best = table.sort_values(["grouped_source_cv_mae", "alpha"], kind="stable").iloc[0]
    return float(best.alpha), table


def crossfit_predictions(frame: pd.DataFrame, features: tuple[str, ...], response: str, alpha: float) -> np.ndarray:
    prediction = np.full(len(frame), np.nan, dtype=float)
    for held in sorted(frame.dataset.unique()):
        train_mask = frame.dataset != held
        valid_mask = frame.dataset == held
        model = Ridge(alpha=alpha).fit(frame.loc[train_mask, list(features)], frame.loc[train_mask, response])
        prediction[valid_mask.to_numpy()] = model.predict(frame.loc[valid_mask, list(features)])
    if not np.isfinite(prediction).all():
        raise RuntimeError("source cross-fit left non-finite predictions")
    return prediction


def top20(score: np.ndarray, error: np.ndarray) -> tuple[int, float]:
    k = max(1, int(math.ceil(0.2 * len(score))))
    selected = error[np.argsort(-score, kind="stable")[:k]]
    return k, float(selected.mean() / error.mean())


def bootstrap_delta(candidate: np.ndarray, baseline: np.ndarray, error: np.ndarray, seed: int, n_boot: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        index = rng.integers(0, len(error), len(error))
        value = spearman(candidate[index], error[index]) - spearman(baseline[index], error[index])
        if math.isfinite(value):
            values.append(value)
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def fit_one(source: pd.DataFrame, target: pd.DataFrame, target_error: str) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    response = f"rank_error__{target_error}"
    base_alpha, base_cv = grouped_cv_alpha(source, BASE_FEATURES, response)
    source = source.copy()
    source["base_oof"] = crossfit_predictions(source, BASE_FEATURES, response, base_alpha)
    source["residual_oof_target"] = source[response] - source["base_oof"]
    residual_alpha, residual_cv = grouped_cv_alpha(source, RESIDUAL_FEATURES, "residual_oof_target")

    base_model = Ridge(alpha=base_alpha).fit(source[list(BASE_FEATURES)], source[response])
    residual_model = Ridge(alpha=residual_alpha).fit(source[list(RESIDUAL_FEATURES)], source["residual_oof_target"])
    baseline = base_model.predict(target[list(BASE_FEATURES)])
    residual = residual_model.predict(target[list(RESIDUAL_FEATURES)])
    candidate = baseline + residual
    disagreement_only = target["rank_model_disagreement"].to_numpy(float)
    fixed_rank_average = (
        target["rank_gears_magnitude"].to_numpy(float)
        + target["rank_scgpt_magnitude"].to_numpy(float)
        + target["rank_model_disagreement"].to_numpy(float)
    ) / 3.0
    actual = target[target_error].to_numpy(float)
    scores = {
        "magnitude_only": baseline,
        "disagreement_only": disagreement_only,
        "fixed_rank_average": fixed_rank_average,
        "predictor_aware_residual": candidate,
    }

    summary_rows, prediction_rows, coverage_rows = [], [], []
    for name, score in scores.items():
        k, enrichment = top20(score, actual)
        summary_rows.append({
            "target_dataset": str(target.dataset.iloc[0]), "target_error": target_error,
            "risk_model": name, "target_spearman": spearman(score, actual),
            "top20_k": k, "top20_error_enrichment": enrichment,
            "target_truth_used_for_fit_feature_ranking_or_selection": False,
        })
        for task, risk, error in zip(target.perturbation, score, actual):
            prediction_rows.append({
                "target_dataset": str(target.dataset.iloc[0]), "target_error": target_error,
                "risk_model": name, "perturbation": task, "predicted_risk_rank_score": float(risk),
                "observed_error_evaluation_only": float(error),
            })
        order = np.argsort(score, kind="stable")
        for fraction in (0.0, 0.1, 0.2, 0.3):
            reject = int(math.ceil(fraction * len(order)))
            kept = order[:len(order) - reject] if reject else order
            coverage_rows.append({
                "target_dataset": str(target.dataset.iloc[0]), "target_error": target_error,
                "risk_model": name, "reject_fraction": fraction, "n_kept": len(kept),
                "kept_mean_error": float(actual[kept].mean()),
                "kept_vs_all_error_ratio": float(actual[kept].mean() / actual.mean()),
            })
    diagnostics = {
        "target_dataset": str(target.dataset.iloc[0]), "target_error": target_error,
        "source_datasets": ";".join(sorted(source.dataset.unique())),
        "base_alpha": base_alpha, "residual_alpha": residual_alpha,
        "base_features": ";".join(BASE_FEATURES), "residual_features": ";".join(RESIDUAL_FEATURES),
        "base_coefficients": json.dumps(base_model.coef_.tolist()),
        "residual_coefficients": json.dumps(residual_model.coef_.tolist()),
        "base_intercept": float(base_model.intercept_), "residual_intercept": float(residual_model.intercept_),
    }
    cv_rows = []
    for stage, table in (("magnitude_base", base_cv), ("disagreement_residual", residual_cv)):
        for _, row in table.iterrows():
            cv_rows.append({"target_dataset": str(target.dataset.iloc[0]), "target_error": target_error, "stage": stage, **row.to_dict()})
    return diagnostics, summary_rows, prediction_rows, coverage_rows, cv_rows


def write_report(summary: pd.DataFrame, delta: pd.DataFrame) -> None:
    lines = [
        "# E73｜Predictor-aware residual risk：三数据集冻结外推", "",
        "每轮留出一个完整数据集。来源数据先拟合 magnitude-only 风险排序，再用跨数据集 out-of-fold 残差训练模型分歧校正器。目标数据的误差、真实表达和效应不参与特征、标准化、选参或拟合。", "",
        "## 目标域结果", "", "| target | error | method | ρ | top20 enrichment |", "|---|---|---|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(f"| {row.target_dataset} | {row.target_error} | {row.risk_model} | {row.target_spearman:.3f} | {row.top20_error_enrichment:.3f} |")
    lines += ["", "## 相对 magnitude-only 的增量", "", "| target | error | Δρ | bootstrap 95% CI | 稳定为正 |", "|---|---|---:|---:|---|"]
    for _, row in delta.iterrows():
        lines.append(f"| {row.target_dataset} | {row.target_error} | {row.delta_spearman:.3f} | [{row.bootstrap_ci95_low:.3f}, {row.bootstrap_ci95_high:.3f}] | {'是' if row.reliably_positive else '否'} |")
    lines += [
        "", "## 解释", "",
        "这个实验的主判断是三份完整目标数据集上的增量是否方向一致，以及区间是否支持它。单个数据集的正结果不作为方法成立的证据。batch 内特征排序需要拿到待筛选任务的一批预测，但不读取目标真值。",
    ]
    (REPORTS / "E73_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "README_先看这个.md").write_text("# E73 predictor-aware residual risk\n\n先读 `reports/E73_REPORT.md`。\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()
    for directory in (TABLES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    frames = {name: enrich_features(pd.read_csv(path), name) for name, path in DATASETS.items()}
    diagnostics, summaries, predictions, coverages, cv_rows = [], [], [], [], []
    for target_name, target in frames.items():
        source = pd.concat([frame for name, frame in frames.items() if name != target_name], ignore_index=True)
        for error in ERRORS:
            diagnostic, summary, prediction, coverage, cv = fit_one(source, target, error)
            diagnostics.append(diagnostic); summaries.extend(summary); predictions.extend(prediction); coverages.extend(coverage); cv_rows.extend(cv)
    summary = pd.DataFrame(summaries)
    prediction = pd.DataFrame(predictions)
    delta_rows = []
    for (target, error), group in prediction.groupby(["target_dataset", "target_error"]):
        pivot = group.pivot(index="perturbation", columns="risk_model", values="predicted_risk_rank_score")
        actual = group.drop_duplicates("perturbation").set_index("perturbation").loc[pivot.index, "observed_error_evaluation_only"].to_numpy(float)
        baseline = pivot["magnitude_only"].to_numpy(float)
        candidate = pivot["predictor_aware_residual"].to_numpy(float)
        mean_delta, low, high = bootstrap_delta(candidate, baseline, actual, 20260773, args.n_boot)
        delta_rows.append({
            "target_dataset": target, "target_error": error,
            "delta_spearman": spearman(candidate, actual) - spearman(baseline, actual),
            "bootstrap_delta_mean": mean_delta, "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
            "reliably_positive": bool(low > 0),
        })
    delta = pd.DataFrame(delta_rows)
    summary.to_csv(TABLES / "E73_SUMMARY.csv", index=False)
    prediction.to_csv(TABLES / "E73_TARGET_PREDICTIONS.csv", index=False)
    pd.DataFrame(coverages).to_csv(TABLES / "E73_RISK_COVERAGE.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(TABLES / "E73_MODEL_DIAGNOSTICS.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(TABLES / "E73_SOURCE_GROUPED_CV.csv", index=False)
    delta.to_csv(TABLES / "E73_INCREMENTAL_DELTA.csv", index=False)
    status = {
        "experiment": "E73_predictor_aware_residual_risk", "generated_at": now(), "git_head_before_run": git_head(),
        "datasets": list(DATASETS), "n_tasks": {name: len(frame) for name, frame in frames.items()},
        "evaluation": "leave-one-entire-dataset-out", "target_truth_used_for_fit_feature_ranking_or_selection": False,
        "feature_transform": "within-screening-batch percentile ranks; target deployable features allowed, target labels forbidden",
        "n_boot": args.n_boot,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(summary, delta)
    print(delta.to_string(index=False))


if __name__ == "__main__":
    main()
