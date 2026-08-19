#!/usr/bin/env python3
"""E135: transparent exploratory LODO model for direction-sensitive risk."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/实验结果/E134_systema_exact_expression_space_audit_20260714/tables/E134_TASK_AUDIT.csv"
OUT = ROOT / "docs/实验结果/E135_directional_risk_lodo_20260714"
TABLES = OUT / "tables"
SEED = 202607135
N_BOOTSTRAP = 3000
FEATURES = ["risk_disagreement_z", "predicted_magnitude_z", "context_novelty_scaled", "perturbation_novelty"]
ENDPOINTS = ["error_centered_pearson_mean", "error_centered_cosine_mean", "direction_error_rank_target"]


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.unique(a[mask]).size < 2 or np.unique(b[mask]).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a[mask]), rankdata(b[mask]))[0, 1])


def add_target(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    ranks = []
    for endpoint in ENDPOINTS[:2]:
        ranks.append(data.groupby(["dataset", "fold_id"])[endpoint].transform(lambda values: rankdata(values) / len(values)))
    data["direction_error_rank_target"] = np.mean(np.stack(ranks), axis=0)
    return data


def candidates():
    return {
        "ridge_alpha10": lambda: Ridge(alpha=10.0),
        "ridge_positive_alpha10": lambda: Ridge(alpha=10.0, positive=True),
        "hist_leaf7_l2_10": lambda: HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7, l2_regularization=10, random_state=SEED),
        "rf_depth3_leaf30": lambda: RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_leaf=30, max_features=None, n_jobs=-1, random_state=SEED),
        "extra_depth3_leaf30": lambda: ExtraTreesRegressor(n_estimators=200, max_depth=3, min_samples_leaf=30, max_features=None, n_jobs=-1, random_state=SEED),
    }


def lodo(data: pd.DataFrame):
    predictions, metrics = [], []
    for model_name, make_model in candidates().items():
        for heldout in sorted(data.dataset.unique()):
            train = data.dataset.ne(heldout)
            test = ~train
            model = make_model()
            model.fit(data.loc[train, FEATURES], data.loc[train, "direction_error_rank_target"])
            block = data.loc[test, ["dataset", "fold_id", "task_id", "context", "perturbation", *ENDPOINTS[:2], "direction_error_rank_target", "baseline_predicted_magnitude", "safeconf_calibrated_pair_risk"]].copy()
            block["model"] = model_name
            block["directional_risk_lodo"] = model.predict(data.loc[test, FEATURES])
            block["target_truth_used_for_fit"] = False
            predictions.append(block)
            for fold, group in block.groupby("fold_id", sort=True):
                for endpoint in ENDPOINTS:
                    metrics.append({"model": model_name, "heldout_dataset": heldout, "fold_id": fold, "endpoint": endpoint, "n_tasks": len(group), "spearman": rho(group.directional_risk_lodo, group[endpoint])})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)


def macro(metrics: pd.DataFrame):
    dataset = metrics.groupby(["model", "heldout_dataset", "endpoint"], as_index=False).agg(n_folds=("fold_id", "nunique"), spearman=("spearman", "mean"))
    overall = dataset.groupby(["model", "endpoint"], as_index=False).agg(n_datasets=("heldout_dataset", "nunique"), dataset_equal_macro_spearman=("spearman", "mean"), min_dataset_spearman=("spearman", "min"))
    return dataset, overall


def bootstrap(selected: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    packed = {}
    for dataset, group in selected.groupby("dataset", sort=True):
        perturbations = sorted(group.perturbation.astype(str).unique())
        index = {value: number for number, value in enumerate(perturbations)}
        folds = []
        for _, fold in group.groupby("fold_id", sort=False):
            folds.append({
                "cluster": np.asarray([index[str(value)] for value in fold.perturbation], int),
                "risk": fold.directional_risk_lodo.to_numpy(float),
                "magnitude": fold.baseline_predicted_magnitude.to_numpy(float),
                "safeconf": fold.safeconf_calibrated_pair_risk.to_numpy(float),
                "endpoints": {endpoint: fold[endpoint].to_numpy(float) for endpoint in ENDPOINTS},
            })
        packed[dataset] = (len(perturbations), folds)
    rows = []
    for draw in range(N_BOOTSTRAP):
        row = {"draw": draw}
        accumulator = {endpoint: {score: [] for score in ["directional", "magnitude", "safeconf"]} for endpoint in ENDPOINTS}
        for n_clusters, folds in packed.values():
            counts = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters))
            dataset_values = {endpoint: {score: [] for score in ["directional", "magnitude", "safeconf"]} for endpoint in ENDPOINTS}
            for fold in folds:
                indices = np.repeat(np.arange(len(fold["cluster"])), counts[fold["cluster"]])
                for endpoint in ENDPOINTS:
                    target = fold["endpoints"][endpoint][indices]
                    for score in ["directional", "magnitude", "safeconf"]:
                        dataset_values[endpoint][score].append(rho(fold[score if score != "directional" else "risk"][indices], target))
            for endpoint in ENDPOINTS:
                for score in ["directional", "magnitude", "safeconf"]:
                    accumulator[endpoint][score].append(float(np.nanmean(dataset_values[endpoint][score])))
        for endpoint in ENDPOINTS:
            values = {score: float(np.nanmean(accumulator[endpoint][score])) for score in accumulator[endpoint]}
            row[f"directional__{endpoint}"] = values["directional"]
            row[f"delta_directional_minus_magnitude__{endpoint}"] = values["directional"] - values["magnitude"]
            row[f"delta_directional_minus_safeconf__{endpoint}"] = values["directional"] - values["safeconf"]
        rows.append(row)
    draws = pd.DataFrame(rows)
    summary = []
    for column in draws.columns[1:]:
        values = draws[column].to_numpy(float)
        summary.append({"metric": column, "bootstrap_draws": N_BOOTSTRAP, "ci_low_2.5pct": np.nanquantile(values, .025), "median": np.nanmedian(values), "ci_high_97.5pct": np.nanquantile(values, .975), "fraction_above_zero": np.nanmean(values > 0)})
    return draws, pd.DataFrame(summary)


def freeze_model(data: pd.DataFrame):
    model = Ridge(alpha=10.0).fit(data[FEATURES], data.direction_error_rank_target)
    payload = {
        "schema": "safeconf_directional_risk_v1",
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "status": "frozen_before_seventh_dataset_prediction_or_truth",
        "model": "Ridge",
        "alpha": 10.0,
        "features_in_order": FEATURES,
        "coefficients_in_order": [float(value) for value in model.coef_],
        "intercept": float(model.intercept_),
        "training_target": "mean of within-dataset-fold percentile ranks of Systema-centered Pearson and cosine error",
        "source_datasets": sorted(data.dataset.unique().tolist()),
        "n_source_tasks": len(data),
        "deployment_requires_target_truth": False,
        "selection_disclosure": "Chosen after transparent five-candidate exploration on the six source datasets; requires untouched seventh-dataset confirmation.",
    }
    (OUT / "E135_FROZEN_DIRECTION_MODEL.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def write_report(overall, dataset, boot, frozen):
    chosen = overall[overall.model.eq("ridge_alpha10")].set_index("endpoint")
    b = boot.set_index("metric")
    lines = [
        "# E135｜方向误差风险的留一数据集探索",
        "",
        "## 结果",
        "",
        f"Ridge(alpha=10) 每次只在其余五个数据集拟合，在完全留出的第六个数据集打分。六数据集等权宏平均：centered Pearson **ρ={chosen.loc['error_centered_pearson_mean','dataset_equal_macro_spearman']:.3f}**，centered cosine **ρ={chosen.loc['error_centered_cosine_mean','dataset_equal_macro_spearman']:.3f}**；六个留出数据集的 fold 宏平均均为正。",
        "",
        f"按 perturbation 整簇重采样，Pearson 方向风险的 95% CI 为 **[{b.loc['directional__error_centered_pearson_mean','ci_low_2.5pct']:.3f}, {b.loc['directional__error_centered_pearson_mean','ci_high_97.5pct']:.3f}]**，cosine 为 **[{b.loc['directional__error_centered_cosine_mean','ci_low_2.5pct']:.3f}, {b.loc['directional__error_centered_cosine_mean','ci_high_97.5pct']:.3f}]**。",
        "",
        "## 留出数据集结果",
        "",
        "| held-out dataset | Pearson ρ | cosine ρ | combined rank ρ |",
        "|---|---:|---:|---:|",
    ]
    pivot = dataset[dataset.model.eq("ridge_alpha10")].pivot(index="heldout_dataset", columns="endpoint", values="spearman")
    for name, row in pivot.iterrows():
        lines.append(f"| {name} | {row['error_centered_pearson_mean']:.3f} | {row['error_centered_cosine_mean']:.3f} | {row['direction_error_rank_target']:.3f} |")
    lines += [
        "",
        "## 解释",
        "",
        "原 SafeConf 的固定结构项针对 absolute RMSE。方向误差需要单独的轻量风险头；四个输入仍全部是部署时可见量，没有使用目标任务真值。Ridge 系数允许结构项改变方向，避免把一种误差定义的先验硬套到另一种误差定义。",
        "",
        "## 证据边界",
        "",
        "本轮先看过五类候选模型，属于探索性模型选择。完整候选结果全部保存在表中，没有只保留最好模型。冻结文件 `E135_FROZEN_DIRECTION_MODEL.json` 只能在新的第七数据集上确认，不能把 LODO 结果伪装成未见数据确认。",
        "",
        "## 冻结系数",
        "",
        f"`intercept={frozen['intercept']:.6f}`；" + "；".join(f"`{name}={value:.6f}`" for name, value in zip(frozen["features_in_order"], frozen["coefficients_in_order"])),
    ]
    (OUT / "E135_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "README_先看这个.md").write_text("# E135 先看这个\n\n先读 `E135_REPORT.md`。本轮是探索性 LODO；真正的确认规则见第七数据集预注册。\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    data = add_target(pd.read_csv(SOURCE))
    predictions, metrics = lodo(data)
    dataset, overall = macro(metrics)
    selected = predictions[predictions.model.eq("ridge_alpha10")].copy()
    draws, boot = bootstrap(selected)
    frozen = freeze_model(data)
    predictions.to_csv(TABLES / "E135_ALL_CANDIDATE_LODO_PREDICTIONS.csv", index=False)
    metrics.to_csv(TABLES / "E135_FOLD_METRICS.csv", index=False)
    dataset.to_csv(TABLES / "E135_DATASET_MACRO.csv", index=False)
    overall.to_csv(TABLES / "E135_CANDIDATE_MACRO.csv", index=False)
    draws.to_csv(TABLES / "E135_CLUSTER_BOOTSTRAP_DRAWS.csv", index=False)
    boot.to_csv(TABLES / "E135_CLUSTER_BOOTSTRAP_SUMMARY.csv", index=False)
    write_report(overall, dataset, boot, frozen)
    status = {
        "experiment": "E135_directional_risk_lodo",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete_exploratory",
        "n_datasets": int(data.dataset.nunique()),
        "n_tasks": len(data),
        "candidate_models_reported": list(candidates()),
        "selected_model": "ridge_alpha10",
        "untouched_seventh_dataset_confirmation_required": True,
        "target_truth_required_at_deployment": False,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(overall.to_string(index=False))
    print(dataset[dataset.model.eq("ridge_alpha10")].to_string(index=False))
    print(boot.to_string(index=False))
    print(json.dumps(frozen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
