#!/usr/bin/env python3
"""E108: formal SafeConf audit from context-aware scGPT and GEARS outputs."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code/20260426_154505_perturb_transport_final_push"
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

from safetrans_confidence.data.records import validate_prediction_record_artifacts

E106 = ROOT / "docs/实验结果/E106_frangieh_context_scgpt_20260713"
E107 = ROOT / "docs/实验结果/E107_frangieh_context_gears_20260713"
CONTRACT = ROOT / "docs/实验结果/E97_frangieh_gene_cartesian_contract_20260713/manifests/E97_TASK_MANIFEST.csv"
SOURCE = Path("/home/yyf/data/scgpt_formal_frangieh_fixed_panel_20260711/frangieh_e72_fixed512/perturb_processed.h5ad")
OUT = ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713"
TABLES, ARRAYS, REPORTS, FIGURES = OUT / "tables", OUT / "arrays", OUT / "reports", OUT / "figures"
SEED = 202607108
N_BOOTSTRAP = 3000


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def cosine_error(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(1.0 - np.dot(a, b) / denominator) if denominator > 1e-12 else 1.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.unique(a[mask]).size < 2 or np.unique(b[mask]).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a[mask]), rankdata(b[mask]))[0, 1])


def robust_z(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, float)
    center = float(np.median(reference))
    mad = float(np.median(np.abs(reference - center)))
    scale = max(1.4826 * mad, float(np.std(reference)), 1e-8)
    return np.clip((np.asarray(values, float) - center) / scale, -5.0, 5.0)


def dense_mean(matrix: Any) -> np.ndarray:
    if issparse(matrix):
        return np.asarray(matrix.mean(axis=0)).reshape(-1).astype(np.float32)
    return np.asarray(matrix, np.float32).mean(axis=0)


def load_controls() -> tuple[dict[str, np.ndarray], list[str]]:
    adata = sc.read_h5ad(SOURCE)
    contexts = adata.obs["cell_type"].astype(str)
    conditions = adata.obs["condition"].astype(str)
    controls = {
        context: dense_mean(adata[contexts.eq(context).to_numpy() & conditions.eq("ctrl").to_numpy()].X)
        for context in sorted(contexts.unique())
    }
    genes = adata.var["gene_name"].astype(str).tolist()
    return controls, genes


def structural_features(frame: pd.DataFrame, controls: dict[str, np.ndarray]) -> pd.DataFrame:
    distances = [1.0 - cosine(controls[a], controls[b]) for i, a in enumerate(sorted(controls)) for b in sorted(controls)[i + 1:]]
    scale = max(float(np.median([value for value in distances if value > 1e-10])), 1e-8)
    result = frame.copy()
    novelty, support = [], []
    for row in result.itertuples(index=False):
        train = result[(result.fold_id == row.fold_id) & result.split.eq("train")]
        train_contexts = sorted(train.context.astype(str).unique())
        max_sim = max(cosine(controls[str(row.context)], controls[value]) for value in train_contexts)
        novelty.append(0.0 if str(row.context) in train_contexts else min((1.0 - max_sim) / scale, 5.0))
        support.append(int(train.perturbation.astype(str).eq(str(row.perturbation)).sum()))
    result["context_novelty_scaled"] = novelty
    result["training_support_count"] = support
    result["perturbation_novelty"] = 1.0 / (1.0 + result.training_support_count.to_numpy(float))
    return result


def build_tasks() -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    manifest = pd.read_csv(CONTRACT)
    controls, genes = load_controls()
    panel_hash = "sha256:" + hashlib.sha256("\n".join(genes).encode()).hexdigest()
    task_rows, records = [], []
    pred_export: dict[str, np.ndarray] = {}
    true_export: dict[str, np.ndarray] = {}
    model_specs = [
        ("scGPT_context_mean_finetuned", E106),
        ("GEARS_context_mean_trainonly_graphs", E107),
    ]
    for fold in sorted(manifest.fold_id.unique()):
        stores = {}
        metrics = {}
        truths = {}
        for model_name, root in model_specs:
            directory = root / "folds" / fold
            metrics[model_name] = pd.read_csv(directory / "ALL_TASK_METRICS.csv")
            stores[model_name] = np.load(directory / "predicted_effects.npz")
            truths[model_name] = np.load(directory / "true_effects.npz")
        left, right = [metrics[name] for name, _ in model_specs]
        keys = ["fold_id", "task_id", "split", "context", "perturbation", "setting"]
        if not left[keys].equals(right[keys]):
            raise RuntimeError(f"E106/E107 task contract differs in {fold}")
        for index, base in left.iterrows():
            split, task_id = str(base.split), str(base.task_id)
            array_key = f"{split}::{task_id}"
            pred = {name: np.asarray(stores[name][array_key], np.float32) for name, _ in model_specs}
            truth_a = np.asarray(truths[model_specs[0][0]][array_key], np.float32)
            truth_b = np.asarray(truths[model_specs[1][0]][array_key], np.float32)
            if not np.allclose(truth_a, truth_b, atol=1e-6):
                raise RuntimeError(f"model truth mismatch: {fold} {task_id}")
            errors = {name: rmse(vector, truth_a) for name, vector in pred.items()}
            disagreement = rmse(pred[model_specs[0][0]], pred[model_specs[1][0]])
            ensemble = np.mean(np.stack(list(pred.values())), axis=0)
            task_rows.append({
                "fold_id": fold,
                "task_id": task_id,
                "split": split,
                "context": str(base.context),
                "perturbation": str(base.perturbation),
                "setting": str(base.setting),
                "risk_model_disagreement": disagreement,
                "baseline_predicted_magnitude": float(np.sqrt(np.mean(ensemble ** 2))),
                "error_scgpt_rmse": errors[model_specs[0][0]],
                "error_gears_rmse": errors[model_specs[1][0]],
                "error_two_predictor_mean_rmse": float(np.mean(list(errors.values()))),
                "error_two_predictor_max_rmse": float(np.max(list(errors.values()))),
                "deployable_features_use_target_perturbed_truth": False,
            })
            if split == "test":
                truth_key = f"E108::{fold}::{task_id}::truth"
                true_export[truth_key] = truth_a
                match = manifest[(manifest.fold_id == fold) & (manifest.context.astype(str) == str(base.context)) & (manifest.perturbation.astype(str) == str(base.perturbation))]
                n_cells = int(match.n_cells.iloc[0])
                for model_name, _ in model_specs:
                    pred_key = f"E108::{fold}::{task_id}::{model_name}::prediction"
                    pred_export[pred_key] = pred[model_name]
                    records.append({
                        "schema_version": "safeconf_prediction_record_v1",
                        "record_id": pred_key.removesuffix("::prediction"),
                        "task_id": task_id,
                        "task_key": f"E108::{fold}::{task_id}",
                        "dataset_name": "Frangieh2019_E97_formal512",
                        "dataset_group": "frangieh_context_gene_cartesian_formal",
                        "fold_id": fold,
                        "split": "test",
                        "context": str(base.context),
                        "perturbation": str(base.perturbation),
                        "predictor_name": model_name,
                        "run_type": "formal",
                        "gene_panel_id": "frangieh_context_safe_fixed512",
                        "gene_order_hash": panel_hash,
                        "effect_definition": "mean_diff",
                        "normalization_id": "frangieh_mean_expression_minus_same_context_ctrl512_v1",
                        "error_normalization": "raw_rmse",
                        "predicted_effect_key": pred_key,
                        "true_effect_key": truth_key,
                        "true_error_rmse": errors[model_name],
                        "true_error_cosine": cosine_error(pred[model_name], truth_a),
                        "n_cells": n_cells,
                    })
        for store in [*stores.values(), *truths.values()]:
            store.close()
    tasks = structural_features(pd.DataFrame(task_rows), controls)
    return tasks, records, pred_export, true_export, genes


def calibrate(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs, calibrators = [], []
    for fold, group in tasks.groupby("fold_id", sort=True):
        group = group.copy()
        reference = group.split.eq("val")
        group["risk_disagreement_z"] = robust_z(group.risk_model_disagreement, group.loc[reference, "risk_model_disagreement"])
        group["predicted_magnitude_z"] = robust_z(group.baseline_predicted_magnitude, group.loc[reference, "baseline_predicted_magnitude"])
        group["safeconf_frozen_pair_risk"] = group.risk_disagreement_z + group.context_novelty_scaled + group.perturbation_novelty
        x_val = group.loc[reference, ["risk_disagreement_z", "predicted_magnitude_z"]].to_numpy(float)
        y_val = group.loc[reference, "error_two_predictor_mean_rmse"].to_numpy(float)
        model = Ridge(alpha=1.0, positive=True).fit(x_val, y_val)
        base = model.predict(group[["risk_disagreement_z", "predicted_magnitude_z"]].to_numpy(float))
        structural_scale = max(float(np.std(y_val)), 0.05 * float(np.mean(y_val)), 1e-6)
        support_ref = float(group.loc[reference, "perturbation_novelty"].median())
        structural = group.context_novelty_scaled + np.maximum(group.perturbation_novelty - support_ref, 0.0)
        group["safeconf_calibrated_pair_risk"] = base + structural_scale * structural
        threshold = float(group.loc[reference, "safeconf_calibrated_pair_risk"].quantile(.80))
        group["accepted_by_validation_q80"] = group.safeconf_calibrated_pair_risk <= threshold
        group["validation_q80_threshold"] = threshold
        outputs.append(group)
        calibrators.append({
            "fold_id": fold,
            "n_validation_tasks": int(reference.sum()),
            "ridge_alpha": 1.0,
            "coefficient_disagreement_z": float(model.coef_[0]),
            "coefficient_predicted_magnitude_z": float(model.coef_[1]),
            "intercept": float(model.intercept_),
            "structural_scale": structural_scale,
            "validation_q80_threshold": threshold,
            "test_truth_used": False,
        })
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(calibrators)


def coverage_improvement(score: np.ndarray, error: np.ndarray, coverage: float = .80) -> float:
    keep = max(1, int(math.ceil(len(score) * coverage)))
    selected = np.argsort(score, kind="stable")[:keep]
    return float(100 * (error.mean() - error[selected].mean()) / max(error.mean(), 1e-12))


def summarize(tasks: pd.DataFrame) -> pd.DataFrame:
    scores = ["safeconf_calibrated_pair_risk", "safeconf_frozen_pair_risk", "risk_model_disagreement", "baseline_predicted_magnitude"]
    rows = []
    test = tasks[tasks.split.eq("test")]
    for fold, fold_group in test.groupby("fold_id", sort=True):
        groups = list(fold_group.groupby("setting", sort=True)) + [("all_test_settings_pooled", fold_group)]
        for setting, group in groups:
            error = group.error_two_predictor_mean_rmse.to_numpy(float)
            for score in scores:
                value = group[score].to_numpy(float)
                rows.append({
                    "fold_id": fold,
                    "setting": setting,
                    "score": score,
                    "n_tasks": len(group),
                    "spearman": spearman(value, error),
                    "mean_error": float(error.mean()),
                    "risk_coverage80_improve_pct": coverage_improvement(value, error),
                    "accepted_q80_fraction": float(group.accepted_by_validation_q80.mean()),
                    "accepted_q80_mean_error": float(group.loc[group.accepted_by_validation_q80, "error_two_predictor_mean_rmse"].mean()) if group.accepted_by_validation_q80.any() else float("nan"),
                })
    return pd.DataFrame(rows)


def bootstrap(tasks: pd.DataFrame) -> pd.DataFrame:
    test = tasks[tasks.split.eq("test")].copy()
    primary = "safeconf_calibrated_pair_risk"
    comparators = ["safeconf_frozen_pair_risk", "risk_model_disagreement", "baseline_predicted_magnitude"]
    cache = {}
    for fold, group in test.groupby("fold_id"):
        group = group.reset_index(drop=True)
        clusters = [np.flatnonzero(group.perturbation.astype(str).to_numpy() == pert) for pert in sorted(group.perturbation.astype(str).unique())]
        cache[str(fold)] = (group, clusters)
    rng = np.random.default_rng(SEED)
    folds = sorted(cache)
    rows = []
    for comparator in comparators:
        observed = []
        for group, _ in cache.values():
            error = group.error_two_predictor_mean_rmse.to_numpy(float)
            observed.append(spearman(group[primary], error) - spearman(group[comparator], error))
        draws = []
        for _ in range(N_BOOTSTRAP):
            deltas = []
            for fold in rng.choice(folds, len(folds), replace=True):
                group, clusters = cache[str(fold)]
                index = np.concatenate([clusters[int(i)] for i in rng.integers(0, len(clusters), len(clusters))])
                error = group.error_two_predictor_mean_rmse.to_numpy(float)[index]
                deltas.append(spearman(group[primary].to_numpy(float)[index], error) - spearman(group[comparator].to_numpy(float)[index], error))
            draws.append(float(np.nanmean(deltas)))
        values = np.asarray(draws)
        rows.append({
            "primary": primary,
            "comparator": comparator,
            "bootstrap_unit": "outer_fold_plus_perturbation_cluster",
            "n_folds": len(folds),
            "n_test_task_rows": len(test),
            "observed_macro_delta_spearman": float(np.nanmean(observed)),
            "ci95_low": float(np.nanquantile(values, .025)),
            "ci95_high": float(np.nanquantile(values, .975)),
            "probability_delta_gt_zero": float(np.mean(values > 0)),
            "n_bootstrap": N_BOOTSTRAP,
        })
    return pd.DataFrame(rows)


def write_figure(summary: pd.DataFrame) -> None:
    macro = summary.groupby(["setting", "score"], as_index=False).spearman.mean()
    settings = ["random_missing_pair", "context_unseen_row", "perturbation_unseen_column", "context_and_perturbation_unseen"]
    labels = ["随机缺失", "新背景", "新扰动", "背景+扰动双未见"]
    scores = ["safeconf_calibrated_pair_risk", "risk_model_disagreement", "baseline_predicted_magnitude"]
    colors = ["#386f88", "#76988a", "#b08355"]
    width, height, x0, y0, plot_w, plot_h = 1120, 650, 105, 110, 930, 430
    low, high = -.4, 1.0
    sy = lambda value: y0 + (high - value) / (high - low) * plot_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="#fff"/>', '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#27343c}.t{font-size:26px;font-weight:700}.s{font-size:15px;fill:#657078}.l{font-size:15px}</style>', '<text x="50" y="42" class="t">E108｜正式 scGPT–GEARS 风险排序</text>', '<text x="50" y="70" class="s">Frangieh 3 个外层背景留出 fold；柱高为 fold 内 Spearman 的宏平均。</text>']
    for tick in [-.4, 0, .4, .8, 1.0]:
        y = sy(tick); parts += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" stroke="#dce3e6"/>', f'<text x="{x0-12}" y="{y+5:.1f}" text-anchor="end" class="s">{tick:.1f}</text>']
    group_w = plot_w / len(settings)
    for gi, (setting, label) in enumerate(zip(settings, labels)):
        center = x0 + group_w * (gi + .5)
        for si, (score, color) in enumerate(zip(scores, colors)):
            hit = macro[(macro.setting == setting) & (macro.score == score)]
            value = float(hit.spearman.iloc[0])
            x = center + (si - 1) * 54 - 21
            z = sy(0); y = sy(value)
            parts.append(f'<rect x="{x:.1f}" y="{min(y,z):.1f}" width="42" height="{max(abs(z-y),1):.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x+21:.1f}" y="{min(y,z)-7:.1f}" text-anchor="middle" class="s">{value:.2f}</text>')
        parts.append(f'<text x="{center:.1f}" y="575" text-anchor="middle" class="l">{label}</text>')
    for i, (label, color) in enumerate(zip(["SafeConf 校准风险", "模型分歧", "预测幅度"], colors)):
        x = 250 + i * 245; parts.append(f'<rect x="{x}" y="610" width="18" height="12" fill="{color}"/><text x="{x+27}" y="622" class="s">{label}</text>')
    parts.append('</svg>')
    (FIGURES / "F1_formal_risk_by_setting.svg").write_text("\n".join(parts))


def main() -> None:
    for directory in (TABLES, ARRAYS, REPORTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    raw_tasks, records, predictions, truths, genes = build_tasks()
    tasks, calibrators = calibrate(raw_tasks)
    summary = summarize(tasks)
    boot = bootstrap(tasks)
    test = tasks[tasks.split.eq("test")].copy()
    pd.DataFrame(records).to_csv(TABLES / "PREDICTION_RECORDS.csv", index=False)
    tasks.to_csv(TABLES / "E108_ALL_TASK_RISK_TABLE.csv", index=False)
    test.to_csv(TABLES / "E108_TEST_TASK_RISK_TABLE.csv", index=False)
    calibrators.to_csv(TABLES / "E108_CALIBRATORS.csv", index=False)
    summary.to_csv(TABLES / "E108_FOLD_SETTING_SUMMARY.csv", index=False)
    boot.to_csv(TABLES / "E108_CLUSTER_BOOTSTRAP.csv", index=False)
    np.savez_compressed(ARRAYS / "predicted_effects.npz", **predictions)
    np.savez_compressed(ARRAYS / "true_effects.npz", **truths)
    issues = validate_prediction_record_artifacts(OUT, records=pd.DataFrame(records), strict=True)
    pd.DataFrame({"strict_issue": issues}).to_csv(TABLES / "E108_STRICT_CONTRACT_ISSUES.csv", index=False)
    write_figure(summary)
    macro = summary.groupby(["setting", "score"], as_index=False).spearman.mean()
    pooled = macro[macro.setting.eq("all_test_settings_pooled")].sort_values("spearman", ascending=False)
    status = {
        "experiment": "E108_formal_dual_model_risk_audit",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "models": ["scGPT_context_mean_finetuned", "GEARS_context_mean_trainonly_graphs"],
        "n_outer_folds": int(test.fold_id.nunique()),
        "n_test_task_rows": len(test),
        "n_prediction_records": len(records),
        "strict_issue_count": len(issues),
        "test_truth_used_for_prediction_score_or_threshold": False,
        "validation_truth_used_for_calibration_and_q80_threshold": True,
        "primary_score": "safeconf_calibrated_pair_risk",
        "pooled_setting_macro_spearman": dict(zip(pooled.score, pooled.spearman)),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# E108｜正式 scGPT–GEARS 双模型风险审计", "",
        "E108 使用 E106/E107 的同背景输入正式预测。外层测试共有 3×279=837 个任务、1674 条严格 PredictionRecord，合同问题数为 " + str(len(issues)) + "。预测、风险分数和 q80 阈值冻结后才读取测试扰动表达。", "",
        "校准只读取每折 30 个 source validation pair。新背景与新扰动的结构项沿用冻结规则，因此这部分仍需后续 inner row/column/double 校准实验验证，不能把当前校准结果写成跨设置保证。", "",
        "## 3-fold 宏平均 Spearman", "", "| setting | score | ρ |", "|---|---|---:|",
    ]
    for row in macro.itertuples(index=False):
        lines.append(f"| {row.setting} | {row.score} | {row.spearman:.3f} |")
    lines += ["", "## 聚类 bootstrap：校准风险相对基线", "", "| comparator | Δρ | 95% CI | P(Δ>0) |", "|---|---:|---:|---:|"]
    for row in boot.itertuples(index=False):
        lines.append(f"| {row.comparator} | {row.observed_macro_delta_spearman:.3f} | [{row.ci95_low:.3f}, {row.ci95_high:.3f}] | {row.probability_delta_gt_zero:.3f} |")
    lines += ["", "完整任务表见 `tables/E108_TEST_TASK_RISK_TABLE.csv`，校准参数见 `tables/E108_CALIBRATORS.csv`，白底图见 `figures/F1_formal_risk_by_setting.svg`。"]
    (REPORTS / "E108_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "README_先看这个.md").write_text("# E108 先看这个\n\n先读 `reports/E108_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(pooled.to_string(index=False))
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
