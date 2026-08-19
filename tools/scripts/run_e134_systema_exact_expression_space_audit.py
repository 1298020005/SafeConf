#!/usr/bin/env python3
"""E134: exact expression-space implementation of Systema's perturbed centroid."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E134_systema_exact_expression_space_audit_20260714"
TABLES, FIGURES = OUT / "tables", OUT / "figures"


def load_e133():
    path = ROOT / "tools/scripts/run_e133_systema_aware_baseline_audit.py"
    spec = importlib.util.spec_from_file_location("e133_for_e134", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E133 = load_e133()


def external_state_maps(cache_path: Path):
    with np.load(cache_path) as store:
        contexts = store["contexts"].astype(str).tolist()
        perturbations = store["perturbations"].astype(str).tolist()
        controls = np.asarray(store["controls"], np.float32)
        effects = np.asarray(store["effects"], np.float32).reshape(len(contexts), len(perturbations), -1)
    control_map = {context: controls[index] for index, context in enumerate(contexts)}
    effect_map = {(context, perturbation): effects[i, j] for i, context in enumerate(contexts) for j, perturbation in enumerate(perturbations)}
    return control_map, effect_map


def frangieh_state_maps(manifest: pd.DataFrame):
    source = Path("/home/yyf/data/scgpt_formal_frangieh_fixed_panel_20260711/frangieh_e72_fixed512/perturb_processed.h5ad")
    data = ad.read_h5ad(source)
    contexts = data.obs["cell_type"].astype(str).to_numpy()
    conditions = data.obs["condition"].astype(str).to_numpy()
    required_contexts = sorted(manifest.context.unique())
    required_conditions = sorted(manifest.perturbation.unique())
    keep = np.isin(contexts, required_contexts) & (np.isin(conditions, required_conditions) | (conditions == "ctrl"))
    matrix = data.X[keep]
    matrix = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)
    labels = np.asarray([f"{context}\x1f{condition}" for context, condition in zip(contexts[keep], conditions[keep])])
    groups, codes = np.unique(labels, return_inverse=True)
    membership = sp.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))), shape=(len(groups), len(codes)))
    counts = np.bincount(codes, minlength=len(groups)).astype(np.float32)
    means = np.asarray((membership @ matrix).multiply((1.0 / counts)[:, None]).toarray(), np.float32)
    mean_map = {label: means[index] for index, label in enumerate(groups)}
    controls = {context: mean_map[f"{context}\x1fctrl"] for context in required_contexts}
    effects = {
        (context, perturbation): mean_map[f"{context}\x1f{perturbation}"] - controls[context]
        for context in required_contexts for perturbation in required_conditions
    }
    return controls, effects


def training_centroids(manifest: pd.DataFrame, controls: dict, effects: dict):
    result = {}
    for fold, group in manifest.groupby("fold_id", sort=True):
        train = group[group.split.astype(str).eq("train")].copy()
        if "in_train_fraction_100" in train.columns:
            train = train[train.in_train_fraction_100.astype(bool)]
        states = [controls[str(row.context)] + effects[(str(row.context), str(row.perturbation))] for row in train.itertuples(index=False)]
        if not states:
            raise RuntimeError(f"no training states for {fold}")
        result[str(fold)] = np.mean(np.stack(states), axis=0).astype(np.float32)
    return result


def audit_dataset(dataset: str, spec: dict):
    root = spec["root"]
    tasks = pd.read_csv(root / spec["task"])
    tasks["fold_id"], tasks["task_id"] = tasks.fold_id.astype(str), tasks.task_id.astype(str)
    records = pd.read_csv(root / spec["records"])
    manifest = E133.load_manifest(dataset, spec)
    controls, effects = frangieh_state_maps(manifest) if spec["cache"] is None else external_state_maps(spec["cache"])
    centroids = training_centroids(manifest, controls, effects)
    predictions = np.load(root / "arrays/predicted_effects.npz")
    truths = np.load(root / "arrays/true_effects.npz")
    grouped_records = records.groupby([records.fold_id.astype(str), records.task_id.astype(str)], sort=False)
    rows, max_truth_difference = [], 0.0
    for task in tasks.itertuples(index=False):
        key = (str(task.fold_id), str(task.task_id))
        block = grouped_records.get_group(key)
        if len(block) != 2:
            raise RuntimeError(f"{dataset} {key}: expected two records")
        control = controls[str(task.context)]
        centroid = centroids[str(task.fold_id)]
        pred, truth, direction = {}, None, {}
        for record in block.itertuples(index=False):
            slot = E133.predictor_slot(str(record.predictor_name))
            effect_prediction = np.asarray(predictions[str(record.predicted_effect_key)], np.float32)
            current_truth = np.asarray(truths[str(record.true_effect_key)], np.float32)
            truth = current_truth if truth is None else truth
            predicted_state, true_state = control + effect_prediction, control + current_truth
            direction[slot] = {
                "pearson": 1.0 - E133.pearson(predicted_state - centroid, true_state - centroid),
                "cosine": 1.0 - E133.cosine(predicted_state - centroid, true_state - centroid),
            }
            pred[slot] = effect_prediction
        expected = effects[(str(task.context), str(task.perturbation))]
        max_truth_difference = max(max_truth_difference, float(np.max(np.abs(truth - expected))))
        baseline_effect = centroid - control
        ensemble = (pred["scgpt"] + pred["gears"]) / 2.0
        rmse_scgpt, rmse_gears = E133.rmse(pred["scgpt"], truth), E133.rmse(pred["gears"], truth)
        individual_mean = float(np.mean([rmse_scgpt, rmse_gears]))
        baseline_rmse = E133.rmse(baseline_effect, truth)
        row = task._asdict()
        row.update({
            "dataset": dataset,
            "rmse_scgpt_recomputed": rmse_scgpt,
            "rmse_gears_recomputed": rmse_gears,
            "rmse_two_predictor_individual_mean": individual_mean,
            "rmse_two_predictor_ensemble": E133.rmse(ensemble, truth),
            "rmse_training_perturbed_mean": baseline_rmse,
            "excess_rmse_mean_vs_training_perturbed_mean": individual_mean - baseline_rmse,
            "excess_rmse_ensemble_vs_training_perturbed_mean": E133.rmse(ensemble, truth) - baseline_rmse,
            "error_centered_pearson_scgpt": direction["scgpt"]["pearson"],
            "error_centered_pearson_gears": direction["gears"]["pearson"],
            "error_centered_pearson_mean": float(np.mean([direction["scgpt"]["pearson"], direction["gears"]["pearson"]])),
            "error_centered_cosine_scgpt": direction["scgpt"]["cosine"],
            "error_centered_cosine_gears": direction["gears"]["cosine"],
            "error_centered_cosine_mean": float(np.mean([direction["scgpt"]["cosine"], direction["gears"]["cosine"]])),
            "truth_alignment_with_training_perturbed_mean": E133.cosine(control + truth, centroid),
            "training_reference_uses_validation_or_test_truth": False,
        })
        rows.append(row)
    predictions.close()
    truths.close()
    frame = pd.DataFrame(rows)
    return frame, {
        "dataset": dataset,
        "n_folds": int(frame.fold_id.nunique()),
        "n_test_tasks": len(frame),
        "max_abs_saved_truth_vs_reconstructed_truth": max_truth_difference,
        "truth_reconstruction_pass_atol_1e-5": bool(max_truth_difference <= 1e-5),
    }


def write_report(tasks, baseline, overall, boot, audits):
    safe = overall[overall.score.eq("safeconf_calibrated_pair_risk")].set_index("endpoint").dataset_equal_macro_spearman
    b = boot.set_index("metric")
    advantage = b.loc["ensemble_advantage_over_training_perturbed_mean"]
    lines = [
        "# E134｜Systema 表达空间精确定义审计",
        "",
        "## 结果",
        "",
        f"按训练受扰动表达质心的精确定义，SafeConf 对 centered Pearson 方向误差的六数据集等权宏平均 Spearman 为 **{safe['error_centered_pearson_mean']:.3f}**，对 centered cosine 方向误差为 **{safe['error_centered_cosine_mean']:.3f}**，对正式模型相对简单质心预测器的超额 RMSE 为 **{safe['excess_rmse_mean_vs_training_perturbed_mean']:.3f}**。",
        "",
        f"两模型 ensemble 相对训练受扰动表达质心的 RMSE 优势，扰动聚类 bootstrap 中位数为 **{advantage['median']:.4f}**，95% CI **[{advantage['ci_low_2.5pct']:.4f}, {advantage['ci_high_97.5pct']:.4f}]**。正值表示 ensemble 更好。",
        "",
        "## 每数据集简单基线",
        "",
        "| 数据集 | tasks | ensemble RMSE | perturbed-centroid RMSE | ensemble − simple | ensemble 胜出任务比例 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in baseline.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.n_tasks} | {row.rmse_ensemble_mean:.4f} | {row.rmse_training_perturbed_mean:.4f} | {row.ensemble_minus_simple_baseline:+.4f} | {row.fraction_tasks_ensemble_beats_simple_baseline:.1%} |")
    lines += [
        "",
        "## 六数据集等权排序",
        "",
        "| 分数 | original RMSE | centered Pearson | centered cosine | excess RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    pivot = overall.pivot(index="score", columns="endpoint", values="dataset_equal_macro_spearman")
    for score, row in pivot.iterrows():
        lines.append(f"| {score} | {row['error_two_predictor_mean_rmse']:.3f} | {row['error_centered_pearson_mean']:.3f} | {row['error_centered_cosine_mean']:.3f} | {row['excess_rmse_mean_vs_training_perturbed_mean']:.3f} |")
    lines += [
        "",
        "## 与 E133 的关系",
        "",
        "E133 检查的是训练平均效应，E134 检查的是训练受扰动表达质心。跨背景时两者不同，两个结果并列保留。E134 是 E133 解封后对文献公式的技术校正，不能写成最初主终点。",
        "",
        "## 完整性",
        "",
    ]
    for item in audits:
        lines.append(f"- {item['dataset']}：{item['n_folds']} folds，{item['n_test_tasks']} tasks；真值重建最大绝对差 {item['max_abs_saved_truth_vs_reconstructed_truth']:.2e}，检查={'通过' if item['truth_reconstruction_pass_atol_1e-5'] else '失败'}。")
    (OUT / "E134_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "README_先看这个.md").write_text("# E134 先看这个\n\n先读 `E134_REPORT.md`，冻结说明见 `PREREG.md`。\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    frames, audits = [], []
    for dataset, spec in E133.DATASETS.items():
        print(f"[E134] auditing {dataset}", flush=True)
        frame, audit = audit_dataset(dataset, spec)
        frames.append(frame)
        audits.append(audit)
    tasks = pd.concat(frames, ignore_index=True, sort=False)
    if not all(item["truth_reconstruction_pass_atol_1e-5"] for item in audits):
        raise RuntimeError("source truth reconstruction failed")
    folds = E133.fold_correlations(tasks)
    baseline = E133.baseline_summary(tasks)
    dataset_macro, overall = E133.macro_summary(folds)
    draws, boot = E133.bootstrap(tasks)
    tasks.to_csv(TABLES / "E134_TASK_AUDIT.csv", index=False)
    folds.to_csv(TABLES / "E134_FOLD_CORRELATIONS.csv", index=False)
    baseline.to_csv(TABLES / "E134_SIMPLE_BASELINE_SUMMARY.csv", index=False)
    dataset_macro.to_csv(TABLES / "E134_DATASET_MACRO.csv", index=False)
    overall.to_csv(TABLES / "E134_SIX_DATASET_MACRO.csv", index=False)
    draws.to_csv(TABLES / "E134_CLUSTER_BOOTSTRAP_DRAWS.csv", index=False)
    boot.to_csv(TABLES / "E134_CLUSTER_BOOTSTRAP_SUMMARY.csv", index=False)
    pd.DataFrame(audits).to_csv(TABLES / "E134_SOURCE_TRUTH_AUDIT.csv", index=False)
    old_out, old_figures = E133.OUT, E133.FIGURES
    E133.OUT, E133.FIGURES = OUT, FIGURES
    E133.svg_figure(dataset_macro, baseline)
    E133.OUT, E133.FIGURES = old_out, old_figures
    write_report(tasks, baseline, overall, boot, audits)
    status = {
        "experiment": "E134_systema_exact_expression_space_audit",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "n_datasets": int(tasks.dataset.nunique()),
        "n_folds": int(tasks[["dataset", "fold_id"]].drop_duplicates().shape[0]),
        "n_test_tasks": len(tasks),
        "bootstrap_draws": E133.N_BOOTSTRAP,
        "safeconf_or_predictor_refit": False,
        "training_reference_uses_validation_or_test_truth": False,
        "all_source_truth_checks_pass": True,
        "e133_negative_result_overwritten": False,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(overall.to_string(index=False))
    print(baseline.to_string(index=False))
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
