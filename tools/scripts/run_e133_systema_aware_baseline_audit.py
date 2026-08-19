#!/usr/bin/env python3
"""E133: literature-motivated simple-baseline and direction-metric audit.

The analysis is intentionally post-model: it consumes frozen folds and saved
prediction vectors, never refits SafeConf or either upstream predictor.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E133_systema_aware_baseline_audit_20260714"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SEED = 202607133
N_BOOTSTRAP = 3000


DATASETS = {
    "Frangieh": {
        "root": ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713",
        "task": "tables/E108_TEST_TASK_RISK_TABLE.csv",
        "records": "tables/PREDICTION_RECORDS.csv",
        "manifest": ROOT / "docs/实验结果/E97_frangieh_gene_cartesian_contract_20260713/manifests/E97_TASK_MANIFEST.csv",
        "cache": None,
    },
    "Lara_exvivo": {
        "root": ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/Lara_exvivo",
        "task": "TASK_RISK_TABLE.csv",
        "records": "PREDICTION_RECORDS.csv",
        "manifest": ROOT / "docs/实验结果/E99_multicontext_external_contract_20260713/manifests/E99_TASK_MANIFEST.csv",
        "cache": Path("/home/yyf/data/safeconf_e112_external/Lara_exvivo_CONTROL_ONLY_512.npz"),
    },
    "Santinha": {
        "root": ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/Santinha",
        "task": "TASK_RISK_TABLE.csv",
        "records": "PREDICTION_RECORDS.csv",
        "manifest": ROOT / "docs/实验结果/E99_multicontext_external_contract_20260713/manifests/E99_TASK_MANIFEST.csv",
        "cache": Path("/home/yyf/data/safeconf_e112_external/Santinha_CONTROL_ONLY_512.npz"),
    },
    "Shifrut": {
        "root": ROOT / "docs/实验结果/E120_shifrut_formal_dual_models_20260714/Shifrut",
        "task": "TASK_RISK_TABLE.csv",
        "records": "PREDICTION_RECORDS.csv",
        "manifest": ROOT / "docs/实验结果/E119_shifrut_four_context_contract_20260714/manifests/E119_TASK_MANIFEST.csv",
        "cache": Path("/home/yyf/data/safeconf_e112_external/Shifrut_CONTROL_ONLY_512.npz"),
    },
    "Liang": {
        "root": ROOT / "docs/实验结果/E123_liang_formal_dual_models_20260714/Liang",
        "task": "TASK_RISK_TABLE.csv",
        "records": "PREDICTION_RECORDS.csv",
        "manifest": ROOT / "docs/实验结果/E122_liang_nine_context_contract_20260714/manifests/E122_TASK_MANIFEST.csv",
        "cache": Path("/home/yyf/data/safeconf_e112_external/Liang_CONTROL_ONLY_512.npz"),
    },
    "Tian_CRISPRi": {
        "root": ROOT / "docs/实验结果/E129_tian_crispri_formal_dual_models_20260714/Tian_CRISPRi",
        "task": "TASK_RISK_TABLE.csv",
        "records": "PREDICTION_RECORDS.csv",
        "manifest": ROOT / "docs/实验结果/E128_tian_crispri_four_batch_contract_20260714/manifests/E128_TASK_MANIFEST.csv",
        "cache": Path("/home/yyf/data/safeconf_e112_external/Tian_CRISPRi_CONTROL_ONLY_512.npz"),
    },
}

SCORES = [
    "safeconf_calibrated_pair_risk",
    "safeconf_frozen_pair_risk",
    "risk_model_disagreement",
    "baseline_predicted_magnitude",
]
ENDPOINTS = [
    "error_two_predictor_mean_rmse",
    "error_centered_pearson_mean",
    "error_centered_cosine_mean",
    "excess_rmse_mean_vs_training_perturbed_mean",
]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.unique(a[mask]).size < 2 or np.unique(b[mask]).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a[mask]), rankdata(b[mask]))[0, 1])


def load_manifest(dataset: str, spec: dict) -> pd.DataFrame:
    manifest = pd.read_csv(spec["manifest"], keep_default_na=False)
    if "dataset" in manifest.columns:
        manifest = manifest[manifest.dataset.astype(str).eq(dataset)]
    manifest = manifest.copy()
    manifest["context"] = manifest.context.astype(str)
    manifest["perturbation"] = manifest.perturbation.astype(str)
    return manifest


def external_effect_map(cache_path: Path) -> dict[tuple[str, str], np.ndarray]:
    with np.load(cache_path) as store:
        contexts = store["contexts"].astype(str).tolist()
        perturbations = store["perturbations"].astype(str).tolist()
        effects = np.asarray(store["effects"], np.float32).reshape(len(contexts), len(perturbations), -1)
    return {(context, perturbation): effects[i, j] for i, context in enumerate(contexts) for j, perturbation in enumerate(perturbations)}


def frangieh_effect_map(manifest: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    source = Path("/home/yyf/data/scgpt_formal_frangieh_fixed_panel_20260711/frangieh_e72_fixed512/perturb_processed.h5ad")
    data = ad.read_h5ad(source)
    contexts = data.obs["cell_type"].astype(str).to_numpy()
    conditions = data.obs["condition"].astype(str).to_numpy()
    required_contexts = sorted(manifest.context.unique())
    required_conditions = sorted(manifest.perturbation.unique())
    keep = np.isin(contexts, required_contexts) & (np.isin(conditions, required_conditions) | (conditions == "ctrl"))
    matrix = data.X[keep]
    matrix = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)
    labels = np.asarray([f"{c}\x1f{p}" for c, p in zip(contexts[keep], conditions[keep])])
    groups, codes = np.unique(labels, return_inverse=True)
    membership = sp.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))), shape=(len(groups), len(codes)))
    counts = np.bincount(codes, minlength=len(groups)).astype(np.float32)
    means = np.asarray((membership @ matrix).multiply((1.0 / counts)[:, None]).toarray(), np.float32)
    mean_map = {label: means[index] for index, label in enumerate(groups)}
    return {
        (context, perturbation): mean_map[f"{context}\x1f{perturbation}"] - mean_map[f"{context}\x1fctrl"]
        for context in required_contexts for perturbation in required_conditions
    }


def training_references(manifest: pd.DataFrame, effect_map: dict[tuple[str, str], np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for fold, group in manifest.groupby("fold_id", sort=True):
        train = group[group.split.astype(str).eq("train")].copy()
        if "in_train_fraction_100" in train.columns:
            train = train[train.in_train_fraction_100.astype(bool)]
        vectors = [effect_map[(str(row.context), str(row.perturbation))] for row in train.itertuples(index=False)]
        if not vectors:
            raise RuntimeError(f"no training effects for {fold}")
        result[str(fold)] = np.mean(np.stack(vectors), axis=0).astype(np.float32)
    return result


def predictor_slot(name: str) -> str:
    lower = name.lower()
    if "scgpt" in lower:
        return "scgpt"
    if "gears" in lower:
        return "gears"
    raise ValueError(f"unknown predictor: {name}")


def audit_dataset(dataset: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    root = spec["root"]
    tasks = pd.read_csv(root / spec["task"])
    tasks["fold_id"] = tasks.fold_id.astype(str)
    tasks["task_id"] = tasks.task_id.astype(str)
    records = pd.read_csv(root / spec["records"])
    manifest = load_manifest(dataset, spec)
    effect_map = frangieh_effect_map(manifest) if spec["cache"] is None else external_effect_map(spec["cache"])
    references = training_references(manifest, effect_map)
    predictions = np.load(root / "arrays/predicted_effects.npz")
    truths = np.load(root / "arrays/true_effects.npz")
    grouped_records = records.groupby([records.fold_id.astype(str), records.task_id.astype(str)], sort=False)
    rows, max_truth_difference = [], 0.0
    for task in tasks.itertuples(index=False):
        key = (str(task.fold_id), str(task.task_id))
        block = grouped_records.get_group(key)
        if len(block) != 2:
            raise RuntimeError(f"{dataset} {key}: expected two records, found {len(block)}")
        reference = references[str(task.fold_id)]
        pred, truth = {}, None
        direction = {}
        for record in block.itertuples(index=False):
            slot = predictor_slot(str(record.predictor_name))
            vector = np.asarray(predictions[str(record.predicted_effect_key)], np.float32)
            current_truth = np.asarray(truths[str(record.true_effect_key)], np.float32)
            truth = current_truth if truth is None else truth
            if not np.allclose(truth, current_truth, atol=1e-6):
                raise RuntimeError(f"truth mismatch in {dataset} {key}")
            pred[slot] = vector
            centered_prediction, centered_truth = vector - reference, current_truth - reference
            direction[slot] = {
                "pearson": 1.0 - pearson(centered_prediction, centered_truth),
                "cosine": 1.0 - cosine(centered_prediction, centered_truth),
            }
        expected = effect_map[(str(task.context), str(task.perturbation))]
        max_truth_difference = max(max_truth_difference, float(np.max(np.abs(truth - expected))))
        ensemble = (pred["scgpt"] + pred["gears"]) / 2.0
        rmse_scgpt = rmse(pred["scgpt"], truth)
        rmse_gears = rmse(pred["gears"], truth)
        rmse_individual_mean = float(np.mean([rmse_scgpt, rmse_gears]))
        rmse_baseline = rmse(reference, truth)
        row = task._asdict()
        row.update({
            "dataset": dataset,
            "rmse_scgpt_recomputed": rmse_scgpt,
            "rmse_gears_recomputed": rmse_gears,
            "rmse_two_predictor_individual_mean": rmse_individual_mean,
            "rmse_two_predictor_ensemble": rmse(ensemble, truth),
            "rmse_training_perturbed_mean": rmse_baseline,
            "excess_rmse_mean_vs_training_perturbed_mean": rmse_individual_mean - rmse_baseline,
            "excess_rmse_ensemble_vs_training_perturbed_mean": rmse(ensemble, truth) - rmse_baseline,
            "error_centered_pearson_scgpt": direction["scgpt"]["pearson"],
            "error_centered_pearson_gears": direction["gears"]["pearson"],
            "error_centered_pearson_mean": float(np.mean([direction["scgpt"]["pearson"], direction["gears"]["pearson"]])),
            "error_centered_cosine_scgpt": direction["scgpt"]["cosine"],
            "error_centered_cosine_gears": direction["gears"]["cosine"],
            "error_centered_cosine_mean": float(np.mean([direction["scgpt"]["cosine"], direction["gears"]["cosine"]])),
            "truth_alignment_with_training_perturbed_mean": cosine(truth, reference),
            "training_reference_uses_validation_or_test_truth": False,
        })
        rows.append(row)
    predictions.close()
    truths.close()
    audited = pd.DataFrame(rows)
    return audited, {
        "dataset": dataset,
        "n_folds": int(audited.fold_id.nunique()),
        "n_test_tasks": len(audited),
        "max_abs_saved_truth_vs_reconstructed_truth": max_truth_difference,
        "truth_reconstruction_pass_atol_1e-5": bool(max_truth_difference <= 1e-5),
    }


def fold_correlations(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, fold), group in tasks.groupby(["dataset", "fold_id"], sort=True):
        for score in SCORES:
            for endpoint in ENDPOINTS:
                rows.append({
                    "dataset": dataset,
                    "fold_id": fold,
                    "score": score,
                    "endpoint": endpoint,
                    "n_tasks": len(group),
                    "spearman": spearman(group[score], group[endpoint]),
                })
    return pd.DataFrame(rows)


def baseline_summary(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in tasks.groupby("dataset", sort=True):
        rows.append({
            "dataset": dataset,
            "n_tasks": len(group),
            "rmse_scgpt_mean": group.rmse_scgpt_recomputed.mean(),
            "rmse_gears_mean": group.rmse_gears_recomputed.mean(),
            "rmse_individual_mean": group.rmse_two_predictor_individual_mean.mean(),
            "rmse_ensemble_mean": group.rmse_two_predictor_ensemble.mean(),
            "rmse_training_perturbed_mean": group.rmse_training_perturbed_mean.mean(),
            "ensemble_minus_simple_baseline": (group.rmse_two_predictor_ensemble - group.rmse_training_perturbed_mean).mean(),
            "fraction_tasks_ensemble_beats_simple_baseline": (group.rmse_two_predictor_ensemble < group.rmse_training_perturbed_mean).mean(),
            "truth_alignment_with_training_mean": group.truth_alignment_with_training_perturbed_mean.mean(),
        })
    return pd.DataFrame(rows)


def macro_summary(folds: pd.DataFrame) -> pd.DataFrame:
    dataset_macro = folds.groupby(["dataset", "score", "endpoint"], as_index=False).agg(
        n_folds=("fold_id", "nunique"), spearman=("spearman", "mean")
    )
    overall = dataset_macro.groupby(["score", "endpoint"], as_index=False).agg(
        n_datasets=("dataset", "nunique"), dataset_equal_macro_spearman=("spearman", "mean")
    )
    return dataset_macro, overall


def fast_spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    ra, rb = rankdata(a[mask]), rankdata(b[mask])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denominator = float(np.sqrt(np.dot(ra, ra) * np.dot(rb, rb)))
    return float(np.dot(ra, rb) / denominator) if denominator > 1e-12 else float("nan")


def bootstrap_arrays(tasks: pd.DataFrame) -> dict[str, dict]:
    result = {}
    for dataset, group in tasks.groupby("dataset", sort=True):
        perturbations = sorted(group.perturbation.astype(str).unique())
        perturbation_index = {value: index for index, value in enumerate(perturbations)}
        folds = []
        for _, fold in group.groupby("fold_id", sort=False):
            folds.append({
                "cluster": np.asarray([perturbation_index[str(value)] for value in fold.perturbation], dtype=int),
                "safeconf": fold.safeconf_calibrated_pair_risk.to_numpy(float),
                "magnitude": fold.baseline_predicted_magnitude.to_numpy(float),
                "endpoints": {endpoint: fold[endpoint].to_numpy(float) for endpoint in ENDPOINTS[1:]},
                "advantage": (fold.rmse_training_perturbed_mean - fold.rmse_two_predictor_ensemble).to_numpy(float),
            })
        result[dataset] = {"n_clusters": len(perturbations), "folds": folds}
    return result


def bootstrap(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    endpoints = ENDPOINTS[1:]
    draws = []
    grouped = bootstrap_arrays(tasks)
    for draw in range(N_BOOTSTRAP):
        row = {"draw": draw}
        dataset_stats, advantages = [], []
        for packed in grouped.values():
            n_clusters = packed["n_clusters"]
            counts = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters))
            safe_fold = {endpoint: [] for endpoint in endpoints}
            magnitude_fold = {endpoint: [] for endpoint in endpoints}
            advantage_sum, advantage_weight = 0.0, 0
            for fold in packed["folds"]:
                repetitions = counts[fold["cluster"]]
                indices = np.repeat(np.arange(len(repetitions)), repetitions)
                if len(indices) < 3:
                    continue
                safe = fold["safeconf"][indices]
                magnitude = fold["magnitude"][indices]
                for endpoint in endpoints:
                    values = fold["endpoints"][endpoint][indices]
                    safe_fold[endpoint].append(fast_spearman(safe, values))
                    magnitude_fold[endpoint].append(fast_spearman(magnitude, values))
                advantage_sum += float(np.sum(fold["advantage"] * repetitions))
                advantage_weight += int(np.sum(repetitions))
            dataset_stats.append({
                endpoint: (float(np.nanmean(safe_fold[endpoint])), float(np.nanmean(magnitude_fold[endpoint])))
                for endpoint in endpoints
            })
            advantages.append(advantage_sum / advantage_weight)
        for endpoint in endpoints:
            safe = float(np.nanmean([item[endpoint][0] for item in dataset_stats]))
            magnitude = float(np.nanmean([item[endpoint][1] for item in dataset_stats]))
            row[f"safeconf__{endpoint}"] = safe
            row[f"delta_safeconf_minus_magnitude__{endpoint}"] = safe - magnitude
        row["ensemble_advantage_over_training_perturbed_mean"] = float(np.mean(advantages))
        draws.append(row)
    draws = pd.DataFrame(draws)
    summaries = []
    for metric in draws.columns[1:]:
        values = draws[metric].to_numpy(float)
        summaries.append({
            "metric": metric,
            "bootstrap_draws": N_BOOTSTRAP,
            "ci_low_2.5pct": float(np.nanquantile(values, .025)),
            "median": float(np.nanmedian(values)),
            "ci_high_97.5pct": float(np.nanquantile(values, .975)),
            "fraction_above_zero": float(np.nanmean(values > 0)),
        })
    return draws, pd.DataFrame(summaries)


def svg_figure(dataset_macro: pd.DataFrame, baseline: pd.DataFrame) -> None:
    safe = dataset_macro[dataset_macro.score.eq("safeconf_calibrated_pair_risk")]
    endpoint_labels = {
        "error_centered_pearson_mean": "Pearson direction error",
        "error_centered_cosine_mean": "Cosine direction error",
        "excess_rmse_mean_vs_training_perturbed_mean": "Excess RMSE vs simple mean",
    }
    colors = {"Frangieh": "#3B6FB6", "Lara_exvivo": "#5B8E7D", "Santinha": "#A36A43", "Shifrut": "#7C6BAE", "Liang": "#4D8796", "Tian_CRISPRi": "#B15D62"}
    width, height = 1180, 620
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family:Arial,"Noto Sans SC",sans-serif;fill:#202124}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#5f6368}.axis{stroke:#9aa0a6;stroke-width:1}.zero{stroke:#5f6368;stroke-width:1.5}.label{font-size:14px}.small{font-size:12px}</style>', '<text x="45" y="42" class="title">E133 · Systema-aware robustness audit</text>', '<text x="45" y="68" class="sub">Frozen six-dataset folds; training-only perturbed mean; no SafeConf refit</text>']
    left_x, panel_w, top, panel_h = 75, 610, 120, 390
    min_v, max_v = -.45, .65
    x = lambda value: left_x + (value - min_v) / (max_v - min_v) * panel_w
    parts += [f'<line x1="{left_x}" y1="{top+panel_h}" x2="{left_x+panel_w}" y2="{top+panel_h}" class="axis"/>', f'<line x1="{x(0)}" y1="{top-15}" x2="{x(0)}" y2="{top+panel_h}" class="zero"/>', '<text x="75" y="103" class="label" font-weight="700">A  SafeConf Spearman by dataset</text>']
    selected = list(endpoint_labels)
    row = 0
    for endpoint in selected:
        parts.append(f'<text x="75" y="{top+row*115}" class="small" font-weight="700">{endpoint_labels[endpoint]}</text>')
        block = safe[safe.endpoint.eq(endpoint)]
        for j, item in enumerate(block.itertuples(index=False)):
            cy = top + 24 + row * 115 + j * 13
            cx = x(float(item.spearman))
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy}" r="4.5" fill="{colors[item.dataset]}"/><text x="{cx+8:.1f}" y="{cy+4}" class="small">{item.dataset}</text>')
        row += 1
    for tick in [-.4, -.2, 0, .2, .4, .6]:
        parts.append(f'<text x="{x(tick):.1f}" y="{top+panel_h+24}" class="small" text-anchor="middle">{tick:.1f}</text>')
    bx, bw = 760, 350
    parts += ['<text x="760" y="103" class="label" font-weight="700">B  Ensemble advantage over simple mean</text>', '<text x="760" y="123" class="small">positive bar = ensemble has lower RMSE</text>']
    scale = max(float(np.abs(baseline.ensemble_minus_simple_baseline).max()), .01)
    zero_x = bx + bw / 2
    parts.append(f'<line x1="{zero_x}" y1="145" x2="{zero_x}" y2="470" class="zero"/>')
    for j, item in enumerate(baseline.itertuples(index=False)):
        advantage = -float(item.ensemble_minus_simple_baseline)
        length = advantage / (2 * scale) * bw
        x0, w = (zero_x, length) if length >= 0 else (zero_x + length, -length)
        cy = 165 + j * 48
        parts.append(f'<text x="{bx}" y="{cy+4}" class="small">{item.dataset}</text><rect x="{x0:.1f}" y="{cy-10}" width="{w:.1f}" height="18" fill="{colors[item.dataset]}" opacity="0.82"/><text x="{(x0+w+6 if advantage>=0 else x0-6):.1f}" y="{cy+4}" class="small" text-anchor="{("start" if advantage>=0 else "end")}">{advantage:.3f}</text>')
    parts += ['<text x="45" y="585" class="sub">Reference-sensitive direction metrics complement RMSE; they do not replace the preregistered primary endpoint.</text>', '</svg>']
    (FIGURES / "F1_systema_aware_audit.svg").write_text("\n".join(parts) + "\n")


def write_report(tasks: pd.DataFrame, baseline: pd.DataFrame, dataset_macro: pd.DataFrame, overall: pd.DataFrame, boot: pd.DataFrame, audits: list[dict]) -> None:
    safe = overall[overall.score.eq("safeconf_calibrated_pair_risk")].set_index("endpoint").dataset_equal_macro_spearman
    boot_index = boot.set_index("metric")
    model_adv = boot_index.loc["ensemble_advantage_over_training_perturbed_mean"]
    lines = [
        "# E133｜Systema-aware 简单基线与方向误差审计",
        "",
        "## 结论",
        "",
        f"六数据集、{tasks.fold_id.nunique()} 个不重复 fold 标签、{len(tasks):,} 个测试任务均按冻结方案复核。SafeConf 对训练扰动均值中心化后的 Pearson 方向误差，数据集等权宏平均 Spearman 为 **{safe['error_centered_pearson_mean']:.3f}**；对 cosine 方向误差为 **{safe['error_centered_cosine_mean']:.3f}**。",
        "",
        f"对“正式模型相对简单扰动均值的超额 RMSE”，SafeConf 宏平均 Spearman 为 **{safe['excess_rmse_mean_vs_training_perturbed_mean']:.3f}**。该结论只描述排序能力，不把 SafeConf 写成新的扰动预测器。",
        "",
        f"两模型均值预测相对 training perturbed-mean 的数据集等权平均 RMSE 优势，聚类 bootstrap 中位数为 **{model_adv['median']:.4f}**，95% CI **[{model_adv['ci_low_2.5pct']:.4f}, {model_adv['ci_high_97.5pct']:.4f}]**。",
        "",
        "## 每个数据集的简单基线对照",
        "",
        "| 数据集 | tasks | scGPT RMSE | GEARS RMSE | ensemble RMSE | training perturbed-mean RMSE | ensemble − simple | ensemble 胜出任务比例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in baseline.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.n_tasks} | {row.rmse_scgpt_mean:.4f} | {row.rmse_gears_mean:.4f} | {row.rmse_ensemble_mean:.4f} | {row.rmse_training_perturbed_mean:.4f} | {row.ensemble_minus_simple_baseline:+.4f} | {row.fraction_tasks_ensemble_beats_simple_baseline:.1%} |")
    lines += [
        "",
        "## 六数据集等权排序结果",
        "",
        "| 风险分数 | 原 RMSE | centered Pearson error | centered cosine error | excess RMSE vs simple |",
        "|---|---:|---:|---:|---:|",
    ]
    pivot = overall.pivot(index="score", columns="endpoint", values="dataset_equal_macro_spearman")
    for score, row in pivot.iterrows():
        lines.append(f"| {score} | {row['error_two_predictor_mean_rmse']:.3f} | {row['error_centered_pearson_mean']:.3f} | {row['error_centered_cosine_mean']:.3f} | {row['excess_rmse_mean_vs_training_perturbed_mean']:.3f} |")
    lines += [
        "",
        "## 数据边界检查",
        "",
    ]
    for item in audits:
        lines.append(f"- {item['dataset']}：{item['n_folds']} folds，{item['n_test_tasks']} tasks；保存真值与从原始数据重建真值的最大绝对差 {item['max_abs_saved_truth_vs_reconstructed_truth']:.2e}，一致性检查={'通过' if item['truth_reconstruction_pass_atol_1e-5'] else '失败'}。")
    lines += [
        "- training perturbed-mean 只由各 fold 的训练任务真值构造；验证和测试真值未进入参考量。",
        "- RMSE 对共同平移不敏感，因此没有报告数学等价的“中心化 RMSE”。",
        "",
        "## 使用边界",
        "",
        "E133 是看到 2025 年方法学文献后增加的次要稳健性审计，不能回写成最初预注册的主终点。它用于回答审稿人对简单基线、系统性平均变化和误差定义的质疑。完整逐任务值、fold 统计与 3,000 次聚类 bootstrap 均在 `tables/`。",
    ]
    (OUT / "E133_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "README_先看这个.md").write_text("# E133 先看这个\n\n先读 `E133_REPORT.md`，再看 `figures/F1_systema_aware_audit.svg`。冻结规则在 `PREREG.md`。\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    frames, audits = [], []
    for dataset, spec in DATASETS.items():
        print(f"[E133] auditing {dataset}", flush=True)
        frame, audit = audit_dataset(dataset, spec)
        frames.append(frame)
        audits.append(audit)
    tasks = pd.concat(frames, ignore_index=True, sort=False)
    if not all(item["truth_reconstruction_pass_atol_1e-5"] for item in audits):
        raise RuntimeError("saved truth does not match reconstructed source truth")
    folds = fold_correlations(tasks)
    baseline = baseline_summary(tasks)
    dataset_macro, overall = macro_summary(folds)
    draws, boot = bootstrap(tasks)
    tasks.to_csv(TABLES / "E133_TASK_AUDIT.csv", index=False)
    folds.to_csv(TABLES / "E133_FOLD_CORRELATIONS.csv", index=False)
    baseline.to_csv(TABLES / "E133_SIMPLE_BASELINE_SUMMARY.csv", index=False)
    dataset_macro.to_csv(TABLES / "E133_DATASET_MACRO.csv", index=False)
    overall.to_csv(TABLES / "E133_SIX_DATASET_MACRO.csv", index=False)
    draws.to_csv(TABLES / "E133_CLUSTER_BOOTSTRAP_DRAWS.csv", index=False)
    boot.to_csv(TABLES / "E133_CLUSTER_BOOTSTRAP_SUMMARY.csv", index=False)
    pd.DataFrame(audits).to_csv(TABLES / "E133_SOURCE_TRUTH_AUDIT.csv", index=False)
    svg_figure(dataset_macro, baseline)
    write_report(tasks, baseline, dataset_macro, overall, boot, audits)
    status = {
        "experiment": "E133_systema_aware_baseline_audit",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "n_datasets": int(tasks.dataset.nunique()),
        "n_folds": int(tasks[["dataset", "fold_id"]].drop_duplicates().shape[0]),
        "n_test_tasks": len(tasks),
        "bootstrap_draws": N_BOOTSTRAP,
        "safeconf_or_predictor_refit": False,
        "training_reference_uses_validation_or_test_truth": False,
        "all_source_truth_checks_pass": True,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(overall.to_string(index=False))
    print(baseline.to_string(index=False))
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
