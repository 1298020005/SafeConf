#!/usr/bin/env python3
"""E148: Nadig split-half and cross-guide reproducibility audit.

Freeze mode reads labels, cell counts, guide metadata and deployable scores only.
Analysis mode then opens expression values, measures experimental reproducibility,
and asks whether directional risk still tracks model error after noise adjustment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E148_nadig_noise_reproducibility_20260714"
TABLES, REPORTS = OUT / "tables", OUT / "reports"
SOURCES = {
    "HepG2": Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/NadigOConner2024_hepg2.h5ad"),
    "Jurkat": Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/NadigOConner2024_jurkat.h5ad"),
}
SELECTION = ROOT / "docs/实验结果/E136_nadig_two_cellline_contract_20260714/tables/E136_SELECTED_PERTURBATIONS.csv"
E138_TASKS = ROOT / "docs/实验结果/E138_nadig_formal_dual_models_20260714/Nadig_two_cellline/TASK_RISK_TABLE.csv"
E139_SCORES = ROOT / "docs/实验结果/E139_nadig_directional_confirmation_20260714/tables/E139_DIRECTIONAL_SCORES_BEFORE_TRUTH.csv"
E139_TASKS = ROOT / "docs/实验结果/E139_nadig_directional_confirmation_20260714/tables/E139_TASK_AUDIT.csv"
PANEL = ROOT / "docs/实验结果/E138_nadig_formal_dual_models_20260714/Nadig_two_cellline/GENE_PANEL.csv"
SEED = 202607148
N_SPLIT_REPEATS = 50
N_BOOTSTRAP = 3000
MIN_CELLS_PER_HALF = 20
MIN_CELLS_PER_GUIDE = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rho(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.unique(a[mask]).size < 2 or np.unique(b[mask]).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a[mask]), rankdata(b[mask]))[0, 1])


def vector_errors(left, right) -> tuple[float, float, float]:
    left, right = np.asarray(left, float), np.asarray(right, float)
    rmse = float(np.sqrt(np.mean((left - right) ** 2)))
    centered_left, centered_right = left - left.mean(), right - right.mean()
    pearson = float(np.corrcoef(centered_left, centered_right)[0, 1]) if np.std(centered_left) > 1e-12 and np.std(centered_right) > 1e-12 else 0.0
    denominator = float(np.linalg.norm(centered_left) * np.linalg.norm(centered_right))
    cosine = float(np.dot(centered_left, centered_right) / denominator) if denominator > 1e-12 else 0.0
    return rmse, 1 - pearson, 1 - cosine


def normalize_log1p(x):
    x = sp.csr_matrix(x).astype(np.float32)
    totals = np.asarray(x.sum(axis=1)).ravel()
    scale = np.divide(1e4, totals, out=np.zeros_like(totals, dtype=np.float32), where=totals > 0)
    x = (sp.diags(scale) @ x).tocsr()
    x.data = np.log1p(x.data)
    return x


def stratified_halves(indices: np.ndarray, batches: np.ndarray, rng: np.random.Generator):
    left, right = [], []
    for batch in sorted(set(batches[indices].astype(str))):
        block = indices[batches[indices].astype(str) == batch].copy()
        rng.shuffle(block)
        left.extend(block[::2]); right.extend(block[1::2])
    return np.asarray(left, int), np.asarray(right, int)


def freeze() -> None:
    for directory in [OUT, TABLES, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)
    selected = set(pd.read_csv(SELECTION).perturbation.astype(str))
    metadata_rows = []
    source_status = {}
    for context, path in SOURCES.items():
        data = ad.read_h5ad(path, backed="r")
        obs = data.obs
        source_status[context] = {"path": str(path), "sha256": sha256(path), "shape": [int(data.n_obs), int(data.n_vars)]}
        subset = obs[obs.perturbation.astype(str).isin(selected)].copy()
        for perturbation, group in subset.groupby(subset.perturbation.astype(str), observed=True):
            transcript_counts = group.transcript.astype(str).value_counts()
            metadata_rows.append({
                "context": context, "perturbation": str(perturbation), "n_cells": len(group),
                "n_batches": int(group.batch.astype(str).nunique()), "n_guide_ids": int(group.guide_id.astype(str).nunique()),
                "n_transcript_groups": int(group.transcript.astype(str).nunique()),
                "n_transcript_groups_ge10": int((transcript_counts >= MIN_CELLS_PER_GUIDE).sum()),
                "expression_values_opened_for_freeze": False,
            })
        data.file.close()
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(TABLES / "E148_METADATA_ONLY_TASKS.csv", index=False)
    direction = pd.read_csv(E139_SCORES)
    deployable = pd.read_csv(E138_TASKS, usecols=["fold_id", "task_id", "context", "perturbation",
        "safeconf_calibrated_pair_risk", "safeconf_frozen_pair_risk", "risk_model_disagreement", "baseline_predicted_magnitude"])
    scores = direction.merge(deployable, on=["fold_id", "task_id", "context", "perturbation"], validate="one_to_one")
    scores["target_truth_used_for_score_or_noise_transform"] = False
    scores.to_csv(TABLES / "E148_SCORES_BEFORE_EXPRESSION_NOISE.csv", index=False)
    status = {
        "experiment": "E148_nadig_noise_reproducibility", "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "frozen_before_expression_values_opened_for_noise_endpoints", "sources": source_status,
        "selection_sha256": sha256(SELECTION), "panel_sha256": sha256(PANEL),
        "score_snapshot_sha256": sha256(TABLES / "E148_SCORES_BEFORE_EXPRESSION_NOISE.csv"),
        "n_unique_context_perturbation_tasks": len(metadata), "n_split_repeats": N_SPLIT_REPEATS,
        "minimum_cells_per_half": MIN_CELLS_PER_HALF, "minimum_cells_per_guide": MIN_CELLS_PER_GUIDE,
        "primary_diagnostic": "partial Spearman directional risk vs directional model error controlling split-half noise and log cell count",
        "bootstrap_unit": "perturbation_gene_cluster_across_contexts_and_folds", "n_bootstrap": N_BOOTSTRAP,
        "independence_note": "new technical-noise endpoint on previously used Nadig data; not an independent biological validation",
    }
    (OUT / "FREEZE_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    (OUT / "ANALYSIS_CONTRACT.md").write_text(
        "# E148 分析合同｜Nadig 实验噪声与模型风险\n\n"
        "在不打开表达矩阵数值的阶段冻结 96 个基因、两个细胞背景、guide/batch 元数据和全部风险分数。\n\n"
        "- 每个 context×perturbation 的细胞在 batch 内随机二分；control 同样独立二分。\n"
        "- 在 E138 固定 512 基因轴上重复 50 次，计算 split-half RMSE、centered Pearson error、centered cosine error。\n"
        "- 主诊断：控制 split-half 方向噪声和 log(n_cells) 后，Directional-SafeConf 与模型方向误差的 partial Spearman。\n"
        "- 补充：高复现任务子集、风险与实验噪声直接关联，以及满足每组≥10细胞的跨 transcript/guide 一致性。\n"
        "- 同一扰动基因跨背景、fold 和重复整体聚类 bootstrap；细胞不是统计独立单位。\n"
        "- 本数据已用于 E139，故属于新技术终点审计。结果不能重新包装为独立确认。\n"
    )
    (OUT / "README_先看这个.md").write_text("# E148 先看这个\n\n先读 `ANALYSIS_CONTRACT.md`；完成后读 `reports/E148_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def load_context_matrix(context: str):
    panel = pd.read_csv(PANEL).raw_gene.astype(str).tolist()
    selected = set(pd.read_csv(SELECTION).perturbation.astype(str))
    raw = ad.read_h5ad(SOURCES[context])
    obs_pert = raw.obs.perturbation.astype(str)
    keep = obs_pert.isin(selected | {"control"}).to_numpy()
    gene_index = {gene: index for index, gene in enumerate(raw.var_names.astype(str))}
    columns = [gene_index[gene] for gene in panel]
    x = normalize_log1p(raw.X[keep][:, columns])
    obs = raw.obs.loc[keep, ["perturbation", "batch", "transcript", "guide_id"]].copy().reset_index(drop=True)
    return x, obs


def measure_context(context: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, obs = load_context_matrix(context)
    perturbations = sorted(set(obs.perturbation.astype(str)) - {"control"})
    batches = obs.batch.astype(str).to_numpy()
    control_indices = np.flatnonzero(obs.perturbation.astype(str).eq("control").to_numpy())
    split_rows = []
    for repeat in range(N_SPLIT_REPEATS):
        rng = np.random.default_rng(SEED + repeat)
        ctrl_left, ctrl_right = stratified_halves(control_indices, batches, rng)
        if min(len(ctrl_left), len(ctrl_right)) < MIN_CELLS_PER_HALF:
            raise RuntimeError(f"{context}: insufficient control split")
        control_left = np.asarray(x[ctrl_left].mean(axis=0)).ravel()
        control_right = np.asarray(x[ctrl_right].mean(axis=0)).ravel()
        for perturbation in perturbations:
            indices = np.flatnonzero(obs.perturbation.astype(str).eq(perturbation).to_numpy())
            left, right = stratified_halves(indices, batches, rng)
            if min(len(left), len(right)) < MIN_CELLS_PER_HALF:
                continue
            effect_left = np.asarray(x[left].mean(axis=0)).ravel() - control_left
            effect_right = np.asarray(x[right].mean(axis=0)).ravel() - control_right
            rmse, pearson_error, cosine_error = vector_errors(effect_left, effect_right)
            split_rows.append({"context": context, "perturbation": perturbation, "repeat": repeat,
                               "n_cells": len(indices), "n_left": len(left), "n_right": len(right),
                               "split_half_rmse": rmse, "split_half_pearson_error": pearson_error,
                               "split_half_cosine_error": cosine_error})
    split = pd.DataFrame(split_rows)
    aggregate = split.groupby(["context", "perturbation"], as_index=False).agg(
        n_cells=("n_cells", "first"), n_split_repeats=("repeat", "nunique"),
        split_half_rmse_mean=("split_half_rmse", "mean"), split_half_rmse_sd=("split_half_rmse", "std"),
        split_half_pearson_error_mean=("split_half_pearson_error", "mean"),
        split_half_cosine_error_mean=("split_half_cosine_error", "mean"))

    control_full = np.asarray(x[control_indices].mean(axis=0)).ravel()
    guide_rows = []
    for perturbation in perturbations:
        block = obs[obs.perturbation.astype(str).eq(perturbation)]
        eligible = block.transcript.astype(str).value_counts()
        eligible = eligible[eligible >= MIN_CELLS_PER_GUIDE].index.astype(str).tolist()
        effects = []
        for transcript in eligible:
            indices = block.index[block.transcript.astype(str).eq(transcript)].to_numpy(int)
            effects.append((transcript, len(indices), np.asarray(x[indices].mean(axis=0)).ravel() - control_full))
        for left_index in range(len(effects)):
            for right_index in range(left_index + 1, len(effects)):
                left_name, left_n, left_effect = effects[left_index]
                right_name, right_n, right_effect = effects[right_index]
                rmse, pearson_error, cosine_error = vector_errors(left_effect, right_effect)
                guide_rows.append({"context": context, "perturbation": perturbation,
                                   "guide_group_left": left_name, "guide_group_right": right_name,
                                   "n_left": left_n, "n_right": right_n, "cross_guide_rmse": rmse,
                                   "cross_guide_pearson_error": pearson_error, "cross_guide_cosine_error": cosine_error})
    return aggregate, pd.DataFrame(guide_rows)


def partial_spearman(score, endpoint, covariates) -> float:
    frame = pd.DataFrame({"score": score, "endpoint": endpoint})
    for index, values in enumerate(covariates):
        frame[f"cov{index}"] = values
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 5:
        return float("nan")
    ranked = frame.rank(method="average")
    x = np.column_stack([np.ones(len(ranked)), *[ranked[column] for column in ranked if column.startswith("cov")]])
    score_residual = ranked.score.to_numpy() - x @ np.linalg.lstsq(x, ranked.score.to_numpy(), rcond=None)[0]
    endpoint_residual = ranked.endpoint.to_numpy() - x @ np.linalg.lstsq(x, ranked.endpoint.to_numpy(), rcond=None)[0]
    return float(np.corrcoef(score_residual, endpoint_residual)[0, 1])


def summarize_metrics(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold_id, context), group in tasks.groupby(["fold_id", "context"], sort=True):
        threshold = group.split_half_pearson_error_mean.median()
        high_repro = group[group.split_half_pearson_error_mean <= threshold]
        rows.extend([
            {"fold_id": fold_id, "context": context, "metric": "directional_risk_vs_model_direction_error",
             "n_tasks": len(group), "estimate": rho(group.directional_risk_frozen, group.direction_error_rank_target)},
            {"fold_id": fold_id, "context": context, "metric": "directional_risk_vs_split_half_direction_noise",
             "n_tasks": len(group), "estimate": rho(group.directional_risk_frozen, group.split_half_pearson_error_mean)},
            {"fold_id": fold_id, "context": context, "metric": "partial_directional_risk_vs_model_error_control_noise_ncells",
             "n_tasks": len(group), "estimate": partial_spearman(group.directional_risk_frozen, group.direction_error_rank_target,
                [group.split_half_pearson_error_mean, np.log1p(group.n_cells_noise)])},
            {"fold_id": fold_id, "context": context, "metric": "high_repro_directional_risk_vs_model_direction_error",
             "n_tasks": len(high_repro), "estimate": rho(high_repro.directional_risk_frozen, high_repro.direction_error_rank_target)},
        ])
    return pd.DataFrame(rows)


def cluster_bootstrap(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 1000)
    perturbations = sorted(tasks.perturbation.astype(str).unique())
    rows = []
    for draw in range(N_BOOTSTRAP):
        sampled = rng.choice(perturbations, len(perturbations), replace=True)
        pieces = [tasks[tasks.perturbation.astype(str).eq(gene)] for gene in sampled]
        sample = pd.concat(pieces, ignore_index=True)
        metrics = summarize_metrics(sample).groupby("metric").estimate.mean()
        row = {"draw": draw, **metrics.to_dict()}
        rows.append(row)
    draws = pd.DataFrame(rows)
    summary = []
    for column in draws.columns[1:]:
        values = draws[column].to_numpy(float)
        summary.append({"metric": column, "n_bootstrap": N_BOOTSTRAP, "median": np.nanmedian(values),
                        "ci95_low": np.nanquantile(values, .025), "ci95_high": np.nanquantile(values, .975),
                        "fraction_above_zero": np.nanmean(values > 0)})
    return draws, pd.DataFrame(summary)


def analyze() -> None:
    status = json.loads((OUT / "FREEZE_STATUS.json").read_text())
    score_file = TABLES / "E148_SCORES_BEFORE_EXPRESSION_NOISE.csv"
    if sha256(score_file) != status["score_snapshot_sha256"]:
        raise RuntimeError("score snapshot changed after freeze")
    aggregates, guide_pairs = [], []
    for context in SOURCES:
        aggregate, guides = measure_context(context)
        aggregates.append(aggregate); guide_pairs.append(guides)
    noise = pd.concat(aggregates, ignore_index=True).rename(columns={"n_cells": "n_cells_noise"})
    guides = pd.concat(guide_pairs, ignore_index=True)
    scores = pd.read_csv(score_file)
    model_truth = pd.read_csv(E139_TASKS, usecols=["fold_id", "task_id", "context", "perturbation", "direction_error_rank_target",
        "error_centered_pearson_mean", "error_centered_cosine_mean"])
    all_tasks = scores.merge(model_truth, on=["fold_id", "task_id", "context", "perturbation"], validate="one_to_one")
    all_tasks = all_tasks.merge(noise, on=["context", "perturbation"], how="inner", validate="many_to_one", suffixes=("", "_noise"))
    # A biological context x perturbation can also appear as a source-context
    # random/unseen diagnostic in the opposite fold.  The frozen contract calls
    # for unique context x perturbation tasks, so the primary analysis retains
    # only the genuinely held-out context rows.  Source-context rows remain in
    # a separate audit table and are never counted as independent evidence.
    primary_settings = {"context_unseen", "context_and_perturbation_unseen"}
    tasks = all_tasks[all_tasks.setting.astype(str).isin(primary_settings)].copy()
    if tasks.duplicated(["context", "perturbation"]).any():
        raise RuntimeError("primary E148 rows are not unique context x perturbation tasks")
    metrics = summarize_metrics(tasks)
    macro = metrics.groupby("metric", as_index=False).agg(n_strata=("estimate", "count"), macro_estimate=("estimate", "mean"))
    draws, bootstrap = cluster_bootstrap(tasks)
    guide_summary = pd.DataFrame()
    if len(guides):
        guide_summary = guides.groupby(["context", "perturbation"], as_index=False).agg(
            n_guide_pairs=("guide_group_left", "size"), cross_guide_rmse_mean=("cross_guide_rmse", "mean"),
            cross_guide_pearson_error_mean=("cross_guide_pearson_error", "mean"),
            cross_guide_cosine_error_mean=("cross_guide_cosine_error", "mean"))
    b = bootstrap.set_index("metric")
    partial_key = "partial_directional_risk_vs_model_error_control_noise_ncells"
    raw_key = "directional_risk_vs_model_direction_error"
    noise_key = "directional_risk_vs_split_half_direction_noise"
    high_key = "high_repro_directional_risk_vs_model_direction_error"
    lines = ["# E148｜Nadig split-half 与跨 guide 复现性", "",
             "## 核心结果", "",
             "| 诊断 | 宏平均 | cluster bootstrap 95% CI |", "|---|---:|---:|"]
    for label, key in [("风险→模型方向误差", raw_key), ("风险→split-half方向噪声", noise_key),
                       ("控制噪声与细胞数后的偏相关", partial_key), ("高复现任务中的风险→模型误差", high_key)]:
        value = macro.loc[macro.metric.eq(key), "macro_estimate"].iloc[0]
        ci = b.loc[key]
        lines.append(f"| {label} | {value:.3f} | [{ci.ci95_low:.3f}, {ci.ci95_high:.3f}] |")
    lines += ["", "## 覆盖", "",
              f"- split-half：{noise.context.nunique()} 个背景、{noise.perturbation.nunique()} 个基因、{len(noise)} 个唯一背景×基因任务，每任务 {N_SPLIT_REPEATS} 次分半。",
              f"- 主分析：{len(tasks)} 个真正留出细胞背景且唯一的背景×基因任务；另有 {len(all_tasks) - len(tasks)} 个 source-context 诊断行只落盘，不进入主估计。",
              f"- 跨 transcript/guide：{len(guide_summary)} 个满足每组至少 {MIN_CELLS_PER_GUIDE} 个细胞的背景×基因任务；覆盖不足时只作探索。", "",
              "## 解释规则", "",
              "split-half 误差衡量同一公开实验内部的测量稳定性，不是模型理论上限。若风险在控制噪声后仍与模型误差正相关，支持其不只是在识别低细胞数或实验噪声；若消失，则必须收缩主张。该分析复用 Nadig 数据，不能算新的外部确认。"]
    (REPORTS / "E148_REPORT.md").write_text("\n".join(lines) + "\n")
    noise.to_csv(TABLES / "E148_SPLIT_HALF_TASK_SUMMARY.csv", index=False)
    guides.to_csv(TABLES / "E148_CROSS_GUIDE_PAIRS.csv", index=False)
    guide_summary.to_csv(TABLES / "E148_CROSS_GUIDE_TASK_SUMMARY.csv", index=False)
    all_tasks.to_csv(TABLES / "E148_ALL_MODEL_AND_NOISE_DIAGNOSTICS.csv", index=False)
    tasks.to_csv(TABLES / "E148_TASK_MODEL_AND_NOISE.csv", index=False)
    metrics.to_csv(TABLES / "E148_STRATUM_METRICS.csv", index=False)
    macro.to_csv(TABLES / "E148_MACRO_METRICS.csv", index=False)
    draws.to_csv(TABLES / "E148_CLUSTER_BOOTSTRAP_DRAWS.csv", index=False)
    bootstrap.to_csv(TABLES / "E148_CLUSTER_BOOTSTRAP_SUMMARY.csv", index=False)
    run = {**status, "generated_at": datetime.now().isoformat(timespec="seconds"), "status": "complete",
           "n_noise_tasks": len(noise), "n_primary_unique_model_task_rows": len(tasks),
           "n_source_context_diagnostic_rows_excluded_from_primary": len(all_tasks) - len(tasks),
           "n_cross_guide_tasks": len(guide_summary),
           "postfreeze_implementation_correction": "primary restricted to held-out context settings to enforce unique context x perturbation tasks",
           "score_refit": False, "noise_endpoint_used_to_change_score": False}
    (OUT / "RUN_STATUS.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(run, ensure_ascii=False, indent=2)); print(macro.to_string(index=False)); print(bootstrap.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--freeze-only", action="store_true"); args = parser.parse_args()
    freeze() if args.freeze_only else analyze()


if __name__ == "__main__":
    main()
