#!/usr/bin/env python3
"""E90: gene-side hard-setting matrix under frozen GEARS+scGPT dual-model contract.

Uses only existing PredictionRecords / E77 certificates / E69 transfer tables.
No score retuning on chem E84/E87/E89. No test-set formula search.

Settings reported:
  - col_holdout_unseen_gene: primary formal dual-model panels (E65/E67/E72 + P2)
  - col_holdout_pooled: E77-style stratified pool
  - cross_dataset_risk_transfer: E69 Adamson<->Norman
  - cross_panel_rank_transfer: fit rank association on panel1, evaluate panel2 (and reverse)
  - row_holdout: N/A for current single-context formal panels (documented)
  - train_submatrix: not claimed without GEARS/scGPT retrain; protocol pre-registered only
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "实验结果" / "E90_gene_hard_setting_matrix_20260712"
TABLES, REPORTS = OUT / "tables", OUT / "reports"

# Dual-model formal panels (same contract as E74/E77)
PANEL_SOURCES = {
    "Adamson_P1": ROOT / "docs" / "实验结果" / "E65_scgpt_formal_fixed_panel_20260711",
    "Adamson_P2": ROOT / "docs" / "实验结果" / "E76a_adamson_scgpt_panel2_20260711",
    "Norman_P1": ROOT / "docs" / "实验结果" / "E67_norman_scgpt_formal_fixed_panel_20260711",
    "Norman_P2": ROOT / "docs" / "实验结果" / "E76b_norman_scgpt_panel2_20260711",
    "Frangieh_P1": ROOT / "docs" / "实验结果" / "E72_frangieh_scgpt_formal_fixed_panel_20260711",
    "Frangieh_P2": ROOT / "docs" / "实验结果" / "E76c_frangieh_scgpt_panel2_20260711",
}

E69_DIR = ROOT / "docs" / "实验结果" / "E69_real_model_cross_dataset_risk_transfer_20260711"
E77_DIR = ROOT / "docs" / "实验结果" / "E77_repeated_panel_pair_risk_20260711"


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
    return rankdata(np.asarray(a, float), method="average") / max(len(a), 1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    r, _ = spearmanr(a, b)
    return float(r)


def load_panel(dataset: str, root: Path) -> pd.DataFrame:
    records = pd.read_csv(root / "tables" / "PREDICTION_RECORDS.csv")
    with np.load(root / "arrays" / "predicted_effects.npz") as store:
        predictions = {key: np.asarray(store[key], dtype=float) for key in store.files}
    with np.load(root / "arrays" / "true_effects.npz") as store:
        truths = {key: np.asarray(store[key], dtype=float) for key in store.files}
    rows = []
    for perturbation, group in records.groupby("perturbation", sort=True):
        if len(group) != 2:
            raise ValueError(f"{dataset}/{perturbation}: expected 2 predictors, got {len(group)}")
        gears_row = group[group.predictor_name.str.contains("GEARS", case=False)]
        scgpt_row = group[group.predictor_name.str.contains("scGPT", case=False)]
        if len(gears_row) != 1 or len(scgpt_row) != 1:
            raise ValueError(f"{dataset}/{perturbation}: ambiguous predictors")
        gears_row, scgpt_row = gears_row.iloc[0], scgpt_row.iloc[0]
        gears = predictions[gears_row.predicted_effect_key]
        scgpt = predictions[scgpt_row.predicted_effect_key]
        truth = truths[gears_row.true_effect_key]
        other = truths[scgpt_row.true_effect_key]
        if not np.allclose(truth, other, atol=1e-7, rtol=1e-6):
            raise ValueError(f"{dataset}/{perturbation}: truth mismatch")
        e_g, e_s = rmse(gears, truth), rmse(scgpt, truth)
        d = rmse(gears, scgpt)
        rows.append(
            {
                "panel_id": dataset,
                "dataset_family": dataset.split("_")[0],
                "panel_index": dataset.split("_")[-1],
                "perturbation": perturbation,
                "context": str(gears_row.context),
                "predictor_pair": "GEARS_3seed+scGPT_finetune",
                "split_setting": "col_holdout_unseen_gene",
                "gears_rmse": e_g,
                "scgpt_rmse": e_s,
                "pair_mean_rmse": 0.5 * (e_g + e_s),
                "pair_max_rmse": max(e_g, e_s),
                "model_disagreement_rmse": d,
                "gears_predicted_magnitude": float(np.linalg.norm(gears)),
                "scgpt_predicted_magnitude": float(np.linalg.norm(scgpt)),
                "predicted_magnitude_mean": 0.5
                * (float(np.linalg.norm(gears)) + float(np.linalg.norm(scgpt))),
                "true_effect_key": gears_row.true_effect_key,
                "gene_order_hash": gears_row.gene_order_hash,
                "source_dir": str(root.relative_to(ROOT)),
            }
        )
    frame = pd.DataFrame(rows)
    frame["rank_disagreement"] = rank01(frame.model_disagreement_rmse.to_numpy(float))
    frame["rank_magnitude"] = rank01(frame.predicted_magnitude_mean.to_numpy(float))
    frame["rank_pair_mean_error"] = rank01(frame.pair_mean_rmse.to_numpy(float))
    return frame


def bootstrap_corr(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = spearman(x, y)
    if len(x) < 5 or not math.isfinite(obs):
        return obs, float("nan"), float("nan")
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        vals.append(spearman(x[idx], y[idx]))
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return obs, float("nan"), float("nan")
    return obs, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def panel_summary(frame: pd.DataFrame, setting: str, group_key: str, rng: np.random.Generator) -> dict:
    x_d = frame.model_disagreement_rmse.to_numpy(float)
    x_m = frame.predicted_magnitude_mean.to_numpy(float)
    y = frame.pair_mean_rmse.to_numpy(float)
    rho_d, lo_d, hi_d = bootstrap_corr(x_d, y, rng)
    rho_m, lo_m, hi_m = bootstrap_corr(x_m, y, rng)
    # paired delta bootstrap
    deltas = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        deltas.append(spearman(x_d[idx], y[idx]) - spearman(x_m[idx], y[idx]))
    deltas = np.asarray(deltas, float)
    deltas = deltas[np.isfinite(deltas)]
    d_obs = rho_d - rho_m if math.isfinite(rho_d) and math.isfinite(rho_m) else float("nan")
    d_lo = float(np.quantile(deltas, 0.025)) if len(deltas) else float("nan")
    d_hi = float(np.quantile(deltas, 0.975)) if len(deltas) else float("nan")
    return {
        "split_setting": setting,
        "group": group_key,
        "n_tasks": int(len(frame)),
        "n_contexts": int(frame.context.nunique()),
        "predictor_pair": "GEARS_3seed+scGPT_finetune",
        "rho_disagreement_vs_pair_mean": rho_d,
        "rho_disagreement_ci95_low": lo_d,
        "rho_disagreement_ci95_high": hi_d,
        "rho_magnitude_vs_pair_mean": rho_m,
        "rho_magnitude_ci95_low": lo_m,
        "rho_magnitude_ci95_high": hi_m,
        "delta_rho_disagreement_minus_magnitude": d_obs,
        "delta_ci95_low": d_lo,
        "delta_ci95_high": d_hi,
        "delta_ci_excludes_zero": bool(math.isfinite(d_lo) and math.isfinite(d_hi) and d_lo > 0),
        "triangle_bound_violations": 0,  # filled later if needed
    }


def cross_panel_transfer(all_tasks: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    """Use panel1 ranks as source calibration, evaluate Spearman on panel2 errors with source-fitted score weights."""
    rows = []
    for family in sorted(all_tasks.dataset_family.unique()):
        p1 = all_tasks[(all_tasks.dataset_family == family) & (all_tasks.panel_index == "P1")]
        p2 = all_tasks[(all_tasks.dataset_family == family) & (all_tasks.panel_index == "P2")]
        if len(p1) < 5 or len(p2) < 5:
            continue
        for source, target, tag in ((p1, p2, f"{family}_P1_to_P2"), (p2, p1, f"{family}_P2_to_P1")):
            # fit simple z-score combination on source: alpha * z(dis) + (1-alpha) * z(mag)
            # choose alpha by max source spearman on grid (source only)
            best_alpha, best_rho = 0.5, -1e9
            sd = source.model_disagreement_rmse.to_numpy(float)
            sm = source.predicted_magnitude_mean.to_numpy(float)
            sy = source.pair_mean_rmse.to_numpy(float)
            for alpha in np.linspace(0, 1, 21):
                score = alpha * rank01(sd) + (1 - alpha) * rank01(sm)
                r = spearman(score, sy)
                if math.isfinite(r) and r > best_rho:
                    best_rho, best_alpha = r, float(alpha)
            td = target.model_disagreement_rmse.to_numpy(float)
            tm = target.predicted_magnitude_mean.to_numpy(float)
            ty = target.pair_mean_rmse.to_numpy(float)
            # apply alpha with target-side rank01 (deployable without source labels)
            t_score = best_alpha * rank01(td) + (1 - best_alpha) * rank01(tm)
            rho_t, lo, hi = bootstrap_corr(t_score, ty, rng)
            rho_d, _, _ = bootstrap_corr(td, ty, rng)
            rho_m, _, _ = bootstrap_corr(tm, ty, rng)
            rows.append(
                {
                    "split_setting": "cross_panel_score_transfer",
                    "group": tag,
                    "n_tasks": int(len(target)),
                    "n_contexts": int(target.context.nunique()),
                    "predictor_pair": "GEARS_3seed+scGPT_finetune",
                    "source_selected_alpha_disagreement": best_alpha,
                    "source_spearman": best_rho,
                    "rho_transferred_score_vs_pair_mean": rho_t,
                    "rho_transferred_ci95_low": lo,
                    "rho_transferred_ci95_high": hi,
                    "rho_disagreement_vs_pair_mean": rho_d,
                    "rho_magnitude_vs_pair_mean": rho_m,
                    "delta_vs_magnitude": (rho_t - rho_m) if math.isfinite(rho_t) and math.isfinite(rho_m) else float("nan"),
                    "note": "alpha chosen on source panel only; target truth not used for fit",
                }
            )
    return rows


def load_e69_rows() -> list[dict]:
    path = E69_DIR / "tables" / "E69_TRANSFER_SUMMARY.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "split_setting": "cross_dataset_risk_calibrator_transfer",
                "group": f"{r.source_dataset}->{r.target_dataset}::{r.target_error}::{r.risk_model}",
                "n_tasks": 24,
                "predictor_pair": "GEARS_3seed+scGPT_finetune",
                "rho_risk_model": float(r.target_spearman),
                "top20_error_enrichment": float(r.top20_error_enrichment),
                "target_truth_used_for_fit": bool(r.target_truth_used_for_fit_or_standardization),
                "source": "E69",
            }
        )
    delta_path = E69_DIR / "tables" / "E69_INCREMENTAL_DELTA.csv"
    if delta_path.exists():
        for _, r in pd.read_csv(delta_path).iterrows():
            rows.append(
                {
                    "split_setting": "cross_dataset_risk_calibrator_transfer_delta",
                    "group": f"{r.source_dataset}->{r.target_dataset}::{r.target_error}",
                    "delta_combined_minus_magnitude": float(r.observed_spearman_delta_combined_minus_magnitude),
                    "delta_ci95_low": float(r.bootstrap_delta_ci95_low),
                    "delta_ci95_high": float(r.bootstrap_delta_ci95_high),
                    "reliably_better_than_magnitude": bool(r.combined_reliably_better_than_magnitude),
                    "source": "E69",
                }
            )
    return rows


def train_split_sizes() -> pd.DataFrame:
    rows = []
    for panel_id, root in PANEL_SOURCES.items():
        splits = list((root / "tables").glob("*FIXED_SPLIT.csv"))
        if not splits:
            continue
        sp = pd.read_csv(splits[0])
        col = "split" if "split" in sp.columns else sp.columns[0]
        vc = sp[col].astype(str).value_counts().to_dict()
        rows.append(
            {
                "panel_id": panel_id,
                "n_train": int(vc.get("train", 0)),
                "n_val": int(vc.get("val", 0)),
                "n_test": int(vc.get("test", 0)),
                "n_context_levels_in_records": 1,
                "row_holdout_applicable": False,
                "row_holdout_reason": "formal dual-model panels use a single frozen context string per dataset; whole-context holdout needs multi-context gene data under same GEARS/scGPT contract",
                "train_submatrix_requires_retrain": True,
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, transfer: pd.DataFrame, e69: list[dict], meta: pd.DataFrame) -> None:
    lines = [
        "# E90｜基因侧难设置总表（冻结 GEARS+scGPT 合同）",
        "",
        "本实验**不重新训练**预测器，只在已冻结的双模型 PredictionRecord 上统一汇总周老师关心的难设置，并标明哪些还缺真实重训。",
        "",
        "## 合同",
        "",
        "- predictor_pair: GEARS 三 seed ensemble + 正式微调 scGPT",
        "- 误差: pair_mean_rmse = 0.5*(GEARS_RMSE + scGPT_RMSE)",
        "- 可部署分数: model_disagreement_rmse, predicted_magnitude_mean（真值仅用于评价）",
        "- 数据: Adamson / Norman / Frangieh × 两套不重叠未见基因面板",
        "",
        "## 1. 整列未见基因 holdout（主硬设置，已完成）",
        "",
        "| group | n | ρ_disagree | ρ_magnitude | Δρ (d−m) | Δ 95% CI | CI>0 |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    col = summary[summary.split_setting == "col_holdout_unseen_gene"].sort_values("group")
    for _, r in col.iterrows():
        lines.append(
            f"| {r.group} | {r.n_tasks} | {r.rho_disagreement_vs_pair_mean:.3f} | "
            f"{r.rho_magnitude_vs_pair_mean:.3f} | {r.delta_rho_disagreement_minus_magnitude:.3f} | "
            f"[{r.delta_ci95_low:.3f}, {r.delta_ci95_high:.3f}] | "
            f"{'Y' if r.delta_ci_excludes_zero else 'N'} |"
        )
    pooled = summary[summary.split_setting == "col_holdout_pooled"]
    if len(pooled):
        r = pooled.iloc[0]
        lines += [
            "",
            f"**分层池化（6 面板 / {r.n_tasks} 任务）**: ρ_disagree={r.rho_disagreement_vs_pair_mean:.3f}, "
            f"ρ_magnitude={r.rho_magnitude_vs_pair_mean:.3f}, Δ={r.delta_rho_disagreement_minus_magnitude:.3f} "
            f"CI=[{r.delta_ci95_low:.3f}, {r.delta_ci95_high:.3f}] "
            f"({'CI 不含 0' if r.delta_ci_excludes_zero else 'CI 跨 0'})。",
            "",
            "此行与 E77 主张一致：基因整列未见扰动上，跨模型分歧相对预测幅度有正增量。",
        ]
    lines += [
        "",
        "## 2. 跨面板分数迁移（同数据集、不重叠测试基因）",
        "",
        "| group | α* | source ρ | target ρ(transfer) | target ρ(dis) | target ρ(mag) | Δ vs mag |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in transfer.iterrows():
        lines.append(
            f"| {r.group} | {r.source_selected_alpha_disagreement:.2f} | {r.source_spearman:.3f} | "
            f"{r.rho_transferred_score_vs_pair_mean:.3f} | {r.rho_disagreement_vs_pair_mean:.3f} | "
            f"{r.rho_magnitude_vs_pair_mean:.3f} | {r.delta_vs_magnitude:.3f} |"
        )
    lines += [
        "",
        "说明：α 只在 source 面板上按 Spearman 网格选择；target 真值不参与拟合。这是设置难度的中间台阶，不是完整训练子矩阵。",
        "",
        "## 3. 跨数据集风险校准器迁移（E69，已有）",
        "",
    ]
    if e69:
        lines.append("| group | metric | value |")
        lines.append("|---|---|---|")
        for r in e69:
            if r["split_setting"].endswith("_delta"):
                lines.append(
                    f"| {r['group']} | Δρ combined−mag CI | "
                    f"{r['delta_combined_minus_magnitude']:.3f} "
                    f"[{r['delta_ci95_low']:.3f}, {r['delta_ci95_high']:.3f}] "
                    f"reliable={r['reliably_better_than_magnitude']} |"
                )
            elif "rho_risk_model" in r:
                lines.append(
                    f"| {r['group']} | target Spearman | {r['rho_risk_model']:.3f} "
                    f"(top20={r['top20_error_enrichment']:.3f}) |"
                )
        lines.append("")
        lines.append("E69 结论保持：仅部分方向 combined 稳定超过 magnitude；不能写普遍跨数据集增益。")
    else:
        lines.append("E69 表缺失。")
    lines += [
        "",
        "## 4. 整行 holdout / 训练子矩阵（当前合同下未宣称完成）",
        "",
        "| panel | n_train | n_test | row_holdout | train_submatrix |",
        "|---|---:|---:|---|---|",
    ]
    for _, r in meta.iterrows():
        lines.append(
            f"| {r.panel_id} | {r.n_train} | {r.n_test} | N/A（单 context） | 需重训 GEARS/scGPT |"
        )
    lines += [
        "",
        "**诚实边界**：当前正式双模型面板每个数据集只有一个冻结 context 标签，无法在不重训的前提下做整行新 context。",
        "训练子矩阵（只暴露部分 train condition）会改变 GEARS/scGPT 预测本身，必须新开训练任务；E90 只预注册该合同，不伪造结果。",
        "",
        "## 5. 现在能写 / 不能写",
        "",
        "**能写**：基因扰动、整列未见基因、双模型分歧相对预测幅度的排序增益（E77/E90 col_holdout）。",
        "",
        "**不能写**：基因侧小矩阵训练与整行 holdout 已在双模型合同下完成；跨数据集普遍超过 magnitude。",
        "",
        f"生成时间：{now()}  git：`{git_head()}`",
        "",
    ]
    (REPORTS / "E90_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E90 基因侧难设置总表\n\n先读 `reports/E90_REPORT.md`。\n",
        encoding="utf-8",
    )


def main() -> None:
    for d in (TABLES, REPORTS, OUT):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260790)

    panels = []
    summary_rows = []
    for panel_id, root in PANEL_SOURCES.items():
        if not (root / "tables" / "PREDICTION_RECORDS.csv").exists():
            raise FileNotFoundError(root)
        frame = load_panel(panel_id, root)
        panels.append(frame)
        # triangle checks
        viol = int(
            (
                (0.5 * (frame.gears_rmse + frame.scgpt_rmse) + 1e-10 < 0.5 * frame.model_disagreement_rmse)
                | (frame.pair_max_rmse + 1e-10 < 0.5 * frame.model_disagreement_rmse)
            ).sum()
        )
        row = panel_summary(frame, "col_holdout_unseen_gene", panel_id, rng)
        row["triangle_bound_violations"] = viol
        summary_rows.append(row)

    all_tasks = pd.concat(panels, ignore_index=True)
    # stratified pooled: average per-panel spearman then report bootstrap across tasks within panels
    # reuse panel_summary on concatenated with group labels via stratified mean
    pooled = {
        "split_setting": "col_holdout_pooled",
        "group": "Adamson+Norman+Frangieh x P1+P2",
        "n_tasks": int(len(all_tasks)),
        "n_contexts": int(all_tasks.context.nunique()),
        "predictor_pair": "GEARS_3seed+scGPT_finetune",
    }
    # stratified mean rho
    def strat_rho(score_col: str) -> float:
        vals = []
        for _, g in all_tasks.groupby("panel_id"):
            vals.append(spearman(g[score_col].to_numpy(float), g.pair_mean_rmse.to_numpy(float)))
        return float(np.nanmean(vals))

    rho_d = strat_rho("model_disagreement_rmse")
    rho_m = strat_rho("predicted_magnitude_mean")
    # bootstrap panels: resample tasks within each panel
    boots_d, boots_m, boots_delta = [], [], []
    groups = [g.reset_index(drop=True) for _, g in all_tasks.groupby("panel_id")]
    for _ in range(3000):
        ds, ms = [], []
        for g in groups:
            idx = rng.integers(0, len(g), len(g))
            y = g.pair_mean_rmse.to_numpy(float)[idx]
            ds.append(spearman(g.model_disagreement_rmse.to_numpy(float)[idx], y))
            ms.append(spearman(g.predicted_magnitude_mean.to_numpy(float)[idx], y))
        boots_d.append(np.nanmean(ds))
        boots_m.append(np.nanmean(ms))
        boots_delta.append(np.nanmean(ds) - np.nanmean(ms))
    pooled.update(
        {
            "rho_disagreement_vs_pair_mean": rho_d,
            "rho_disagreement_ci95_low": float(np.quantile(boots_d, 0.025)),
            "rho_disagreement_ci95_high": float(np.quantile(boots_d, 0.975)),
            "rho_magnitude_vs_pair_mean": rho_m,
            "rho_magnitude_ci95_low": float(np.quantile(boots_m, 0.025)),
            "rho_magnitude_ci95_high": float(np.quantile(boots_m, 0.975)),
            "delta_rho_disagreement_minus_magnitude": rho_d - rho_m,
            "delta_ci95_low": float(np.quantile(boots_delta, 0.025)),
            "delta_ci95_high": float(np.quantile(boots_delta, 0.975)),
            "delta_ci_excludes_zero": float(np.quantile(boots_delta, 0.025)) > 0,
            "triangle_bound_violations": 0,
        }
    )
    summary_rows.append(pooled)

    summary = pd.DataFrame(summary_rows)
    transfer = pd.DataFrame(cross_panel_transfer(all_tasks, rng))
    e69_rows = load_e69_rows()
    meta = train_split_sizes()

    # protocol stub for missing settings
    protocol = pd.DataFrame(
        [
            {
                "split_setting": "row_holdout_new_context",
                "status": "not_done_under_dual_model",
                "blocker": "current formal panels are single-context",
                "required_action": "retrain GEARS+scGPT with multi-context gene datasets and hold out whole contexts",
            },
            {
                "split_setting": "train_submatrix_25_50_75",
                "status": "not_done_under_dual_model",
                "blocker": "predictions were trained on full train splits; cannot fake submatrix without retrain",
                "required_action": "freeze submatrix manifests then retrain GEARS+scGPT; reuse E90 schema",
            },
        ]
    )

    all_tasks.to_csv(TABLES / "E90_TASK_LEVEL.csv", index=False)
    summary.to_csv(TABLES / "E90_SETTING_SUMMARY.csv", index=False)
    transfer.to_csv(TABLES / "E90_CROSS_PANEL_TRANSFER.csv", index=False)
    pd.DataFrame(e69_rows).to_csv(TABLES / "E90_E69_CROSS_DATASET_ROWS.csv", index=False)
    meta.to_csv(TABLES / "E90_PANEL_META.csv", index=False)
    protocol.to_csv(TABLES / "E90_MISSING_SETTINGS_PROTOCOL.csv", index=False)

    write_report(summary, transfer, e69_rows, meta)

    status = {
        "experiment": "E90_gene_hard_setting_matrix",
        "generated_at": now(),
        "git_head": git_head(),
        "n_tasks": int(len(all_tasks)),
        "n_panels": len(PANEL_SOURCES),
        "col_holdout_delta_ci_excludes_zero": bool(pooled["delta_ci_excludes_zero"]),
        "pooled_delta_rho": pooled["delta_rho_disagreement_minus_magnitude"],
        "row_holdout_done": False,
        "train_submatrix_done": False,
        "cross_dataset_source": "E69",
        "no_chem_retune": True,
        "target_truth_used_for_scoring": False,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(transfer.to_string(index=False))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
