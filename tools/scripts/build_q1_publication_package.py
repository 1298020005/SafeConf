#!/usr/bin/env python3
"""Build a Q1/CCF-A oriented publication upgrade package for SafeConf.

The script does not rerun heavy perturbation prediction models.  It audits the
already frozen/result-ready CSVs and turns them into a compact submission workbench:

* baseline ladder across existing reliability scores
* magnitude-controlled incremental value summary
* cost-effectiveness / top-risk retrieval summary
* Tahoe chemical boundary table
* Q1 readiness and next-experiment action matrix

Outputs are written under docs/投稿升级/Q1_CCFA_upgrade_20260707/.
"""

from __future__ import annotations

import html
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "投稿升级" / "Q1_CCFA_upgrade_20260707"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"


SRC = {
    "baseline_ladder": ROOT
    / "docs/实验结果/Reliability_model_corrected_20260610/tables/RELIABILITY_BASELINE_LADDER.csv",
    "within_magnitude": ROOT
    / "docs/实验结果/Reliability_model_corrected_20260610/tables/RELIABILITY_WITHIN_MAGNITUDE_STRATUM.csv",
    "formal_main": ROOT
    / "docs/实验结果/Evidence_to_Claim_20260615/figure_ready_tables/FIG2_FORMAL_MAIN_FOREST.csv",
    "e2_residual": ROOT
    / "docs/实验结果/Evidence_to_Claim_20260615/figure_ready_tables/FIG3_E2_MAGNITUDE_RESIDUAL.csv",
    "cost_effectiveness": ROOT
    / "docs/实验结果/Evidence_to_Claim_20260615/figure_ready_tables/FIG5_COST_EFFECTIVENESS.csv",
    "risk_coverage": ROOT
    / "docs/实验结果/Evidence_to_Claim_20260615/figure_ready_tables/SFIG_RISK_COVERAGE.csv",
    "e8b_external": ROOT
    / "docs/实验结果/Evidence_to_Claim_20260615/figure_ready_tables/FIG4_E8B_EXTERNAL_BENCHMARK.csv",
    "tahoe_d5": ROOT
    / "docs/实验结果/Tahoe_D5_combined_triage_20260627/tables/TAHOE_D5_POINT_SUMMARY.csv",
    "source_files": ROOT
    / "docs/实验结果/Evidence_to_Claim_20260615/figure_ready_tables/SOURCE_FILES_USED.csv",
}


SCORE_LABEL = {
    "random": "Random",
    "predicted_magnitude": "Magnitude-only",
    "protocol_v0_2_family_confidence": "Frozen v0.2",
    "safeconf_lodo_risk": "SafeConf LODO",
    "safeconf_lodo_linear_risk": "SafeConf LODO linear",
    "safeconf_perdataset_risk": "SafeConf per-dataset",
    "oracle_magnitude_diagnostic": "Oracle true magnitude",
    "safeconf_full": "SafeConf full",
    "combined_equal": "Combined equal",
    "combined_magnitude75": "Combined 75% magnitude",
    "combined_safeconf75": "Combined 75% SafeConf",
}


def read_csv(key: str) -> pd.DataFrame:
    path = SRC[key]
    if not path.exists():
        raise FileNotFoundError(f"Missing source file for {key}: {path}")
    return pd.read_csv(path)


def rel(path: Path) -> str:
    return path.relative_to(OUT).as_posix()


def rel_from_root(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLES / name
    df.to_csv(path, index=False)
    return path


def fmt_num(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}%"


def compact_table_html(df: pd.DataFrame, columns: Iterable[str] | None = None, max_rows: int = 20) -> str:
    view = df.copy()
    if columns is not None:
        view = view[list(columns)]
    if len(view) > max_rows:
        view = view.head(max_rows)
    return view.to_html(index=False, escape=True, classes="data-table")


def markdown_table(df: pd.DataFrame, columns: Iterable[str] | None = None, max_rows: int = 20) -> str:
    """Small dependency-free Markdown table writer."""
    view = df.copy()
    if columns is not None:
        view = view[list(columns)]
    if len(view) > max_rows:
        view = view.head(max_rows)
    cols = list(view.columns)

    def cell(value: object) -> str:
        text = "" if pd.isna(value) else str(value)
        return text.replace("|", "\\|").replace("\n", "<br>")

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(cell(row[c]) for c in cols) + " |" for _, row in view.iterrows()]
    return "\n".join([header, sep, *rows])


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#1f2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
        }
    )


def save_svg_clean(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path)
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def build_baseline_summary() -> pd.DataFrame:
    df = read_csv("baseline_ladder")
    rows = []
    for score, g in df.groupby("score_name", sort=False):
        rows.append(
            {
                "score_name": score,
                "score_label": SCORE_LABEL.get(score, score),
                "role": {
                    "random": "sanity baseline",
                    "predicted_magnitude": "strong deployable competitor",
                    "protocol_v0_2_family_confidence": "frozen protocol score",
                    "safeconf_lodo_risk": "cross-dataset transfer score",
                    "safeconf_lodo_linear_risk": "simpler transfer variant",
                    "safeconf_perdataset_risk": "within-dataset upper reference",
                    "oracle_magnitude_diagnostic": "non-deployable oracle diagnostic",
                }.get(score, "candidate score"),
                "n_datasets": int(g["dataset_name"].nunique()),
                "median_aligned_rho": float(g["aligned_rho"].median()),
                "median_partial_rho_control_magnitude": float(
                    g["partial_rho_control_magnitude"].median()
                ),
                "median_aurc_reduction_vs_random_pct": float(
                    g["aurc_reduction_vs_random_pct"].median()
                ),
                "positive_aligned_datasets": int((g["aligned_rho"] > 0).sum()),
                "positive_partial_after_magnitude_datasets": int(
                    (g["partial_rho_control_magnitude"] > 0).sum()
                ),
                "deployment_status": "deployable"
                if score
                not in {"random", "oracle_magnitude_diagnostic", "safeconf_perdataset_risk"}
                else (
                    "non-deployable oracle"
                    if score == "oracle_magnitude_diagnostic"
                    else "reference only"
                ),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["deployment_status", "median_partial_rho_control_magnitude", "median_aligned_rho"],
        ascending=[True, False, False],
    )
    return out


def build_formal_evidence() -> pd.DataFrame:
    df = read_csv("formal_main")
    out = df[
        [
            "dataset_name",
            "dataset_family",
            "n",
            "aligned_rho",
            "partial_rho",
            "partial_rho_ci_low",
            "partial_rho_ci_high",
            "magnitude_only_rho",
            "rc80_improve_pct",
            "is_failure_boundary",
        ]
    ].copy()
    out["formal_claim"] = np.where(
        out["partial_rho_ci_low"] > 0,
        "positive after magnitude control",
        "boundary / failure",
    )
    return out


def build_incremental_value() -> pd.DataFrame:
    df = read_csv("e2_residual")
    if "calibration_method" in df.columns:
        df = df[df["calibration_method"].eq("isotonic")].copy()
    df = df.rename(
        columns={
            "residual_partial_rho": "residual_partial_rho_point",
            "residual_partial_ci_low": "residual_partial_rho_ci_low",
            "residual_partial_ci_high": "residual_partial_rho_ci_high",
            "aurc_diff_magnitude_minus_combined": "aurc_improvement_magnitude_minus_combined_point",
            "aurc_diff_ci_low": "aurc_improvement_magnitude_minus_combined_ci_low",
            "aurc_diff_ci_high": "aurc_improvement_magnitude_minus_combined_ci_high",
        }
    )
    keep = [
        "dataset_name",
        "calibration_method",
        "n",
        "n_task_clusters",
        "residual_partial_rho_point",
        "residual_partial_rho_ci_low",
        "residual_partial_rho_ci_high",
        "aurc_improvement_magnitude_minus_combined_point",
        "aurc_improvement_magnitude_minus_combined_ci_low",
        "aurc_improvement_magnitude_minus_combined_ci_high",
    ]
    out = df[keep].copy()
    out["incremental_claim"] = np.where(
        out["residual_partial_rho_ci_low"] > 0,
        "adds information beyond magnitude",
        "needs caution",
    )
    return out


def build_cost_summary() -> pd.DataFrame:
    df = read_csv("cost_effectiveness")
    out = df[df["is_macro_summary"].eq(True)].copy()
    out = out[out["top_fraction"].isin([0.05, 0.10, 0.20])]
    out["score_label"] = out["score_name"].map(SCORE_LABEL).fillna(out["score_label"])
    return out[
        [
            "score_name",
            "score_label",
            "strategy_role",
            "top_fraction",
            "precision",
            "random_expected_precision",
            "enrichment_fold",
            "hits",
            "n_top_risk",
            "n_worst_error",
        ]
    ].sort_values(["top_fraction", "enrichment_fold"], ascending=[True, False])


def build_tahoe_boundary() -> pd.DataFrame:
    df = read_csv("tahoe_d5")
    df = df.copy()
    df["score_label"] = df["score_name"].map(SCORE_LABEL).fillna(df["score_name"])
    mag_ref = (
        df[df["score_name"].eq("predicted_magnitude")]
        .set_index("top_fraction")[["enrichment", "aligned_rho"]]
        .rename(columns={"enrichment": "magnitude_enrichment_same_top_fraction", "aligned_rho": "magnitude_rho"})
    )
    df = df.join(mag_ref, on="top_fraction")
    df["delta_enrichment_vs_magnitude_same_top_fraction"] = (
        df["enrichment"] - df["magnitude_enrichment_same_top_fraction"]
    )
    df["delta_rho_vs_magnitude_reference"] = df["aligned_rho"] - df["magnitude_rho"]
    df["interpretation"] = np.where(
            df["score_name"].eq("predicted_magnitude"),
            "chemical dominant baseline",
            np.where(
            df["delta_enrichment_vs_magnitude_same_top_fraction"] >= 0,
            "competitive with magnitude",
            "below magnitude; keep as boundary evidence",
        ),
    )
    return df[
        [
            "score_name",
            "score_label",
            "top_fraction",
            "n_records",
            "aligned_rho",
            "precision",
            "random_expected_precision",
            "enrichment",
            "magnitude_enrichment_same_top_fraction",
            "delta_enrichment_vs_magnitude_same_top_fraction",
            "delta_rho_vs_magnitude_reference",
            "interpretation",
        ]
    ]


def build_gap_actions() -> pd.DataFrame:
    rows = [
        {
            "priority": 1,
            "gap": "SafeConf vs magnitude 的边界仍会被审稿人追问",
            "why_it_matters": "Tahoe chemical 中 magnitude 更强；必须把边界变成可信叙事，而不是让审稿人觉得我们回避。",
            "action": "新增统一强基线审计：报告 SafeConf、magnitude、support、context、disagreement、combined；按 gene / chemical / cross-context 分层。",
            "deliverable": "E9_STRONG_BASELINE_AUDIT.csv + Fig: baseline ladder by domain",
            "status": "next experiment",
        },
        {
            "priority": 2,
            "gap": "外部验证仍偏聚合层面",
            "why_it_matters": "E8b 说明外部 aggregate association，但一区审稿人会想看冻结外部任务级验证。",
            "action": "接入一个冻结 benchmark：优先 scPerturBench 可落地子集；失败也要形成资源审计和可复现实验日志。",
            "deliverable": "E10_EXTERNAL_TASK_VALIDATION/",
            "status": "next experiment",
        },
        {
            "priority": 3,
            "gap": "方法学贡献需要从打分推进到可控风险",
            "why_it_matters": "CCF-A/更高一区需要更像 ML 方法，而不只是经验特征组合。",
            "action": "已生成 retrospective selective prediction audit；下一步补 calibration split 与 risk-control guarantee。",
            "deliverable": "E11_selective_prediction_audit_20260707/；formal conformal guarantee 待补",
            "status": "audit generated; guarantee pending",
        },
        {
            "priority": 4,
            "gap": "生物学故事仍偏方法验证",
            "why_it_matters": "Genome Biology/Cell Systems 等更看重一个可解释 biological case。",
            "action": "选择 2–3 个高风险任务做案例：预测器分歧、历史支持、细胞背景、通路/marker 是否解释错误。",
            "deliverable": "E12_BIOLOGICAL_CASE_STUDIES/",
            "status": "paper narrative",
        },
        {
            "priority": 5,
            "gap": "代码与结果虽然多，但投稿时需要一键复现边界",
            "why_it_matters": "审稿人和编辑会把可复现性当作加分项；也能保护自己不被质疑挑结果。",
            "action": "建立 paper/ 级别 manifest：每张主图对应源 CSV、脚本、commit、运行命令。",
            "deliverable": "PAPER_REPRODUCIBILITY_MANIFEST.md",
            "status": "engineering",
        },
    ]
    return pd.DataFrame(rows)


def build_venue_strategy() -> pd.DataFrame:
    rows = [
        {
            "track": "Q1 bioinformatics method",
            "target": "Bioinformatics / Briefings in Bioinformatics / PLOS Computational Biology",
            "fit": "最高。SafeConf 是单细胞扰动预测的可靠性评估与实验决策工具。",
            "required_upgrade": "统一强基线、外部验证、risk-budget 决策曲线、可复现包。",
            "risk": "如果写成预测模型，会被要求直接超过 SOTA；必须写成 post-prediction risk auditing。",
        },
        {
            "track": "higher biological computation",
            "target": "Genome Biology / Cell Systems",
            "fit": "中等偏高。需要把方法和真实生物案例绑紧。",
            "required_upgrade": "补 biological case study：错误富集任务对应通路、药物机制或细胞背景差异。",
            "risk": "纯方法审计可能被认为生物发现不足。",
        },
        {
            "track": "CCF-A AI",
            "target": "AAAI / IJCAI / NeurIPS workshop-to-main route / ICML-style ML venue",
            "fit": "当前中等。需要把 SafeConf 升级成 risk-controlled selective prediction 方法。",
            "required_upgrade": "conformal/selective risk control、跨域分布偏移理论化、公开 benchmark 对比。",
            "risk": "现有版本更像应用方法；直接投 CCF-A 主会很危险。",
        },
        {
            "track": "fallback but still useful",
            "target": "BMC Bioinformatics / GigaScience / Scientific Reports",
            "fit": "保底。适合在一区冲刺失败后快速转投。",
            "required_upgrade": "不建议先走这条；会浪费当前证据积累。",
            "risk": "安全但上限低。",
        },
    ]
    return pd.DataFrame(rows)


def build_scorecard() -> pd.DataFrame:
    rows = [
        {
            "dimension": "problem importance",
            "score_5": 5,
            "evidence": "单细胞扰动预测和基础模型快速发展，预测结果可靠性是明确痛点。",
        },
        {
            "dimension": "dataset breadth",
            "score_5": 4,
            "evidence": "七主数据集 + Tahoe chemical + E8b 外部聚合证据；任务类型覆盖较广。",
        },
        {
            "dimension": "baseline strength",
            "score_5": 3,
            "evidence": "已有 magnitude、LODO、oracle 等阶梯，但还需统一强基线审计和分层报告。",
        },
        {
            "dimension": "external validation",
            "score_5": 3,
            "evidence": "E8b 有外部关联证据；仍缺冻结任务级外部验证。",
        },
        {
            "dimension": "method novelty for Q1",
            "score_5": 4,
            "evidence": "risk auditing/triage 叙事清楚；若加入 selective/conformal 风险控制可更强。",
        },
        {
            "dimension": "method novelty for CCF-A",
            "score_5": 2,
            "evidence": "当前偏应用审计；需要理论化为可控风险选择性预测。",
        },
        {
            "dimension": "reproducibility",
            "score_5": 4,
            "evidence": "已有大量 frozen CSV、manifest 和 commit 记录；需要整理成 paper-level 一键入口。",
        },
    ]
    return pd.DataFrame(rows)


def plot_baseline(summary: pd.DataFrame) -> Path:
    plot_df = summary[~summary["score_name"].eq("random")].copy()
    plot_df = plot_df.sort_values("median_partial_rho_control_magnitude")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = [
        "#b45309" if "magnitude" in s and "oracle" not in s else "#047857"
        if "safeconf" in s or "protocol" in s
        else "#6b7280"
        for s in plot_df["score_name"]
    ]
    ax.barh(plot_df["score_label"], plot_df["median_partial_rho_control_magnitude"], color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Median partial rho after controlling predicted magnitude")
    ax.set_title("Baseline ladder: incremental information beyond magnitude")
    fig.tight_layout()
    path = FIGURES / "fig1_baseline_ladder_partial_rho.svg"
    save_svg_clean(fig, path)
    plt.close(fig)
    return path


def plot_incremental(e2: pd.DataFrame) -> Path:
    df = e2.sort_values("residual_partial_rho_point")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = df["residual_partial_rho_point"].to_numpy()
    lo = df["residual_partial_rho_ci_low"].to_numpy()
    hi = df["residual_partial_rho_ci_high"].to_numpy()
    ax.errorbar(
        x,
        y,
        xerr=np.vstack([x - lo, hi - x]),
        fmt="o",
        color="#047857",
        ecolor="#94a3b8",
        capsize=3,
    )
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["dataset_name"])
    ax.set_xlabel("Residual partial rho")
    ax.set_title("E2: SafeConf residual signal after magnitude control")
    fig.tight_layout()
    path = FIGURES / "fig2_e2_residual_partial_rho.svg"
    save_svg_clean(fig, path)
    plt.close(fig)
    return path


def plot_cost(cost: pd.DataFrame) -> Path:
    df = cost[cost["top_fraction"].eq(0.10)].copy()
    df = df.sort_values("enrichment_fold")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = [
        "#b45309" if s == "predicted_magnitude" else "#047857" if "safeconf" in s or "protocol" in s else "#9ca3af"
        for s in df["score_name"]
    ]
    ax.barh(df["score_label"], df["enrichment_fold"], color=colors)
    ax.axvline(1, color="#111827", linewidth=0.8)
    ax.set_xlabel("Top-10% high-error enrichment over random")
    ax.set_title("Cost-effectiveness: high-risk triage at 10% budget")
    fig.tight_layout()
    path = FIGURES / "fig3_cost_effectiveness_top10.svg"
    save_svg_clean(fig, path)
    plt.close(fig)
    return path


def plot_tahoe(tahoe: pd.DataFrame) -> Path:
    df = tahoe[tahoe["top_fraction"].eq(0.10)].copy()
    df = df.sort_values("enrichment")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = [
        "#b45309" if s == "predicted_magnitude" else "#047857" if "safeconf" in s else "#2563eb"
        for s in df["score_name"]
    ]
    ax.barh(df["score_label"], df["enrichment"], color=colors)
    ax.axvline(1, color="#111827", linewidth=0.8)
    ax.set_xlabel("Top-10% enrichment")
    ax.set_title("Tahoe chemical boundary: magnitude remains stronger")
    fig.tight_layout()
    path = FIGURES / "fig4_tahoe_chemical_boundary.svg"
    save_svg_clean(fig, path)
    plt.close(fig)
    return path


def plot_scorecard(scorecard: pd.DataFrame) -> Path:
    df = scorecard.copy()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.barh(df["dimension"], df["score_5"], color="#047857")
    ax.set_xlim(0, 5)
    ax.set_xlabel("Readiness score (0-5)")
    ax.set_title("Publication readiness scorecard")
    fig.tight_layout()
    path = FIGURES / "fig5_publication_readiness_scorecard.svg"
    save_svg_clean(fig, path)
    plt.close(fig)
    return path


def make_report(
    baseline: pd.DataFrame,
    formal: pd.DataFrame,
    e2: pd.DataFrame,
    cost: pd.DataFrame,
    tahoe: pd.DataFrame,
    actions: pd.DataFrame,
    venues: pd.DataFrame,
    scorecard: pd.DataFrame,
    figs: dict[str, Path],
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    median_safeconf = baseline.loc[
        baseline["score_name"].eq("protocol_v0_2_family_confidence"),
        "median_partial_rho_control_magnitude",
    ].iloc[0]
    median_mag = baseline.loc[
        baseline["score_name"].eq("predicted_magnitude"),
        "median_partial_rho_control_magnitude",
    ].iloc[0]
    e2_pos = int((e2["residual_partial_rho_ci_low"] > 0).sum())
    e2_total = len(e2)
    tahoe_top10 = tahoe[tahoe["top_fraction"].eq(0.10)].copy()
    safeconf_tahoe = tahoe_top10.loc[tahoe_top10["score_name"].eq("safeconf_full"), "enrichment"].iloc[0]
    mag_tahoe = tahoe_top10.loc[
        tahoe_top10["score_name"].eq("predicted_magnitude"), "enrichment"
    ].iloc[0]

    report = f"""# SafeConf 一区 / CCF-A 投稿升级审计报告

生成时间：{now}

## 1. 当前判断

SafeConf 现在适合按一区生信方法论文推进，暂不适合直接按 CCF-A 主会版本硬投。核心原因很清楚：现有证据能支持“预测后风险审计”和“实验复核优先级排序”，但还不足以支持“通用新预测模型”或“稳定击败所有强基线”。

当前最稳的论文主张：

> SafeConf is a task-level risk auditing framework for single-cell perturbation prediction. It identifies predictions likely to fail and supports selective verification under limited experimental budget.

中文表达：SafeConf 的职责是给已有扰动预测结果做风险审计，告诉研究者哪些预测更可能错、哪些结果值得优先复核。

## 2. 现有证据中最能打的部分

- 七主数据集：formal corrected 结果支持 SafeConf 风险信号在多数数据集中为正。
- E2 magnitude residual：{e2_pos}/{e2_total} 个数据集在控制 predicted magnitude 后仍为正，说明 SafeConf 并非只是在重复“扰动幅度越大越容易错”。
- 成本有效性：top-risk triage 能把有限复核预算集中到高错误任务上，适合讲“湿实验资源有限时如何决策”。
- Tahoe chemical：SafeConf 能筛高错误任务，但 magnitude 更强。这一结果应作为边界写入论文，而不是藏起来。
- E8b 外部证据：已有外部 benchmark 关联证据，但还需要任务级冻结外部验证来提高一区安全性。

## 3. 最危险的问题

1. Tahoe chemical 中 magnitude top-10 enrichment = {mag_tahoe:.2f}，SafeConf full = {safeconf_tahoe:.2f}。如果论文写成“SafeConf 全面更强”，会被审稿人直接打穿。
2. 当前外部验证偏聚合关联，缺少冻结任务级独立验证。
3. CCF-A 需要更强方法学形式。当前 SafeConf 是优秀的可靠性工程框架，但还需要 selective prediction / conformal risk control 才更像 AI 方法论文。

## 4. 投稿路线

优先路线：Bioinformatics / Briefings in Bioinformatics / PLOS Computational Biology。  
高风险冲刺：Genome Biology / Cell Systems，需要补生物案例。  
CCF-A 路线：AAAI / IJCAI / NeurIPS/ICML 风格，需要把 SafeConf 升级为可控风险选择性预测方法。

## 5. 下一轮实验优先级

{markdown_table(actions)}

## 6. 自动生成文件

- HTML 工作台：`Q1_PUBLICATION_WORKBENCH.html`
- 强基线阶梯：`tables/TABLE_Q1_BASELINE_LADDER_SUMMARY.csv`
- E2 增量价值：`tables/TABLE_INCREMENTAL_VALUE_E2.csv`
- Tahoe chemical 边界：`tables/TABLE_TAHOE_CHEMICAL_BOUNDARY.csv`
- 投稿路线：`tables/TABLE_TARGET_VENUE_STRATEGY.csv`
- 补实验清单：`tables/TABLE_Q1_GAP_AND_ACTIONS.csv`

## 7. 图

- ![]({rel(figs["baseline"])})
- ![]({rel(figs["incremental"])})
- ![]({rel(figs["cost"])})
- ![]({rel(figs["tahoe"])})
- ![]({rel(figs["scorecard"])})
"""
    (OUT / "Q1_READINESS_REPORT.md").write_text(report, encoding="utf-8")

    css = """
*{box-sizing:border-box}
body{margin:0;background:#f7f8f6;color:#17202a;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"Noto Sans SC",sans-serif;line-height:1.72;overflow-x:hidden}
.top{background:#12312b;color:white;padding:26px 42px}
.top h1{margin:0;font-size:30px}.top p{margin:8px 0 0;color:#d8eee8}
.wrap{max-width:1180px;margin:0 auto;padding:28px}
.card{background:white;border:1px solid #d8e0dc;border-radius:16px;padding:24px;margin:18px 0;box-shadow:0 8px 22px rgba(15,23,42,.06);overflow-x:auto}
h2{border-bottom:3px solid #0f766e;padding-bottom:8px;margin-top:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
.metric{border:1px solid #d7e5df;border-radius:14px;padding:18px;background:#fbfdfc}
.metric b{font-size:26px;color:#0f766e}.warn{border-left:6px solid #b45309;background:#fff8eb}.ok{border-left:6px solid #047857;background:#f1fbf6}
.figgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}.figgrid img{width:100%;max-width:100%;background:white;border:1px solid #d8e0dc;border-radius:12px}
.data-table{border-collapse:collapse;width:100%;font-size:14px}.data-table th,.data-table td{border:1px solid #dbe3df;padding:8px;vertical-align:top}.data-table th{background:#eef6f3}
code{background:#eef2f7;padding:2px 6px;border-radius:5px}
a{color:#0f766e}.small{color:#5f6b66;font-size:14px}
"""
    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SafeConf Q1 / CCF-A 投稿升级工作台</title>
<style>{css}</style>
</head>
<body>
<div class="top">
  <h1>SafeConf Q1 / CCF-A 投稿升级工作台</h1>
  <p>目标：把现有 SafeConf 从“实验结果很多”整理成“一区投稿可辩护的证据链”。</p>
</div>
<div class="wrap">
  <div class="card ok">
    <h2>当前结论</h2>
    <p><b>主线冲一区生信方法论文，CCF-A 作为方法升级线。</b> SafeConf 的强点是任务级风险审计、复核优先级和失败边界识别；弱点是 chemical 场景中 magnitude 更强，外部任务级验证还要补。</p>
  </div>

  <div class="grid">
    <div class="metric"><span>Formal datasets</span><br><b>{formal['dataset_name'].nunique()}</b><p>七主数据集 evidence chain。</p></div>
    <div class="metric"><span>E2 positive after magnitude control</span><br><b>{e2_pos}/{e2_total}</b><p>控制 predicted magnitude 后仍保留信息。</p></div>
    <div class="metric"><span>Frozen v0.2 median partial rho</span><br><b>{fmt_num(median_safeconf)}</b><p>来自 baseline ladder。</p></div>
    <div class="metric"><span>Magnitude median partial rho</span><br><b>{fmt_num(median_mag)}</b><p>强竞争基线，不能回避。</p></div>
  </div>

  <div class="card warn">
    <h2>论文定位红线</h2>
    <p>论文中避免把 SafeConf 写成新的扰动预测器，也避免宣称全面超过 magnitude。更稳的写法：SafeConf 对已有扰动预测做风险审计，在 gene/cross-context 场景有增量价值，在 chemical 场景需要和 magnitude 形成边界或组合。</p>
  </div>

  <div class="card">
    <h2>核心图</h2>
    <div class="figgrid">
      <img src="{rel(figs['baseline'])}" alt="baseline ladder">
      <img src="{rel(figs['incremental'])}" alt="E2 residual">
      <img src="{rel(figs['cost'])}" alt="cost effectiveness">
      <img src="{rel(figs['tahoe'])}" alt="Tahoe chemical boundary">
      <img src="{rel(figs['scorecard'])}" alt="readiness scorecard">
    </div>
  </div>

  <div class="card">
    <h2>强基线阶梯</h2>
    {compact_table_html(baseline, ["score_label","role","n_datasets","median_aligned_rho","median_partial_rho_control_magnitude","median_aurc_reduction_vs_random_pct","deployment_status"])}
    <p class="small"><a href="{rel(TABLES / 'TABLE_Q1_BASELINE_LADDER_SUMMARY.csv')}">打开 CSV</a></p>
  </div>

  <div class="card">
    <h2>E2：控制 magnitude 后的增量价值</h2>
    {compact_table_html(e2, ["dataset_name","n","residual_partial_rho_point","residual_partial_rho_ci_low","residual_partial_rho_ci_high","aurc_improvement_magnitude_minus_combined_point","incremental_claim"])}
    <p class="small"><a href="{rel(TABLES / 'TABLE_INCREMENTAL_VALUE_E2.csv')}">打开 CSV</a></p>
  </div>

  <div class="card">
    <h2>Tahoe chemical 边界</h2>
    {compact_table_html(tahoe[tahoe["top_fraction"].eq(0.10)], ["score_label","top_fraction","aligned_rho","precision","enrichment","magnitude_enrichment_same_top_fraction","delta_enrichment_vs_magnitude_same_top_fraction","interpretation"])}
    <p class="small"><a href="{rel(TABLES / 'TABLE_TAHOE_CHEMICAL_BOUNDARY.csv')}">打开 CSV</a></p>
  </div>

  <div class="card">
    <h2>下一轮必须补什么</h2>
    {compact_table_html(actions, ["priority","gap","action","deliverable","status"])}
    <p class="small"><a href="{rel(TABLES / 'TABLE_Q1_GAP_AND_ACTIONS.csv')}">打开 CSV</a></p>
  </div>

  <div class="card">
    <h2>投稿路线</h2>
    {compact_table_html(venues)}
    <p class="small"><a href="{rel(TABLES / 'TABLE_TARGET_VENUE_STRATEGY.csv')}">打开 CSV</a></p>
  </div>

  <div class="card">
    <h2>来源文件</h2>
    <ul>
      {''.join(f'<li><code>{html.escape(k)}</code>: {html.escape(rel_from_root(v))}</li>' for k, v in SRC.items() if v.exists())}
    </ul>
  </div>
</div>
</body>
</html>
"""
    (OUT / "Q1_PUBLICATION_WORKBENCH.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    style_matplotlib()

    baseline = build_baseline_summary()
    formal = build_formal_evidence()
    e2 = build_incremental_value()
    cost = build_cost_summary()
    tahoe = build_tahoe_boundary()
    actions = build_gap_actions()
    venues = build_venue_strategy()
    scorecard = build_scorecard()
    source_files = read_csv("source_files")

    save_table(baseline, "TABLE_Q1_BASELINE_LADDER_SUMMARY.csv")
    save_table(formal, "TABLE_FORMAL_DATASET_EVIDENCE.csv")
    save_table(e2, "TABLE_INCREMENTAL_VALUE_E2.csv")
    save_table(cost, "TABLE_COST_EFFECTIVENESS_MACRO.csv")
    save_table(tahoe, "TABLE_TAHOE_CHEMICAL_BOUNDARY.csv")
    save_table(actions, "TABLE_Q1_GAP_AND_ACTIONS.csv")
    save_table(venues, "TABLE_TARGET_VENUE_STRATEGY.csv")
    save_table(scorecard, "TABLE_Q1_READINESS_SCORECARD.csv")
    save_table(source_files, "TABLE_SOURCE_FILES_USED.csv")

    figs = {
        "baseline": plot_baseline(baseline),
        "incremental": plot_incremental(e2),
        "cost": plot_cost(cost),
        "tahoe": plot_tahoe(tahoe),
        "scorecard": plot_scorecard(scorecard),
    }
    make_report(baseline, formal, e2, cost, tahoe, actions, venues, scorecard, figs)

    readme = """# Q1 / CCF-A 投稿升级包

入口文件：

- `Q1_PUBLICATION_WORKBENCH.html`：浏览器打开，适合快速查看。
- `Q1_READINESS_REPORT.md`：文字报告，适合复制给老师或继续改成论文计划。
- `tables/`：所有自动生成的审计表。
- `figures/`：所有自动生成的 SVG 图。

生成命令：

```bash
python3 tools/scripts/build_q1_publication_package.py
```
"""
    (OUT / "README_先看这个.md").write_text(readme, encoding="utf-8")
    status = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": rel_from_root(Path(__file__).resolve()),
        "output_dir": rel_from_root(OUT),
        "input_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip(),
        "source_files": {key: rel_from_root(path) for key, path in SRC.items() if path.exists()},
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote Q1 package to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
