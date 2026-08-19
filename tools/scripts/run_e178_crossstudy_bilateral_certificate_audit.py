#!/usr/bin/env python3
"""E178: synthesize E176/E177 without tuning on their evaluation truth.

This is a descriptive, post-evaluation audit.  It quantifies four questions raised
in the advisor meeting: model-specific versus shared task difficulty, target-truth
independence at scoring time, hard split coverage, and the lower/upper certificate.
No model, score weight, calibration quantile, or decision threshold is changed.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import beta, fisher_exact, spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E178_crossstudy_bilateral_certificate_audit_20260722"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

E176 = ROOT / "docs/实验结果/E176_four_donor_fresh_confirmation_20260719"
E177 = ROOT / "docs/实验结果/E177_sunshine_external_certificate_20260719"
INPUTS = {
    "runner": Path(__file__).resolve(),
    "e176_tasks": E176 / "final_evaluation/tables/EVALUATION_TASK_METRICS.csv",
    "e176_coverage": E176 / "final_evaluation/tables/CONFORMAL_COVERAGE_EFFICIENCY.csv",
    "e176_final_status": E176 / "final_evaluation/RUN_STATUS.json",
    "e176_calibration_status": E176 / "calibration_release/RUN_STATUS.json",
    "e177_tasks": E177 / "final_evaluation/tables/EVALUATION_TASK_RESULTS.csv",
    "e177_clusters": E177 / "final_evaluation/tables/EVALUATION_TARGET_CLUSTER_COVERAGE.csv",
    "e177_summary": E177 / "final_evaluation/E177_FINAL_SUMMARY.json",
    "e177_calibration": E177 / "calibration_release/CALIBRATION_MODEL.json",
    "e177_access": E177 / "final_evaluation/ACCESS_ATTESTATION.json",
}

BLUE = "#3C5488"
TEAL = "#00A087"
RED = "#E64B35"
ORANGE = "#F39B7F"
GRAY = "#4D4D4D"
LIGHT = "#F4F6F8"
MID = "#B7C2CC"
SEED = 178
N_BOOT = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exact_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


def rho(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def cluster_bootstrap_rho(
    df: pd.DataFrame,
    cluster: str,
    x: str,
    y: str,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float]:
    groups = {key: part.index.to_numpy() for key, part in df.groupby(cluster, sort=True)}
    keys = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[key] for key in sampled])
        val = rho(df.loc[idx, x], df.loc[idx, y])
        if np.isfinite(val):
            draws.append(val)
    if not draws:
        return float("nan"), float("nan")
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def choose_font() -> None:
    candidates = ["Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((font for font in candidates if font in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": selected,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "svg.fonttype": "none",
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    svg_path = FIGURES / f"{name}.svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def load_tasks() -> tuple[pd.DataFrame, pd.DataFrame]:
    e176 = pd.read_csv(INPUTS["e176_tasks"])
    e176 = e176.loc[e176["evaluation_test_task"].astype(bool)].copy()
    e176["study"] = "E176 四供体"
    e176["cluster_id"] = e176["panel_id"].astype(str) + "::" + e176["perturbed_gene_id"].astype(str)
    e176["setting"] = np.where(
        e176["target_stratum"].eq("COLUMN_UNSEEN"), "供体与靶点双未见", "完整留出供体"
    )

    e177 = pd.read_csv(INPUTS["e177_tasks"])
    e177["study"] = "E177 独立研究"
    e177["cluster_id"] = e177["perturbation"].astype(str)
    e177["setting"] = "独立研究内未见靶点"
    return e176, e177


def model_specificity(e176: pd.DataFrame, e177: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    concordance: list[dict[str, object]] = []
    scores = ["model_disagreement_rmse", "predicted_magnitude", "safeconf_risk"]
    outcomes = ["scgpt_rmse", "gears_rmse", "ensemble_rmse", "pair_mean_rmse", "pair_max_rmse"]
    labels = {
        "model_disagreement_rmse": "模型分歧",
        "predicted_magnitude": "预测幅度",
        "safeconf_risk": "冻结风险分数",
        "scgpt_rmse": "scGPT误差",
        "gears_rmse": "GEARS误差",
        "ensemble_rmse": "集成误差",
        "pair_mean_rmse": "两模型平均误差",
        "pair_max_rmse": "两模型最大误差",
    }
    for study_i, df in enumerate([e176, e177]):
        study = str(df["study"].iloc[0])
        for score in scores:
            for outcome in outcomes:
                lo, hi = cluster_bootstrap_rho(df, "cluster_id", score, outcome, seed=SEED + study_i)
                rows.append(
                    {
                        "study": study,
                        "score": score,
                        "score_cn": labels[score],
                        "outcome": outcome,
                        "outcome_cn": labels[outcome],
                        "n_tasks": len(df),
                        "n_target_clusters": df["cluster_id"].nunique(),
                        "spearman_task": rho(df[score], df[outcome]),
                        "cluster_boot_ci95_low": lo,
                        "cluster_boot_ci95_high": hi,
                    }
                )

        target = (
            df.groupby("cluster_id", as_index=False)[["scgpt_rmse", "gears_rmse"]]
            .mean()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        sc_med = float(df["scgpt_rmse"].median())
        ge_med = float(df["gears_rmse"].median())
        sc_high = df["scgpt_rmse"] >= sc_med
        ge_high = df["gears_rmse"] >= ge_med
        sc_top = df["scgpt_rmse"] >= df["scgpt_rmse"].quantile(0.8)
        ge_top = df["gears_rmse"] >= df["gears_rmse"].quantile(0.8)
        union = int((sc_top | ge_top).sum())
        concordance.append(
            {
                "study": study,
                "n_tasks": len(df),
                "n_target_clusters": df["cluster_id"].nunique(),
                "task_error_spearman": rho(df["scgpt_rmse"], df["gears_rmse"]),
                "target_mean_error_spearman": rho(target["scgpt_rmse"], target["gears_rmse"]),
                "both_above_median_fraction": float((sc_high & ge_high).mean()),
                "both_below_median_fraction": float((~sc_high & ~ge_high).mean()),
                "discordant_median_fraction": float((sc_high ^ ge_high).mean()),
                "top20_overlap_fraction_of_tasks": float((sc_top & ge_top).mean()),
                "top20_jaccard": float((sc_top & ge_top).sum() / union) if union else float("nan"),
                "scgpt_lower_error_fraction": float((df["scgpt_rmse"] < df["gears_rmse"]).mean()),
                "gears_lower_error_fraction": float((df["gears_rmse"] < df["scgpt_rmse"]).mean()),
                "ties_fraction": float((df["gears_rmse"] == df["scgpt_rmse"]).mean()),
                "median_abs_model_error_gap": float((df["scgpt_rmse"] - df["gears_rmse"]).abs().median()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(concordance)


def certificate_table(e176: pd.DataFrame, e177: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for df in [e176, e177]:
        lower = df["pair_lower_bound_rmse"].to_numpy(float)
        mean = df["pair_mean_rmse"].to_numpy(float)
        maximum = df["pair_max_rmse"].to_numpy(float)
        ratio = np.divide(lower, mean, out=np.full_like(lower, np.nan), where=mean > 0)
        rows.append(
            {
                "study": str(df["study"].iloc[0]),
                "n_tasks": len(df),
                "pair_mean_violations": int(np.sum(lower > mean + 1e-10)),
                "pair_max_violations": int(np.sum(lower > maximum + 1e-10)),
                "median_pair_lower": float(np.nanmedian(lower)),
                "median_pair_mean_error": float(np.nanmedian(mean)),
                "median_lower_tightness": float(np.nanmedian(ratio)),
                "q25_lower_tightness": float(np.nanquantile(ratio, 0.25)),
                "q75_lower_tightness": float(np.nanquantile(ratio, 0.75)),
                "max_lower_tightness": float(np.nanmax(ratio)),
            }
        )
    return pd.DataFrame(rows)


def coverage_table(e176: pd.DataFrame, e177: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    cov176 = pd.read_csv(INPUTS["e176_coverage"])
    cov176 = cov176[
        (cov176["outcome"] == "ensemble_rmse")
        & (cov176["model_spec"] == "magnitude")
        & cov176["population"].isin(["ALL_640", "H01", "H02", "H03", "H04", "COLUMN_UNSEEN_128"])
    ].copy()
    label_map = {
        "ALL_640": "E176 总体",
        "H01": "E176 供体H01",
        "H02": "E176 供体H02",
        "H03": "E176 供体H03",
        "H04": "E176 供体H04",
        "COLUMN_UNSEEN_128": "E176 双未见靶点",
    }
    rows: list[dict[str, object]] = []
    for _, row in cov176.iterrows():
        rows.append(
            {
                "population": label_map[row["population"]],
                "study": "E176 四供体",
                "n_clusters": int(row["n_targets"]),
                "covered_clusters": int(row["targets_all_states_covered"]),
                "cluster_coverage": float(row["target_simultaneous_coverage"]),
                "ci95_low": float(row["exact_binomial_ci95_lower"]),
                "ci95_high": float(row["exact_binomial_ci95_upper"]),
                "task_coverage": float(row["task_marginal_coverage"]),
                "mean_upper": float(row["mean_upper"]),
                "mean_width_above_lower": float(row["mean_interval_width_above_lower"]),
                "calibration_scope": "供体专属40靶点",
                "confirmatory_status": "预冻结总体通过" if row["population"] == "ALL_640" else "预冻结分层描述",
            }
        )

    clusters177 = pd.read_csv(INPUTS["e177_clusters"])
    k177 = int(clusters177["ensemble_cluster_covered"].sum())
    n177 = len(clusters177)
    lo177, hi177 = exact_ci(k177, n177)
    rows.append(
        {
            "population": "E177 独立研究",
            "study": "E177 独立研究",
            "n_clusters": n177,
            "covered_clusters": k177,
            "cluster_coverage": k177 / n177,
            "ci95_low": lo177,
            "ci95_high": hi177,
            "task_coverage": float(e177["ensemble_covered"].mean()),
            "mean_upper": float(e177["ensemble_upper_bound"].mean()),
            "mean_width_above_lower": float(e177["ensemble_upper_bound"].mean()),
            "calibration_scope": "独立研究内30靶点",
            "confirmatory_status": "预冻结边界结果",
        }
    )

    all176 = next(row for row in rows if row["population"] == "E176 总体")
    pooled_k = int(all176["covered_clusters"]) + k177
    pooled_n = int(all176["n_clusters"]) + n177
    pooled_lo, pooled_hi = exact_ci(pooled_k, pooled_n)
    rows.append(
        {
            "population": "两研究描述性合计",
            "study": "E176+E177",
            "n_clusters": pooled_n,
            "covered_clusters": pooled_k,
            "cluster_coverage": pooled_k / pooled_n,
            "ci95_low": pooled_lo,
            "ci95_high": pooled_hi,
            "task_coverage": float("nan"),
            "mean_upper": float("nan"),
            "mean_width_above_lower": float("nan"),
            "calibration_scope": "各研究分别校准",
            "confirmatory_status": "仅描述，不形成新的覆盖保证",
        }
    )

    table = [[int(all176["covered_clusters"]), int(all176["n_clusters"]) - int(all176["covered_clusters"])], [k177, n177 - k177]]
    _, p_value = fisher_exact(table, alternative="two-sided")
    extras = {
        "pooled_coverage": pooled_k / pooled_n,
        "pooled_ci_low": pooled_lo,
        "pooled_ci_high": pooled_hi,
        "fisher_exact_p_e176_vs_e177": float(p_value),
        "coverage_difference_e176_minus_e177": float(all176["cluster_coverage"]) - k177 / n177,
    }
    return pd.DataFrame(rows), extras


def efficiency_table(e176: pd.DataFrame, e177: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for df in [e176, e177]:
        study = str(df["study"].iloc[0])
        for outcome, upper in [
            ("ensemble_rmse", "ensemble_upper_bound"),
            ("pair_mean_rmse", "pair_mean_upper_bound"),
        ]:
            if upper not in df.columns:
                spec = "magnitude"
                upper = f"upper_{outcome}__{spec}"
            lower = np.zeros(len(df)) if outcome == "ensemble_rmse" else df["pair_lower_bound_rmse"].to_numpy(float)
            widths = df[upper].to_numpy(float) - lower
            errors = df[outcome].to_numpy(float)
            rows.append(
                {
                    "study": study,
                    "outcome": outcome,
                    "n_tasks": len(df),
                    "median_error": float(np.median(errors)),
                    "mean_upper": float(np.mean(df[upper])),
                    "median_upper": float(np.median(df[upper])),
                    "mean_width_above_lower": float(np.mean(widths)),
                    "median_width_above_lower": float(np.median(widths)),
                    "mean_width_over_median_error": float(np.mean(widths) / np.median(errors)),
                }
            )
    return pd.DataFrame(rows)


def write_static_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    provenance = pd.DataFrame(
        [
            ["细胞背景相似度", "目标背景的未扰动control表达 + 训练背景control表达", "否", "需有目标背景control；E177不把技术组冒充生物背景"],
            ["扰动历史支持量", "训练集标签计数", "否", "全新扰动时允许为0；不回填评价结果"],
            ["模型分歧", "scGPT与GEARS的预测向量", "否", "同时产生确定性下界；不能判断哪个模型错"],
            ["预测幅度", "预测效应向量相对零向量的RMSE", "否", "老师追问的第四项；不是实测效应幅度"],
            ["conformal余量", "独立校准靶点的历史误差", "否（对评价靶点）", "允许用校准真值；冻结后不能再看评价真值调分位数"],
            ["评价误差", "目标扰动后的实测表达", "是", "只在最终评价阶段计算，不进入部署时打分"],
        ],
        columns=["quantity", "runtime_input", "uses_target_evaluation_truth", "boundary"],
    )
    difficulty = pd.DataFrame(
        [
            ["随机缺失pair", "同一矩阵随机留点", "E98/E100/E103", "已测", "用于基础可计算性；部分为embedding/transfer预测器"],
            ["训练子矩阵", "只给25%/50%/75%矩阵", "E98/E100/E103", "已测", "展示数据量敏感性；不等于所有端到端模型重训"],
            ["整行留出", "完整新背景/新供体", "E176", "正式scGPT–GEARS已测", "同一Primary CD4研究内四供体"],
            ["整列留出", "完整新扰动", "E90/E176/E177", "正式双模型已测", "E176含128个零训练支持靶点"],
            ["双未见", "新背景与新扰动同时出现", "E176", "正式双模型已测", "128靶点、384任务；分层覆盖88.28%"],
            ["跨数据集迁移", "源数据开发，目标数据直接评价", "E69/E87/E89/E101", "已测但结果混合", "不能写普遍跨数据集增益"],
            ["独立研究复现", "在新研究重新冻结、校准、评价", "E177", "已完成", "技术组不是供体；覆盖点估计88%"],
            ["不同扰动模态", "基因、细胞因子、化学扰动", "E84/E87/E89/E102/E103/E176/E177", "覆盖三类", "chemical中magnitude更强，不能合并生物学结论"],
        ],
        columns=["difficulty_level", "definition", "evidence", "status", "remaining_boundary"],
    )
    return provenance, difficulty


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.4))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, detail: str, edge: str, fill: str = "white") -> None:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", ec=edge, fc=fill, lw=1.6)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", fontsize=11, weight="bold", color=GRAY)
        ax.text(x + w / 2, y + h * 0.31, detail, ha="center", va="center", fontsize=8.6, color=GRAY, linespacing=1.35)

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str = MID) -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.4, color=color))

    box(0.25, 2.0, 2.05, 1.45, "新任务 x", "目标背景control\n扰动标签", BLUE, LIGHT)
    box(2.85, 3.25, 2.0, 1.25, "scGPT", "输出预测向量 p₁", BLUE)
    box(2.85, 0.95, 2.0, 1.25, "GEARS", "输出预测向量 p₂", TEAL)
    arrow(2.3, 2.75, 2.85, 3.75, BLUE)
    arrow(2.3, 2.65, 2.85, 1.55, TEAL)
    box(5.45, 3.05, 2.15, 1.55, "部署时可计算", "预测幅度 ‖(p₁+p₂)/2‖\n模型分歧 d(p₁,p₂)", TEAL, "#F2FBF8")
    arrow(4.85, 3.85, 5.45, 3.85, BLUE)
    arrow(4.85, 1.55, 5.45, 3.45, TEAL)
    box(5.45, 0.55, 2.15, 1.45, "历史校准", "只读取校准靶点真值\n冻结分位数 q", ORANGE, "#FFF8F5")
    box(8.2, 2.05, 2.15, 1.45, "双边证书", "下界 d(p₁,p₂)/2\n上界 base(x)+q", RED, "#FFF6F4")
    arrow(7.6, 3.7, 8.2, 3.0, TEAL)
    arrow(7.6, 1.3, 8.2, 2.45, ORANGE)
    box(10.95, 2.05, 2.0, 1.45, "输出", "至少有多危险？\n大概率不超过多少？", BLUE, LIGHT)
    arrow(10.35, 2.78, 10.95, 2.78, BLUE)
    ax.text(6.5, 5.08, "目标评价真值不进入以上路径", ha="center", color=RED, fontsize=11, weight="bold")
    ax.plot([4.8, 8.2], [4.86, 4.86], color=RED, lw=1.4, ls="--")
    ax.text(6.5, 0.12, "评价真值只在全部规则冻结后用于核验覆盖率、误差和违例数", ha="center", color=GRAY, fontsize=9.5)
    ax.set_title("E178｜SafeConf 当前可复核的信息流", loc="left", fontsize=15, weight="bold", color=GRAY, pad=12)
    savefig(fig, "F1_deployment_information_flow")


def fig_error_concordance(e176: pd.DataFrame, e177: pd.DataFrame, concordance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.1))
    for ax, df in zip(axes, [e176, e177]):
        study = str(df["study"].iloc[0])
        metric = concordance.loc[concordance["study"] == study].iloc[0]
        ax.scatter(df["scgpt_rmse"], df["gears_rmse"], s=13, alpha=0.42, color=BLUE, edgecolors="none")
        lo = min(df["scgpt_rmse"].min(), df["gears_rmse"].min())
        hi = max(df["scgpt_rmse"].max(), df["gears_rmse"].max())
        ax.plot([lo, hi], [lo, hi], color=MID, lw=1, ls="--")
        ax.set_xlabel("scGPT RMSE")
        ax.set_ylabel("GEARS RMSE")
        ax.set_title(study, weight="bold", color=GRAY)
        ax.text(
            0.04,
            0.95,
            f"任务级 Spearman = {metric['task_error_spearman']:.3f}\n高误差Top20% Jaccard = {metric['top20_jaccard']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=MID, alpha=0.92),
        )
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("两个模型会不会在同一任务上一起出错？", x=0.06, ha="left", fontsize=15, weight="bold", color=GRAY)
    fig.text(0.06, 0.01, "点越靠近对角线，越像共享任务难度；远离对角线的点更偏模型特异性。", fontsize=9.5, color=GRAY)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    savefig(fig, "F2_model_error_concordance")


def fig_score_heatmap(spec: pd.DataFrame) -> None:
    scores = ["模型分歧", "预测幅度", "冻结风险分数"]
    outcomes = ["scGPT误差", "GEARS误差", "集成误差", "两模型平均误差", "两模型最大误差"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.5), constrained_layout=True)
    for ax, study in zip(axes, ["E176 四供体", "E177 独立研究"]):
        sub = spec[spec["study"] == study].pivot(index="score_cn", columns="outcome_cn", values="spearman_task").reindex(index=scores, columns=outcomes)
        im = ax.imshow(sub.to_numpy(float), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        ax.set_xticks(range(len(outcomes)), outcomes, rotation=28, ha="right")
        ax.set_yticks(range(len(scores)), scores)
        for i in range(len(scores)):
            for j in range(len(outcomes)):
                val = sub.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.28 else GRAY, fontsize=9, weight="bold")
        ax.set_title(study, weight="bold", color=GRAY)
        ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("Spearman ρ（越大越能排序高误差）")
    fig.suptitle("一个总分不能自动变成每个模型的置信度", x=0.05, ha="left", fontsize=15, weight="bold", color=GRAY)
    savefig(fig, "F3_score_model_specificity_heatmap")


def fig_certificate(cert: pd.DataFrame, efficiency: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    x = np.arange(len(cert))
    axes[0].bar(x, cert["median_lower_tightness"] * 100, color=[BLUE, TEAL], width=0.55)
    axes[0].vlines(x, cert["q25_lower_tightness"] * 100, cert["q75_lower_tightness"] * 100, color=GRAY, lw=3)
    axes[0].set_xticks(x, cert["study"])
    axes[0].set_ylabel("下界 / 两模型平均误差（%）")
    axes[0].set_title("确定性下界：正确，但偏松", weight="bold", color=GRAY)
    for i, row in cert.iterrows():
        axes[0].text(i, row["median_lower_tightness"] * 100 + 1.2, f"{row['median_lower_tightness']*100:.1f}%\n0违例", ha="center", fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    pair = efficiency[efficiency["outcome"] == "pair_mean_rmse"].reset_index(drop=True)
    axes[1].bar(x, pair["mean_width_over_median_error"], color=[BLUE, TEAL], width=0.55)
    axes[1].axhline(1, color=MID, lw=1, ls="--")
    axes[1].set_xticks(x, pair["study"])
    axes[1].set_ylabel("平均区间宽度 / 中位真实误差")
    axes[1].set_title("校准上界：覆盖较高，区间仍宽", weight="bold", color=GRAY)
    for i, row in pair.iterrows():
        axes[1].text(i, row["mean_width_over_median_error"] + 0.06, f"{row['mean_width_over_median_error']:.2f}×", ha="center", fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("双边证书的有效性与实用性必须分开报告", x=0.06, ha="left", fontsize=15, weight="bold", color=GRAY)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "F4_bilateral_certificate_efficiency")


def fig_coverage(coverage: pd.DataFrame) -> None:
    order = ["E176 供体H01", "E176 供体H02", "E176 供体H03", "E176 供体H04", "E176 双未见靶点", "E176 总体", "E177 独立研究", "两研究描述性合计"]
    df = coverage.set_index("population").loc[order].reset_index()
    y = np.arange(len(df))[::-1]
    colors = [BLUE] * 6 + [TEAL, GRAY]
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    for yi, (_, row), color in zip(y, df.iterrows(), colors):
        ax.plot([row["ci95_low"], row["ci95_high"]], [yi, yi], color=color, lw=2)
        ax.scatter(row["cluster_coverage"], yi, s=55, color=color, zorder=3)
        ax.text(1.015, yi, f"{int(row['covered_clusters'])}/{int(row['n_clusters'])}", va="center", fontsize=9, color=GRAY)
    ax.axvline(0.9, color=RED, ls="--", lw=1.4)
    ax.set_yticks(y, df["population"])
    ax.set_xlim(0.72, 1.075)
    ax.set_xlabel("靶点簇同时覆盖率（精确二项95% CI）")
    ax.set_title("校准上界在内部达到目标，外部结果位于边界", loc="left", fontsize=15, weight="bold", color=GRAY)
    ax.text(0.755, y.max() + 0.12, "红色虚线：预设 90%", color=RED, fontsize=9, ha="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.29, 0.015, "合计行仅描述两项研究的总数，不构成新的 conformal 覆盖保证。", fontsize=9, color=GRAY)
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    savefig(fig, "F5_cluster_coverage_forest")


def fig_difficulty_ladder() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.7))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 5.7)
    ax.axis("off")
    levels = [
        (0.4, 0.65, 2.2, 1.1, "1  随机缺失pair", "E98/E100/E103\n已测，基础难度", BLUE),
        (2.85, 1.35, 2.2, 1.1, "2  整行或整列", "E90/E176/E177\n正式双模型已测", BLUE),
        (5.3, 2.05, 2.2, 1.1, "3  背景+扰动双未见", "E176：128靶点\n384任务", TEAL),
        (7.75, 2.75, 2.2, 1.1, "4  跨数据集迁移", "E69/E87/E89/E101\n结果混合", ORANGE),
        (10.2, 3.45, 2.2, 1.1, "5  独立研究复现", "E177：完整冻结流程\n覆盖点估计88%", RED),
    ]
    for x, y, w, h, title, detail, color in levels:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", fc="white", ec=color, lw=1.8))
        ax.text(x + 0.15, y + 0.76, title, ha="left", va="center", fontsize=10.5, weight="bold", color=GRAY)
        ax.text(x + 0.15, y + 0.32, detail, ha="left", va="center", fontsize=8.7, color=GRAY)
        if x < 10:
            ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (x + 2.45, y + 0.7 + h / 2), arrowstyle="-|>", mutation_scale=13, color=MID, lw=1.5))
    ax.text(6.4, 5.25, "周老师要求的任务难度阶梯", ha="center", fontsize=15, weight="bold", color=GRAY)
    ax.text(6.4, 0.18, "难度越高，越接近真正外推；结果也越弱。跨数据集与外部覆盖必须保留负面和边界结果。", ha="center", fontsize=9.5, color=GRAY)
    savefig(fig, "F6_advisor_difficulty_ladder")


def build_report(
    specificity: pd.DataFrame,
    concordance: pd.DataFrame,
    certificate: pd.DataFrame,
    coverage: pd.DataFrame,
    coverage_extra: dict[str, float],
    efficiency: pd.DataFrame,
) -> str:
    c176 = concordance.set_index("study").loc["E176 四供体"]
    c177 = concordance.set_index("study").loc["E177 独立研究"]
    cv176 = coverage.set_index("population").loc["E176 总体"]
    cv177 = coverage.set_index("population").loc["E177 独立研究"]
    cert176 = certificate.set_index("study").loc["E176 四供体"]
    cert177 = certificate.set_index("study").loc["E177 独立研究"]
    dis176 = specificity[(specificity.study == "E176 四供体") & (specificity.score == "model_disagreement_rmse")].set_index("outcome")
    dis177 = specificity[(specificity.study == "E177 独立研究") & (specificity.score == "model_disagreement_rmse")].set_index("outcome")
    pair_eff = efficiency[efficiency.outcome == "pair_mean_rmse"].set_index("study")
    return f"""# E178｜跨研究双边证书与模型特异性审计

完成时间：2026-07-22
性质：**冻结结果的描述性综合，不是新的确认集，也没有根据评价真值修改模型、权重、分位数或阈值。**

## 结论

E176 与 E177 合计包含 690 个评价靶点簇、2,320 个任务。两个模型的距离除以 2 在两项研究中对 pair-mean 和 pair-max RMSE 均为零违例。E176 的靶点簇覆盖为 {int(cv176.covered_clusters)}/{int(cv176.n_clusters)}={cv176.cluster_coverage:.2%}，E177 为 {int(cv177.covered_clusters)}/{int(cv177.n_clusters)}={cv177.cluster_coverage:.2%}。两项研究各自校准后的描述性合计是 {coverage_extra['pooled_coverage']:.2%}，但这不是新的 conformal 保证。

老师问“这个分数针对某个模型，还是说明任务总体难”。现在可以给出定量回答：scGPT 与 GEARS 的误差相关在 E176 为 ρ={c176.task_error_spearman:.3f}，E177 为 ρ={c177.task_error_spearman:.3f}；高误差 Top-20% 集合的 Jaccard 分别为 {c176.top20_jaccard:.3f} 和 {c177.top20_jaccard:.3f}。两模型确有共享难度，但并非总在同一任务上失败。模型分歧与 scGPT/GEARS 误差的相关在 E176 分别为 {dis176.loc['scgpt_rmse','spearman_task']:.3f}/{dis176.loc['gears_rmse','spearman_task']:.3f}，E177 分别为 {dis177.loc['scgpt_rmse','spearman_task']:.3f}/{dis177.loc['gears_rmse','spearman_task']:.3f}。因此当前 SafeConf 应称为**模型对层面的任务风险证书**，不能称为某个单模型的置信度。

## 老师的问题逐项回答

### 1. 最后和什么做相关？真实错误来自哪个模型？

真实错误是预测效应向量与冻结评价真值之间的 RMSE。E176/E177 分别报告 scGPT RMSE、GEARS RMSE、两模型集成 RMSE、两模型平均 RMSE和最大 RMSE。排序分数对每一种误差分别计算，不能只写一个含糊的“预测误差”。主证书关注 pair mean 与 pair max；单模型结果作为模型特异性诊断保留。

![两模型误差一致性](../figures/F2_model_error_concordance.png)

### 2. 模型分歧说明任务难，还是说明某个模型错？

分歧 `d(p1,p2)` 是对称量。它能证明 `d(p1,p2)/2 ≤ pair mean RMSE` 且 `d(p1,p2)/2 ≤ pair max RMSE`，所以分歧大时至少有一个模型不能很准；它不能指出是哪一个模型错。两项研究的下界违例均为 0，但中位紧度只有 {cert176.median_lower_tightness:.2%} 和 {cert177.median_lower_tightness:.2%}。小分歧仍可能是两个模型一起犯错。

![分数与各模型误差](../figures/F3_score_model_specificity_heatmap.png)

### 3. 预测幅度是不是要先做完目标实验才能计算？

不需要。当前 `predicted_magnitude` 是预测效应向量相对零向量的 RMSE，只读取 scGPT/GEARS 的预测输出。目标扰动后的实测表达仅用于最后计算真实误差。E176 的 1,920 个评价任务全部标记为 `QUERY_ONLY`；E177 在 pretruth 阶段保留 640 个无 y 查询任务，并在校准分位数冻结后才读取 400 个评价任务真值。

![部署信息流](../figures/F1_deployment_information_flow.png)

### 4. 随机留点、整行整列、双未见和跨数据集是否都做了？

已经形成从随机缺失 pair 到独立研究复现的难度阶梯。E176 首次用正式 scGPT–GEARS 在完整留出供体上完成 640 个靶点评价，其中 128 个靶点在训练供体中也没有支持，构成 384 个双未见任务；该分层的靶点簇覆盖为 113/128=88.28%。跨数据集直接迁移已有 E69/E87/E89/E101，结果并不普遍为正。E177 是独立研究内重新冻结、重新校准、最终评价，不应冒充“把 E176 的校准器直接搬过去”。

![难度阶梯](../figures/F6_advisor_difficulty_ladder.png)

## 双边证书结果

![双边证书效率](../figures/F4_bilateral_certificate_efficiency.png)

| 研究 | 评价任务 | 下界违例 | 下界中位紧度 | pair-mean平均区间宽度/中位误差 |
|---|---:|---:|---:|---:|
| E176 四供体 | {len(pd.read_csv(INPUTS['e176_tasks'])):,} | 0 | {cert176.median_lower_tightness:.2%} | {pair_eff.loc['E176 四供体','mean_width_over_median_error']:.2f}× |
| E177 独立研究 | {len(pd.read_csv(INPUTS['e177_tasks'])):,} | 0 | {cert177.median_lower_tightness:.2%} | {pair_eff.loc['E177 独立研究','mean_width_over_median_error']:.2f}× |

下界的逻辑保证很强，紧度较弱；上界的覆盖率较高，宽度仍大。二者共同限定风险范围，但当前还不足以授权真实实验或临床决策。

## 覆盖率

![覆盖率](../figures/F5_cluster_coverage_forest.png)

E176 总体点估计达到预设 90%；E177 点估计低 2 个百分点，精确二项 95% CI 仍包含 90%。两研究覆盖率差为 {coverage_extra['coverage_difference_e176_minus_e177']:.2%}，Fisher 精确检验 p={coverage_extra['fisher_exact_p_e176_vs_e177']:.3f}。这只能说明目前没有足够证据认定两项研究的覆盖率不同，不能把“不显著”解释成二者相同，也不能把 E177 写成强阳性。

## 当前论文主张

可以保留：

- 正式 scGPT–GEARS 预测记录、五随机种子和分阶段真值访问审计；
- 模型对距离给出的确定性 pair-mean/pair-max 误差下界；
- 同一研究四供体上的供体专属 conformal 覆盖，以及独立研究的边界复现；
- 随机留点、整行、整列、双未见、跨数据集和多扰动模态的结果矩阵；
- 当排序增量门失败时自动停止“优于 magnitude”的主张。

不能保留：

- fixed SafeConf 稳定超过 predicted magnitude；
- 模型分歧是某一个模型的置信度；
- 小分歧代表预测安全；
- E177 的技术组等同于供体、患者或生物学背景；
- 当前上下界已经足够紧，可直接替代真实实验。

## 投稿判断

E178 把周老师的问题、E176 内部确认和 E177 外部边界结果统一到了同一个可复核框架中。它提升的是论证完整性，不会凭空提高方法新颖性。当前稿件最可信的题目方向是“单细胞扰动预测的模型对双边误差证书与失败关闭审计”，经验排序降为次要诊断。较强期刊仍会追问：上界能否更窄、与现有 uncertainty/conformal 方法相比是否有增量、能否在新的真正生物背景复现。录用概率不能通过继续堆同类靶点变成百分之百。

## 文件

- [模型特异性相关](../tables/E178_MODEL_SPECIFICITY.csv)
- [共享难度诊断](../tables/E178_SHARED_DIFFICULTY.csv)
- [证书审计](../tables/E178_CERTIFICATE_SUMMARY.csv)
- [覆盖率汇总](../tables/E178_COVERAGE_SUMMARY.csv)
- [区间效率](../tables/E178_INTERVAL_EFFICIENCY.csv)
- [部署输入来源](../tables/E178_INPUT_PROVENANCE.csv)
- [任务难度与证据](../tables/E178_DIFFICULTY_EVIDENCE.csv)
- [输入哈希](../tables/INPUT_HASHES.csv)
"""


def main() -> None:
    for path in INPUTS.values():
        if not path.exists():
            raise FileNotFoundError(path)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    choose_font()

    e176, e177 = load_tasks()
    specificity, concordance = model_specificity(e176, e177)
    certificate = certificate_table(e176, e177)
    coverage, coverage_extra = coverage_table(e176, e177)
    efficiency = efficiency_table(e176, e177)
    provenance, difficulty = write_static_tables()

    specificity.to_csv(TABLES / "E178_MODEL_SPECIFICITY.csv", index=False)
    concordance.to_csv(TABLES / "E178_SHARED_DIFFICULTY.csv", index=False)
    certificate.to_csv(TABLES / "E178_CERTIFICATE_SUMMARY.csv", index=False)
    coverage.to_csv(TABLES / "E178_COVERAGE_SUMMARY.csv", index=False)
    efficiency.to_csv(TABLES / "E178_INTERVAL_EFFICIENCY.csv", index=False)
    provenance.to_csv(TABLES / "E178_INPUT_PROVENANCE.csv", index=False)
    difficulty.to_csv(TABLES / "E178_DIFFICULTY_EVIDENCE.csv", index=False)
    pd.DataFrame(
        [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in INPUTS.values()]
    ).to_csv(TABLES / "INPUT_HASHES.csv", index=False)

    fig_architecture()
    fig_error_concordance(e176, e177, concordance)
    fig_score_heatmap(specificity)
    fig_certificate(certificate, efficiency)
    fig_coverage(coverage)
    fig_difficulty_ladder()

    report = build_report(specificity, concordance, certificate, coverage, coverage_extra, efficiency)
    (REPORTS / "E178_REPORT.md").write_text(report, encoding="utf-8")
    readme = """# E178｜跨研究双边证书与模型特异性审计

先读：[中文完整报告](./reports/E178_REPORT.md)

本实验只综合已经冻结并完成的 E176/E177 结果，不重新拟合模型、分数、校准分位数或阈值。用途是逐项回答周老师关于真实误差来源、模型特异性、目标真值依赖和任务难度设置的问题。
"""
    (OUT / "README_先看这个.md").write_text(readme, encoding="utf-8")

    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    status = {
        "schema": "safeconf_e178_crossstudy_bilateral_audit_v1",
        "experiment": "E178_crossstudy_bilateral_certificate_audit",
        "status": "COMPLETE",
        "analysis_type": "post_evaluation_descriptive_synthesis",
        "method_or_quantile_tuned_on_evaluation_truth": False,
        "n_studies": 2,
        "n_evaluation_target_clusters": int(e176.cluster_id.nunique() + e177.cluster_id.nunique()),
        "n_evaluation_tasks": int(len(e176) + len(e177)),
        "pair_mean_bound_violations": int(certificate.pair_mean_violations.sum()),
        "pair_max_bound_violations": int(certificate.pair_max_violations.sum()),
        "e176_cluster_coverage": float(coverage.set_index("population").loc["E176 总体", "cluster_coverage"]),
        "e177_cluster_coverage": float(coverage.set_index("population").loc["E177 独立研究", "cluster_coverage"]),
        "pooled_coverage_descriptive_only": coverage_extra["pooled_coverage"],
        "legacy_ranking_superiority_claim_supported": False,
        "deployment_authorized": False,
        "git_head_at_run": git_head,
        "python": sys.version,
        "platform": platform.platform(),
        "bootstrap_seed": SEED,
        "bootstrap_replicates": N_BOOT,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    artifacts = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "MANIFEST.sha256")
    manifest = "\n".join(f"{sha256(path)}  {path.relative_to(OUT)}" for path in artifacts) + "\n"
    (OUT / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
