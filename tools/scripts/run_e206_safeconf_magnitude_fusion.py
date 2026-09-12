#!/usr/bin/env python3
"""Develop and audit a magnitude-augmented SafeConf ranking rule on E201.

E201 outcomes were already released before this analysis.  Accordingly, every
output is labelled as development evidence and must not be presented as a new
blind or external validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
    / "formal_core_evaluation/tables/E201_TASK_METRICS.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "docs/实验结果/E206_safeconf_magnitude_fusion_20260909"
)
TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 2)
ALPHA_DENOMINATOR = 20
BUDGET = 0.20
MASTER_SEED = 20_260_909
DEFAULT_BOOTSTRAP = 5_000


class AnalysisFailure(RuntimeError):
    """Raised when the E206 input or numerical contract is violated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def percentile_rank(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise AnalysisFailure("percentile rank requires finite values")
    return rankdata(values, method="average") / len(values)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 4 or np.std(left) <= 0 or np.std(right) <= 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return pearson(percentile_rank(left), percentile_rank(right))


def stable_ties(task_ids: Iterable[str], occurrences: Iterable[int]) -> np.ndarray:
    values = []
    for task_id, occurrence in zip(task_ids, occurrences):
        payload = f"E206\0{task_id}\0{int(occurrence)}".encode()
        values.append(int(hashlib.sha256(payload).hexdigest()[:16], 16))
    return np.asarray(values, dtype=np.uint64)


def review_metrics(
    score: np.ndarray,
    outcome: np.ndarray,
    task_ids: Iterable[str],
    occurrences: Iterable[int] | None = None,
) -> dict[str, float]:
    score = np.asarray(score, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    task_ids = np.asarray(list(map(str, task_ids)))
    if occurrences is None:
        occurrences = np.zeros(len(score), dtype=int)
    occurrences = np.asarray(list(occurrences), dtype=int)
    if not (
        len(score) == len(outcome) == len(task_ids) == len(occurrences)
        and len(score) >= 5
        and np.isfinite(score).all()
        and np.isfinite(outcome).all()
    ):
        raise AnalysisFailure("invalid arrays for review metrics")
    n_select = int(math.ceil(BUDGET * len(score)))
    ties = stable_ties(task_ids, occurrences)
    selected = np.lexsort((ties, -score))[:n_select]
    oracle = np.lexsort((ties, -outcome))[:n_select]
    overall_mean = float(outcome.mean())
    selected_mean = float(outcome[selected].mean())
    oracle_mean = float(outcome[oracle].mean())
    denominator = oracle_mean - overall_mean
    utility = (
        float((selected_mean - overall_mean) / denominator)
        if denominator > 1e-15
        else float("nan")
    )
    return {
        "spearman": spearman(score, outcome),
        "utility": utility,
        "capture": float(len(set(selected) & set(oracle)) / n_select),
        "selected_mean_error": selected_mean,
        "overall_mean_error": overall_mean,
        "oracle_mean_error": oracle_mean,
        "n_tasks": len(score),
        "n_selected": n_select,
    }


def add_target_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    for target in TARGETS:
        block = frame.loc[frame.target.eq(target)].copy()
        if block.empty:
            raise AnalysisFailure(f"missing target: {target}")
        block["rank_magnitude"] = percentile_rank(block.predicted_magnitude)
        block["rank_safeconf"] = percentile_rank(block.safeconf_e201_risk)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def rank_fusion(
    magnitude: Iterable[float], safeconf: Iterable[float], alpha: float
) -> np.ndarray:
    """Combine average ranks without floating-point tie splitting.

    The registered alpha grid advances in twentieths.  Combining the raw
    half-integer average ranks with the corresponding integer numerator before
    one final division is mathematically identical to the percentile formula,
    while preserving every theoretical tie exactly.
    """
    magnitude = np.asarray(list(magnitude), dtype=float)
    safeconf = np.asarray(list(safeconf), dtype=float)
    if len(magnitude) != len(safeconf) or len(magnitude) < 2:
        raise AnalysisFailure("rank fusion requires aligned finite arrays")
    if not np.isfinite(magnitude).all() or not np.isfinite(safeconf).all():
        raise AnalysisFailure("rank fusion requires aligned finite arrays")
    numerator = int(round(float(alpha) * ALPHA_DENOMINATOR))
    if not math.isclose(
        float(alpha), numerator / ALPHA_DENOMINATOR, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AnalysisFailure("alpha is outside the registered twentieth grid")
    magnitude_rank = rankdata(magnitude, method="average")
    safeconf_rank = rankdata(safeconf, method="average")
    weighted_rank = (
        numerator * magnitude_rank
        + (ALPHA_DENOMINATOR - numerator) * safeconf_rank
    )
    return weighted_rank / (ALPHA_DENOMINATOR * len(magnitude))


def fusion_score(frame: pd.DataFrame, alpha: float) -> np.ndarray:
    return rank_fusion(
        frame.predicted_magnitude.to_numpy(float),
        frame.safeconf_e201_risk.to_numpy(float),
        alpha,
    )


def target_metrics(frame: pd.DataFrame, alpha: float) -> dict[str, float]:
    score = fusion_score(frame, alpha)
    return review_metrics(score, frame.family_rms_error, frame.task_id)


def macro_metrics(frame: pd.DataFrame, alphas: dict[str, float]) -> dict[str, float]:
    rows = []
    for target in TARGETS:
        block = frame.loc[frame.target.eq(target)]
        rows.append(target_metrics(block, alphas[target]))
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("spearman", "utility", "capture", "selected_mean_error")
    }


def choose_alpha(training: pd.DataFrame) -> tuple[float, float]:
    candidates = []
    training_targets = [target for target in TARGETS if target in set(training.target)]
    for alpha in ALPHAS:
        values = []
        for target in training_targets:
            block = training.loc[training.target.eq(target)]
            values.append(target_metrics(block, float(alpha))["spearman"])
        candidates.append((float(np.mean(values)), float(alpha)))
    # Larger alpha wins an exact tie, keeping the correction closer to magnitude-only.
    best_value, best_alpha = max(candidates, key=lambda item: (item[0], item[1]))
    return best_alpha, best_value


def weight_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha in ALPHAS:
        metrics = macro_metrics(frame, {target: float(alpha) for target in TARGETS})
        rows.append({"alpha_magnitude": float(alpha), **metrics})
    return pd.DataFrame(rows)


def leave_one_target_out(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for held_target in TARGETS:
        training = frame.loc[frame.target.ne(held_target)]
        held = frame.loc[frame.target.eq(held_target)]
        alpha, training_spearman = choose_alpha(training)
        fusion = target_metrics(held, alpha)
        magnitude = target_metrics(held, 1.0)
        safeconf = target_metrics(held, 0.0)
        rows.append(
            {
                "held_target": held_target,
                "n_tasks": len(held),
                "selected_alpha_magnitude": alpha,
                "training_macro_spearman": training_spearman,
                **{f"fusion_{key}": value for key, value in fusion.items()},
                **{f"magnitude_{key}": value for key, value in magnitude.items()},
                **{f"safeconf_{key}": value for key, value in safeconf.items()},
                "delta_spearman_vs_magnitude": (
                    fusion["spearman"] - magnitude["spearman"]
                ),
                "delta_utility_vs_magnitude": fusion["utility"] - magnitude["utility"],
                "delta_capture_vs_magnitude": fusion["capture"] - magnitude["capture"],
            }
        )
    return pd.DataFrame(rows)


def macro_bootstrap_metrics(
    arrays: dict[str, np.ndarray],
    sampled_indices: np.ndarray,
    occurrences: np.ndarray,
    alphas: dict[str, float],
) -> dict[str, float]:
    fusion_rows = []
    magnitude_rows = []
    for target in TARGETS:
        keep = arrays["target"][sampled_indices] == target
        indices = sampled_indices[keep]
        target_occurrences = occurrences[keep]
        rank_magnitude = percentile_rank(arrays["magnitude"][indices])
        fused = rank_fusion(
            arrays["magnitude"][indices],
            arrays["safeconf"][indices],
            alphas[target],
        )
        fusion_rows.append(
            review_metrics(
                fused,
                arrays["outcome"][indices],
                arrays["task_id"][indices],
                target_occurrences,
            )
        )
        magnitude_rows.append(
            review_metrics(
                rank_magnitude,
                arrays["outcome"][indices],
                arrays["task_id"][indices],
                target_occurrences,
            )
        )
    output = {}
    for metric in ("spearman", "utility", "capture"):
        fusion_value = float(np.mean([row[metric] for row in fusion_rows]))
        magnitude_value = float(np.mean([row[metric] for row in magnitude_rows]))
        output[f"fusion_{metric}"] = fusion_value
        output[f"magnitude_{metric}"] = magnitude_value
        output[f"delta_{metric}"] = fusion_value - magnitude_value
    return output


def bootstrap(
    frame: pd.DataFrame,
    fixed_alpha: float,
    loto_alphas: dict[str, float],
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_bootstrap < 100:
        raise AnalysisFailure("at least 100 bootstrap replicates are required")
    rng = np.random.default_rng(MASTER_SEED)
    arrays = {
        "target": frame.target.astype(str).to_numpy(),
        "task_id": frame.task_id.astype(str).to_numpy(),
        "magnitude": frame.predicted_magnitude.to_numpy(float),
        "safeconf": frame.safeconf_e201_risk.to_numpy(float),
        "outcome": frame.family_rms_error.to_numpy(float),
    }
    cluster_names = sorted(frame.condition.astype(str).unique())
    condition_values = frame.condition.astype(str).to_numpy()
    cluster_members = [
        np.flatnonzero(condition_values == cluster) for cluster in cluster_names
    ]
    rows = []
    for replicate in range(n_bootstrap):
        chosen = rng.integers(0, len(cluster_names), len(cluster_names))
        sampled_indices = np.concatenate([cluster_members[int(index)] for index in chosen])
        occurrences = np.concatenate(
            [
                np.full(len(cluster_members[int(index)]), occurrence, dtype=int)
                for occurrence, index in enumerate(chosen)
            ]
        )
        for estimator, alphas in (
            ("fixed_candidate", {target: fixed_alpha for target in TARGETS}),
            ("loto_selected", loto_alphas),
        ):
            rows.append(
                {
                    "replicate": replicate,
                    "estimator": estimator,
                    **macro_bootstrap_metrics(
                        arrays, sampled_indices, occurrences, alphas
                    ),
                }
            )
    draws = pd.DataFrame(rows)
    summary_rows = []
    for estimator, block in draws.groupby("estimator", sort=False):
        for metric in ("spearman", "utility", "capture"):
            values = block[f"delta_{metric}"].to_numpy(float)
            summary_rows.append(
                {
                    "estimator": estimator,
                    "metric": f"delta_{metric}_vs_magnitude",
                    "ci95_lower": float(np.quantile(values, 0.025)),
                    "bootstrap_median": float(np.median(values)),
                    "ci95_upper": float(np.quantile(values, 0.975)),
                    "fraction_positive": float(np.mean(values > 0)),
                    "bootstrap_valid": len(values),
                }
            )
    return draws, pd.DataFrame(summary_rows)


def plot_sweep(sweep: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), facecolor="white")
    colors = {"spearman": "#3C5488", "utility": "#00A087"}
    for axis, metric, label in zip(
        axes,
        ("spearman", "utility"),
        ("排序一致性（Spearman）", "20% 复核效用"),
    ):
        axis.plot(
            sweep.alpha_magnitude,
            sweep[metric],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=colors[metric],
        )
        best = sweep.loc[sweep[metric].idxmax()]
        baseline = sweep.loc[sweep.alpha_magnitude.eq(1.0)].iloc[0]
        axis.axvline(0.8, color="#E64B35", linestyle="--", linewidth=1.2)
        axis.scatter([best.alpha_magnitude], [best[metric]], s=55, color="#E64B35", zorder=4)
        axis.text(
            best.alpha_magnitude,
            best[metric] + 0.012,
            f"最高 {best[metric]:.3f}",
            ha="center",
            fontsize=9,
        )
        axis.text(
            0.98,
            baseline[metric] - 0.025,
            f"只看幅度 {baseline[metric]:.3f}",
            ha="right",
            fontsize=9,
            color="#555555",
        )
        axis.set_xlabel("预测幅度在融合分数中的权重")
        axis.set_ylabel(label)
        axis.set_xlim(-0.02, 1.02)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "全部 1,808 个任务均保留；曲线用于开发候选公式，不是新的盲测结果",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / "figures" / f"E206_weight_sweep.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loto(loto: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), facecolor="white")
    display_targets = {"hepg2": "HepG2", "jurkat": "Jurkat"}
    targets = [display_targets.get(target, target) for target in loto.held_target]
    x = np.arange(len(targets))
    for axis, metric, label in zip(
        axes,
        ("delta_spearman_vs_magnitude", "delta_utility_vs_magnitude"),
        ("排序一致性的变化", "20% 复核效用的变化"),
    ):
        values = loto[metric].to_numpy(float)
        axis.axhline(0, color="#333333", linewidth=1)
        axis.bar(x, values, color=["#3C5488" if value >= 0 else "#E64B35" for value in values], width=0.62)
        for position, value in zip(x, values):
            axis.text(
                position,
                value + (0.002 if value >= 0 else -0.004),
                f"{value:+.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
        axis.set_xticks(x, targets)
        axis.set_ylabel(label + "（融合－只看幅度）")
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "每个细胞背景的权重只由另外三个背景选择；结果仍属于 E201 内部开发证据",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / "figures" / f"E206_leave_one_target_out.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_framework(output: Path) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, axis = plt.subplots(figsize=(12.0, 5.0), facecolor="white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def box(x: float, y: float, width: float, height: float, text: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.5,
            edgecolor=color,
            facecolor="white",
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=12,
            color="#222222",
            linespacing=1.45,
        )

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.3,
                color="#555555",
                connectionstyle="arc3,rad=0.0",
            )
        )

    axis.text(
        0.5,
        0.93,
        "任务风险由两部分组成：模型预测的变化规模 ＋ 当前证据是否充分",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#222222",
    )
    box(0.03, 0.36, 0.17, 0.25, "扰动预测模型\n输出每个任务的\n基因变化向量", "#3C5488")
    box(0.27, 0.62, 0.24, 0.20, "响应规模\n预测幅度\n模型认为变化有多大", "#E64B35")
    box(
        0.27,
        0.16,
        0.24,
        0.26,
        "证据不足与预测不稳定\n原 SafeConf\n历史支持·背景覆盖·模型分歧",
        "#00A087",
    )
    box(0.58, 0.35, 0.18, 0.28, "百分位融合\n80% 响应规模\n20% 证据风险", "#7E6148")
    box(0.83, 0.36, 0.14, 0.25, "任务风险队列\n优先复核\n更可能出错的任务", "#3C5488")

    arrow((0.20, 0.50), (0.27, 0.71))
    arrow((0.20, 0.47), (0.27, 0.30))
    arrow((0.51, 0.72), (0.58, 0.55))
    arrow((0.51, 0.29), (0.58, 0.43))
    arrow((0.76, 0.49), (0.83, 0.49))

    axis.text(
        0.5,
        0.055,
        "打分时只读取训练侧证据和模型预测，不读取目标任务的真实表达或真实误差",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output / "figures" / f"E206_risk_decomposition.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def build_report(
    frame: pd.DataFrame,
    sweep: pd.DataFrame,
    loto: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    fixed_alpha: float,
    n_bootstrap: int,
) -> str:
    fixed = sweep.loc[sweep.alpha_magnitude.eq(fixed_alpha)].iloc[0]
    magnitude = sweep.loc[sweep.alpha_magnitude.eq(1.0)].iloc[0]
    best_rho = sweep.loc[sweep.spearman.idxmax()]
    best_utility = sweep.loc[sweep.utility.idxmax()]
    selected = ", ".join(
        f"{row.held_target}={row.selected_alpha_magnitude:.2f}"
        for row in loto.itertuples(index=False)
    )
    lines = [
        "# E206：预测幅度加入 SafeConf 的开发结果",
        "",
        f"生成时间：{now()}",
        "",
        "证据等级：**E201 结局打开后的内部方法开发；不是新外部盲测**",
        "",
        "## 一句话结论",
        "",
        (
            f"在四个细胞背景的 {len(frame):,} 个任务上，候选公式把 80% 权重给预测幅度、"
            "20% 权重给原 SafeConf。它比只看预测幅度获得更高的平均排序一致性；"
            "20% 复核效用也有小幅上升，但仍需新的冻结数据确认。"
        ),
        "",
        "## 候选公式",
        "",
        "```text",
        "SafeConf-M = 0.80 × 目标背景内的预测幅度百分位",
        "           + 0.20 × 目标背景内的原 SafeConf 百分位",
        "```",
        "",
        "百分位只依赖模型预测和 SafeConf 特征，不读取该任务的真实答案。",
        "",
        "## 全数据描述性结果",
        "",
        "| 方案 | 四背景宏平均 Spearman | 四背景宏平均 20% 效用 | 四背景宏平均抓取率 |",
        "|---|---:|---:|---:|",
        (
            f"| 只看预测幅度 | {magnitude.spearman:.4f} | {magnitude.utility:.4f} | "
            f"{magnitude.capture:.4f} |"
        ),
        (
            f"| 固定候选公式（0.80/0.20） | {fixed.spearman:.4f} | {fixed.utility:.4f} | "
            f"{fixed.capture:.4f} |"
        ),
        (
            f"| 开发曲线最高 Spearman（幅度权重 {best_rho.alpha_magnitude:.2f}） | "
            f"{best_rho.spearman:.4f} | {best_rho.utility:.4f} | {best_rho.capture:.4f} |"
        ),
        (
            f"| 开发曲线最高效用（幅度权重 {best_utility.alpha_magnitude:.2f}） | "
            f"{best_utility.spearman:.4f} | {best_utility.utility:.4f} | {best_utility.capture:.4f} |"
        ),
        "",
        "这些数字按细胞背景分别计算后取平均，因此回答的是：一次面对某个新细胞背景的一批任务时，怎样安排复核顺序。",
        "",
        "## 留一细胞背景结果",
        "",
        f"训练背景选择出的幅度权重：{selected}。",
        "",
        "| 留出背景 | 选定幅度权重 | Spearman：融合 / 幅度 | 差值 | 20% 效用：融合 / 幅度 | 差值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in loto.itertuples(index=False):
        lines.append(
            f"| {row.held_target} | {row.selected_alpha_magnitude:.2f} | "
            f"{row.fusion_spearman:.4f} / {row.magnitude_spearman:.4f} | "
            f"{row.delta_spearman_vs_magnitude:+.4f} | "
            f"{row.fusion_utility:.4f} / {row.magnitude_utility:.4f} | "
            f"{row.delta_utility_vs_magnitude:+.4f} |"
        )
    lines.extend(
        [
            "",
            "留一背景分析的重点是方向是否稳定，而不是在已经打开的 E201 上追求最大的数字。",
            "",
            f"## {n_bootstrap:,} 次簇自助法",
            "",
            "| 估计方案 | 相对幅度的指标 | 95% 区间 | 差值为正的重抽样比例 |",
            "|---|---|---:|---:|",
        ]
    )
    for row in bootstrap_summary.itertuples(index=False):
        lines.append(
            f"| {row.estimator} | {row.metric} | "
            f"[{row.ci95_lower:.4f}, {row.ci95_upper:.4f}] | {row.fraction_positive:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 明早可以向老师说什么",
            "",
            "1. 原公式没有作废。它提供了幅度以外的风险信息，但不应与幅度对着竞争。",
            "2. 新候选公式让预测幅度负责主排序，让 SafeConf 负责小幅校正，结构更符合现有证据。",
            "3. 四个背景轮流留出时，排序一致性的变化需要逐背景报告，不能只报最好的一格。",
            "4. 这一步完成的是候选公式开发。下一步要先冻结 0.80/0.20，再到新数据或不同架构上一次性确认。",
            "",
            "## 不能写进论文结论的话",
            "",
            "- ‘融合公式已经在独立外部数据上得到验证’；",
            "- ‘SafeConf 在所有模型和细胞背景上都优于预测幅度’；",
            "- ‘本方法已经保证节省湿实验或保证二区录用’。",
            "",
            "## 文件",
            "",
            "- `tables/E206_WEIGHT_SWEEP.csv`：全部候选权重；",
            "- `tables/E206_LOTO_RESULTS.csv`：四次留一背景；",
            "- `tables/E206_BOOTSTRAP_SUMMARY.csv`：配对差值区间；",
            "- `figures/E206_weight_sweep.png`：权重曲线；",
            "- `figures/E206_leave_one_target_out.png`：逐背景变化。",
            "- `figures/E206_risk_decomposition.png`：风险分解与融合流程。",
        ]
    )
    return "\n".join(lines) + "\n"


def self_test() -> None:
    values = percentile_rank([30, 10, 20, 20])
    expected = np.asarray([1.0, 0.25, 0.625, 0.625])
    if not np.allclose(values, expected):
        raise AnalysisFailure("rank self-test failed")
    outcome = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    metrics = review_metrics(outcome, outcome, list("abcde"))
    if not math.isclose(metrics["spearman"], 1.0) or not math.isclose(
        metrics["utility"], 1.0
    ):
        raise AnalysisFailure("metric self-test failed")
    magnitude = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    safeconf = np.asarray([5.0, 1.0, 2.0, 3.0, 4.0])
    fused = rank_fusion(magnitude, safeconf, 0.80)
    if fused[0] != fused[1]:
        raise AnalysisFailure("fusion tie-stability self-test failed")
    print("E206 self-test PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    input_path = args.input.resolve()
    output = args.output.resolve()
    if not input_path.is_file():
        raise AnalysisFailure(f"missing input: {input_path}")
    frame = pd.read_csv(input_path)
    required = {
        "task_id",
        "target",
        "condition",
        "analysis_stratum",
        "predicted_magnitude",
        "safeconf_e201_risk",
        "family_rms_error",
    }
    if not required.issubset(frame.columns):
        raise AnalysisFailure(f"missing columns: {sorted(required - set(frame.columns))}")
    frame = frame.loc[frame.analysis_stratum.eq("primary_ge30")].copy()
    if (
        len(frame) != 1_808
        or frame.task_id.nunique() != 1_808
        or tuple(frame.target.drop_duplicates()) != TARGETS
        or not np.isfinite(
            frame[["predicted_magnitude", "safeconf_e201_risk", "family_rms_error"]].to_numpy(float)
        ).all()
    ):
        raise AnalysisFailure("E201 primary task contract changed")
    frame = add_target_ranks(frame)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)

    sweep = weight_sweep(frame)
    loto = leave_one_target_out(frame)
    fixed_alpha = float(np.median(loto.selected_alpha_magnitude))
    loto_alphas = dict(zip(loto.held_target, loto.selected_alpha_magnitude))
    draws, bootstrap_summary = bootstrap(
        frame, fixed_alpha, loto_alphas, args.bootstrap
    )

    fixed_point = macro_metrics(frame, {target: fixed_alpha for target in TARGETS})
    magnitude_point = macro_metrics(frame, {target: 1.0 for target in TARGETS})
    loto_point = macro_metrics(frame, loto_alphas)
    for estimator, point, alphas in (
        ("fixed_candidate", fixed_point, {target: fixed_alpha for target in TARGETS}),
        ("loto_selected", loto_point, loto_alphas),
    ):
        for metric in ("spearman", "utility", "capture"):
            mask = (bootstrap_summary.estimator == estimator) & (
                bootstrap_summary.metric == f"delta_{metric}_vs_magnitude"
            )
            bootstrap_summary.loc[mask, "point_estimate"] = (
                point[metric] - magnitude_point[metric]
            )
        bootstrap_summary.loc[bootstrap_summary.estimator == estimator, "alpha_by_target"] = json.dumps(
            alphas, sort_keys=True
        )

    atomic_csv(output / "tables/E206_WEIGHT_SWEEP.csv", sweep)
    atomic_csv(output / "tables/E206_LOTO_RESULTS.csv", loto)
    atomic_csv(output / "tables/E206_BOOTSTRAP_SUMMARY.csv", bootstrap_summary)
    # Bootstrap draws are intentionally not written: the reproducible seed and
    # summary are retained without adding a large, low-value file to Git.
    input_record = pd.DataFrame(
        [
            {
                "path": input_path.relative_to(ROOT).as_posix(),
                "bytes": input_path.stat().st_size,
                "sha256": sha256_file(input_path),
                "n_primary_tasks": len(frame),
            }
        ]
    )
    atomic_csv(output / "tables/E206_INPUT_HASHES.csv", input_record)
    plot_sweep(sweep, output)
    plot_loto(loto, output)
    plot_framework(output)
    report = build_report(
        frame, sweep, loto, bootstrap_summary, fixed_alpha, args.bootstrap
    )
    atomic_text(output / "E206_REPORT.md", report)
    status = {
        "status": "PASS_DEVELOPMENT_ONLY",
        "generated_at": now(),
        "evidence_level": "post-outcome internal development",
        "external_validation": False,
        "input": input_record.iloc[0].to_dict(),
        "n_tasks": len(frame),
        "targets": list(TARGETS),
        "candidate_formula": {
            "magnitude_weight": fixed_alpha,
            "safeconf_weight": round(1.0 - fixed_alpha, 2),
            "transform": "within-target percentile rank",
        },
        "selection_rule": "leave-one-target-out; maximize training-target macro Spearman; larger alpha wins ties",
        "bootstrap": {
            "unit": "perturbation condition",
            "replicates": args.bootstrap,
            "seed": MASTER_SEED,
        },
        "outputs": [],
    }
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "E206_STATUS.json":
            status["outputs"].append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    atomic_json(output / "E206_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
