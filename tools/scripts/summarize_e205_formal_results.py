#!/usr/bin/env python3
"""Render a fixed, result-agnostic E205 report from formal evaluation tables."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SummaryFailure(RuntimeError):
    pass


PREDICTOR_LABELS = {
    "safeconf_m_4to1": "SafeConf-M",
    "predicted_magnitude": "Predicted magnitude",
    "safeconf_e205_risk": "Original SafeConf",
    "family_disagreement": "Exphormer seed disagreement",
    "gat_family_disagreement": "GAT seed disagreement",
    "registered_family_disagreement": "Registered-family lower bound",
    "cross_family_disagreement": "Cross-architecture disagreement",
}

PALETTE = {
    "safeconf_m_4to1": "#118A7E",
    "predicted_magnitude": "#D55E5E",
    "safeconf_e205_risk": "#54769A",
    "family_disagreement": "#7E6AAD",
    "gat_family_disagreement": "#B07C45",
    "registered_family_disagreement": "#3F8F72",
    "cross_family_disagreement": "#8A8F98",
}
TARGET_ORDER = ("K562", "RPE1", "hepg2", "jurkat")
REGISTERED_ROUTER_LABELS = {
    "context_holdout_router": "Context-holdout SafeConf",
    "architecture_aware_router": "Architecture-aware SafeConf",
    "historical_nonnegative_router": "Historical nonnegative SafeConf",
    "certificate_priority_q80": "Certificate first + magnitude",
    "registered_predicted_magnitude": "Registered-family magnitude",
    "registered_family_disagreement": "Registered-family lower bound",
}
REGISTERED_ROUTER_COLORS = {
    "context_holdout_router": "#B07C45",
    "architecture_aware_router": "#54769A",
    "historical_nonnegative_router": "#8E6C3A",
    "certificate_priority_q80": "#118A7E",
    "registered_predicted_magnitude": "#D55E5E",
    "registered_family_disagreement": "#7E6AAD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    return parser.parse_args()


def finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise SummaryFailure(f"expected finite value, got {value}")
    return result


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def extract_unique(frame: pd.DataFrame, **filters: object) -> pd.Series:
    block = frame
    for column, value in filters.items():
        block = block.loc[block[column].eq(value)]
    if len(block) != 1:
        raise SummaryFailure(f"expected one row for {filters}, found {len(block)}")
    return block.iloc[0]


def clean_axis(axis: plt.Axes, grid_axis: str | None = None) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=0.8, length=3, color="#4A4A4A")
    if grid_axis is not None:
        axis.grid(axis=grid_axis, color="#E6E8EB", linewidth=0.6)
        axis.set_axisbelow(True)


def save_figure(fig: plt.Figure, path: Path, extra_formats: bool = True) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = [path]
    if extra_formats:
        outputs.extend([path.with_suffix(".pdf"), path.with_suffix(".png")])
    for output in outputs:
        if output.exists():
            raise SummaryFailure(f"refusing to overwrite: {output}")
        options = {"bbox_inches": "tight", "facecolor": "white"}
        if output.suffix.lower() == ".png":
            options["dpi"] = 300
        fig.savefig(output, **options)
    return outputs


def annotate_panel(axis: plt.Axes, letter: str) -> None:
    axis.text(
        -0.16,
        1.05,
        letter,
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=13,
        va="top",
    )


def render_context_matrix(
    associations: pd.DataFrame,
    utilities: pd.DataFrame,
    predictors: list[str],
    output: Path,
) -> None:
    selected = [
        name
        for name in (
            "safeconf_m_4to1",
            "predicted_magnitude",
            "safeconf_e205_risk",
            "registered_family_disagreement",
        )
        if name in predictors
    ]
    labels = [PREDICTOR_LABELS[name] for name in selected]
    assoc = (
        associations.loc[
            associations.scope.isin(TARGET_ORDER)
            & associations.predictor.isin(selected)
        ]
        .pivot(index="scope", columns="predictor", values="spearman")
        .reindex(index=TARGET_ORDER, columns=selected)
    )
    utility = (
        utilities.loc[
            utilities.scope.isin(TARGET_ORDER)
            & utilities.predictor.isin(selected)
            & np.isclose(utilities.budget.astype(float), 0.20)
        ]
        .pivot(index="scope", columns="predictor", values="oracle_normalized_utility")
        .reindex(index=TARGET_ORDER, columns=selected)
    )
    if assoc.isna().any().any() or utility.isna().any().any():
        raise SummaryFailure("per-context result matrix is incomplete")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), facecolor="white")
    for axis, frame, label, letter in (
        (axes[0], assoc, "Spearman correlation", "a"),
        (axes[1], utility, "Review utility at 20% budget", "b"),
    ):
        values = frame.to_numpy(float)
        bound = max(0.25, float(np.nanmax(np.abs(values))))
        image = axis.imshow(
            values,
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
        axis.set_yticks(np.arange(len(TARGET_ORDER)), TARGET_ORDER)
        axis.set_xlabel(label)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                color = "white" if abs(values[row, column]) > 0.62 * bound else "#222222"
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=color,
                )
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.outline.set_linewidth(0.6)
        annotate_panel(axis, letter)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.tight_layout(w_pad=2.3)
    save_figure(fig, output)
    plt.close(fig)


def render_operating_curves(
    utilities: pd.DataFrame,
    curves: pd.DataFrame,
    predictors: list[str],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), facecolor="white")
    selected = [
        name
        for name in ("safeconf_m_4to1", "predicted_magnitude", "safeconf_e205_risk")
        if name in predictors
    ]
    pooled = utilities.loc[utilities.scope.eq("pooled")]
    for predictor in selected:
        block = pooled.loc[pooled.predictor.eq(predictor)].sort_values("budget")
        axes[0].plot(
            100 * block.budget.to_numpy(float),
            block.oracle_normalized_utility.to_numpy(float),
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=PALETTE[predictor],
            label=PREDICTOR_LABELS[predictor],
        )
    axes[0].set_xlabel("Review budget (%)")
    axes[0].set_ylabel("Oracle-normalized utility")
    axes[0].legend(frameon=False, fontsize=8.5)
    clean_axis(axes[0], "both")
    annotate_panel(axes[0], "a")

    for scope in ("pooled", *TARGET_ORDER):
        block = curves.loc[
            curves.unit.eq("task") & curves.scope.eq(scope)
        ].sort_values("certified_high_coverage")
        if block.empty:
            continue
        color = "#222222" if scope == "pooled" else PALETTE[
            ("safeconf_m_4to1", "predicted_magnitude", "safeconf_e205_risk", "family_disagreement")[
                TARGET_ORDER.index(scope)
            ]
        ]
        axes[1].plot(
            block.certified_high_coverage.to_numpy(float),
            block.certified_high_recall.to_numpy(float),
            marker="o" if scope == "pooled" else None,
            linewidth=2.2 if scope == "pooled" else 1.2,
            color=color,
            alpha=1.0 if scope == "pooled" else 0.72,
            label=scope,
        )
    axes[1].set_xlabel("Certified fraction")
    axes[1].set_ylabel("Recall of observed high-error tasks")
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=8.5, ncol=2)
    clean_axis(axes[1], "both")
    annotate_panel(axes[1], "b")
    fig.tight_layout(w_pad=2.5)
    save_figure(fig, output)
    plt.close(fig)


def render_certificate_priority_router(
    utilities: pd.DataFrame,
    output: Path,
) -> None:
    predictors = list(REGISTERED_ROUTER_LABELS)
    pooled = utilities.loc[
        utilities.scope.eq("pooled") & utilities.predictor.isin(predictors)
    ]
    if set(pooled.predictor) != set(predictors):
        raise SummaryFailure("registered-family routing table is incomplete")
    at_twenty = utilities.loc[
        utilities.scope.isin(("pooled", *TARGET_ORDER))
        & np.isclose(utilities.budget.astype(float), 0.20)
        & utilities.predictor.isin(
            (
                "architecture_aware_router",
                "context_holdout_router",
                "historical_nonnegative_router",
                "certificate_priority_q80",
                "registered_predicted_magnitude",
            )
        )
    ]
    matrix = (
        at_twenty.pivot(
            index="scope", columns="predictor", values="oracle_normalized_utility"
        )
        .reindex(index=("pooled", *TARGET_ORDER))
    )
    if matrix.isna().any().any():
        raise SummaryFailure("registered-family per-context routing table is incomplete")
    certificate_delta = matrix["certificate_priority_q80"] - matrix[
        "registered_predicted_magnitude"
    ]
    architecture_delta = matrix["architecture_aware_router"] - matrix[
        "registered_predicted_magnitude"
    ]
    context_delta = matrix["context_holdout_router"] - matrix[
        "registered_predicted_magnitude"
    ]
    nonnegative_delta = matrix["historical_nonnegative_router"] - matrix[
        "registered_predicted_magnitude"
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), facecolor="white")
    positions = np.arange(len(certificate_delta))
    axes[0].bar(
        positions - 0.27,
        context_delta.to_numpy(float),
        color=REGISTERED_ROUTER_COLORS["context_holdout_router"],
        width=0.18,
        label=REGISTERED_ROUTER_LABELS["context_holdout_router"],
    )
    axes[0].bar(
        positions - 0.09,
        architecture_delta.to_numpy(float),
        color=REGISTERED_ROUTER_COLORS["architecture_aware_router"],
        width=0.18,
        label=REGISTERED_ROUTER_LABELS["architecture_aware_router"],
    )
    axes[0].bar(
        positions + 0.09,
        nonnegative_delta.to_numpy(float),
        color=REGISTERED_ROUTER_COLORS["historical_nonnegative_router"],
        width=0.18,
        label=REGISTERED_ROUTER_LABELS["historical_nonnegative_router"],
    )
    axes[0].bar(
        positions + 0.27,
        certificate_delta.to_numpy(float),
        color=REGISTERED_ROUTER_COLORS["certificate_priority_q80"],
        width=0.18,
        label=REGISTERED_ROUTER_LABELS["certificate_priority_q80"],
    )
    axes[0].set_xticks(
        positions, certificate_delta.index, rotation=25, ha="right"
    )
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_ylabel("20% review-utility difference")
    axes[0].legend(frameon=False, fontsize=7.5)
    clean_axis(axes[0], "y")
    annotate_panel(axes[0], "a")

    for predictor in predictors:
        block = pooled.loc[pooled.predictor.eq(predictor)].sort_values("budget")
        axes[1].plot(
            100 * block.budget.to_numpy(float),
            block.oracle_normalized_utility.to_numpy(float),
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=REGISTERED_ROUTER_COLORS[predictor],
            label=REGISTERED_ROUTER_LABELS[predictor],
        )
    axes[1].set_xlabel("Review budget (%)")
    axes[1].set_ylabel("Registered-family review utility")
    axes[1].legend(frameon=False, fontsize=8.0)
    clean_axis(axes[1], "both")
    annotate_panel(axes[1], "b")
    fig.tight_layout(w_pad=2.5)
    save_figure(fig, output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    evaluation = args.evaluation_dir.resolve()
    report = args.output_report.resolve()
    figure = args.output_figure.resolve()
    for path in (report, figure):
        if path.exists():
            raise SummaryFailure(f"refusing to overwrite: {path}")

    status = json.loads(
        (evaluation / "E205_FORMAL_EVALUATION_STATUS.json").read_text(
            encoding="utf-8"
        )
    )
    if status.get("execution_status") != "PASS":
        raise SummaryFailure("formal evaluation did not finish with PASS")
    associations = pd.read_csv(evaluation / "E205_RISK_ASSOCIATIONS.csv")
    utilities = pd.read_csv(evaluation / "E205_REVIEW_UTILITY.csv")
    intervals = pd.read_csv(evaluation / "E205_INCREMENTAL_INTERVALS.csv")
    curves = pd.read_csv(evaluation / "E205_REGISTERED_CERTIFICATE_CURVES.csv")
    registered_utilities = pd.read_csv(
        evaluation / "E205_REGISTERED_ROUTING_UTILITY.csv"
    )
    registered_intervals = pd.read_csv(
        evaluation / "E205_REGISTERED_ROUTING_INTERVALS.csv"
    )

    predictors = [name for name in PREDICTOR_LABELS if name in set(associations.predictor)]
    pooled_assoc = associations.loc[
        associations.scope.eq("pooled") & associations.predictor.isin(predictors)
    ].set_index("predictor").loc[predictors]
    pooled_utility = utilities.loc[
        utilities.scope.eq("pooled")
        & utilities.predictor.isin(predictors)
        & np.isclose(utilities.budget.astype(float), 0.20)
    ].set_index("predictor").loc[predictors]
    utility_interval = extract_unique(intervals, measure="delta_utility_20")
    spearman_interval = extract_unique(intervals, measure="delta_spearman")
    task_curves = curves.loc[
        curves.unit.eq("task") & curves.scope.eq("pooled")
    ].sort_values("quantile")
    if len(task_curves) != 9:
        raise SummaryFailure("expected nine frozen certificate thresholds")

    colors = [PALETTE[p] for p in predictors]
    labels = [PREDICTOR_LABELS[p] for p in predictors]
    y = np.arange(len(predictors))
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), facecolor="white")
    axes[0].barh(y, pooled_assoc.spearman.to_numpy(float), color=colors, height=0.64)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Spearman correlation with family error")
    axes[0].axvline(0, color="#333333", linewidth=0.8)
    axes[0].text(-0.18, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=13)

    axes[1].barh(y, pooled_utility.oracle_normalized_utility.to_numpy(float), color=colors, height=0.64)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Review utility at 20% budget")
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].text(-0.18, 1.04, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=13)

    axes[2].plot(task_curves["quantile"], task_curves["certified_high_recall"], marker="o", color="#118A7E", label="Recall")
    axes[2].plot(task_curves["quantile"], task_curves["certified_high_coverage"], marker="s", color="#54769A", label="Issued fraction")
    axes[2].set_xlabel("Frozen threshold quantile")
    axes[2].set_ylabel("Fraction")
    axes[2].set_ylim(bottom=0)
    axes[2].legend(frameon=False)
    axes[2].text(-0.18, 1.04, "c", transform=axes[2].transAxes, fontweight="bold", fontsize=13)
    for axis in axes:
        clean_axis(axis, "x")
    fig.tight_layout()
    overview_outputs = save_figure(fig, figure)
    plt.close(fig)

    figure_dir = figure.parent / "figures"
    context_figure = figure_dir / "E205_CONTEXT_RESOLVED_RESULTS.svg"
    operating_figure = figure_dir / "E205_OPERATING_CHARACTERISTICS.svg"
    router_figure = figure_dir / "E205_CERTIFICATE_PRIORITY_ROUTER.svg"
    render_context_matrix(
        associations,
        utilities,
        predictors,
        context_figure,
    )
    render_operating_curves(
        utilities,
        curves,
        predictors,
        operating_figure,
    )
    render_certificate_priority_router(registered_utilities, router_figure)

    certificate_word = (
        "SUPPORTED"
        if status.get("registered_family_certificate_status") == "SUPPORTED"
        else "NOT_SUPPORTED"
    )
    ranking_word = (
        "SUPPORTED"
        if status.get("primary_increment_status") == "SUPPORTED"
        else "NOT_SUPPORTED"
    )
    router_word = (
        "SUPPORTED"
        if status.get("certificate_priority_router_status") == "SUPPORTED"
        else "NOT_SUPPORTED"
    )
    architecture_word = (
        "SUPPORTED"
        if status.get("architecture_aware_router_status") == "SUPPORTED"
        else "NOT_SUPPORTED"
    )
    context_word = (
        "SUPPORTED"
        if status.get("context_holdout_router_status") == "SUPPORTED"
        else "NOT_SUPPORTED"
    )
    nonnegative_word = (
        "SUPPORTED"
        if status.get("historical_nonnegative_router_status") == "SUPPORTED"
        else "NOT_SUPPORTED"
    )
    conclusion = []
    if int(status.get("registered_family_lower_bound_violations", -1)) == 0:
        conclusion.append("注册家族确定性下界在本次跨结构评价中没有发生违反。")
    else:
        conclusion.append("注册家族下界出现违反，数值实现或误差对象必须停止并复核。")
    if certificate_word == "SUPPORTED":
        conclusion.append("下界同时达到预先登记的紧致度与非零高风险签发门。")
    else:
        conclusion.append("下界的操作性门未通过；理论恒等式与实际筛查价值必须分开陈述。")
    if ranking_word == "SUPPORTED":
        conclusion.append("固定 SafeConf-M 在 20% 复核预算下确认了相对预测幅度的增量。")
    else:
        conclusion.append("固定 SafeConf-M 未确认相对预测幅度的复核增量，排序模块不得作为主要胜利。")
    if router_word == "SUPPORTED":
        conclusion.append("证书优先、幅度组内排序确认了对注册家族幅度的次要复核增量。")
    else:
        conclusion.append("证书优先路由未确认复核增量，不影响确定性下界本身的有效性。")
    if architecture_word == "SUPPORTED":
        conclusion.append("架构感知单向修正确认了相对注册家族幅度的复核增量。")
    else:
        conclusion.append("架构感知单向修正未通过跨结构确认，E217 公式不得按本次结果再调权。")
    if context_word == "SUPPORTED":
        conclusion.append("整背景留出专用路由通过了预先冻结的首要复核效用门。")
    else:
        conclusion.append("整背景留出专用路由未通过首要复核效用门，不能改用次要公式代替判定。")
    if nonnegative_word == "SUPPORTED":
        conclusion.append("历史非负学习路由在次要稳健性分析中也确认了增量。")
    else:
        conclusion.append("历史非负学习路由未通过次要门，不根据 E205 重新拟合权重。")

    router_utility_interval = extract_unique(
        registered_intervals,
        predictor="certificate_priority_q80",
        measure="delta_utility_20",
    )
    router_spearman_interval = extract_unique(
        registered_intervals,
        predictor="certificate_priority_q80",
        measure="delta_spearman",
    )
    architecture_utility_interval = extract_unique(
        registered_intervals,
        predictor="architecture_aware_router",
        measure="delta_utility_20",
    )
    architecture_spearman_interval = extract_unique(
        registered_intervals,
        predictor="architecture_aware_router",
        measure="delta_spearman",
    )
    context_utility_interval = extract_unique(
        registered_intervals,
        predictor="context_holdout_router",
        measure="delta_utility_20",
    )
    context_spearman_interval = extract_unique(
        registered_intervals,
        predictor="context_holdout_router",
        measure="delta_spearman",
    )
    nonnegative_utility_interval = extract_unique(
        registered_intervals,
        predictor="historical_nonnegative_router",
        measure="delta_utility_20",
    )
    nonnegative_spearman_interval = extract_unique(
        registered_intervals,
        predictor="historical_nonnegative_router",
        measure="delta_spearman",
    )

    rows = []
    for predictor in predictors:
        a = pooled_assoc.loc[predictor]
        u = pooled_utility.loc[predictor]
        rows.append(
            f"| {PREDICTOR_LABELS[predictor]} | {finite(a.spearman):.4f} | "
            f"{finite(u.oracle_normalized_utility):.4f} |"
        )
    text = f"""# E205 跨结构正式结果

生成依据：`E205_FORMAL_EVALUATION_STATUS.json` 与同目录正式 CSV。本文档只按预先冻结的门解释结果，不重新挑权重、目标或阈值。

## 一句话结论

{' '.join(conclusion)}

## 预注册门

| 项目 | 结果 |
| --- | --- |
| 正式评价执行 | {status['execution_status']} |
| 排序增量 | {ranking_word} |
| 整背景留出路由（首要确认） | {context_word} |
| 架构感知路由确认 | {architecture_word} |
| 历史非负学习路由（次要） | {nonnegative_word} |
| 证书优先路由（次要） | {router_word} |
| 注册家族证书 | {certificate_word} |
| 主任务数 | {int(status['n_primary_tasks'])} |
| 注册家族下界违反 | {int(status['registered_family_lower_bound_violations'])} |
| 注册家族恒等式失败 | {int(status['registered_family_identity_failures'])} |
| 下界紧致度中位数 | {finite(status['registered_family_lower_tightness_median']):.4f} |

## 风险排序与复核效用

| 信号 | 与家族真实误差的 Spearman | 20% 预算复核效用 |
| --- | ---: | ---: |
{chr(10).join(rows)}

SafeConf-M 相对预测幅度的 Spearman 差为 {finite(spearman_interval['estimate']):.4f}，扰动簇自助法 95% 区间为 [{finite(spearman_interval['ci95_lower']):.4f}, {finite(spearman_interval['ci95_upper']):.4f}]。20% 复核效用差为 {finite(utility_interval['estimate']):.4f}，95% 区间为 [{finite(utility_interval['ci95_lower']):.4f}, {finite(utility_interval['ci95_upper']):.4f}]。

## 整背景留出专用路由（E217 首要确认）

相对注册家族预测幅度的 Spearman 差为 {finite(context_spearman_interval['estimate']):.4f}，95% 区间为 [{finite(context_spearman_interval['ci95_lower']):.4f}, {finite(context_spearman_interval['ci95_upper']):.4f}]；20% 复核效用差为 {finite(context_utility_interval['estimate']):.4f}，95% 区间为 [{finite(context_utility_interval['ci95_lower']):.4f}, {finite(context_utility_interval['ci95_upper']):.4f}]。这是 E205 的首要路由确认量。

## 架构感知单向路由（E217 真值前冻结）

在注册跨架构家族自身的 RMS 误差上，架构感知路由相对注册家族预测幅度的 Spearman 差为 {finite(architecture_spearman_interval['estimate']):.4f}，95% 区间为 [{finite(architecture_spearman_interval['ci95_lower']):.4f}, {finite(architecture_spearman_interval['ci95_upper']):.4f}]；20% 复核效用差为 {finite(architecture_utility_interval['estimate']):.4f}，95% 区间为 [{finite(architecture_utility_interval['ci95_lower']):.4f}, {finite(architecture_utility_interval['ci95_upper']):.4f}]。公式在 E205 真值授权前固定，不按本次结果重新调权。

## 历史非负学习路由（E218 次要分析）

用八个已释放研究固定的非负权重路由相对注册家族预测幅度的 Spearman 差为 {finite(nonnegative_spearman_interval['estimate']):.4f}，95% 区间为 [{finite(nonnegative_spearman_interval['ci95_lower']):.4f}, {finite(nonnegative_spearman_interval['ci95_upper']):.4f}]；20% 复核效用差为 {finite(nonnegative_utility_interval['estimate']):.4f}，95% 区间为 [{finite(nonnegative_utility_interval['ci95_lower']):.4f}, {finite(nonnegative_utility_interval['ci95_upper']):.4f}]。它只是次要稳健性分析，不替换整背景留出的首要判定。

## 证书优先路由（预登记次要分析）

在注册跨架构家族自身的 RMS 误差上，证书优先路由相对注册家族预测幅度的 Spearman 差为 {finite(router_spearman_interval['estimate']):.4f}，95% 区间为 [{finite(router_spearman_interval['ci95_lower']):.4f}, {finite(router_spearman_interval['ci95_upper']):.4f}]；20% 复核效用差为 {finite(router_utility_interval['estimate']):.4f}，95% 区间为 [{finite(router_utility_interval['ci95_lower']):.4f}, {finite(router_utility_interval['ci95_upper']):.4f}]。该分析不替换固定 SafeConf-M 的主要门，也不替换证书正确性门。

## 可用于汇报的逻辑

1. E201 只有同一 GAT 架构的随机种子；E205 加入 Exphormer 后，检验对象变成预先注册的跨结构模型家族。
2. 确定性下界只依赖冻结预测之间的分歧，不读取目标真值。揭盲后用恒等式残差和下界违反数核对实现是否正确。
3. “数学下界成立”“下界足够紧，能签发高风险证书”“经验排序超过预测幅度”是三个不同结论，按上表分别报告。
4. 即使排序增量未通过，跨结构证书结果仍可独立解释；如果证书操作性门也未通过，则 E205 给出的有效结论是适用边界，而不是性能提升。

## 图件

- 总体排序、20% 复核效用与冻结证书阈值：`{figure.name}`（同时生成 PDF 与 300 dpi PNG）。
- 四个细胞背景的分解结果：`figures/{context_figure.name}`（同时生成 PDF 与 300 dpi PNG）。
- 不同复核预算及证书覆盖—召回关系：`figures/{operating_figure.name}`（同时生成 PDF 与 300 dpi PNG）。
- 证书优先路由相对注册家族幅度：`figures/{router_figure.name}`（同时生成 PDF 与 300 dpi PNG）。

图件共 {len(overview_outputs) + 9} 个文件；所有面板由同一批正式 CSV 自动生成，未按结果删除细胞背景、风险信号或预算点。
"""
    atomic_text(report, text)


if __name__ == "__main__":
    main()
