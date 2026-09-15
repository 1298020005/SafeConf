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

    colors = ["#118A7E" if p == "safeconf_m_4to1" else "#D55E5E" if p == "predicted_magnitude" else "#54769A" for p in predictors]
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
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="x", color="#E6E8EB", linewidth=0.6)
        axis.set_axisbelow(True)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure, bbox_inches="tight")
    plt.close(fig)

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

## 可用于汇报的逻辑

1. E201 只有同一 GAT 架构的随机种子；E205 加入 Exphormer 后，检验对象变成预先注册的跨结构模型家族。
2. 确定性下界只依赖冻结预测之间的分歧，不读取目标真值。揭盲后用恒等式残差和下界违反数核对实现是否正确。
3. “数学下界成立”“下界足够紧，能签发高风险证书”“经验排序超过预测幅度”是三个不同结论，按上表分别报告。
4. 即使排序增量未通过，跨结构证书结果仍可独立解释；如果证书操作性门也未通过，则 E205 给出的有效结论是适用边界，而不是性能提升。

图：`{figure.name}`。
"""
    atomic_text(report, text)


if __name__ == "__main__":
    main()
