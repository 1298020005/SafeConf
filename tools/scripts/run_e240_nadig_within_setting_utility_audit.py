#!/usr/bin/env python3
"""Post-hoc audit: does the Nadig direction score help *within* task regimes?

E237 and E239 were unsealed before this analysis was conceived. This script is
diagnostic only; it cannot turn either cohort into a new confirmatory test.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/实验结果"
OUT = RESULTS / "E240_nadig_within_setting_utility_audit_20260923"
INPUTS = {
    "E237": RESULTS / "E237_nadig_disjoint_gene_confirmation_20260923/E237_TASK_AUDIT.csv",
    "E239": RESULTS / "E239_nadig_third_gene_confirmation_20260923/E239_TASK_AUDIT.csv",
}
METHODS = ["directional_risk_frozen", "magnitude", "source_seen"]


def correlation(score: np.ndarray, target: np.ndarray) -> float:
    if np.unique(score).size < 2 or np.unique(target).size < 2:
        return float("nan")
    return float(spearmanr(score, target).statistic)


def capture(score: np.ndarray, target: np.ndarray) -> float:
    k = math.ceil(.2 * len(target))
    selected = np.argsort(-score, kind="stable")[:k]
    return float(target[selected].sum() / np.sort(target)[-k:].sum())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cohort, path in INPUTS.items():
        frame = pd.read_csv(path)
        frame["source_seen"] = frame.setting.isin(["perturbation_unseen", "random_seen_pair"]).astype(int)
        assert len(frame) == 332 and frame.perturbation.nunique() == 128
        for fold, group in frame.groupby("fold_id", sort=True):
            for scope, part in [
                ("mixed", group),
                ("source_seen_only", group[group.source_seen.eq(1)]),
                ("heldout_context_only", group[group.source_seen.eq(0)]),
            ]:
                target = part.direction_error_rank_target.to_numpy(float)
                for method in METHODS:
                    score = part[method].to_numpy(float)
                    if method == "source_seen" and scope != "mixed":
                        continue  # Constant within the task regime.
                    rows.append({
                        "cohort": cohort, "fold_id": fold, "scope": scope, "method": method,
                        "n_tasks": len(part), "n_genes": part.perturbation.nunique(),
                        "spearman": correlation(score, target),
                        "top20_oracle_normalized_capture_stable_ties": capture(score, target),
                        "top20_capture_expected_random_ties": (
                            float(math.ceil(.2 * len(part)) * part.loc[part.source_seen.eq(1),
                                  "direction_error_rank_target"].mean()
                                  / np.sort(target)[-math.ceil(.2 * len(part)):].sum())
                            if method == "source_seen" and scope == "mixed" else float("nan")
                        ),
                        "target_mean": float(target.mean()),
                    })
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "E240_FOLD_AND_SETTING.csv", index=False)
    summary = detail.groupby(["cohort", "scope", "method"], as_index=False).agg(
        n_tasks=("n_tasks", "sum"), fold_macro_spearman=("spearman", "mean"),
        fold_macro_top20_capture=("top20_oracle_normalized_capture_stable_ties", "mean"),
        fold_macro_expected_random_tie_capture=("top20_capture_expected_random_ties", "mean"),
        target_mean=("target_mean", "mean"),
    )
    summary.to_csv(OUT / "E240_SUMMARY.csv", index=False)
    text = [
        "# E240｜Nadig 两批未用基因的任务类型与复核效用审计", "",
        "证据身份：**事后诊断**。E237、E239 均已解封；本审计不修改它们的主判定。",
        "每批 128 个互不重叠基因、332 个测试任务；两折等权。",
        "`source_seen` 是无需预测输出的二元任务元数据：该细胞背景是否在训练来源出现。",
        "20%捕获率除以同折同范围的理想选择结果；二元分数并列时采用固定表顺序，",
        "所以它只是一个可复算的展示值，不能声称二元基线具备并列内部排序能力。",
        "同时另报并列内部随机打破时的期望捕获率。", "",
        "| 批次 | 范围 | 风险分 ρ / 捕获 | 幅度 ρ / 捕获 | 仅背景见过 ρ / 捕获 |",
        "|---|---|---:|---:|---:|",
    ]
    for cohort in INPUTS:
        for scope in ["mixed", "source_seen_only", "heldout_context_only"]:
            values = {row.method: row for row in summary[(summary.cohort == cohort) & (summary.scope == scope)].itertuples()}
            fmt = lambda name: (f"{values[name].fold_macro_spearman:.3f} / "
                                f"{values[name].fold_macro_top20_capture:.3f}") if name in values else "—"
            text.append(f"| {cohort} | {scope} | {fmt('directional_risk_frozen')} | "
                        f"{fmt('magnitude')} | {fmt('source_seen')} |")
    text += [
        "", "只看背景是否见过的二元基线，在混合队列中随机打破并列后的",
        "20%期望捕获率：" + "；".join(
            f"{cohort} {summary[(summary.cohort == cohort) & (summary.scope == 'mixed') & (summary.method == 'source_seen')].fold_macro_expected_random_tie_capture.iloc[0]:.3f}"
            for cohort in INPUTS) + "。",
        "", "## 可支持的结论", "",
        "混合队列中风险分的相关性确实比强简单组合略高（E237 事后强基线审计、E239 预定主比较）；",
        "但任务类型本身几乎将高方向误差隔离出来。",
        "在来源已见和目标背景留出两个范围内分别排序时，这两批均未显示风险分优于单幅度。",
        "混合队列前 20% 捕获率相对仅看背景是否见过也没有稳定的实用增益。",
        "因此不能仅凭混合队列的相关性提升主张复核资源节约或跨背景一般可靠性。",
        "如果把方向任务作为论文主体，必须在另一个**独立研究**、预先规定的任务混合和",
        "真正有增益的复核指标下再验证；不通过就应停止该主张。", "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
