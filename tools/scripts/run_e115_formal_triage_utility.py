#!/usr/bin/env python3
"""E115: practical triage utility on the three formal gene datasets."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
E108 = ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713/tables/E108_TEST_TASK_RISK_TABLE.csv"
E112 = ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/E112_ALL_TASKS.csv"
OUT = ROOT / "docs/实验结果/E115_formal_triage_utility_20260713"
TABLES, REPORTS, FIGURES = OUT / "tables", OUT / "reports", OUT / "figures"
ERROR = "error_two_predictor_mean_rmse"
SCORES = {
    "SafeConf": "safeconf_calibrated_pair_risk",
    "predicted_magnitude": "baseline_predicted_magnitude",
    "model_disagreement": "risk_model_disagreement",
}
COVERAGES = np.round(np.arange(0.50, 1.001, 0.05), 2)
SEED = 202607115
N_BOOT = 10000


def load() -> pd.DataFrame:
    a = pd.read_csv(E108)
    a["dataset"] = "Frangieh"
    b = pd.read_csv(E112)
    data = pd.concat([a, b], ignore_index=True, sort=False)
    required = ["dataset", "fold_id", "perturbation", ERROR, *SCORES.values()]
    missing = [x for x in required if x not in data]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return data


def metrics_for(g: pd.DataFrame, score: str) -> tuple[dict, list[dict]]:
    x = g.sort_values([score, "task_id"], kind="stable")
    full = float(g[ERROR].mean())
    curves = []
    for coverage in COVERAGES:
        n = max(1, int(np.ceil(coverage * len(x))))
        curves.append({"coverage": coverage, "retained_mean_error": float(x.iloc[:n][ERROR].mean())})
    y = np.asarray([r["retained_mean_error"] for r in curves], float)
    aurc = float(np.trapezoid(y, COVERAGES) / (COVERAGES[-1] - COVERAGES[0]))
    nhigh = max(1, int(np.ceil(0.20 * len(x))))
    high, low = x.iloc[-nhigh:], x.iloc[:-nhigh]
    return {
        "n_tasks": len(g),
        "normalized_aurc_50_100": aurc / full,
        "top20_error_enrichment": float(high[ERROR].mean() / full),
        "reject20_remaining_error_reduction": float((full - low[ERROR].mean()) / full),
        "top20_total_error_capture": float(high[ERROR].sum() / g[ERROR].sum()),
    }, curves


def compute(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, curves = [], []
    for (dataset, fold), g in data.groupby(["dataset", "fold_id"], sort=True):
        for score_name, score_col in SCORES.items():
            values, curve = metrics_for(g, score_col)
            rows.append({"dataset": dataset, "fold_id": fold, "score": score_name, **values})
            for item in curve:
                curves.append({"dataset": dataset, "fold_id": fold, "score": score_name, **item})
    return pd.DataFrame(rows), pd.DataFrame(curves)


def macro(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = ["normalized_aurc_50_100", "top20_error_enrichment", "reject20_remaining_error_reduction", "top20_total_error_capture"]
    by_dataset = folds.groupby(["dataset", "score"], as_index=False)[metrics].mean()
    overall = by_dataset.groupby("score", as_index=False)[metrics].mean()
    overall.insert(0, "scope", "three_dataset_equal_macro")
    dataset_rows = by_dataset.rename(columns={"dataset": "scope"})
    return pd.concat([dataset_rows, overall], ignore_index=True)


def bootstrap(folds: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    datasets = sorted(folds.dataset.unique())
    metrics = ["normalized_aurc_50_100", "top20_error_enrichment", "reject20_remaining_error_reduction", "top20_total_error_capture"]
    arrays = {}
    for dataset in datasets:
        d = folds[folds.dataset.eq(dataset)].copy()
        fold_order = sorted(d.fold_id.unique())
        arrays[dataset] = {
            metric: {
                score: np.asarray([d[d.fold_id.eq(f) & d.score.eq(score)][metric].iloc[0] for f in fold_order], float)
                for score in SCORES
            }
            for metric in metrics
        }
    boots = {m: {c: [] for c in ("predicted_magnitude", "model_disagreement")} for m in metrics}
    for _ in range(N_BOOT):
        sampled = rng.choice(datasets, size=len(datasets), replace=True)
        values = {m: {s: [] for s in SCORES} for m in metrics}
        for dataset in sampled:
            n_folds = len(arrays[dataset][metrics[0]]["SafeConf"])
            chosen = rng.integers(0, n_folds, size=n_folds)
            for metric in metrics:
                for score in SCORES:
                    value = float(arrays[dataset][metric][score][chosen].mean())
                    values[metric][score].append(value)
        for metric in metrics:
            safe = float(np.mean(values[metric]["SafeConf"]))
            for comparator in ("predicted_magnitude", "model_disagreement"):
                other = float(np.mean(values[metric][comparator]))
                favorable = other - safe if metric == "normalized_aurc_50_100" else safe - other
                boots[metric][comparator].append(favorable)
    rows = []
    point = macro(folds).query("scope == 'three_dataset_equal_macro'").set_index("score")
    for metric in metrics:
        for comparator in ("predicted_magnitude", "model_disagreement"):
            safe = float(point.loc["SafeConf", metric])
            other = float(point.loc[comparator, metric])
            delta = other - safe if metric == "normalized_aurc_50_100" else safe - other
            b = np.asarray(boots[metric][comparator], float)
            rows.append({
                "metric": metric,
                "comparator": comparator,
                "favorable_delta_definition": "comparator_minus_safeconf" if metric == "normalized_aurc_50_100" else "safeconf_minus_comparator",
                "favorable_delta": delta,
                "ci95_low": float(np.quantile(b, 0.025)),
                "ci95_high": float(np.quantile(b, 0.975)),
                "probability_favorable": float(np.mean(b > 0)),
                "bootstrap_unit": "dataset_population_plus_fold",
                "n_bootstrap": N_BOOT,
            })
    return pd.DataFrame(rows)


def figure(summary: pd.DataFrame) -> None:
    d = summary[summary.scope.eq("three_dataset_equal_macro")].set_index("score")
    metrics = [("normalized_aurc_50_100", "normalized AURC\n(低为好)"), ("top20_error_enrichment", "top-20% 错误富集\n(高为好)"), ("top20_total_error_capture", "top-20% 错误捕获\n(高为好)")]
    colors = {"SafeConf": "#2f7f76", "predicted_magnitude": "#a5782b", "model_disagreement": "#55758a"}
    w, h = 1120, 500
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">', '<rect width="100%" height="100%" fill="#fff"/>', '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#26343c}.t{font-size:25px;font-weight:700}.s{font-size:14px;fill:#637179}.v{font-size:14px;font-weight:700}</style>', '<text x="45" y="42" class="t">E115｜三正式数据集的分诊效用</text>', '<text x="45" y="70" class="s">数据集与 fold 等权；分数不使用测试真值重新拟合。</text>']
    for j, (metric, label) in enumerate(metrics):
        x0 = 70 + j * 350
        vals = [float(d.loc[s, metric]) for s in SCORES]
        lower = min(0.9 if metric == "normalized_aurc_50_100" else 0.0, min(vals) * 0.95)
        upper = max(vals) * 1.08
        scale = lambda v: 365 - (v - lower) / max(upper - lower, 1e-9) * 245
        parts += [f'<text x="{x0+135}" y="110" text-anchor="middle" class="s">{label.split(chr(10))[0]}</text>', f'<line x1="{x0}" y1="365" x2="{x0+270}" y2="365" stroke="#cfd9dc"/>']
        for i, score in enumerate(SCORES):
            x = x0 + 20 + i * 82
            y = scale(vals[i])
            parts += [f'<rect x="{x}" y="{y:.1f}" width="55" height="{365-y:.1f}" fill="{colors[score]}"/>', f'<text x="{x+27.5}" y="{y-7:.1f}" text-anchor="middle" class="v">{vals[i]:.3f}</text>', f'<text x="{x+27.5}" y="390" text-anchor="middle" class="s">{["SafeConf","幅度","分歧"][i]}</text>']
    parts.append('</svg>')
    (FIGURES / "F1_formal_triage_utility.svg").write_text("\n".join(parts))


def main() -> None:
    for d in (TABLES, REPORTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    data = load()
    folds, curves = compute(data)
    summary = macro(folds)
    boot = bootstrap(folds)
    folds.to_csv(TABLES / "E115_FOLD_METRICS.csv", index=False)
    curves.to_csv(TABLES / "E115_RISK_COVERAGE_CURVES.csv", index=False)
    summary.to_csv(TABLES / "E115_MACRO_SUMMARY.csv", index=False)
    boot.to_csv(TABLES / "E115_BOOTSTRAP.csv", index=False)
    figure(summary)
    point = summary[summary.scope.eq("three_dataset_equal_macro")].set_index("score")
    primary = boot[boot.metric.isin(["normalized_aurc_50_100", "top20_total_error_capture"])]
    point_better_both = all(
        point.loc["SafeConf", "normalized_aurc_50_100"] < point.loc[c, "normalized_aurc_50_100"]
        and point.loc["SafeConf", "top20_total_error_capture"] > point.loc[c, "top20_total_error_capture"]
        for c in ("predicted_magnitude", "model_disagreement")
    )
    ci_positive_for_one = any(
        (primary[primary.comparator.eq(c)].ci95_low > 0).all()
        for c in ("predicted_magnitude", "model_disagreement")
    )
    passed = bool(point_better_both and ci_positive_for_one)
    lines = [
        "# E115｜三正式数据集的实际分诊效用",
        "",
        "E115 不重训预测器、不修改风险分数。它把 E108/E112 已冻结的三套 gene 数据测试任务转换为 risk–coverage 和 top-risk 资源分诊指标。",
        "",
        "## 三数据集等权宏平均",
        "",
        "| score | normalized AURC↓ | top-20% error enrichment↑ | reject-20% remaining error reduction↑ | top-20% total error capture↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for score, r in point.iterrows():
        lines.append(f"| {score} | {r.normalized_aurc_50_100:.4f} | {r.top20_error_enrichment:.4f} | {r.reject20_remaining_error_reduction:.4f} | {r.top20_total_error_capture:.4f} |")
    lines += ["", "## 成对 bootstrap", "", "正的 favorable delta 表示 SafeConf 更好。", "", "| metric | comparator | favorable Δ | 95% CI | P(Δ>0) |", "|---|---|---:|---:|---:|"]
    for r in boot.itertuples(index=False):
        lines.append(f"| {r.metric} | {r.comparator} | {r.favorable_delta:.4f} | [{r.ci95_low:.4f}, {r.ci95_high:.4f}] | {r.probability_favorable:.3f} |")
    lines += ["", "## 预设判定", "", f"- 通过：**{'是' if passed else '否'}**。", "- 点估计只有在 AURC 与 top-20% error capture 同时超过两个基线时才算方向一致。", "- 还要求相对至少一个强基线的两个主效用指标区间均不跨 0。未通过时只保留描述性趋势。"]
    (REPORTS / "E115_REPORT.md").write_text("\n".join(lines) + "\n")
    status = {"experiment": "E115_formal_triage_utility", "generated_at": datetime.now().isoformat(timespec="seconds"), "status": "complete", "n_datasets": int(data.dataset.nunique()), "n_folds": int(data.fold_id.nunique()), "n_tasks": len(data), "risk_scores_refit_on_test_truth": False, "preregistered_gate_passed": passed}
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README_先看这个.md").write_text("# E115 先看这个\n\n先读 `reports/E115_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(point.to_string())
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
