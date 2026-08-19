#!/usr/bin/env python3
"""E101: no-retuning meta-audit across three independent gene matrices."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E101_gene_cartesian_meta_audit_20260713"
TABLES, REPORTS, FIGURES = OUT / "tables", OUT / "reports", OUT / "figures"
E98 = ROOT / "docs/实验结果/E98_frangieh_gene_cartesian_predictions_20260713/tables/E98_TASK_RISK_TABLE.csv"
E100 = ROOT / "docs/实验结果/E100_gene_external_cartesian_predictions_20260713/tables/E100_TASK_RISK_TABLE.csv"
PRIMARY = "safeconf_frozen_pair_risk"
COMPARATORS = ("baseline_predicted_magnitude", "risk_model_disagreement")
N_BOOTSTRAP = 5000
SEED = 20260713101


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or len(np.unique(a[mask])) < 2 or len(np.unique(b[mask])) < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a[mask]), rankdata(b[mask]))[0, 1])


def load_tasks() -> pd.DataFrame:
    frangieh = pd.read_csv(E98)
    frangieh["dataset_meta"] = "Frangieh"
    external = pd.read_csv(E100)
    external["dataset_meta"] = external["dataset"].astype(str)
    tasks = pd.concat([frangieh, external], ignore_index=True, sort=False)
    return tasks[tasks["split"].eq("test")].copy()


def summarize(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, fraction, fold), group in tasks.groupby(["dataset_meta", "train_fraction", "fold_id"], sort=True):
        error = group["error_two_predictor_mean_rmse"].to_numpy(float)
        for score in (PRIMARY, *COMPARATORS):
            rows.append(
                {"dataset": dataset, "train_fraction": fraction, "fold_id": fold, "score": score,
                 "n_tasks": len(group), "spearman": spearman(group[score].to_numpy(float), error)}
            )
    return pd.DataFrame(rows)


def make_cache(tasks: pd.DataFrame) -> dict[str, dict[str, tuple[pd.DataFrame, list[np.ndarray]]]]:
    cache = {}
    for dataset, dataset_group in tasks.groupby("dataset_meta", sort=True):
        fold_cache = {}
        for fold, group in dataset_group.groupby("fold_id", sort=True):
            group = group.reset_index(drop=True)
            labels = group["perturbation"].astype(str).to_numpy()
            clusters = [np.flatnonzero(labels == value) for value in sorted(np.unique(labels))]
            fold_cache[str(fold)] = (group, clusters)
        cache[str(dataset)] = fold_cache
    return cache


def sampled_dataset_delta(
    fold_cache: dict[str, tuple[pd.DataFrame, list[np.ndarray]]],
    comparator: str,
    rng: np.random.Generator,
) -> float:
    folds = sorted(fold_cache)
    deltas = []
    for fold in rng.choice(folds, len(folds), replace=True):
        group, clusters = fold_cache[str(fold)]
        draws = rng.integers(0, len(clusters), len(clusters))
        index = np.concatenate([clusters[int(draw)] for draw in draws])
        error = group["error_two_predictor_mean_rmse"].to_numpy(float)[index]
        deltas.append(
            spearman(group[PRIMARY].to_numpy(float)[index], error)
            - spearman(group[comparator].to_numpy(float)[index], error)
        )
    return float(np.nanmean(deltas))


def observed_dataset_delta(fold_cache: dict[str, tuple[pd.DataFrame, list[np.ndarray]]], comparator: str) -> float:
    values = []
    for group, _ in fold_cache.values():
        error = group["error_two_predictor_mean_rmse"].to_numpy(float)
        values.append(spearman(group[PRIMARY].to_numpy(float), error) - spearman(group[comparator].to_numpy(float), error))
    return float(np.nanmean(values))


def bootstrap(tasks: pd.DataFrame) -> pd.DataFrame:
    full = tasks[np.isclose(tasks["train_fraction"], 1.0)]
    cache = make_cache(full)
    datasets = sorted(cache)
    rng = np.random.default_rng(SEED)
    rows = []
    for comparator in COMPARATORS:
        observed = {dataset: observed_dataset_delta(cache[dataset], comparator) for dataset in datasets}
        dataset_samples = {dataset: [] for dataset in datasets}
        fixed_samples, population_samples = [], []
        for _ in range(N_BOOTSTRAP):
            one = {dataset: sampled_dataset_delta(cache[dataset], comparator, rng) for dataset in datasets}
            for dataset in datasets:
                dataset_samples[dataset].append(one[dataset])
            fixed_samples.append(float(np.mean(list(one.values()))))
            sampled_datasets = rng.choice(datasets, len(datasets), replace=True)
            population_samples.append(
                float(np.mean([sampled_dataset_delta(cache[str(dataset)], comparator, rng) for dataset in sampled_datasets]))
            )
        for dataset in datasets:
            values = np.asarray(dataset_samples[dataset])
            rows.append(
                {"scope": dataset, "primary": PRIMARY, "comparator": comparator,
                 "bootstrap_unit": "outer_fold_plus_perturbation_cluster", "n_datasets": 1,
                 "n_folds": len(cache[dataset]), "observed_macro_delta_spearman": observed[dataset],
                 "ci95_low": float(np.quantile(values, .025)), "ci95_high": float(np.quantile(values, .975)),
                 "probability_delta_gt_zero": float(np.mean(values > 0)), "n_bootstrap": N_BOOTSTRAP}
            )
        for unit, values in [
            ("fixed_datasets_fold_plus_perturbation", np.asarray(fixed_samples)),
            ("dataset_population_plus_fold_plus_perturbation", np.asarray(population_samples)),
        ]:
            rows.append(
                {"scope": "three_dataset_macro", "primary": PRIMARY, "comparator": comparator,
                 "bootstrap_unit": unit, "n_datasets": len(datasets),
                 "n_folds": sum(len(cache[dataset]) for dataset in datasets),
                 "observed_macro_delta_spearman": float(np.mean(list(observed.values()))),
                 "ci95_low": float(np.quantile(values, .025)), "ci95_high": float(np.quantile(values, .975)),
                 "probability_delta_gt_zero": float(np.mean(values > 0)), "n_bootstrap": N_BOOTSTRAP}
            )
    return pd.DataFrame(rows)


def leave_one_dataset_out(summary: pd.DataFrame) -> pd.DataFrame:
    full = summary[np.isclose(summary["train_fraction"], 1.0)]
    macro = full.groupby(["dataset", "score"], as_index=False)["spearman"].mean()
    datasets = sorted(macro["dataset"].unique())
    rows = []
    for removed in datasets:
        kept = macro[macro["dataset"].ne(removed)]
        values = kept.groupby("score")["spearman"].mean()
        for comparator in COMPARATORS:
            rows.append(
                {"removed_dataset": removed, "kept_datasets": "+".join(sorted(set(datasets) - {removed})),
                 "primary": PRIMARY, "comparator": comparator,
                 "macro_delta_spearman": float(values[PRIMARY] - values[comparator])}
            )
    return pd.DataFrame(rows)


def write_figure(bootstrap_table: pd.DataFrame) -> None:
    data = bootstrap_table[
        bootstrap_table["comparator"].eq("baseline_predicted_magnitude")
        & bootstrap_table["bootstrap_unit"].isin(
            ["outer_fold_plus_perturbation_cluster", "fixed_datasets_fold_plus_perturbation"]
        )
    ].copy()
    order = ["Frangieh", "Lara_exvivo", "Santinha", "three_dataset_macro"]
    data["order"] = data["scope"].map({value: index for index, value in enumerate(order)})
    data = data.sort_values("order")
    width, height, left, right = 1040, 520, 280, 90
    low, high = -0.35, 0.35
    def sx(value: float) -> float:
        return left + (value - low) / (high - low) * (width - left - right)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#26343d}.t{font-size:25px;font-weight:700}.s{font-size:15px;fill:#5f6b72}.l{font-size:16px}.v{font-size:14px;fill:#52616a}</style>',
        '<text x="48" y="42" class="t">E101｜三套遗传扰动矩阵：Frozen pair risk 相对 magnitude</text>',
        '<text x="48" y="70" class="s">点为 fold 宏平均 ΔSpearman；线为 fold+扰动聚类 bootstrap 95% CI。</text>',
        f'<line x1="{sx(0):.1f}" y1="100" x2="{sx(0):.1f}" y2="430" stroke="#87949b" stroke-dasharray="5 5"/>',
    ]
    for tick in [-.3, -.2, -.1, 0, .1, .2, .3]:
        x = sx(tick); parts.append(f'<line x1="{x:.1f}" y1="430" x2="{x:.1f}" y2="438" stroke="#69777f"/>')
        parts.append(f'<text x="{x:.1f}" y="459" text-anchor="middle" class="v">{tick:.1f}</text>')
    for index, row in enumerate(data.itertuples(index=False)):
        y = 130 + index * 82
        label = "三数据集固定集合宏平均" if row.scope == "three_dataset_macro" else row.scope
        parts.append(f'<text x="55" y="{y+5}" class="l">{label}</text>')
        parts.append(f'<line x1="{sx(row.ci95_low):.1f}" y1="{y}" x2="{sx(row.ci95_high):.1f}" y2="{y}" stroke="#456f82" stroke-width="4"/>')
        parts.append(f'<circle cx="{sx(row.observed_macro_delta_spearman):.1f}" cy="{y}" r="7" fill="#2f6f8f"/>')
        parts.append(f'<text x="{width-45}" y="{y+5}" text-anchor="end" class="v">{row.observed_macro_delta_spearman:.3f} [{row.ci95_low:.3f}, {row.ci95_high:.3f}]</text>')
    parts.append(f'<text x="{(left+width-right)/2:.1f}" y="495" text-anchor="middle" class="l">Δρ = frozen pair risk − predicted magnitude</text>')
    parts.append('</svg>')
    (FIGURES / "F1_frozen_vs_magnitude_forest.svg").write_text("\n".join(parts), encoding="utf-8")


def write_report(summary: pd.DataFrame, bootstrap_table: pd.DataFrame, lodo: pd.DataFrame, status: dict) -> None:
    full = summary[np.isclose(summary["train_fraction"], 1.0)]
    macro = full.groupby(["dataset", "score"], as_index=False)["spearman"].mean()
    pivot = macro.pivot(index="dataset", columns="score", values="spearman").reset_index()
    overall = macro.groupby("score")["spearman"].mean()
    lines = [
        "# E101｜三套独立多背景遗传扰动矩阵元分析", "",
        "E101 不重新拟合分数。主分数是 E98/E100 在 test truth 解封前已经计算的 `safeconf_frozen_pair_risk`，由模型分歧、背景 control 新颖度和训练支持组成；强基线为同一双预测器输出的 predicted magnitude。每个 fold 先算 Spearman，再在数据集内和数据集间做等权宏平均。", "",
        "## 100% 训练量 pooled setting", "",
        "| dataset | frozen pair risk ρ | magnitude ρ | disagreement ρ | Δρ vs magnitude |", "|---|---:|---:|---:|---:|",
    ]
    for row in pivot.itertuples(index=False):
        delta = row.safeconf_frozen_pair_risk - row.baseline_predicted_magnitude
        lines.append(f"| {row.dataset} | {row.safeconf_frozen_pair_risk:.3f} | {row.baseline_predicted_magnitude:.3f} | {row.risk_model_disagreement:.3f} | {delta:.3f} |")
    lines += [
        f"| 三数据集宏平均 | {overall[PRIMARY]:.3f} | {overall['baseline_predicted_magnitude']:.3f} | {overall['risk_model_disagreement']:.3f} | {overall[PRIMARY]-overall['baseline_predicted_magnitude']:.3f} |",
        "", "## Bootstrap", "",
        "| scope | comparator | unit | Δρ | 95% CI | P(Δ>0) |", "|---|---|---|---:|---:|---:|",
    ]
    for row in bootstrap_table.itertuples(index=False):
        lines.append(f"| {row.scope} | {row.comparator} | {row.bootstrap_unit} | {row.observed_macro_delta_spearman:.3f} | [{row.ci95_low:.3f}, {row.ci95_high:.3f}] | {row.probability_delta_gt_zero:.3f} |")
    lines += [
        "", "固定三数据集 bootstrap 回答这三份数据上的测量不确定性；dataset-population bootstrap 额外重采样数据集，回答推广到未来数据集的不确定性。只有 3 个数据集时，后者应作为主边界。", "",
        "## Leave-one-dataset-out 敏感性", "",
        "| removed | kept | comparator | macro Δρ |", "|---|---|---|---:|",
    ]
    for row in lodo.itertuples(index=False):
        lines.append(f"| {row.removed_dataset} | {row.kept_datasets} | {row.comparator} | {row.macro_delta_spearman:.3f} |")
    lines += [
        "", "## 结论边界", "",
        "Frozen pair risk 在 Frangieh、Lara 为正增量，在 Santinha 略低于 magnitude。若固定这三套数据，宏平均可以衡量现有证据；若把数据集视作未来总体的随机样本，三个数据集仍不足以给出稳定推广保证。E101 不用校准后的分数替换 frozen 主分数，因此没有根据 Santinha 失败回调权重。", "",
        "- `tables/E101_FOLD_SUMMARY.csv`", "- `tables/E101_BOOTSTRAP.csv`",
        "- `tables/E101_LEAVE_ONE_DATASET_OUT.csv`", "- `figures/F1_frozen_vs_magnitude_forest.svg`",
    ]
    (REPORTS / "E101_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "README_先看这个.md").write_text("# E101 先看这个\n\n先读 `reports/E101_REPORT.md`。\n", encoding="utf-8")


def main() -> None:
    for path in (TABLES, REPORTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    summary = summarize(tasks)
    bootstrap_table = bootstrap(tasks)
    lodo = leave_one_dataset_out(summary)
    summary.to_csv(TABLES / "E101_FOLD_SUMMARY.csv", index=False)
    bootstrap_table.to_csv(TABLES / "E101_BOOTSTRAP.csv", index=False)
    lodo.to_csv(TABLES / "E101_LEAVE_ONE_DATASET_OUT.csv", index=False)
    write_figure(bootstrap_table)
    status = {
        "experiment": "E101_gene_cartesian_meta_audit", "generated_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": sorted(tasks["dataset_meta"].unique()), "n_datasets": int(tasks["dataset_meta"].nunique()),
        "n_folds_at_full_fraction": int(tasks[np.isclose(tasks["train_fraction"], 1.0)]["fold_id"].nunique()),
        "n_test_task_rows_at_full_fraction": int(np.isclose(tasks["train_fraction"], 1.0).sum()),
        "primary_score": PRIMARY, "comparators": list(COMPARATORS), "n_bootstrap": N_BOOTSTRAP,
        "score_refit_in_E101": False, "test_truth_used_to_choose_score_or_weight": False,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(summary, bootstrap_table, lodo, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
