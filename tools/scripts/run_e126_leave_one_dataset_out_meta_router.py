#!/usr/bin/env python3
"""E126: leave-one-dataset-out meta calibration of deployable risk features."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E126_leave_one_dataset_out_meta_router_20260714"
TABLES, REPORTS, FIGURES = OUT / "tables", OUT / "reports", OUT / "figures"
ERROR = "error_two_predictor_mean_rmse"
FEATURES = {
    "safeconf_rank": "safeconf_calibrated_pair_risk",
    "disagreement_rank": "risk_model_disagreement",
    "magnitude_rank": "baseline_predicted_magnitude",
    "context_novelty_rank": "context_novelty_scaled",
    "perturbation_novelty_rank": "perturbation_novelty",
    "low_support_rank": "low_support_risk",
}
SCORES = {
    "MetaSafeConf_LODO": "metasafeconf_lodo_risk",
    "SafeConf": "safeconf_calibrated_pair_risk",
    "predicted_magnitude": "baseline_predicted_magnitude",
    "model_disagreement": "risk_model_disagreement",
}
COVERAGES = np.round(np.arange(0.50, 1.001, 0.05), 2)
SEED, N_BOOT = 202607126, 10000


def load() -> pd.DataFrame:
    paths = [
        ("Frangieh", ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713/tables/E108_TEST_TASK_RISK_TABLE.csv"),
        (None, ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/E112_ALL_TASKS.csv"),
        ("Shifrut", ROOT / "docs/实验结果/E120_shifrut_formal_dual_models_20260714/Shifrut/TASK_RISK_TABLE.csv"),
        ("Liang", ROOT / "docs/实验结果/E123_liang_formal_dual_models_20260714/Liang/TASK_RISK_TABLE.csv"),
    ]
    parts = []
    for dataset, path in paths:
        d = pd.read_csv(path)
        if dataset is not None:
            d["dataset"] = dataset
        parts.append(d)
    data = pd.concat(parts, ignore_index=True, sort=False)
    data["low_support_risk"] = -np.log1p(data.training_support_count.astype(float))
    group = data.groupby(["dataset", "fold_id"], sort=False)
    for new, old in FEATURES.items():
        data[new] = group[old].rank(method="average", pct=True)
    data["error_rank_train_target"] = group[ERROR].rank(method="average", pct=True)
    return data


def fit_lodo(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, coefs = [], []
    cols = list(FEATURES)
    for heldout in sorted(data.dataset.unique()):
        train = data[~data.dataset.eq(heldout)].copy()
        test = data[data.dataset.eq(heldout)].copy()
        # Each source dataset contributes total weight one; each fold within it contributes equally.
        fold_counts = train.groupby(["dataset", "fold_id"]).size()
        nfolds = train.groupby("dataset").fold_id.nunique()
        weights = []
        for r in train.itertuples(index=False):
            weights.append(1.0 / (len(nfolds) * nfolds.loc[r.dataset] * fold_counts.loc[(r.dataset, r.fold_id)]))
        model = Ridge(alpha=10.0, positive=True)
        model.fit(train[cols].to_numpy(float), train.error_rank_train_target.to_numpy(float), sample_weight=np.asarray(weights) * len(train))
        test["metasafeconf_lodo_risk"] = model.predict(test[cols].to_numpy(float))
        test["heldout_truth_used_for_fit_or_transform"] = False
        out.append(test)
        coefs.extend({"heldout_dataset": heldout, "term": term, "coefficient": float(value)} for term, value in zip(cols, model.coef_))
        coefs.append({"heldout_dataset": heldout, "term": "intercept", "coefficient": float(model.intercept_)})
    return pd.concat(out, ignore_index=True), pd.DataFrame(coefs)


def triage(g: pd.DataFrame, score: str) -> dict:
    x = g.sort_values([score, "task_id"], kind="stable")
    full = float(g[ERROR].mean())
    retained = []
    for coverage in COVERAGES:
        n = max(1, int(np.ceil(coverage * len(x))))
        retained.append(float(x.iloc[:n][ERROR].mean()))
    aurc = float(np.trapezoid(retained, COVERAGES) / (COVERAGES[-1] - COVERAGES[0])) / full
    n = max(1, int(np.ceil(0.20 * len(x))))
    high, low = x.iloc[-n:], x.iloc[:-n]
    rho = spearmanr(g[score], g[ERROR]).statistic
    return {
        "spearman": float(0 if not np.isfinite(rho) else rho),
        "normalized_aurc_50_100": aurc,
        "top20_error_enrichment": float(high[ERROR].mean() / full),
        "reject20_remaining_error_reduction": float((full - low[ERROR].mean()) / full),
        "top20_total_error_capture": float(high[ERROR].sum() / g[ERROR].sum()),
    }


def metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, fold), g in data.groupby(["dataset", "fold_id"], sort=True):
        for score, col in SCORES.items():
            rows.append({"dataset": dataset, "fold_id": fold, "score": score, "n_tasks": len(g), **triage(g, col)})
    return pd.DataFrame(rows)


def macro(folds: pd.DataFrame) -> pd.DataFrame:
    cols = ["spearman", "normalized_aurc_50_100", "top20_error_enrichment", "reject20_remaining_error_reduction", "top20_total_error_capture"]
    by = folds.groupby(["dataset", "score"], as_index=False)[cols].mean()
    total = by.groupby("score", as_index=False)[cols].mean()
    total.insert(0, "scope", "five_dataset_equal_macro")
    return pd.concat([by.rename(columns={"dataset": "scope"}), total], ignore_index=True)


def bootstrap(folds: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    datasets = sorted(folds.dataset.unique())
    metrics = ["spearman", "normalized_aurc_50_100", "top20_total_error_capture"]
    arrays = {}
    for dataset in datasets:
        d = folds[folds.dataset.eq(dataset)]
        order = sorted(d.fold_id.unique())
        arrays[dataset] = {m: {s: np.asarray([d[(d.fold_id == f) & (d.score == s)][m].iloc[0] for f in order]) for s in SCORES} for m in metrics}
    values = {(m, c): [] for m in metrics for c in ("SafeConf", "predicted_magnitude", "model_disagreement")}
    for _ in range(N_BOOT):
        sampled = rng.choice(datasets, len(datasets), replace=True)
        agg = {m: {s: [] for s in SCORES} for m in metrics}
        for dataset in sampled:
            n = len(arrays[dataset][metrics[0]]["MetaSafeConf_LODO"])
            ix = rng.integers(0, n, n)
            for m in metrics:
                for s in SCORES:
                    agg[m][s].append(float(arrays[dataset][m][s][ix].mean()))
        for m in metrics:
            meta = float(np.mean(agg[m]["MetaSafeConf_LODO"]))
            for c in ("SafeConf", "predicted_magnitude", "model_disagreement"):
                comp = float(np.mean(agg[m][c]))
                values[(m, c)].append(comp - meta if m == "normalized_aurc_50_100" else meta - comp)
    point = macro(folds).query("scope == 'five_dataset_equal_macro'").set_index("score")
    rows = []
    for (metric, comparator), samples in values.items():
        meta, comp = point.loc["MetaSafeConf_LODO", metric], point.loc[comparator, metric]
        delta = comp - meta if metric == "normalized_aurc_50_100" else meta - comp
        x = np.asarray(samples)
        rows.append({"metric": metric, "comparator": comparator, "favorable_delta": float(delta), "ci95_low": float(np.quantile(x, .025)), "ci95_high": float(np.quantile(x, .975)), "probability_favorable": float(np.mean(x > 0)), "bootstrap_unit": "dataset_population_plus_outer_fold", "n_bootstrap": N_BOOT})
    return pd.DataFrame(rows)


def figure(summary: pd.DataFrame) -> None:
    d = summary[summary.scope.eq("five_dataset_equal_macro")].set_index("score")
    w, h = 1120, 500
    colors = {"MetaSafeConf_LODO": "#276c66", "SafeConf": "#6c948e", "predicted_magnitude": "#a47b3c", "model_disagreement": "#58778a"}
    panels = [("spearman", "Spearman ↑"), ("normalized_aurc_50_100", "normalized AURC ↓"), ("top20_total_error_capture", "top-20% error capture ↑")]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">', '<rect width="100%" height="100%" fill="#fff"/>', '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#27343c}.t{font-size:25px;font-weight:700}.s{font-size:13px;fill:#647078}.v{font-size:13px;font-weight:700}</style>', '<text x="45" y="42" class="t">E126｜整套留出数据集的风险路由</text>', '<text x="45" y="70" class="s">每次只用其余四套历史数据学习；目标数据集真值不进入拟合与特征变换。</text>']
    order = list(SCORES)
    for j, (metric, label) in enumerate(panels):
        x0 = 45 + j * 355
        vals = [float(d.loc[s, metric]) for s in order]
        lo = min(0.0, min(vals) * 1.1); hi = max(vals) * 1.15
        sy = lambda v: 370 - (v - lo) / max(hi - lo, 1e-9) * 245
        parts += [f'<text x="{x0+145}" y="110" text-anchor="middle" class="s">{label}</text>', f'<line x1="{x0}" y1="370" x2="{x0+300}" y2="370" stroke="#ccd5d8"/>']
        for i, score in enumerate(order):
            x = x0 + 10 + 72 * i; y = sy(vals[i])
            parts += [f'<rect x="{x}" y="{y:.1f}" width="48" height="{370-y:.1f}" fill="{colors[score]}"/>', f'<text x="{x+24}" y="{y-6:.1f}" text-anchor="middle" class="v">{vals[i]:.3f}</text>', f'<text x="{x+24}" y="394" text-anchor="middle" class="s">{["Meta","Safe","幅度","分歧"][i]}</text>']
    parts.append('</svg>')
    (FIGURES / "F1_lodo_meta_router.svg").write_text("\n".join(parts))


def main() -> None:
    for d in (TABLES, REPORTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    raw = load()
    routed, coef = fit_lodo(raw)
    fold = metrics(routed)
    summary = macro(fold)
    boot = bootstrap(fold)
    routed.to_csv(TABLES / "E126_LODO_TASKS.csv", index=False)
    coef.to_csv(TABLES / "E126_COEFFICIENTS.csv", index=False)
    fold.to_csv(TABLES / "E126_FOLD_METRICS.csv", index=False)
    summary.to_csv(TABLES / "E126_SUMMARY.csv", index=False)
    boot.to_csv(TABLES / "E126_BOOTSTRAP.csv", index=False)
    figure(summary)
    point = summary[summary.scope.eq("five_dataset_equal_macro")].set_index("score")
    lines = ["# E126｜跨数据集风险路由器", "", "每次整套留出一个数据集，只用其余四套历史数据拟合正系数 Ridge。留出数据集的任务真值不参与拟合、特征秩变换或阈值选择。", "", "## 五数据集等权结果", "", "| score | Spearman↑ | normalized AURC↓ | top-20% enrichment↑ | reject-20% reduction↑ | top-20% capture↑ |", "|---|---:|---:|---:|---:|---:|"]
    for score, r in point.iterrows():
        lines.append(f"| {score} | {r.spearman:.4f} | {r.normalized_aurc_50_100:.4f} | {r.top20_error_enrichment:.4f} | {r.reject20_remaining_error_reduction:.4f} | {r.top20_total_error_capture:.4f} |")
    lines += ["", "## 聚类 bootstrap", "", "正的 favorable delta 表示 MetaSafeConf 更好。", "", "| metric | comparator | favorable Δ | 95% CI | P(Δ>0) |", "|---|---|---:|---:|---:|"]
    for r in boot.itertuples(index=False):
        lines.append(f"| {r.metric} | {r.comparator} | {r.favorable_delta:.4f} | [{r.ci95_low:.4f}, {r.ci95_high:.4f}] | {r.probability_favorable:.3f} |")
    lines += ["", "## 定位", "", "E126 是在 E125 后冻结的方法改进；它证明的是跨项目历史校准是否可行，不是未来数据上的事前确认。无论结果方向如何，下一步均需用冻结后的同一实现测试第六套未见数据。"]
    (REPORTS / "E126_REPORT.md").write_text("\n".join(lines) + "\n")
    status = {"experiment": "E126_leave_one_dataset_out_meta_router", "generated_at": datetime.now().isoformat(timespec="seconds"), "status": "complete", "n_datasets": int(routed.dataset.nunique()), "n_folds": int(routed.fold_id.nunique()), "n_tasks": len(routed), "model": "positive Ridge alpha=10", "heldout_dataset_truth_used_for_fit_or_feature_transform": False, "confirmatory": False, "requires_new_sixth_dataset": True}
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README_先看这个.md").write_text("# E126 先看这个\n\n先读 `reports/E126_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2)); print(point.to_string()); print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
