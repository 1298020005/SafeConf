#!/usr/bin/env python3
"""E118: unified meta-audit of formal CPA chemical risk contracts."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E118_chemical_contract_meta_20260713"
TABLES, REPORTS, FIGURES = OUT / "tables", OUT / "reports", OUT / "figures"
P84 = ROOT / "docs/实验结果/E84_cpa_rdkit_cartesian_formal_20260712/tables/E84_TASK_SCORES.csv"
P87 = ROOT / "docs/实验结果/E87_sciplex_to_openproblems_cpa_20260712/tables/E87_TASK_SCORES.csv"
P89 = ROOT / "docs/实验结果/E89_sciplex3_to_sciplex4_cpa_20260712/tables/E89_TASK_SCORES.csv"
STATUSES = [
    ROOT / "docs/实验结果/E84_cpa_rdkit_cartesian_formal_20260712/RUN_STATUS.json",
    ROOT / "docs/实验结果/E87_sciplex_to_openproblems_cpa_20260712/RUN_STATUS.json",
    ROOT / "docs/实验结果/E89_sciplex3_to_sciplex4_cpa_20260712/RUN_STATUS.json",
]
SEED, N_BOOT = 202607118, 10000


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[keep], b[keep]).statistic) if keep.sum() >= 3 else float("nan")


def one_metrics(g, score, error="pair_mean_rmse"):
    x = g.sort_values([score, "task_key"], kind="stable")
    full = float(g[error].mean())
    nhigh = max(1, int(np.ceil(0.2 * len(x))))
    high = x.iloc[-nhigh:]
    return {"spearman": rho(g[score], g[error]), "top20_error_enrichment": float(high[error].mean() / full), "top20_total_error_capture": float(high[error].sum() / g[error].sum())}


def load_sources():
    a = pd.read_csv(P84).rename(columns={"cpa_ridge_disagreement_rmse": "disagreement", "predicted_magnitude_mean": "magnitude"})
    b = pd.read_csv(P87).rename(columns={"model_disagreement_rmse": "disagreement", "predicted_magnitude_mean": "magnitude", "drug": "cluster"})
    c = pd.read_csv(P89).rename(columns={"model_disagreement_rmse": "disagreement", "predicted_magnitude_mean": "magnitude", "drug": "cluster"})
    a["cluster"] = a.perturbation_key.astype(str) + "::" + a.dose_key.astype(str)
    return {"sciPlex3_internal": a, "sciPlex3_to_OpenProblems": b, "sciPlex3_to_sciPlex4": c}


def point_metrics(sources):
    rows = []
    for source, g in sources.items():
        if source == "sciPlex3_internal":
            unit_rows = []
            for (manifest, quadrant), u in g.groupby(["manifest_id", "quadrant"], sort=True):
                for name, col in [("model_disagreement", "disagreement"), ("predicted_magnitude", "magnitude")]:
                    unit_rows.append({"manifest_id": manifest, "quadrant": quadrant, "score": name, **one_metrics(u, col)})
            units = pd.DataFrame(unit_rows)
            for score, u in units.groupby("score"):
                values = u.groupby("quadrant")[["spearman", "top20_error_enrichment", "top20_total_error_capture"]].mean().mean()
                rows.append({"source": source, "score": score, "n_tasks": len(g), **values.to_dict()})
        else:
            for name, col in [("model_disagreement", "disagreement"), ("predicted_magnitude", "magnitude")]:
                rows.append({"source": source, "score": name, "n_tasks": len(g), **one_metrics(g, col)})
    detail = pd.DataFrame(rows)
    macro = detail.groupby("score", as_index=False)[["spearman", "top20_error_enrichment", "top20_total_error_capture"]].mean()
    macro.insert(0, "source", "three_source_equal_macro")
    return pd.concat([detail, macro], ignore_index=True)


def numpy_metrics(error, score):
    error, score = np.asarray(error, float), np.asarray(score, float)
    rank_error, rank_score = rankdata(error), rankdata(score)
    spearman = float(np.corrcoef(rank_error, rank_score)[0, 1])
    order_idx = np.argsort(score, kind="stable")
    nhigh = max(1, int(np.ceil(0.2 * len(error))))
    high = error[order_idx[-nhigh:]]
    return np.asarray([spearman, high.mean() / error.mean(), high.sum() / error.sum()], float)


def bootstrap(sources):
    rng = np.random.default_rng(SEED)
    metrics = ("spearman", "top20_error_enrichment", "top20_total_error_capture")
    vals = {m: [] for m in metrics}
    a = sources["sciPlex3_internal"]
    a_by_manifest = {}
    for manifest, m in a.groupby("manifest_id", sort=True):
        unit = []
        for _, q in m.groupby("quadrant", sort=True):
            unit.append(numpy_metrics(q.pair_mean_rmse, q.disagreement) - numpy_metrics(q.pair_mean_rmse, q.magnitude))
        a_by_manifest[manifest] = np.stack(unit)
    external = {}
    for source in ("sciPlex3_to_OpenProblems", "sciPlex3_to_sciPlex4"):
        g = sources[source]
        groups = [idx.to_numpy() for _, idx in g.groupby(g.cluster.astype(str), sort=True).groups.items()]
        external[source] = {"error": g.pair_mean_rmse.to_numpy(float), "disagreement": g.disagreement.to_numpy(float), "magnitude": g.magnitude.to_numpy(float), "groups": groups}
    manifests = np.asarray(list(a_by_manifest))
    for _ in range(N_BOOT):
        chosen = rng.choice(manifests, size=len(manifests), replace=True)
        source_deltas = [np.concatenate([a_by_manifest[m] for m in chosen], axis=0).mean(axis=0)]
        for source, item in external.items():
            groups = item["groups"]
            selected_groups = rng.integers(0, len(groups), size=len(groups))
            idx = np.concatenate([groups[i] for i in selected_groups])
            source_deltas.append(numpy_metrics(item["error"][idx], item["disagreement"][idx]) - numpy_metrics(item["error"][idx], item["magnitude"][idx]))
        macro_delta = np.stack(source_deltas).mean(axis=0)
        for i, metric in enumerate(metrics):
            vals[metric].append(float(macro_delta[i]))
    point = point_metrics(sources).query("source == 'three_source_equal_macro'").set_index("score")
    rows = []
    for metric, values in vals.items():
        delta = float(point.loc["model_disagreement", metric] - point.loc["predicted_magnitude", metric])
        b = np.asarray(values)
        rows.append({"metric": metric, "delta_disagreement_minus_magnitude": delta, "ci95_low": float(np.quantile(b, .025)), "ci95_high": float(np.quantile(b, .975)), "probability_delta_positive": float((b > 0).mean()), "n_bootstrap": N_BOOT})
    return pd.DataFrame(rows)


def validate_contracts():
    rows = []
    for p in STATUSES:
        d = json.loads(p.read_text())
        issues = int(d.get("strict_issue_count", -1))
        truth = bool(d.get("target_truth_used_for_scores", d.get("target_truth_used_for_training_calibration_score_or_threshold", False)))
        rows.append({"experiment": d.get("experiment", p.parent.name), "strict_issue_count": issues, "target_truth_used_for_scores": truth, "contract_pass": issues == 0 and not truth})
    return pd.DataFrame(rows)


def figure(summary):
    d = summary[summary.source.eq("three_source_equal_macro")].set_index("score")
    metrics = [("spearman", "Spearman ρ"), ("top20_error_enrichment", "top-20% error enrichment"), ("top20_total_error_capture", "top-20% error capture")]
    w, h = 1000, 450
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">', '<rect width="100%" height="100%" fill="#fff"/>', '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#27343c}.t{font-size:24px;font-weight:700}.s{font-size:14px;fill:#647078}.v{font-size:14px;font-weight:700}</style>', '<text x="42" y="40" class="t">E118｜化学扰动统一合同元审计</text>', '<text x="42" y="68" class="s">三来源等权；青色为模型分歧，金色为预测幅度。</text>']
    for j, (metric, label) in enumerate(metrics):
        x = 70 + j * 310
        vals = [float(d.loc[s, metric]) for s in ("model_disagreement", "predicted_magnitude")]
        maxv = max(vals) * 1.15
        parts.append(f'<text x="{x+110}" y="105" text-anchor="middle" class="s">{label}</text>')
        for i, (value, color, name) in enumerate(zip(vals, ("#3b7188", "#a5782b"), ("分歧", "幅度"))):
            bh = 220 * value / max(maxv, 1e-9)
            bx = x + 35 + i * 90
            parts += [f'<rect x="{bx}" y="{350-bh:.1f}" width="55" height="{bh:.1f}" fill="{color}"/>', f'<text x="{bx+27}" y="{342-bh:.1f}" text-anchor="middle" class="v">{value:.3f}</text>', f'<text x="{bx+27}" y="375" text-anchor="middle" class="s">{name}</text>']
    parts.append('</svg>')
    (FIGURES / "F1_chemical_contract_meta.svg").write_text("\n".join(parts))


def main():
    for d in (TABLES, REPORTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    contracts = validate_contracts()
    summary = point_metrics(sources)
    boot = bootstrap(sources)
    contracts.to_csv(TABLES / "E118_CONTRACT_AUDIT.csv", index=False)
    summary.to_csv(TABLES / "E118_SOURCE_SUMMARY.csv", index=False)
    boot.to_csv(TABLES / "E118_BOOTSTRAP.csv", index=False)
    figure(summary)
    primary = boot.set_index("metric")
    passed = bool(contracts.contract_pass.all() and primary.loc["spearman", "ci95_low"] > 0 and primary.loc["top20_total_error_capture", "ci95_low"] > 0)
    macro = summary[summary.source.eq("three_source_equal_macro")].set_index("score")
    lines = ["# E118｜化学扰动统一合同元审计", "", "E118 不学习新权重，只把 E84、E87、E89 的 formal CPA 双预测器结果按同一指标汇总。三个来源分别代表 sciPlex3 内部四象限、sciPlex3→OpenProblems 跨数据集和 sciPlex3→sciPlex4 同族外部验证。", "", "## 合同审计", "", "| experiment | strict issues | truth used for score | pass |", "|---|---:|---|---|"]
    for r in contracts.itertuples(index=False):
        lines.append(f"| {r.experiment} | {r.strict_issue_count} | {r.target_truth_used_for_scores} | {r.contract_pass} |")
    lines += ["", "## 三来源等权宏平均", "", "| score | Spearman | top-20% error enrichment | top-20% total error capture |", "|---|---:|---:|---:|"]
    for score, r in macro.iterrows():
        lines.append(f"| {score} | {r.spearman:.3f} | {r.top20_error_enrichment:.3f} | {r.top20_total_error_capture:.3f} |")
    lines += ["", "## 分歧相对幅度", "", "| metric | Δ | 95% CI | P(Δ>0) |", "|---|---:|---:|---:|"]
    for r in boot.itertuples(index=False):
        lines.append(f"| {r.metric} | {r.delta_disagreement_minus_magnitude:.4f} | [{r.ci95_low:.4f}, {r.ci95_high:.4f}] | {r.probability_delta_positive:.3f} |")
    lines += ["", "## 预设判定", "", f"- chemical 独立增量通过：**{'是' if passed else '否'}**。", "- 合同闭环与方法增量是两件事：strict contract 全部通过，只能证明预测—冻结—解封—评价流程可信；若分歧没有稳定超过 magnitude，就必须作为跨模态负边界。"]
    (REPORTS / "E118_REPORT.md").write_text("\n".join(lines) + "\n")
    status = {"experiment": "E118_chemical_contract_meta", "generated_at": datetime.now().isoformat(timespec="seconds"), "status": "complete", "n_sources": len(sources), "n_source_tasks": {k: len(v) for k, v in sources.items()}, "all_strict_contracts_pass": bool(contracts.contract_pass.all()), "test_truth_used_to_refit_scores": False, "preregistered_chemical_increment_gate_passed": passed}
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README_先看这个.md").write_text("# E118 先看这个\n\n先读 `reports/E118_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
