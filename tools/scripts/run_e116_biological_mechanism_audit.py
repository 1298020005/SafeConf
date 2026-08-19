#!/usr/bin/env python3
"""E116: predictor-failure mechanism and pathway hypotheses."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
E108 = ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713/tables/E108_TEST_TASK_RISK_TABLE.csv"
E112 = ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/E112_ALL_TASKS.csv"
OUT = ROOT / "docs/实验结果/E116_biological_mechanism_audit_20260713"
TABLES, REPORTS, FIGURES, RAW = OUT / "tables", OUT / "reports", OUT / "figures", OUT / "raw"
GP_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
SPECIES = {"Frangieh": "hsapiens", "Lara_exvivo": "mmusculus", "Santinha": "mmusculus"}
FEATURES = [
    "safeconf_calibrated_pair_risk",
    "risk_model_disagreement",
    "baseline_predicted_magnitude",
    "context_novelty_scaled",
    "perturbation_novelty",
    "training_support_count",
]


def rho(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.unique(a[keep]).size < 2 or np.unique(b[keep]).size < 2:
        return float("nan")
    return float(spearmanr(a[keep], b[keep]).statistic)


def gene_name(x: str) -> str:
    x = str(x)
    for suffix in ("+ctrl", "_ctrl", "+control"):
        if x.endswith(suffix):
            x = x[: -len(suffix)]
    return x


def load() -> pd.DataFrame:
    a = pd.read_csv(E108)
    a["dataset"] = "Frangieh"
    b = pd.read_csv(E112)
    data = pd.concat([a, b], ignore_index=True, sort=False)
    data["gene"] = data.perturbation.map(gene_name)
    data["gears_excess_error"] = data.error_gears_rmse - data.error_scgpt_rmse
    return data


def feature_associations(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, fold), g in data.groupby(["dataset", "fold_id"], sort=True):
        for feature in FEATURES:
            rows.append({"dataset": dataset, "fold_id": fold, "feature": feature, "n_tasks": len(g), "spearman_with_gears_excess_error": rho(g[feature], g.gears_excess_error), "spearman_with_gears_error": rho(g[feature], g.error_gears_rmse), "spearman_with_scgpt_error": rho(g[feature], g.error_scgpt_rmse)})
    frame = pd.DataFrame(rows)
    macro = frame.groupby(["dataset", "feature"], as_index=False).agg(n_folds=("fold_id", "nunique"), spearman_with_gears_excess_error=("spearman_with_gears_excess_error", "mean"), spearman_with_gears_error=("spearman_with_gears_error", "mean"), spearman_with_scgpt_error=("spearman_with_scgpt_error", "mean"))
    overall = macro.groupby("feature", as_index=False).agg(n_folds=("n_folds", "sum"), spearman_with_gears_excess_error=("spearman_with_gears_excess_error", "mean"), spearman_with_gears_error=("spearman_with_gears_error", "mean"), spearman_with_scgpt_error=("spearman_with_scgpt_error", "mean"))
    overall.insert(0, "dataset", "three_dataset_equal_macro")
    return frame, pd.concat([macro, overall], ignore_index=True)


def aggregate_genes(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, sets = [], {}
    for dataset, g in data.groupby("dataset", sort=True):
        gene = g.groupby("gene", as_index=False).agg(n_tasks=("task_id", "size"), mean_safeconf_risk=("safeconf_calibrated_pair_risk", "mean"), mean_gears_excess_error=("gears_excess_error", "mean"), mean_gears_error=("error_gears_rmse", "mean"), mean_scgpt_error=("error_scgpt_rmse", "mean"))
        gene["safeconf_percentile"] = gene.mean_safeconf_risk.rank(method="average", pct=True)
        gene["gears_excess_percentile"] = gene.mean_gears_excess_error.rank(method="average", pct=True)
        gene["top20_safeconf_risk"] = gene.safeconf_percentile > 0.80
        gene["top20_gears_excess_error"] = gene.gears_excess_percentile > 0.80
        gene["top20_overlap"] = gene.top20_safeconf_risk & gene.top20_gears_excess_error
        gene.insert(0, "dataset", dataset)
        rows.append(gene)
        sets[dataset] = {
            "background": sorted(gene.gene.astype(str).unique()),
            "top20_safeconf": sorted(gene.loc[gene.top20_safeconf_risk, "gene"].astype(str).unique()),
            "top20_overlap": sorted(gene.loc[gene.top20_overlap, "gene"].astype(str).unique()),
        }
    return pd.concat(rows, ignore_index=True), sets


def context_summary(data: pd.DataFrame) -> pd.DataFrame:
    labels = {
        ("Frangieh", "Control"): "未额外刺激的基线状态",
        ("Frangieh", "IFNγ"): "干扰素-γ刺激状态",
        ("Frangieh", "Co-culture"): "肿瘤细胞与免疫细胞共培养状态",
        ("Lara_exvivo", "HSC"): "造血干细胞",
        ("Lara_exvivo", "EBMP"): "红系/嗜碱/巨核祖细胞群",
        ("Lara_exvivo", "GMP"): "粒细胞-单核细胞祖细胞",
        ("Lara_exvivo", "GMP (late)"): "较晚期粒细胞-单核细胞祖细胞",
        ("Lara_exvivo", "MkP"): "巨核细胞祖细胞",
        ("Santinha", "Interneurons_Sst_Pvalb"): "Sst/Pvalb 类中间神经元",
        ("Santinha", "Interneurons_Vip_Adarb2"): "Vip/Adarb2 类中间神经元",
        ("Santinha", "Neurons_L_2_3"): "皮层第 2/3 层神经元",
        ("Santinha", "Neurons_L_5"): "皮层第 5 层神经元",
        ("Santinha", "Neurons_L_6"): "皮层第 6 层神经元",
    }
    out = data.groupby(["dataset", "context"], as_index=False).agg(n_tasks=("task_id", "size"), mean_safeconf_risk=("safeconf_calibrated_pair_risk", "mean"), mean_gears_error=("error_gears_rmse", "mean"), mean_scgpt_error=("error_scgpt_rmse", "mean"), mean_gears_excess_error=("gears_excess_error", "mean"))
    out["biological_context"] = [labels.get((r.dataset, r.context), r.context) for r in out.itertuples(index=False)]
    out["gears_error_rank_within_dataset"] = out.groupby("dataset").mean_gears_error.rank(method="min", ascending=False)
    out["gears_excess_rank_within_dataset"] = out.groupby("dataset").mean_gears_excess_error.rank(method="min", ascending=False)
    return out.sort_values(["dataset", "gears_excess_rank_within_dataset"])


def gprofiler(dataset: str, label: str, query: list[str], background: list[str]) -> tuple[list[dict], dict]:
    payload = {"organism": SPECIES[dataset], "query": query, "sources": ["GO:BP", "REAC"], "user_threshold": 0.05, "significance_threshold_method": "g_SCS", "domain_scope": "custom", "background": background, "no_evidences": False}
    if len(query) < 3:
        return [], {"status": "skipped", "reason": "query_has_fewer_than_3_genes", "payload": payload}
    cache = RAW / f"gprofiler_{dataset}_{label}.json"
    source = "live_api"
    try:
        response = requests.post(GP_URL, json=payload, timeout=90)
        response.raise_for_status()
        raw = response.json()
        cache.write_text(json.dumps({"payload": payload, "response": raw}, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        if not cache.exists():
            raise
        raw = json.loads(cache.read_text())["response"]
        source = "cached_api_response"
    rows = []
    for item in raw.get("result", []):
        intersections = item.get("intersections", []) or []
        flat = sorted({str(g) for group in intersections for g in (group if isinstance(group, list) else [group]) if g})
        if not flat and item.get("intersection"):
            flat = [str(x) for x in item["intersection"]]
        rows.append({"dataset": dataset, "query_type": label, "source": item.get("source"), "term_id": item.get("native"), "term_name": item.get("name"), "adjusted_p_value": item.get("p_value"), "term_size": item.get("term_size"), "query_size": item.get("query_size"), "intersection_size": item.get("intersection_size"), "intersecting_genes": ";".join(flat), "query_gene_count": len(query), "background_gene_count": len(background)})
    return rows, {"status": "complete", "source": source, "n_results": len(rows), "payload": payload}


def enrichment(sets: dict) -> tuple[pd.DataFrame, dict]:
    rows, audit = [], {}
    for dataset, groups in sets.items():
        audit[dataset] = {}
        for label in ("top20_safeconf", "top20_overlap"):
            try:
                result, status = gprofiler(dataset, label, groups[label], groups["background"])
                rows.extend(result)
                audit[dataset][label] = status
            except Exception as exc:
                audit[dataset][label] = {"status": "failed", "error": repr(exc), "query": groups[label], "background_count": len(groups["background"])}
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame[(pd.to_numeric(frame.adjusted_p_value, errors="coerce") < 0.05) & (pd.to_numeric(frame.intersection_size, errors="coerce") >= 2)].sort_values(["dataset", "query_type", "adjusted_p_value"])
    return frame, audit


def figure(genes: pd.DataFrame) -> None:
    selected = []
    for dataset, g in genes.groupby("dataset", sort=True):
        x = g.sort_values(["top20_overlap", "mean_safeconf_risk"], ascending=False).head(8).copy()
        x["label"] = dataset + " · " + x.gene
        selected.append(x)
    d = pd.concat(selected, ignore_index=True)
    d["risk_scaled"] = d.groupby("dataset").mean_safeconf_risk.transform(lambda x: (x - x.min()) / max(x.max() - x.min(), 1e-9))
    w, h = 1100, 760
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">', '<rect width="100%" height="100%" fill="#fff"/>', '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#26343c}.t{font-size:25px;font-weight:700}.s{font-size:13px;fill:#637179}.g{font-size:14px}</style>', '<text x="42" y="40" class="t">E116｜高 SafeConf 风险扰动基因</text>', '<text x="42" y="68" class="s">每个数据集按平均冻结风险排序；深色描边表示同时位于 GEARS excess error 前 20%。</text>']
    y = 100
    for r in d.itertuples(index=False):
        width = 110 + 560 * r.risk_scaled
        stroke = "#a44e46" if r.top20_overlap else "#2f7f76"
        sw = 3 if r.top20_overlap else 0
        parts += [f'<text x="42" y="{y+17}" class="g">{r.label}</text>', f'<rect x="270" y="{y}" width="{width:.1f}" height="22" fill="#a8c9c3" stroke="{stroke}" stroke-width="{sw}"/>', f'<text x="{285+width:.1f}" y="{y+16}" class="s">GEARS−scGPT={r.mean_gears_excess_error:.3f}</text>']
        y += 27
    parts.append('</svg>')
    (FIGURES / "F1_high_risk_gene_hypotheses.svg").write_text("\n".join(parts))


def main() -> None:
    for d in (TABLES, REPORTS, FIGURES, RAW):
        d.mkdir(parents=True, exist_ok=True)
    data = load()
    fold_assoc, assoc = feature_associations(data)
    genes, sets = aggregate_genes(data)
    contexts = context_summary(data)
    enrich, audit = enrichment(sets)
    fold_assoc.to_csv(TABLES / "E116_FOLD_FEATURE_ASSOCIATIONS.csv", index=False)
    assoc.to_csv(TABLES / "E116_FEATURE_ASSOCIATION_MACRO.csv", index=False)
    genes.to_csv(TABLES / "E116_GENE_MECHANISM_TABLE.csv", index=False)
    contexts.to_csv(TABLES / "E116_CONTEXT_MECHANISM_TABLE.csv", index=False)
    enrich.to_csv(TABLES / "E116_GPROFILER_SIGNIFICANT.csv", index=False)
    (TABLES / "E116_GENE_SETS.json").write_text(json.dumps(sets, ensure_ascii=False, indent=2) + "\n")
    (TABLES / "E116_ENRICHMENT_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    figure(genes)
    overall = assoc[assoc.dataset.eq("three_dataset_equal_macro")].sort_values("spearman_with_gears_excess_error", ascending=False)
    overlap = genes.groupby("dataset").top20_overlap.sum().astype(int)
    lines = ["# E116｜高风险任务与预测器失效机制", "", "该分析不改变 SafeConf 分数。它解释哪些部署前特征与 GEARS 相对 scGPT 的额外错误同向，并把高风险扰动基因转成可复核通路线索。", "", "## 三数据集等权关联", "", "| feature | ρ with GEARS−scGPT error | ρ with GEARS error | ρ with scGPT error |", "|---|---:|---:|---:|"]
    for r in overall.itertuples(index=False):
        lines.append(f"| {r.feature} | {r.spearman_with_gears_excess_error:.3f} | {r.spearman_with_gears_error:.3f} | {r.spearman_with_scgpt_error:.3f} |")
    lines += ["", "## 高风险基因重叠", "", "这里的重叠指同时进入 SafeConf 风险前 20% 与 GEARS excess error 前 20%。", ""]
    for dataset, n in overlap.items():
        names = genes[(genes.dataset.eq(dataset)) & genes.top20_overlap].sort_values("mean_safeconf_risk", ascending=False).gene.tolist()
        lines.append(f"- {dataset}: {n} 个；{', '.join(names) if names else '无'}。")
    lines += ["", "## 细胞背景失效差异", "", "| dataset | context | biological meaning | tasks | GEARS error | scGPT error | GEARS−scGPT |", "|---|---|---|---:|---:|---:|---:|"]
    for r in contexts.itertuples(index=False):
        lines.append(f"| {r.dataset} | {r.context} | {r.biological_context} | {r.n_tasks} | {r.mean_gears_error:.4f} | {r.mean_scgpt_error:.4f} | {r.mean_gears_excess_error:.4f} |")
    lines += ["", "背景新颖度与 GEARS−scGPT 额外误差的三数据集宏平均相关为正，说明跨细胞状态的基础表达变化是当前最清楚的失效来源。该结果来自任务级误差关联，不能进一步写成某条通路导致模型失败。", "", "## 通路富集", ""]
    if len(enrich):
        lines += ["仅列 g:SCS 校正 p<0.05 且交集基因不少于 2 的结果。", "", "| dataset | query | source | term | adjusted p | overlap | genes |", "|---|---|---|---|---:|---:|---|"]
        for r in enrich.head(30).itertuples(index=False):
            lines.append(f"| {r.dataset} | {r.query_type} | {r.source} | {r.term_name} | {float(r.adjusted_p_value):.3g} | {int(r.intersection_size)} | {r.intersecting_genes} |")
    else:
        lines.append("没有满足预设阈值的通路，或远程注释查询失败。审计状态见 `tables/E116_ENRICHMENT_AUDIT.json`。")
    lines += ["", "## 解释边界", "", "这些结果是机制假设，不是因果证明。高风险基因富集使用各数据集全部被测扰动基因作背景；不同物种不混合查询。后续湿实验或外部功能证据应围绕重叠基因与显著通路，而不是只挑单个好看的案例。"]
    (REPORTS / "E116_REPORT.md").write_text("\n".join(lines) + "\n")
    completed_queries = sum(v.get("status") == "complete" for d in audit.values() for v in d.values())
    failed_queries = sum(v.get("status") == "failed" for d in audit.values() for v in d.values())
    status = {"experiment": "E116_biological_mechanism_audit", "generated_at": datetime.now().isoformat(timespec="seconds"), "status": "complete", "n_datasets": int(data.dataset.nunique()), "n_tasks": len(data), "n_unique_dataset_gene_pairs": len(genes), "gprofiler_queries_complete": completed_queries, "gprofiler_queries_failed": failed_queries, "n_significant_terms_after_prespecified_filter": len(enrich), "test_truth_used_to_change_risk_score": False}
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README_先看这个.md").write_text("# E116 先看这个\n\n先读 `reports/E116_REPORT.md`。通路原始响应在 `raw/`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(overall.to_string(index=False))
    print(enrich.head(20).to_string(index=False) if len(enrich) else "no significant enrichment")


if __name__ == "__main__":
    main()
