#!/usr/bin/env python3
"""E29: GEARS–scGPT shared Adamson risk audit.

E28 proved that GEARS and scGPT can be written under one strict
PredictionRecord contract on a shared Adamson task/gene/true-effect panel.

E29 expands that smoke to all available Adamson fold-1 single-gene GEARS tasks
from E25 and asks a narrower SafeConf question:

    When two real model families disagree on the same task, does that
    disagreement identify tasks with larger observed prediction error?

This is still a small evidence package, not a formal benchmark.  Its value is
contract quality and error-analysis logic: shared tasks, shared genes, shared
true effect, strict validator, and deployable risk scores.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code/20260426_154505_perturb_transport_final_push"
TOOLS_ROOT = PROJECT_ROOT / "tools/scripts"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))

import numpy as np
import pandas as pd
import torch

from safetrans_confidence.data.records import validate_prediction_record_artifacts
from run_e27_scgpt_forward_prediction_record_smoke import (
    _cosine_error,
    _load_checkpoint,
    _safe_corr,
)
from run_e28_gears_scgpt_shared_adamson_smoke import (
    ADAMSON_H5AD,
    E25_DIR,
    GEARS_ADAMSON_PROCESSED,
    _hash_gene_order,
    _load_gears_gene_order,
    _rel,
    _run_scgpt_forward,
    _select_adamson_subset,
    _select_gears_tasks,
    _subset_gears_predictions,
)


OUT_DIR = PROJECT_ROOT / "docs/实验结果/E29_gears_scgpt_shared_adamson_risk_audit_20260708"
N_TASKS = 7
N_GENES = 512
CELLS_PER_PERTURBATION = 8


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _git_head() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT)
        return bool(out.decode("utf-8").strip())
    except Exception:
        return True


def _z(values: pd.Series) -> pd.Series:
    arr = values.astype(float)
    std = float(arr.std(ddof=0))
    if not math.isfinite(std) or std <= 1e-12:
        return pd.Series(np.zeros(len(arr)), index=values.index, dtype=float)
    return (arr - float(arr.mean())) / std


def _spearman(x: pd.Series, y: pd.Series) -> float:
    return _safe_corr(x.to_numpy(dtype=float), y.to_numpy(dtype=float), "spearman")


def _pearson(x: pd.Series, y: pd.Series) -> float:
    return _safe_corr(x.to_numpy(dtype=float), y.to_numpy(dtype=float), "pearson")


def _build_records(
    gears_tasks: pd.DataFrame,
    selected_genes: list[str],
    scgpt_pred: dict[str, np.ndarray],
    shared_true: dict[str, np.ndarray],
    gears_pred: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gene_hash = _hash_gene_order(selected_genes)
    gene_panel_id = f"shared::adamson::gears_scgpt::n_genes_{len(selected_genes)}"
    pred_arrays: dict[str, np.ndarray] = {}
    true_arrays: dict[str, np.ndarray] = {}
    record_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []

    for gene in gears_tasks["perturbation_gene"].astype(str).tolist():
        true_vec = shared_true[gene].astype(np.float32)
        true_key = f"E29::Adamson::{gene}::shared_true"
        true_arrays[true_key] = true_vec
        predictors = [
            ("GEARS_formal_seed1_subset512", gears_pred[gene].astype(np.float32)),
            ("scGPT_whole_human_forward_subset512", scgpt_pred[gene].astype(np.float32)),
        ]

        per_task_metrics: list[dict[str, Any]] = []
        for predictor_name, pred_vec in predictors:
            record_id = f"E29::Adamson::{gene}::{predictor_name}"
            pred_key = record_id + "::pred"
            pred_arrays[pred_key] = pred_vec.astype(np.float32)
            rmse = float(np.sqrt(np.mean((pred_vec - true_vec) ** 2)))
            cosine = _cosine_error(pred_vec, true_vec)
            pearson = _safe_corr(pred_vec, true_vec, "pearson")
            spearman = _safe_corr(pred_vec, true_vec, "spearman")
            pred_l2 = float(np.linalg.norm(pred_vec))
            true_l2 = float(np.linalg.norm(true_vec))
            record_rows.append(
                {
                    "schema_version": "safeconf_prediction_record_v1",
                    "record_id": record_id,
                    "task_id": gene,
                    "task_key": f"Adamson::{gene}",
                    "dataset_name": "Adamson_GEARS_scGPT_shared_risk_audit",
                    "dataset_group": "adamson_crispr_shared_risk_audit",
                    "fold_id": 0,
                    "split": "test",
                    "context": "Adamson_shared_panel_risk_audit",
                    "perturbation": gene,
                    "predictor_name": predictor_name,
                    "run_type": "smoke",
                    "gene_panel_id": gene_panel_id,
                    "gene_order_hash": gene_hash,
                    "effect_definition": "mean_diff",
                    "normalization_id": "adamson_shared_expression_delta_subset_v1",
                    "error_normalization": "raw_rmse",
                    "predicted_effect_key": pred_key,
                    "true_effect_key": true_key,
                    "true_error_rmse": rmse,
                    "true_error_cosine": cosine,
                    "n_cells": CELLS_PER_PERTURBATION,
                }
            )
            row = {
                "perturbation": gene,
                "predictor_name": predictor_name,
                "rmse": rmse,
                "cosine_error": cosine,
                "pearson": pearson,
                "spearman": spearman,
                "pred_l2": pred_l2,
                "true_l2": true_l2,
            }
            metric_rows.append(row)
            per_task_metrics.append(row)

        gears_vec = predictors[0][1]
        scgpt_vec = predictors[1][1]
        disagreement_rmse = float(np.sqrt(np.mean((gears_vec - scgpt_vec) ** 2)))
        disagreement_cosine = _cosine_error(gears_vec, scgpt_vec)
        task_rows.append(
            {
                "perturbation": gene,
                "gears_rmse": per_task_metrics[0]["rmse"],
                "scgpt_rmse": per_task_metrics[1]["rmse"],
                "task_mean_rmse": float(np.mean([m["rmse"] for m in per_task_metrics])),
                "task_max_rmse": float(np.max([m["rmse"] for m in per_task_metrics])),
                "disagreement_rmse": disagreement_rmse,
                "disagreement_cosine": disagreement_cosine,
                "gears_pred_l2": per_task_metrics[0]["pred_l2"],
                "scgpt_pred_l2": per_task_metrics[1]["pred_l2"],
                "consensus_pred_l2_mean": float(np.mean([m["pred_l2"] for m in per_task_metrics])),
                "true_l2": float(np.linalg.norm(true_vec)),
            }
        )

    records = pd.DataFrame(record_rows)
    metrics = pd.DataFrame(metric_rows)
    task_scores = pd.DataFrame(task_rows)
    task_scores["risk_disagreement_z"] = _z(task_scores["disagreement_rmse"])
    task_scores["risk_disagreement_plus_pred_magnitude_z"] = (
        _z(task_scores["disagreement_rmse"]) + _z(task_scores["consensus_pred_l2_mean"])
    )
    task_scores["diagnostic_true_magnitude_z"] = _z(task_scores["true_l2"])
    task_scores = task_scores.sort_values(
        ["risk_disagreement_plus_pred_magnitude_z", "risk_disagreement_z"],
        ascending=False,
    ).reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    for score_col in [
        "risk_disagreement_z",
        "risk_disagreement_plus_pred_magnitude_z",
        "diagnostic_true_magnitude_z",
    ]:
        ranked = task_scores.sort_values(score_col, ascending=False).reset_index(drop=True)
        top_k = max(1, math.ceil(len(ranked) * 0.30))
        bottom_k = max(1, math.ceil(len(ranked) * 0.30))
        all_mean = float(ranked["task_mean_rmse"].mean())
        top_mean = float(ranked.head(top_k)["task_mean_rmse"].mean())
        bottom_mean = float(ranked.tail(bottom_k)["task_mean_rmse"].mean())
        summary_rows.append(
            {
                "score": score_col,
                "n_tasks": len(ranked),
                "target": "task_mean_rmse",
                "spearman": _spearman(ranked[score_col], ranked["task_mean_rmse"]),
                "pearson": _pearson(ranked[score_col], ranked["task_mean_rmse"]),
                "top_30pct_k": top_k,
                "top_30pct_mean_rmse": top_mean,
                "bottom_30pct_k": bottom_k,
                "bottom_30pct_mean_rmse": bottom_mean,
                "all_task_mean_rmse": all_mean,
                "top_vs_all_ratio": float(top_mean / all_mean) if all_mean else float("nan"),
                "top_vs_bottom_ratio": float(top_mean / bottom_mean) if bottom_mean else float("nan"),
            }
        )
    risk_summary = pd.DataFrame(summary_rows)
    return records, pred_arrays, true_arrays, metrics, task_scores, risk_summary


def _bar_svg(task_scores: pd.DataFrame) -> str:
    df = task_scores.sort_values("task_mean_rmse", ascending=False).reset_index(drop=True)
    width, height = 920, 360
    left, right, top, bottom = 108, 34, 34, 74
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_val = float(df[["task_mean_rmse", "disagreement_rmse"]].to_numpy().max())
    max_val = max(max_val, 1e-6)
    gap = 18
    group_w = chart_w / max(len(df), 1)
    bar_w = min(28, (group_w - gap) / 2)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="E29 task error and disagreement" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width-right}" y2="{top + chart_h}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#111827" stroke-width="1"/>',
        '<text x="18" y="22" font-size="14" fill="#374151">RMSE</text>',
        '<rect x="690" y="20" width="13" height="13" fill="#315C9B"/><text x="710" y="31" font-size="13" fill="#374151">task mean error</text>',
        '<rect x="690" y="42" width="13" height="13" fill="#B47A3C"/><text x="710" y="53" font-size="13" fill="#374151">GEARS–scGPT disagreement</text>',
    ]
    for i, row in df.iterrows():
        x0 = left + i * group_w + gap / 2
        err_h = chart_h * float(row["task_mean_rmse"]) / max_val
        dis_h = chart_h * float(row["disagreement_rmse"]) / max_val
        parts.append(
            f'<rect x="{x0:.1f}" y="{top + chart_h - err_h:.1f}" width="{bar_w:.1f}" height="{err_h:.1f}" fill="#315C9B"/>'
        )
        parts.append(
            f'<rect x="{x0 + bar_w + 4:.1f}" y="{top + chart_h - dis_h:.1f}" width="{bar_w:.1f}" height="{dis_h:.1f}" fill="#B47A3C"/>'
        )
        label = html.escape(str(row["perturbation"]))
        parts.append(
            f'<text transform="translate({x0 + bar_w:.1f},{top + chart_h + 18}) rotate(35)" font-size="12" fill="#374151">{label}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _write_outputs(
    status: dict[str, Any],
    manifest: pd.DataFrame,
    metrics: pd.DataFrame,
    task_scores: pd.DataFrame,
    risk_summary: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    report_md = f"""# E29 GEARS–scGPT shared Adamson risk audit

生成时间：{_now()}

## 先看结论

E29 把 E25 中 Adamson fold-1 的 7 个单基因任务全部放进 GEARS–scGPT 同任务合同中。

- PredictionRecords：{status['n_prediction_records']}
- Tasks：{status['n_tasks']}
- Genes：{status['n_genes']}
- strict issue_count：{status['strict_issue_count']}

最重要的变化：E29 不只检查格式，还计算了 GEARS 与 scGPT 在同一任务上的预测分歧，并把这个分歧作为可部署风险信号，与真实误差做小样本相关性检查。

边界也很清楚：n=7，只能作为严格合同下的风险审计 smoke，不能作为正式模型优劣结论。
"""
    (OUT_DIR / "reports/E29_GEARS_SCGPT_SHARED_ADAMSON_RISK_AUDIT_REPORT.md").write_text(
        report_md,
        encoding="utf-8",
    )

    css = """
body{margin:0;background:#fff;color:#1f2933;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:42px 28px 76px}
h1{font-size:30px;margin:0 0 10px}h2{font-size:21px;margin:34px 0 12px;border-top:1px solid #e5e7eb;padding-top:24px}
p{line-height:1.75;font-size:16px}.note{border-left:4px solid #315C9B;background:#f8fbff;padding:12px 16px;border-radius:8px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.card{border:1px solid #e5e7eb;border-radius:14px;padding:16px;background:#fafafa}
.k{font-size:26px;font-weight:760;color:#111827}.l{font-size:13px;color:#66788a;margin-top:4px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 22px}th,td{border-bottom:1px solid #e5e7eb;text-align:left;padding:8px 9px;vertical-align:top}th{background:#f7f7f7}
.svgbox{border:1px solid #e5e7eb;border-radius:14px;padding:12px;background:#fff;margin:14px 0 24px}
"""
    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>E29 GEARS scGPT shared Adamson risk audit</title><style>{css}</style></head>
<body><main>
<h1>E29 GEARS–scGPT shared Adamson risk audit</h1>
<p class="note">同任务、同 512-gene panel、同 true effect key 的双预测器风险审计。n=7，结论只用于证据链推进，不当作正式 benchmark。</p>
<div class="cards">
<div class="card"><div class="k">{status['n_prediction_records']}</div><div class="l">PredictionRecords</div></div>
<div class="card"><div class="k">{status['n_tasks']}</div><div class="l">Adamson tasks</div></div>
<div class="card"><div class="k">{status['n_genes']}</div><div class="l">shared genes</div></div>
<div class="card"><div class="k">{status['strict_issue_count']}</div><div class="l">strict issues</div></div>
</div>
<h2>图 1：任务误差与模型分歧</h2>
<p>蓝色是真实任务平均误差，棕色是 GEARS 与 scGPT 的预测分歧。这个图用来观察“两个模型意见不一致的地方，是否更容易出错”。</p>
<div class="svgbox">{_bar_svg(task_scores)}</div>
<h2>风险信号汇总</h2>{risk_summary.to_html(index=False, escape=False)}
<h2>任务级风险排序</h2>{task_scores.to_html(index=False, escape=False)}
<h2>预测器级指标</h2>{metrics.to_html(index=False, escape=False)}
<h2>任务清单</h2>{manifest.to_html(index=False, escape=False)}
<h2>严格校验</h2>{validation.to_html(index=False, escape=False)}
</main></body></html>
"""
    (OUT_DIR / "reports/E29_GEARS_SCGPT_SHARED_ADAMSON_RISK_AUDIT.html").write_text(
        html_text,
        encoding="utf-8",
    )
    (OUT_DIR / "README_先看这个.md").write_text(
        f"""# E29 GEARS–scGPT shared Adamson risk audit

先看结论：E29 将 Adamson fold-1 的全部 7 个可用 GEARS 单基因任务扩展成 GEARS/scGPT 双预测器 strict 合同，并新增任务级风险排序。

- PredictionRecords：{status['n_prediction_records']}
- Tasks：{status['n_tasks']}
- Genes：{status['n_genes']}
- strict issue：{status['strict_issue_count']}

核心表：

- `tables/E29_TASK_RISK_SCORES.csv`：每个扰动的 GEARS/scGPT 误差、模型分歧、风险排序。
- `tables/E29_RISK_AUDIT_SUMMARY.csv`：风险信号与任务平均误差的相关性和 top-risk enrichment。
- `reports/E29_GEARS_SCGPT_SHARED_ADAMSON_RISK_AUDIT.html`：可视化报告。

边界：n=7，只能说明同合同下的风险审计流程跑通，不能作为正式性能 benchmark。
""",
        encoding="utf-8",
    )


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for rel in ["tables", "arrays", "reports"]:
        (OUT_DIR / rel).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(5229)
    checkpoint = _load_checkpoint()
    gears_tasks = _select_gears_tasks(n_tasks=N_TASKS)
    gears_gene_order = _load_gears_gene_order()
    subset_payload = _select_adamson_subset(
        checkpoint["vocab"],
        gears_tasks,
        gears_gene_order,
        cells_per_pert=CELLS_PER_PERTURBATION,
        max_genes=N_GENES,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scgpt_pred, shared_true, scgpt_info = _run_scgpt_forward(checkpoint, subset_payload, device)
    gears_pred = _subset_gears_predictions(gears_tasks, subset_payload["selected_genes"], gears_gene_order)
    records, pred_arrays, true_arrays, metrics, task_scores, risk_summary = _build_records(
        gears_tasks,
        subset_payload["selected_genes"],
        scgpt_pred,
        shared_true,
        gears_pred,
    )

    records.to_csv(OUT_DIR / "tables/PREDICTION_RECORDS.csv", index=False)
    records.to_csv(OUT_DIR / "tables/E29_SHARED_PREDICTION_RECORDS.csv", index=False)
    np.savez_compressed(OUT_DIR / "arrays/predicted_effects.npz", **pred_arrays)
    np.savez_compressed(OUT_DIR / "arrays/true_effects.npz", **true_arrays)
    metrics.to_csv(OUT_DIR / "tables/E29_SHARED_PREDICTOR_METRICS.csv", index=False)
    task_scores.to_csv(OUT_DIR / "tables/E29_TASK_RISK_SCORES.csv", index=False)
    risk_summary.to_csv(OUT_DIR / "tables/E29_RISK_AUDIT_SUMMARY.csv", index=False)
    pd.DataFrame({"gene": subset_payload["selected_genes"]}).to_csv(
        OUT_DIR / "tables/E29_SHARED_GENE_PANEL.csv",
        index=False,
    )
    manifest = gears_tasks[
        ["fold_id", "perturbation", "perturbation_gene", "record_id", "predicted_effect_key"]
    ].copy()
    manifest["scgpt_cells"] = manifest["perturbation_gene"].map(
        subset_payload["cells_per_perturbation"]
    )
    manifest["shared_gene_count"] = len(subset_payload["selected_genes"])
    manifest.to_csv(OUT_DIR / "tables/E29_SHARED_TASK_MANIFEST.csv", index=False)

    issues = validate_prediction_record_artifacts(OUT_DIR, records=records, strict=True)
    validation = pd.DataFrame(
        [
            {
                "scope": "e29_gears_scgpt_shared_adamson_risk_audit",
                "strict": True,
                "issue_count": len(issues),
                "issues": "; ".join(issues),
            }
        ]
    )
    validation.to_csv(OUT_DIR / "tables/E29_SHARED_VALIDATION.csv", index=False)
    status = {
        "status": "ok" if not issues else "has_issues",
        "generated_at": _now(),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "out": os.path.relpath(OUT_DIR, PROJECT_ROOT),
        "device": str(device),
        "n_prediction_records": int(len(records)),
        "n_tasks": int(manifest.shape[0]),
        "n_genes": int(len(subset_payload["selected_genes"])),
        "perturbations": manifest["perturbation_gene"].astype(str).tolist(),
        "strict_issue_count": int(len(issues)),
        "strict_issues": issues,
        "cells_per_perturbation": subset_payload["cells_per_perturbation"],
        "scgpt_matched_key_count": scgpt_info["matched_key_count"],
        "scgpt_total_model_key_count": scgpt_info["total_model_key_count"],
        "sources": {
            "e25": os.path.relpath(E25_DIR, PROJECT_ROOT),
            "adamson_h5ad": _rel(ADAMSON_H5AD),
            "gears_processed": _rel(GEARS_ADAMSON_PROCESSED),
        },
        "risk_summary": risk_summary.to_dict(orient="records"),
    }
    (OUT_DIR / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_outputs(status, manifest, metrics, task_scores, risk_summary, validation)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
