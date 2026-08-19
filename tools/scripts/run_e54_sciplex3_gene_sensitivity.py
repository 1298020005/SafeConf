#!/usr/bin/env python3
"""E54 sciplex3 gene-count sensitivity.

The E50 result is strong at 1000 genes.  This script checks whether the
cell-line holdout signal is stable when the effect vector uses 1000/3000/5000
genes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "scripts"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_e42_e48_local_first_batch_smoke import (  # noqa: E402
    PATHS,
    build_context_tasks,
    leave_context_splits,
    save_csv,
    score_splits,
    summarize_scores,
)


OUT = ROOT / "docs" / "实验结果" / "E54_sciplex3_gene_sensitivity_20260710"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode().strip())
    except Exception:
        return True


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def run_one(n_genes: int, min_cells: int, max_cells_per_group: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = f"sciplex3_cell_line_gene{n_genes}"
    tasks, meta = build_context_tasks(
        path=PATHS["sciplex3"],
        dataset=dataset,
        context_col="cell_line",
        perturbation_col="perturbation",
        n_genes=n_genes,
        min_cells=min_cells,
        max_cells_per_group=max_cells_per_group,
        seed=seed,
    )
    splits = leave_context_splits(tasks)
    scores, status = score_splits(dataset, tasks, splits)
    summary = summarize_scores(scores)
    meta_df = pd.DataFrame([{**meta, "status": "ok" if tasks else "no_tasks"}])
    status = pd.concat([meta_df, status], ignore_index=True, sort=False)
    return scores, summary, status


def write_report(summary: pd.DataFrame, status: pd.DataFrame, gene_stability: pd.DataFrame) -> None:
    lines = []
    lines.append("# E54 sciplex3 基因数敏感性\n")
    lines.append(f"- 生成时间：{now_text()}")
    lines.append(f"- Git：`{git_head()[:12]}`")
    lines.append(f"- 工作区 dirty：`{git_dirty()}`\n")
    lines.append("## 1. 稳定性汇总\n")
    lines.append(gene_stability.to_string(index=False) if not gene_stability.empty else "暂无")
    lines.append("\n## 2. Top summary\n")
    top = summary.sort_values("spearman", ascending=False).head(20) if not summary.empty else summary
    lines.append(top.to_string(index=False) if not top.empty else "暂无")
    lines.append("\n## 3. 构建状态\n")
    lines.append(status.to_string(index=False) if not status.empty else "暂无")
    (REPORTS / "E54_SCIPLEX3_GENE_SENSITIVITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E54 sciplex3 基因数敏感性\n\n"
        "先看 `reports/E54_SCIPLEX3_GENE_SENSITIVITY_REPORT.md`。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-counts", default="1000,3000,5000")
    parser.add_argument("--min-cells", type=int, default=15)
    parser.add_argument("--max-cells-per-group", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ensure_dirs()
    gene_counts = [int(x.strip()) for x in args.gene_counts.split(",") if x.strip()]
    score_frames = []
    summary_frames = []
    status_frames = []
    for n_genes in gene_counts:
        scores, summary, status = run_one(n_genes, args.min_cells, args.max_cells_per_group, args.seed)
        scores["n_genes_setting"] = n_genes
        summary["n_genes_setting"] = n_genes
        status["n_genes_setting"] = n_genes
        score_frames.append(scores)
        summary_frames.append(summary)
        status_frames.append(status)
    scores_all = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    summary_all = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    status_all = pd.concat(status_frames, ignore_index=True) if status_frames else pd.DataFrame()
    key = summary_all[
        summary_all["risk_score_name"].isin(["risk_safeconf_smoke", "risk_predicted_magnitude", "risk_disagreement"])
        & summary_all["target_error"].eq("error_mean_rmse")
    ].copy()
    stability = key[
        [
            "n_genes_setting",
            "risk_score_name",
            "n_tasks",
            "spearman",
            "top20_k",
            "top20_mean_error",
            "top20_enrichment",
        ]
    ].sort_values(["risk_score_name", "n_genes_setting"], kind="stable")
    save_csv(scores_all, TABLES / "E54_SCIPLEX3_GENE_SENSITIVITY_SCORE_TABLE.csv")
    save_csv(summary_all, TABLES / "E54_SCIPLEX3_GENE_SENSITIVITY_SUMMARY.csv")
    save_csv(status_all, TABLES / "E54_SCIPLEX3_GENE_SENSITIVITY_STATUS.csv")
    save_csv(stability, TABLES / "E54_SCIPLEX3_GENE_STABILITY_KEY_ROWS.csv")
    write_report(summary_all, status_all, stability)
    status = {
        "generated_at": now_text(),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "gene_counts": gene_counts,
        "n_score_rows": int(len(scores_all)),
        "n_summary_rows": int(len(summary_all)),
        "output_dir": "docs/实验结果/E54_sciplex3_gene_sensitivity_20260710",
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
