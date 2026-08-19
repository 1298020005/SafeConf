#!/usr/bin/env python3
"""E86: freeze sciPlex3 -> OpenProblems cross-dataset chemical contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/extra_official/"
    "cellular_context_generalization/sciplex3.h5ad"
)
TARGET = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/mega_external/"
    "OpenProblems_NeurIPS2023_single_cell_perturbations/data/raw/sc_counts_processed.h5ad"
)
OUT = ROOT / "docs/实验结果/E86_sciplex_to_openproblems_contract_20260712"


def moments(x):
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    else:
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=0)
        sq = np.square(x).mean(axis=0)
    return mean, np.maximum(sq - np.square(mean), 0.0)


def rank01(values: np.ndarray) -> np.ndarray:
    return stats.rankdata(values, method="average") / len(values)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "reports").mkdir(exist_ok=True)
    source = sc.read_h5ad(SOURCE, backed="r")
    target = sc.read_h5ad(TARGET, backed="r")
    common = sorted(set(source.var_names.astype(str)) & set(target.var_names.astype(str)))
    s_pos = source.var_names.astype(str).get_indexer(common)
    t_pos = target.var_names.astype(str).get_indexer(common)
    s_control = source.obs["perturbation"].astype(str).eq("control").to_numpy()
    t_control = target.obs["control"].astype(bool).to_numpy()
    s_mean, s_var = moments(source.X[s_control][:, s_pos])
    t_mean, t_var = moments(target.X[t_control][:, t_pos])
    score = (rank01(s_var) + rank01(t_var)) / 2.0
    order = np.lexsort((np.asarray(common), -score))[:1000]
    genes = np.asarray(common)[order]
    gene_hash = "sha256:" + hashlib.sha256("\n".join(genes).encode()).hexdigest()
    panel = pd.DataFrame(
        {
            "panel_position": np.arange(len(genes)),
            "gene_id": genes,
            "source_control_variance": s_var[order],
            "target_control_variance": t_var[order],
            "source_control_mean": s_mean[order],
            "target_control_mean": t_mean[order],
            "selection_score_mean_control_variance_rank": score[order],
            "selection_input": "source_and_target_vehicle_controls_only",
            "target_perturbed_truth_used": False,
            "gene_order_hash": gene_hash,
        }
    )
    panel.to_csv(OUT / "tables/E86_GENE_PANEL.csv", index=False)

    so = source.obs.copy()
    so = so.loc[so["perturbation"].astype(str).ne("control")].copy()
    so["context"] = so["cell_line"].astype(str)
    so["drug"] = so["condition2"].astype(str)
    so["dose_nM"] = so["dose"].astype(float)
    source_tasks = so.groupby(["context", "drug", "dose_nM"], observed=True).size().rename("n_cells").reset_index()
    source_tasks["dataset_name"] = "sciPlex3"
    source_tasks["role"] = "source_train"
    source_tasks["task_key"] = (
        "sciPlex3::" + source_tasks["context"] + "::" + source_tasks["drug"] + "::dose_nM=" + source_tasks["dose_nM"].astype(str)
    )

    to = target.obs.copy()
    to = to.loc[~to["control"].astype(bool)].copy()
    to["context"] = to["cell_type"].astype(str)
    to["drug"] = to["sm_name"].astype(str)
    to["dose_nM"] = to["dose_uM"].astype(float) * 1000.0
    target_tasks = to.groupby(["context", "drug", "dose_nM"], observed=True).size().rename("n_cells").reset_index()
    target_tasks = target_tasks.loc[target_tasks["n_cells"] >= 20].copy()
    target_tasks["dataset_name"] = "OpenProblems2023"
    target_tasks["role"] = "target_test"
    target_tasks["task_key"] = (
        "OpenProblems2023::" + target_tasks["context"] + "::" + target_tasks["drug"] + "::dose_nM=" + target_tasks["dose_nM"].astype(str)
    )
    tasks = pd.concat([source_tasks, target_tasks], ignore_index=True)
    tasks["dose_cpa"] = np.log10(np.maximum(tasks["dose_nM"].astype(float), 1.0))
    tasks["target_perturbed_expression_used_for_selection"] = False
    tasks.to_csv(OUT / "tables/E86_CROSS_DATASET_MANIFEST.csv", index=False)

    source_drugs = set(source_tasks["drug"])
    target_drugs = set(target_tasks["drug"])
    summary = pd.DataFrame(
        [
            {"item": "common_genes", "value": len(common)},
            {"item": "frozen_gene_panel", "value": len(genes)},
            {"item": "source_tasks", "value": len(source_tasks)},
            {"item": "target_tasks", "value": len(target_tasks)},
            {"item": "source_contexts", "value": source_tasks["context"].nunique()},
            {"item": "target_contexts", "value": target_tasks["context"].nunique()},
            {"item": "source_drugs", "value": len(source_drugs)},
            {"item": "target_drugs", "value": len(target_drugs)},
            {"item": "shared_drugs_by_exact_name", "value": len(source_drugs & target_drugs)},
        ]
    )
    summary.to_csv(OUT / "tables/E86_SUMMARY.csv", index=False)
    status = {
        "experiment": "E86_sciplex_to_openproblems_contract",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_h5ad": str(SOURCE),
        "target_h5ad": str(TARGET),
        "source_shape": list(source.shape),
        "target_shape": list(target.shape),
        "common_genes": len(common),
        "gene_panel_size": len(genes),
        "gene_order_hash": gene_hash,
        "source_tasks": len(source_tasks),
        "target_tasks": len(target_tasks),
        "target_min_cells": 20,
        "target_controls_allowed": True,
        "target_perturbed_truth_used_for_gene_or_task_selection": False,
        "status": "frozen_before_predictor_training",
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    report = f"""# E86｜sciPlex3 → OpenProblems 跨数据集合同

来源域只使用 sciPlex3 的 perturbed cells；目标域 OpenProblems 的 perturbed cells 全部封存到评价阶段。目标域 vehicle control 可以用于描述新 context。1000 基因面板只按两域 control variance 的平均秩选择，任务只按标签和细胞数筛选。

{markdown_table(summary)}

这是 chemical→chemical 的强迁移：来源是 3 个癌细胞系与 9 种药，目标是 4 类 PBMC 与 141 种药，精确同名药物重叠为 {len(source_drugs & target_drugs)}。模型需要同时处理新实验、新 context 和几乎全部新药；负结果也具有解释价值。

- `tables/E86_GENE_PANEL.csv`
- `tables/E86_CROSS_DATASET_MANIFEST.csv`
"""
    (OUT / "reports/E86_CONTRACT_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E86 先看这个\n\n先读 `reports/E86_CONTRACT_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

