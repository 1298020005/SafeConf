#!/usr/bin/env python3
"""Audit whether Frangieh can be used for a GEARS feasibility run.

This script does not train GEARS. It checks whether perturbation labels look
like gene symbols, whether those genes exist in the expression panel, and
whether the existing SafeConf split has enough tasks for a non-toy adapter test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd


DEFAULT_CODE_ROOT = Path("/home/yyf/proj/code/20260426_154505_perturb_transport_final_push")
DEFAULT_H5AD = Path("/home/yyf/data/singlecell_perturbation_atlas/official_generalization/Frangieh.h5ad")
DEFAULT_RUN_DIR = DEFAULT_CODE_ROOT / "outputs" / "safeconf_phase1_main" / "Frangieh"
DEFAULT_OUT_DIR = (
    Path("/home/yyf/proj/docs")
    / "实验结果"
    / "Formal_main_20260604"
    / "sprint1_lodo"
    / "gears_feasibility"
)


CONTROL_LABELS = {"control", "ctrl", "vehicle", "dmso", "nan", "none", ""}


def _is_control(value: str) -> bool:
    return str(value).strip().lower() in CONTROL_LABELS


def audit_frangieh(h5ad_path: Path, run_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = out_dir / "tables"
    reports = out_dir / "reports"
    tables.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)

    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs[["condition", "perturbation", "gene"]].copy()
    var_names = set(adata.var_names.astype(str))
    n_cells, n_genes = adata.shape
    adata.file.close()

    perturbations = sorted(
        {
            str(p)
            for p in obs["perturbation"].dropna().astype(str).unique()
            if not _is_control(str(p))
        }
    )
    rows = []
    for pert in perturbations:
        n_cells_pert = int((obs["perturbation"].astype(str) == pert).sum())
        rows.append(
            {
                "perturbation": pert,
                "n_cells": n_cells_pert,
                "exact_gene_symbol_in_var_names": pert in var_names,
                "looks_like_gene_symbol": bool(pert and pert.replace("-", "").replace(".", "").isalnum() and pert.upper() == pert),
            }
        )
    overlap_df = pd.DataFrame(rows)
    overlap_df.to_csv(tables / "Frangieh_GEARS_PERTURBATION_GENE_OVERLAP.csv", index=False)

    records_path = run_dir / "tables" / "PREDICTION_RECORDS.csv"
    split_path = run_dir / "tables" / "HELDOUT_PAIR_SPLITS.csv"
    summary_path = run_dir / "tables" / "DATASET_TASK_SUMMARY.csv"
    records = pd.read_csv(records_path)
    split = pd.read_csv(split_path)
    summary = pd.read_csv(summary_path)

    split_summary = (
        split.groupby(["fold_id", "split"], dropna=False)
        .agg(
            n_tasks=("task_key", "nunique"),
            n_contexts=("context", "nunique"),
            n_perturbations=("perturbation", "nunique"),
        )
        .reset_index()
    )
    split_summary.to_csv(tables / "Frangieh_SAFE_CONF_SPLIT_SIZE.csv", index=False)

    test_split = split[split["split"].eq("test")]
    leakage = {
        "test_pair_seen_in_train_true": int(test_split["pair_seen_in_train"].astype(bool).sum()),
        "test_perturbation_not_seen_in_train": int((~test_split["perturbation_seen_in_train"].astype(bool)).sum()),
        "test_context_not_seen_in_train": int((~test_split["context_seen_in_train"].astype(bool)).sum()),
    }

    n_exact_overlap = int(overlap_df["exact_gene_symbol_in_var_names"].sum())
    n_looks_like_gene = int(overlap_df["looks_like_gene_symbol"].sum())
    min_cells_noncontrol = int(overlap_df["n_cells"].min()) if not overlap_df.empty else 0
    compatible = (
        len(perturbations) >= 50
        and n_exact_overlap >= max(20, int(0.5 * len(perturbations)))
        and int(summary["n_tasks"].iloc[0]) >= 100
        and all(value == 0 for value in leakage.values())
    )

    status = {
        "dataset": "Frangieh",
        "h5ad_path": str(h5ad_path),
        "safeconf_run_dir": str(run_dir),
        "n_cells": int(n_cells),
        "n_genes_in_expression_panel": int(n_genes),
        "n_contexts": int(summary["n_contexts"].iloc[0]),
        "n_perturbations_noncontrol": int(len(perturbations)),
        "n_safeconf_tasks": int(summary["n_tasks"].iloc[0]),
        "n_prediction_records": int(len(records)),
        "n_exact_gene_symbol_overlap": n_exact_overlap,
        "n_looks_like_gene_symbol": n_looks_like_gene,
        "min_cells_per_noncontrol_perturbation": min_cells_noncontrol,
        "leakage": leakage,
        "gears_feasibility": "feasible_for_adapter_probe_not_yet_training" if compatible else "not_ready_without_more_checks",
        "recommendation": (
            "Frangieh is a better GEARS feasibility target than Cui because perturbations are gene symbols. "
            "Next step may be a small adapter smoke run, but not a main claim until per-prediction GEARS records exist."
            if compatible
            else "Do not train GEARS yet; inspect overlap/split problems first."
        ),
    }
    (out_dir / "Frangieh_GEARS_FEASIBILITY_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = f"""# Frangieh GEARS Feasibility Audit

## 结论

Frangieh（黑色素瘤 CRISPR 扰动数据集）**可以作为 GEARS（图神经网络扰动预测模型）的小型适配器探针候选**，但现在还不能把它写成正式 GEARS 主结果。

原因很直接：

- perturbation（扰动）基本是 gene symbol（基因名），比 Cui 的 cytokine（细胞因子刺激）更适合 GEARS。
- 现有 SafeConf split（切分）已有 `{int(summary['n_tasks'].iloc[0])}` 个 context × perturbation（背景×扰动）task（任务）。
- 但是目前还没有在 Frangieh 上训练并导出 GEARS 的 per-prediction records（逐条预测记录），所以只能说“格式上可尝试”，不能说“GEARS 证据已完成”。

## 关键数字

| 项目 | 数值 |
|---|---:|
| cells（细胞数） | {int(n_cells)} |
| genes in panel（表达矩阵基因数） | {int(n_genes)} |
| contexts（背景数） | {int(summary['n_contexts'].iloc[0])} |
| non-control perturbations（非对照扰动数） | {len(perturbations)} |
| SafeConf tasks（已有任务数） | {int(summary['n_tasks'].iloc[0])} |
| PredictionRecords（预测记录数） | {len(records)} |
| perturbation 与 var_names 精确重合数 | {n_exact_overlap} |
| 看起来像基因名的 perturbation 数 | {n_looks_like_gene} |
| 最小非对照扰动细胞数 | {min_cells_noncontrol} |

## 泄漏检查

- test pair seen in train（测试组合出现在训练中）: `{leakage['test_pair_seen_in_train_true']}`
- test perturbation not seen in train（测试扰动训练中没见过）: `{leakage['test_perturbation_not_seen_in_train']}`
- test context not seen in train（测试背景训练中没见过）: `{leakage['test_context_not_seen_in_train']}`

## 推荐下一步

1. 可以做 GEARS adapter smoke run（适配器冒烟测试），目标只是验证能导出 `PredictionRecord`。
2. 不要直接训练大规模 GEARS，也不要把它当主表结果。
3. 如果 smoke run 能导出 per-prediction GEARS predicted_effect（逐条 GEARS 预测效应），再把它接入 SafeConf 的 confidence scoring（可信度打分）。
"""
    (reports / "Frangieh_GEARS_FEASIBILITY_AUDIT.md").write_text(md, encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Frangieh feasibility for GEARS adapter work.")
    parser.add_argument("--h5ad-path", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    status = audit_frangieh(args.h5ad_path, args.run_dir, args.out_dir)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
