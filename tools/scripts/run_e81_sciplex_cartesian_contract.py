#!/usr/bin/env python3
"""Freeze a leakage-safe sciPlex3 Cartesian split contract.

This script does not fit a predictor and never reads perturbed expression values.
It uses task labels/cell counts to freeze train/test membership.  The shared gene
panel is selected from vehicle-control cells only, which are explicitly allowed
at deployment for context characterization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/extra_official/"
    "cellular_context_generalization/sciplex3.h5ad"
)
OUT = ROOT / "docs/实验结果/E81_sciplex_cartesian_contract_20260712"


def stable_u01(*parts: object) -> float:
    payload = "||".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) / float(16**16)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_gene_panel(adata, n_genes: int) -> tuple[pd.DataFrame, str]:
    control_mask = adata.obs["perturbation"].astype(str).eq("control").to_numpy()
    if int(control_mask.sum()) == 0:
        raise RuntimeError("No control cells found")
    x = adata[control_mask].X
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        sq_mean = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    else:
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=0)
        sq_mean = np.square(x).mean(axis=0)
    var = np.maximum(sq_mean - np.square(mean), 0.0)
    names = np.asarray(adata.var_names.astype(str))
    order = np.lexsort((names, -mean, -var))[: min(n_genes, adata.n_vars)]
    selected = names[order]
    digest = "sha256:" + hashlib.sha256("\n".join(selected).encode()).hexdigest()
    panel = pd.DataFrame(
        {
            "panel_position": np.arange(len(order)),
            "gene_id": selected,
            "control_variance": var[order],
            "control_mean": mean[order],
            "selection_input": "vehicle_control_expression_only",
            "target_perturbed_truth_used": False,
            "gene_order_hash": digest,
        }
    )
    return panel, digest


def task_table(obs: pd.DataFrame, min_cells: int) -> pd.DataFrame:
    work = obs.loc[obs["perturbation"].astype(str).ne("control")].copy()
    work["context"] = work["cell_line"].astype(str)
    # Freeze whole base drugs as perturbation columns.  Dose is a separate task
    # axis, so another dose of the same drug cannot leak into a column holdout.
    work["perturbation_key"] = work["condition2"].astype(str)
    work["dose_key"] = work["dose"].astype(str)
    grouped = (
        work.groupby(["context", "perturbation_key", "dose_key"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    grouped["task_key"] = (
        grouped["context"] + "::" + grouped["perturbation_key"] + "::dose=" + grouped["dose_key"]
    )
    grouped["eligible"] = grouped["n_cells"] >= min_cells
    grouped["selection_inputs"] = "context_label+perturbation_label+cell_count"
    grouped["target_perturbed_expression_used_for_selection"] = False
    return grouped.sort_values(["context", "perturbation_key"]).reset_index(drop=True)


def make_manifest(
    tasks: pd.DataFrame,
    coverages: list[float],
    repeats: int,
    pair_holdout_fraction: float,
    seed: int,
) -> pd.DataFrame:
    eligible = tasks.loc[tasks["eligible"]].copy()
    contexts = sorted(eligible["context"].unique())
    perturbations = sorted(eligible["perturbation_key"].unique())
    if len(contexts) < 3:
        raise RuntimeError(f"Need >=3 contexts, found {contexts}")
    rows: list[dict] = []

    for repeat in range(repeats):
        held_context = contexts[repeat % len(contexts)]
        seen_contexts = [c for c in contexts if c != held_context]
        ranked_perts = sorted(
            perturbations,
            key=lambda p: (stable_u01(seed, repeat, "perturbation", p), p),
        )
        for coverage in coverages:
            n_seen = max(2, min(len(ranked_perts) - 1, round(len(ranked_perts) * coverage)))
            seen_perts = set(ranked_perts[:n_seen])
            block_tasks = eligible.loc[
                eligible["context"].isin(seen_contexts)
                & eligible["perturbation_key"].isin(seen_perts),
                "task_key",
            ].tolist()
            ranked_block = sorted(
                block_tasks,
                key=lambda task: (stable_u01(seed, repeat, coverage, "pair", task), task),
            )
            n_pair_test = max(4, round(len(ranked_block) * pair_holdout_fraction))
            n_pair_test = min(n_pair_test, len(ranked_block) - 1)
            pair_test_tasks = set(ranked_block[:n_pair_test])
            for rec in eligible.itertuples(index=False):
                context_seen = rec.context in seen_contexts
                perturbation_seen = rec.perturbation_key in seen_perts
                if context_seen and perturbation_seen:
                    is_pair_test = rec.task_key in pair_test_tasks
                    role = "test" if is_pair_test else "train"
                    quadrant = "seen_context_seen_perturbation_pair_holdout" if is_pair_test else "observed_submatrix_train"
                elif (not context_seen) and perturbation_seen:
                    role = "test"
                    quadrant = "new_context_seen_perturbation"
                elif context_seen and (not perturbation_seen):
                    role = "test"
                    quadrant = "seen_context_new_perturbation"
                else:
                    role = "test"
                    quadrant = "new_context_new_perturbation"
                rows.append(
                    {
                        "manifest_id": f"E81_r{repeat+1}_p{int(coverage*100):02d}",
                        "repeat": repeat + 1,
                        "selection_seed": seed,
                        "perturbation_coverage_target": coverage,
                        "context_coverage": len(seen_contexts) / len(contexts),
                        "pair_holdout_fraction_within_submatrix": pair_holdout_fraction,
                        "heldout_context": held_context,
                        "context": rec.context,
                        "perturbation_key": rec.perturbation_key,
                        "dose_key": rec.dose_key,
                        "task_key": rec.task_key,
                        "n_cells": rec.n_cells,
                        "role": role,
                        "quadrant": quadrant,
                        "context_seen_in_training": context_seen,
                        "perturbation_seen_in_training": perturbation_seen,
                        "target_control_available": True,
                        "target_perturbed_expression_used_for_split": False,
                    }
                )
    out = pd.DataFrame(rows)
    # Every frozen setting must have training rows and all four test quadrants.
    expected = {
        "observed_submatrix_train",
        "seen_context_seen_perturbation_pair_holdout",
        "new_context_seen_perturbation",
        "seen_context_new_perturbation",
        "new_context_new_perturbation",
    }
    for manifest_id, group in out.groupby("manifest_id"):
        missing = expected - set(group["quadrant"])
        if missing:
            raise RuntimeError(f"{manifest_id} missing quadrants: {sorted(missing)}")
        if group["task_key"].duplicated().any():
            raise RuntimeError(f"{manifest_id} contains duplicate task rows")
        train = group.loc[group["role"].eq("train")]
        test = group.loc[group["role"].eq("test")]
        held_context = str(group["heldout_context"].iloc[0])
        if train["context"].eq(held_context).any():
            raise RuntimeError(f"{manifest_id} leaks held-out context into training")
        unseen_drugs = set(test.loc[~test["perturbation_seen_in_training"], "perturbation_key"])
        if unseen_drugs & set(train["perturbation_key"]):
            raise RuntimeError(f"{manifest_id} leaks a held-out base drug into training")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--coverages", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pair-holdout-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "reports").mkdir(exist_ok=True)
    adata = sc.read_h5ad(DATA)
    panel, gene_hash = write_gene_panel(adata, args.n_genes)
    tasks = task_table(adata.obs, args.min_cells)
    manifest = make_manifest(
        tasks,
        args.coverages,
        args.repeats,
        args.pair_holdout_fraction,
        args.seed,
    )

    panel.to_csv(OUT / "tables/E81_GENE_PANEL.csv", index=False)
    tasks.to_csv(OUT / "tables/E81_TASK_MATRIX.csv", index=False)
    manifest.to_csv(OUT / "tables/E81_SPLIT_MANIFEST.csv", index=False)
    summary = (
        manifest.groupby(["manifest_id", "repeat", "perturbation_coverage_target", "heldout_context", "role", "quadrant"])
        .size()
        .rename("n_tasks")
        .reset_index()
    )
    summary.to_csv(OUT / "tables/E81_SPLIT_SUMMARY.csv", index=False)

    status = {
        "experiment": "E81_sciplex_cartesian_contract",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_h5ad": str(DATA),
        "source_shape": list(adata.shape),
        "n_contexts": int(tasks["context"].nunique()),
        "n_perturbations": int(tasks["perturbation_key"].nunique()),
        "n_doses": int(tasks["dose_key"].nunique()),
        "n_eligible_tasks": int(tasks["eligible"].sum()),
        "min_cells": args.min_cells,
        "gene_panel_size": len(panel),
        "gene_order_hash": gene_hash,
        "gene_panel_selection": "vehicle control expression only",
        "split_selection_inputs": ["context label", "perturbation label", "cell count", "fixed seed"],
        "target_perturbed_truth_used_for_gene_panel_or_split": False,
        "coverages": args.coverages,
        "repeats": args.repeats,
        "pair_holdout_fraction": args.pair_holdout_fraction,
        "n_manifest_rows": len(manifest),
        "n_manifests": int(manifest["manifest_id"].nunique()),
        "status": "frozen_before_predictor_training",
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")

    pivot = summary.pivot_table(
        index=["manifest_id", "heldout_context"], columns="quadrant", values="n_tasks", fill_value=0
    ).reset_index()
    report = f"""# E81｜sciPlex3 子矩阵四象限正式合同

本轮只冻结任务和基因面板，不训练模型。拆分只读取 context、perturbation、每个任务的细胞数和固定随机种子；1000 基因面板只由 vehicle control 表达选择。目标扰动后的表达没有参与任务选择、基因选择或分组。

- 数据：`{DATA}`
- 原始形状：{adata.n_obs:,} cells × {adata.n_vars:,} genes
- 合格任务：{int(tasks['eligible'].sum())}，context={tasks['context'].nunique()}，base drug={tasks['perturbation_key'].nunique()}，dose={tasks['dose_key'].nunique()}
- 冻结设置：{manifest['manifest_id'].nunique()}（3 个扰动覆盖度 × 3 个 held-out context）
- gene order hash：`{gene_hash}`

每个设置把任务分成一个训练子矩阵和四类测试任务：子矩阵内随机缺失 pair、新 context、新 base drug、context 与 base drug 同时未见。整列留出时，同一种药的所有剂量一起留出。后续 CPA/chemCPA 和基线只能读取 `role=train` 的 perturbed expression；测试真值封存到最终误差计算。

## 任务数

{markdown_table(pivot)}

## 文件

- `tables/E81_TASK_MATRIX.csv`
- `tables/E81_GENE_PANEL.csv`
- `tables/E81_SPLIT_MANIFEST.csv`
- `tables/E81_SPLIT_SUMMARY.csv`
"""
    (OUT / "reports/E81_CONTRACT_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text(
        "# E81 先看这个\n\n先读 `reports/E81_CONTRACT_REPORT.md`。这是模型训练前冻结的 sciPlex3 子矩阵四象限合同。\n"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
