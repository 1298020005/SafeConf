#!/usr/bin/env python3
"""E88: freeze an independent sciPlex3 -> sciPlex4 transfer contract."""

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
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "SrivatsanTrapnell2020_sciplex3.h5ad"
)
TARGET = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "SrivatsanTrapnell2020_sciplex4.h5ad"
)
OUT = ROOT / "docs/实验结果/E88_sciplex3_to_sciplex4_contract_20260712"
SEED = 20268800
DRUG_ALIASES = {
    "Abexinostat (PCI-24781)": "Abexinostat",
    "Pracinostat (SB939)": "Pracinostat",
}


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def sample_indices(indices: np.ndarray, n: int, *seed_parts: object) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= n:
        return np.sort(indices)
    rng = np.random.default_rng(stable_seed(SEED, *seed_parts))
    return np.sort(rng.choice(indices, n, replace=False))


def read_columns(adata, rows: np.ndarray, positions: np.ndarray):
    positions = np.asarray(positions, dtype=int)
    order = np.argsort(positions)
    restore = np.argsort(order)
    matrix = adata.X[rows][:, positions[order]][:, restore]
    return sparse.csr_matrix(matrix, dtype=np.float32)


def log_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.copy().astype(np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(1e4, totals, out=np.zeros_like(totals, dtype=np.float32), where=totals > 0)
    matrix = sparse.diags(scale).dot(matrix).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix


def control_statistics(matrix: sparse.csr_matrix):
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    second = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second - np.square(mean), 0)
    detection = np.asarray((matrix > 0).mean(axis=0)).ravel()
    return variance, detection


def main() -> None:
    source = sc.read_h5ad(SOURCE, backed="r")
    target = sc.read_h5ad(TARGET, backed="r")
    so, to = source.obs.copy(), target.obs.copy()

    shared = np.array(sorted(set(source.var_names.astype(str)) & set(target.var_names.astype(str))))
    sp = source.var_names.astype(str).get_indexer(shared)
    tp = target.var_names.astype(str).get_indexer(shared)

    source_control = (
        so["perturbation"].astype(str).eq("control")
        & so["cell_line"].astype(str).isin(["A549", "MCF7", "K562"])
        & so["time"].astype(float).eq(24.0)
    ).to_numpy()
    target_control = (
        to["perturbation"].astype(str).eq("control")
        & to["perturbation_2"].astype(str).eq("control")
        & to["cell_line"].astype(str).isin(["A549", "MCF7"])
    ).to_numpy()
    source_control_rows = np.concatenate(
        [
            sample_indices(
                np.flatnonzero(source_control & so["cell_line"].astype(str).eq(context).to_numpy()),
                512,
                "source_control",
                context,
            )
            for context in ["A549", "MCF7", "K562"]
        ]
    )
    target_control_rows = np.concatenate(
        [
            sample_indices(
                np.flatnonzero(target_control & to["cell_line"].astype(str).eq(context).to_numpy()),
                512,
                "target_control",
                context,
            )
            for context in ["A549", "MCF7"]
        ]
    )
    source_controls = log_normalize(read_columns(source, source_control_rows, sp))
    target_controls = log_normalize(read_columns(target, target_control_rows, tp))
    source_var, source_detect = control_statistics(source_controls)
    target_var, target_detect = control_statistics(target_controls)
    eligible = (source_detect >= 0.05) & (target_detect >= 0.05)
    if eligible.sum() < 1000:
        raise RuntimeError(f"Only {eligible.sum()} common genes pass control detection")
    source_rank = stats.rankdata(source_var, method="average") / len(source_var)
    target_rank = stats.rankdata(target_var, method="average") / len(target_var)
    joint_rank = (source_rank + target_rank) / 2
    candidates = np.flatnonzero(eligible)
    selected = candidates[np.argsort(-joint_rank[candidates])[:1000]]
    genes = shared[selected]
    gene_hash = "sha256:" + hashlib.sha256("\n".join(genes).encode()).hexdigest()
    panel = pd.DataFrame(
        {
            "gene_order": np.arange(len(genes)),
            "gene_id": genes,
            "source_control_variance": source_var[selected],
            "target_control_variance": target_var[selected],
            "source_control_detection": source_detect[selected],
            "target_control_detection": target_detect[selected],
            "joint_control_variance_rank": joint_rank[selected],
            "gene_order_hash": gene_hash,
            "target_perturbed_expression_used_for_selection": False,
        }
    )

    rows = []
    for raw_drug, drug in DRUG_ALIASES.items():
        mask = (
            so["perturbation"].astype(str).eq(raw_drug)
            & so["cell_line"].astype(str).isin(["A549", "MCF7", "K562"])
            & so["time"].astype(float).eq(24.0)
        )
        for (context, dose), group in so.loc[mask].groupby(["cell_line", "dose_value"], observed=True):
            rows.append(
                {
                    "dataset_name": "sciPlex3_official_raw",
                    "role": "source_train",
                    "context": str(context),
                    "drug": drug,
                    "dose_nM": float(dose),
                    "n_cells": len(group),
                    "time_hours": 24.0,
                    "task_key": f"sciPlex3::{context}::{drug}::dose_nM={float(dose)}",
                    "dose_seen_in_source": True,
                    "context_seen_in_source": True,
                    "drug_seen_in_source": True,
                    "target_perturbed_expression_used_for_selection": False,
                }
            )
    source_doses = {row["dose_nM"] for row in rows}
    target_mask = (
        to["perturbation"].astype(str).eq("control")
        & to["perturbation_2"].astype(str).isin(DRUG_ALIASES.values())
        & to["cell_line"].astype(str).isin(["A549", "MCF7"])
        & to["dose_value_2"].astype(float).gt(0)
    )
    for (context, drug, dose_um), group in to.loc[target_mask].groupby(
        ["cell_line", "perturbation_2", "dose_value_2"], observed=True
    ):
        dose_nm = float(dose_um) * 1000.0
        rows.append(
            {
                "dataset_name": "sciPlex4_official_raw",
                "role": "target_test",
                "context": str(context),
                "drug": str(drug),
                "dose_nM": dose_nm,
                "n_cells": len(group),
                "time_hours": np.nan,
                "task_key": f"sciPlex4::{context}::{drug}::dose_nM={dose_nm}",
                "dose_seen_in_source": dose_nm in source_doses,
                "context_seen_in_source": True,
                "drug_seen_in_source": True,
                "target_perturbed_expression_used_for_selection": False,
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["role", "context", "drug", "dose_nM"]).reset_index(drop=True)
    source_rows = manifest[manifest["role"].eq("source_train")]
    target_rows = manifest[manifest["role"].eq("target_test")]
    if len(source_rows) != 24 or len(target_rows) != 28:
        raise RuntimeError(f"Unexpected task counts: source={len(source_rows)}, target={len(target_rows)}")
    if manifest["task_key"].duplicated().any() or (manifest["n_cells"] < 20).any():
        raise RuntimeError("Task uniqueness or cell-count contract failed")

    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT / "tables/E88_GENE_PANEL.csv", index=False)
    manifest.to_csv(OUT / "tables/E88_TRANSFER_MANIFEST.csv", index=False)
    status = {
        "experiment": "E88_sciplex3_to_sciplex4_contract",
        "phase": "contract_frozen_target_truth_unread",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_tasks": len(source_rows),
        "target_tasks": len(target_rows),
        "source_contexts": int(source_rows["context"].nunique()),
        "target_contexts": int(target_rows["context"].nunique()),
        "shared_drugs": int(target_rows["drug"].nunique()),
        "target_exact_dose_tasks": int(target_rows["dose_seen_in_source"].sum()),
        "target_interpolated_dose_tasks": int((~target_rows["dose_seen_in_source"]).sum()),
        "shared_genes_before_filter": len(shared),
        "gene_panel_size": len(panel),
        "gene_order_hash": gene_hash,
        "gene_selection_uses_controls_only": True,
        "target_perturbed_expression_used": False,
        "target_dose_unit_interpretation": "dose_value_2 is uM; converted to nM",
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    report = f"""# E88｜sciPlex3 → sciPlex4 同族外部合同

这个合同使用同一研究系列中的两个独立筛选文件。源域为 sciPlex3，目标域为 sciPlex4；只保留两边都出现的 Abexinostat、Pracinostat，以及 A549、MCF7 共同细胞系。sciPlex3 额外提供 K562 源训练任务。

- 源训练任务：{len(source_rows)}（3 个 context × 2 个药物 × 4 个剂量）
- 目标测试任务：{len(target_rows)}（2 个 context × 2 个药物 × 7 个剂量）
- 目标精确剂量任务：{int(target_rows['dose_seen_in_source'].sum())}
- 目标插值剂量任务：{int((~target_rows['dose_seen_in_source']).sum())}
- 共同基因：{len(shared)}；冻结 panel：1000
- panel 选择：两个数据集的 control-only 检出率与方差；未读取 sciPlex4 扰动表达
- gene hash：`{gene_hash}`

E87 检验的是无共享药物、无共享细胞体系的强外推；E88 检验独立实验批次间的可迁移性。E88 任务数只有 28，定位为同族外部复核，统计结果必须给 bootstrap 区间，不能替代大规模跨域证据。
"""
    (OUT / "reports/E88_CONTRACT_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E88 先看这个\n\n先读 `reports/E88_CONTRACT_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
