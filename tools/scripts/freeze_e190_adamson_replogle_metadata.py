#!/usr/bin/env python3
"""Freeze E190 task identities using H5AD metadata only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729"
SOURCE = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "AdamsonWeissman2016_GSM2406681_10X010.h5ad"
)
TARGET = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "ReplogleWeissman2022_K562_essential.h5ad"
)
MIN_TARGET_CELLS = 5
MIN_TARGET_CONTROLS = 20
N_FOLDS = 5


def stable_key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def source_gene(label: str) -> str:
    return str(label).split("_", 1)[0].replace("(mod)", "")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = ad.read_h5ad(SOURCE, backed="r")
    target = ad.read_h5ad(TARGET, backed="r")
    try:
        source_obs = source.obs.copy()
        target_obs = target.obs.copy()
        source_obs.insert(0, "source_cell_id", source_obs.index.astype(str))
        source_obs.insert(0, "source_row_index", range(len(source_obs)))
        target_obs.insert(0, "target_cell_id", target_obs.index.astype(str))
        target_obs.insert(0, "target_row_index", range(len(target_obs)))

        source_cells = source_obs.loc[
            source_obs.nperts.astype(str).eq("2")
            & source_obs.perturbation.notna()
            & ~source_obs.perturbation.astype(str).eq("*"),
            ["source_row_index", "source_cell_id", "perturbation"],
        ].copy()
        source_cells["gene"] = source_cells.perturbation.astype(str).map(source_gene)
        target_cells = target_obs.loc[
            target_obs.nperts.astype(str).eq("1"),
            ["target_row_index", "target_cell_id", "batch", "gene"],
        ].copy()
        target_controls = target_obs.loc[
            target_obs.nperts.astype(str).eq("0"),
            ["target_row_index", "target_cell_id", "batch", "gene"],
        ].copy()

        control_counts = target_controls.groupby("batch", observed=True).size()
        eligible_batches = set(
            control_counts.loc[control_counts >= MIN_TARGET_CONTROLS].index.astype(str)
        )
        target_cells["batch"] = target_cells.batch.astype(str)
        target_cells["gene"] = target_cells.gene.astype(str)
        task_counts = (
            target_cells.groupby(["batch", "gene"], observed=True)
            .size()
            .rename("n_target_cells")
            .reset_index()
        )
        task_counts = task_counts.loc[
            task_counts.batch.astype(str).isin(eligible_batches)
            & task_counts.n_target_cells.ge(MIN_TARGET_CELLS)
        ].copy()

        source_genes = set(source_cells.gene.astype(str))
        source_vars = set(source.var_names.astype(str))
        target_vars = set(target.var_names.astype(str))
        eligible_genes = sorted(
            set(task_counts.gene.astype(str))
            & source_genes
            & source_vars
            & target_vars
        )
        task_counts = task_counts.loc[
            task_counts.gene.astype(str).isin(eligible_genes)
        ].copy()
        task_counts["task_id"] = (
            "REPL::"
            + task_counts.batch.astype(str)
            + "::"
            + task_counts.gene.astype(str)
        )
        task_counts = task_counts.sort_values(["gene", "batch"]).reset_index(drop=True)
        if (
            len(eligible_genes) != 47
            or len(task_counts) != 692
            or task_counts.task_id.duplicated().any()
        ):
            raise RuntimeError(
                f"frozen E190 metadata counts changed: genes={len(eligible_genes)}, "
                f"tasks={len(task_counts)}"
            )

        source_cells = source_cells.loc[
            source_cells.gene.astype(str).isin(eligible_genes)
        ].copy()
        fold_rows = []
        for (gene, guide), group in source_cells.groupby(
            ["gene", "perturbation"], observed=True, sort=True
        ):
            ordered = group.assign(
                selection_key=[
                    stable_key("E190", str(gene), str(guide), str(cell_id))
                    for cell_id in group.source_cell_id
                ]
            ).sort_values("selection_key")
            if len(ordered) < N_FOLDS:
                raise RuntimeError(f"source guide has fewer than five cells: {guide}")
            ordered = ordered.copy()
            ordered["fold"] = [
                index % N_FOLDS for index in range(len(ordered))
            ]
            ordered["split"] = ordered.fold.map(
                lambda value: "validation" if value == N_FOLDS - 1 else "train"
            )
            fold_rows.append(ordered)
        source_assignments = pd.concat(fold_rows, ignore_index=True)
        if (
            source_assignments.gene.nunique() != 47
            or source_assignments.perturbation.nunique() != 54
        ):
            raise RuntimeError("source gene/guide counts changed")

        eligible_task_ids = set(task_counts.task_id.astype(str))
        target_cells["task_id"] = (
            "REPL::"
            + target_cells.batch.astype(str)
            + "::"
            + target_cells.gene.astype(str)
        )
        target_assignments = target_cells.loc[
            target_cells.task_id.astype(str).isin(eligible_task_ids)
        ].copy()
        observed_counts = target_assignments.groupby("task_id", observed=True).size()
        expected_counts = task_counts.set_index("task_id").n_target_cells.astype(int)
        if not observed_counts.sort_index().equals(expected_counts.sort_index()):
            raise RuntimeError("target task cell assignments do not match frozen counts")
        target_controls = target_controls.loc[
            target_controls.batch.astype(str).isin(eligible_batches)
        ].copy()

        pd.DataFrame(
            {
                "gene": eligible_genes,
                "selection_rule": "metadata overlap and >=1 eligible target batch",
                "metadata_selection_sha256": [
                    stable_key("E190", "gene", gene) for gene in eligible_genes
                ],
            }
        ).to_csv(OUT / "E190_SELECTED_GENES.csv", index=False)
        task_counts.to_csv(OUT / "E190_QUERY_MANIFEST.csv", index=False)
        source_assignments.to_csv(
            OUT / "E190_SOURCE_CELL_FOLD_ASSIGNMENTS.csv", index=False
        )
        target_assignments.to_csv(
            OUT / "E190_TARGET_CELL_TASK_ASSIGNMENTS.csv", index=False
        )
        target_controls.to_csv(
            OUT / "E190_TARGET_CONTROL_ASSIGNMENTS.csv", index=False
        )
        status = {
            "experiment": "E190",
            "stage": "METADATA_FREEZE",
            "status": "PASS",
            "source_file": str(SOURCE),
            "target_file": str(TARGET),
            "source_shape": list(source.shape),
            "target_shape": list(target.shape),
            "n_common_eligible_genes": len(eligible_genes),
            "n_source_guides": source_assignments.perturbation.nunique(),
            "n_source_selected_cells": len(source_assignments),
            "n_target_tasks": len(task_counts),
            "n_target_selected_cells": len(target_assignments),
            "n_target_control_cells": len(target_controls),
            "n_target_batches": task_counts.batch.nunique(),
            "source_x_values_read": 0,
            "target_x_values_read": 0,
            "selection_used_expression_values": False,
        }
        (OUT / "METADATA_FREEZE_STATUS.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))
    finally:
        source.file.close()
        target.file.close()


if __name__ == "__main__":
    main()
