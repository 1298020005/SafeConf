#!/usr/bin/env python3
"""Build the Jiang24 metadata-only task inventory after the D0 raw audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


PRIMARY_CELL_TYPES = ("k562", "mcf7", "ht29", "hap1")
PRIMARY_TREATMENTS = ("IFNG", "INS", "TGFB")
EXPECTED_H5AD_SHAPE = (1_628_476, 15_476)
EXPECTED_SPLIT_SHA256 = (
    "5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d"
)


class InventoryFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--d0-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    import anndata as ad

    args = parse_args()
    h5ad_path = args.h5ad.resolve()
    split_path = args.split.resolve()
    audit_path = args.d0_audit.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise InventoryFailure(f"refusing to overwrite: {output_dir}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "SCHEMA_REVIEW_REQUIRED"
        or audit.get("gates", {}).get("file_integrity") is not True
        or audit.get("gates", {}).get("split_h5ad_alignment") is not True
        or audit.get("gates", {}).get("expression_values_read") is not False
    ):
        raise InventoryFailure("D0 raw audit gates are incomplete")
    if sha256_file(split_path) != EXPECTED_SPLIT_SHA256:
        raise InventoryFailure("official split changed")

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if (adata.n_obs, adata.n_vars) != EXPECTED_H5AD_SHAPE:
            raise InventoryFailure("Jiang24 matrix shape changed")
        required = {"condition", "cell_type", "treatment"}
        if not required.issubset(adata.obs.columns):
            raise InventoryFailure(
                f"required metadata columns missing: {sorted(required-set(adata.obs.columns))}"
            )
        if not adata.obs_names.is_unique:
            raise InventoryFailure("H5AD cell barcodes are not unique")
        metadata = adata.obs.loc[:, ["condition", "cell_type", "treatment"]].copy()
        metadata.index = metadata.index.astype(str)
    finally:
        adata.file.close()

    split = pd.read_csv(
        split_path,
        header=None,
        names=["cell_barcode", "split"],
        dtype=str,
    )
    if split.cell_barcode.duplicated().any() or set(split.split) != {
        "train",
        "val",
        "test",
    }:
        raise InventoryFailure("official split uniqueness or labels changed")
    split = split.set_index("cell_barcode")
    if set(split.index) != set(metadata.index):
        raise InventoryFailure("split and H5AD barcode sets differ")
    metadata["split"] = split.loc[metadata.index, "split"].to_numpy()
    metadata = metadata.reset_index(names="cell_barcode")
    for column in ("condition", "cell_type", "treatment", "split"):
        metadata[column] = metadata[column].astype(str)
    metadata["is_control"] = metadata.condition.eq("control")
    metadata["is_single_gene"] = (
        ~metadata.is_control & ~metadata.condition.str.contains("+", regex=False)
    )
    metadata["is_primary_context"] = (
        metadata.cell_type.isin(PRIMARY_CELL_TYPES)
        & metadata.treatment.isin(PRIMARY_TREATMENTS)
    )

    task_counts = (
        metadata.groupby(
            ["split", "cell_type", "treatment", "condition", "is_control", "is_single_gene", "is_primary_context"],
            observed=True,
            sort=True,
        )
        .size()
        .rename("n_cells")
        .reset_index()
    )
    task_counts["context_id"] = (
        task_counts.cell_type.astype(str) + "|" + task_counts.treatment.astype(str)
    )
    context_counts = (
        task_counts.groupby(
            ["split", "cell_type", "treatment", "is_primary_context"],
            observed=True,
            sort=True,
        )
        .agg(
            n_cells=("n_cells", "sum"),
            n_conditions=("condition", "nunique"),
            n_single_gene_tasks=("is_single_gene", "sum"),
        )
        .reset_index()
    )
    primary = task_counts.loc[
        task_counts.split.eq("test")
        & task_counts.is_primary_context
        & task_counts.is_single_gene
    ].copy()
    primary["passes_ge30"] = primary.n_cells.ge(30)
    primary_contexts = set(zip(primary.cell_type, primary.treatment))
    expected_contexts = {
        (cell_type, treatment)
        for cell_type in PRIMARY_CELL_TYPES
        for treatment in PRIMARY_TREATMENTS
    }

    train_support = (
        task_counts.loc[task_counts.split.eq("train") & task_counts.is_single_gene]
        .groupby("condition", sort=True)
        .agg(
            n_train_cells=("n_cells", "sum"),
            n_train_contexts=("context_id", "nunique"),
            n_train_cell_types=("cell_type", "nunique"),
            n_train_treatments=("treatment", "nunique"),
        )
        .reset_index()
    )
    primary = primary.merge(
        train_support, on="condition", how="left", validate="many_to_one"
    )
    for column in (
        "n_train_cells",
        "n_train_contexts",
        "n_train_cell_types",
        "n_train_treatments",
    ):
        primary[column] = primary[column].fillna(0).astype(int)

    output_dir.mkdir(parents=True)
    atomic_csv(output_dir / "E208_ALL_TASK_COUNTS.csv", task_counts)
    atomic_csv(output_dir / "E208_CONTEXT_COUNTS.csv", context_counts)
    atomic_csv(output_dir / "E208_PRIMARY_TEST_TASKS.csv", primary)
    summary = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D0_metadata_task_inventory",
        "created_at": now(),
        "status": (
            "PASS"
            if primary_contexts == expected_contexts
            and len(primary) > 0
            and primary.passes_ge30.all()
            else "FAIL"
        ),
        "h5ad_path": str(h5ad_path),
        "h5ad_sha256": audit["h5ad"]["sha256"],
        "split_path": str(split_path),
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "matrix_shape": list(EXPECTED_H5AD_SHAPE),
        "schema": {
            "perturbation_key": "condition",
            "control_value": "control",
            "cell_context_key": "cell_type",
            "treatment_state_key": "treatment",
            "combination_delimiter": "+",
        },
        "primary_contexts": [
            {"cell_type": cell_type, "treatment": treatment}
            for cell_type, treatment in sorted(expected_contexts)
        ],
        "n_primary_contexts_observed": len(primary_contexts),
        "n_primary_single_gene_test_tasks": len(primary),
        "n_primary_tasks_ge30": int(primary.passes_ge30.sum()),
        "n_primary_unique_genes": int(primary.condition.nunique()),
        "n_primary_test_cells": int(primary.n_cells.sum()),
        "n_primary_tasks_without_train_support": int(primary.n_train_cells.eq(0).sum()),
        "expression_values_read": False,
        "target_truth_used_for_selection": False,
        "files": {},
    }
    for name in (
        "E208_ALL_TASK_COUNTS.csv",
        "E208_CONTEXT_COUNTS.csv",
        "E208_PRIMARY_TEST_TASKS.csv",
    ):
        path = output_dir / name
        summary["files"][name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    atomic_text(
        output_dir / "E208_TASK_INVENTORY_STATUS.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
