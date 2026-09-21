#!/usr/bin/env python3
"""Build E208 SafeConf source effects from the official Jiang24 train layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


EXPECTED_H5_BYTES = 93_532_364_449
EXPECTED_SPLIT_SHA256 = "5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d"
EXPECTED_TASK_MANIFEST_SHA256 = "c499c7085bb6b32e4d275dc3007929a79f6617b66d2196686e36915d0ea78964"
EXPECTED_SOURCE_PAIRS = 238
EXPECTED_SOURCE_PERTURBED_CELLS = 210_325
EXPECTED_CONTROL_CONTEXTS = 30
EXPECTED_CONTROL_CELLS = 68_559


class SourceCacheFailure(RuntimeError):
    """A frozen train-source input or output contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values],
        dtype=str,
    )


def categorical(group: h5py.Group) -> np.ndarray:
    categories = decode(group["categories"][:])
    codes = group["codes"][:].astype(np.int64)
    if np.any(codes < 0) or np.any(codes >= len(categories)):
        raise SourceCacheFailure(f"invalid categorical codes: {group.name}")
    return categories[codes]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, value: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    value.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def consecutive_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    if len(rows) == 0:
        return []
    cuts = np.flatnonzero(np.diff(rows) != 1) + 1
    return [
        (int(block[0]), int(block[-1]) + 1)
        for block in np.split(rows, cuts)
    ]


def sparse_row_mean(
    rows: np.ndarray,
    indptr: np.ndarray,
    data: h5py.Dataset,
    indices: h5py.Dataset,
    n_genes: int,
) -> tuple[np.ndarray, int]:
    accumulator = np.zeros(n_genes, dtype=np.float64)
    runs = consecutive_runs(rows)
    for row_start, row_stop in runs:
        value_start = int(indptr[row_start])
        value_stop = int(indptr[row_stop])
        values = data[value_start:value_stop].astype(np.float64, copy=False)
        columns = indices[value_start:value_stop].astype(np.int64, copy=False)
        accumulator += np.bincount(columns, weights=values, minlength=n_genes)
    return (accumulator / len(rows)).astype(np.float32), len(runs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    h5ad = args.h5ad.expanduser().absolute()
    split_path = args.split.expanduser().absolute()
    tasks_path = args.tasks.expanduser().absolute()
    output = args.output_dir.expanduser().absolute()
    if h5ad.stat().st_size != EXPECTED_H5_BYTES:
        raise SourceCacheFailure("Jiang24 H5 byte count changed")
    if sha256_file(split_path) != EXPECTED_SPLIT_SHA256:
        raise SourceCacheFailure("Jiang24 split hash changed")
    if sha256_file(tasks_path) != EXPECTED_TASK_MANIFEST_SHA256:
        raise SourceCacheFailure("E208 frozen prediction task manifest changed")
    if output.exists():
        existing = [path for path in output.iterdir() if path.name != "BUILD.log"]
        if existing:
            raise SourceCacheFailure(f"refusing to overwrite source artifacts: {existing[0]}")

    tasks = pd.read_csv(tasks_path)
    target_genes = sorted(set(tasks.condition.astype(str)))
    if len(tasks) != 224 or len(target_genes) != 53:
        raise SourceCacheFailure("frozen E208 task or gene count changed")
    split = pd.read_csv(split_path, header=None, names=["cell_id", "split"])
    with h5py.File(h5ad, "r") as handle:
        cell_ids = decode(handle["obs/_index"][:])
        if len(cell_ids) != len(split) or not np.array_equal(
            cell_ids, split.cell_id.astype(str).to_numpy()
        ):
            raise SourceCacheFailure("split rows no longer align to H5 observations")
        condition = categorical(handle["obs/condition"])
        cell_type = categorical(handle["obs/cell_type"])
        treatment = categorical(handle["obs/treatment"])
        split_values = split["split"].astype(str).to_numpy()
        train = split_values == "train"
        control_rows = np.flatnonzero(train & (condition == "control"))
        source_rows = np.flatnonzero(
            train
            & np.isin(condition, target_genes)
            & np.asarray(["+" not in value for value in condition])
        )
        control_cells = pd.DataFrame(
            {
                "source_row": control_rows,
                "cell_type": cell_type[control_rows],
                "treatment": treatment[control_rows],
            }
        )
        source_cells = pd.DataFrame(
            {
                "source_row": source_rows,
                "cell_type": cell_type[source_rows],
                "treatment": treatment[source_rows],
                "condition": condition[source_rows],
            }
        )
        control_counts = (
            control_cells.groupby(["cell_type", "treatment"], sort=True)
            .size()
            .rename("n_control_cells")
            .reset_index()
        )
        source_manifest = (
            source_cells.groupby(
                ["cell_type", "treatment", "condition"], sort=True, observed=True
            )
            .size()
            .rename("n_perturbed_cells")
            .reset_index()
            .merge(
                control_counts,
                on=["cell_type", "treatment"],
                how="left",
                validate="many_to_one",
            )
        )
        source_manifest["source_context_id"] = (
            source_manifest.cell_type.astype(str)
            + "|"
            + source_manifest.treatment.astype(str)
        )
        if (
            len(source_manifest) != EXPECTED_SOURCE_PAIRS
            or len(source_cells) != EXPECTED_SOURCE_PERTURBED_CELLS
            or len(control_counts) != EXPECTED_CONTROL_CONTEXTS
            or len(control_cells) != EXPECTED_CONTROL_CELLS
            or source_manifest.condition.nunique() != 53
            or source_manifest.groupby("condition").source_context_id.nunique().min() < 2
            or source_manifest.n_control_cells.isna().any()
        ):
            raise SourceCacheFailure(
                "train source contract changed: "
                f"pairs={len(source_manifest)}, perturb_cells={len(source_cells)}, "
                f"controls={len(control_cells)}, contexts={len(control_counts)}"
            )

        matrix = handle["X"]
        if matrix.attrs.get("encoding-type") != "csr_matrix":
            raise SourceCacheFailure("Jiang24 X is no longer CSR")
        shape = tuple(map(int, matrix.attrs["shape"]))
        if shape != (1_628_476, 15_473):
            raise SourceCacheFailure(f"Jiang24 matrix shape changed: {shape}")
        indptr = matrix["indptr"][:].astype(np.int64)
        data = matrix["data"]
        indices = matrix["indices"]
        gene_names = decode(handle["var/_index"][:])
        control_means: dict[tuple[str, str], np.ndarray] = {}
        source_runs = 0
        for key, block in control_cells.groupby(["cell_type", "treatment"], sort=True):
            mean, runs = sparse_row_mean(
                block.source_row.to_numpy(np.int64), indptr, data, indices, len(gene_names)
            )
            control_means[(str(key[0]), str(key[1]))] = mean
            source_runs += runs
        effects = np.empty((len(source_manifest), len(gene_names)), dtype=np.float32)
        for row_index, row in enumerate(source_manifest.itertuples(index=False)):
            selected_rows = source_cells.loc[
                source_cells.cell_type.eq(row.cell_type)
                & source_cells.treatment.eq(row.treatment)
                & source_cells.condition.eq(row.condition),
                "source_row",
            ].to_numpy(np.int64)
            perturbed_mean, runs = sparse_row_mean(
                selected_rows, indptr, data, indices, len(gene_names)
            )
            source_runs += runs
            effects[row_index] = perturbed_mean - control_means[
                (str(row.cell_type), str(row.treatment))
            ]

    if not np.isfinite(effects).all():
        raise SourceCacheFailure("non-finite source effect values")
    output.mkdir(parents=True, exist_ok=True)
    source_manifest.insert(0, "effect_row", np.arange(len(source_manifest), dtype=int))
    effects_path = output / "E208_SOURCE_EFFECTS.npy"
    manifest_path = output / "E208_SOURCE_EFFECT_MANIFEST.csv"
    genes_path = output / "E208_SOURCE_GENE_AXIS.csv"
    atomic_npy(effects_path, effects)
    atomic_csv(manifest_path, source_manifest)
    atomic_csv(genes_path, pd.DataFrame({"gene_index": np.arange(len(gene_names)), "gene_name": gene_names}))
    status = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_PRETRUTH_SOURCE_EFFECT_CACHE",
        "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_h5ad": str(h5ad),
        "source_h5ad_bytes": h5ad.stat().st_size,
        "source_split_sha256": EXPECTED_SPLIT_SHA256,
        "task_manifest_sha256": EXPECTED_TASK_MANIFEST_SHA256,
        "n_target_genes": len(target_genes),
        "n_source_effects": len(source_manifest),
        "n_source_contexts_with_effects": int(source_manifest.source_context_id.nunique()),
        "n_all_train_control_contexts": len(control_counts),
        "n_source_perturbed_cells": len(source_cells),
        "n_train_control_cells": len(control_cells),
        "minimum_source_contexts_per_gene": int(
            source_manifest.groupby("condition").source_context_id.nunique().min()
        ),
        "sparse_source_row_runs": source_runs,
        "effects": {"path": str(effects_path), "sha256": sha256_file(effects_path)},
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "gene_axis": {"path": str(genes_path), "sha256": sha256_file(genes_path)},
        "train_expression_rows_read": len(source_cells) + len(control_cells),
        "validation_expression_rows_read": 0,
        "test_perturbed_expression_rows_read": 0,
        "target_truth_access": "NOT_AUTHORIZED",
    }
    atomic_json(output / "E208_SOURCE_EFFECT_CACHE_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
