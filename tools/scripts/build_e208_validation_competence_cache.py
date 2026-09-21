#!/usr/bin/env python3
"""Build the frozen Jiang24 validation package for the E208 competence gate."""

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


PRIMARY_CELL_TYPES = {"hap1", "ht29", "k562", "mcf7"}
PRIMARY_TREATMENTS = {"IFNG", "INS", "TGFB"}
SELECTION_PREFIX = "E208-validation-control-v1\0"
EXPECTED_H5_BYTES = 93_532_364_449
EXPECTED_SPLIT_SHA256 = "5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d"
EXPECTED_TASKS = 216
EXPECTED_TASK_CELLS = 181_412
EXPECTED_CONTROL_CANDIDATES = 7_852
EXPECTED_CONTROL_SELECTED = 6_794


class ValidationCacheFailure(RuntimeError):
    """The validation package no longer satisfies the frozen contract."""


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
        raise ValidationCacheFailure(f"invalid categorical codes: {group.name}")
    return categories[codes]


def hash_key(cell_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_PREFIX}{cell_id}".encode()).hexdigest()


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
    """Return half-open consecutive source-row intervals."""
    if len(rows) == 0:
        return []
    cuts = np.flatnonzero(np.diff(rows) != 1) + 1
    blocks = np.split(rows, cuts)
    return [(int(block[0]), int(block[-1]) + 1) for block in blocks]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    h5ad = args.h5ad.expanduser().absolute()
    split_path = args.split.expanduser().absolute()
    output = args.output_dir.expanduser().absolute()
    if h5ad.stat().st_size != EXPECTED_H5_BYTES:
        raise ValidationCacheFailure("Jiang24 H5 byte count changed")
    if sha256_file(split_path) != EXPECTED_SPLIT_SHA256:
        raise ValidationCacheFailure("Jiang24 split hash changed")
    if output.exists() and any(output.iterdir()):
        raise ValidationCacheFailure(f"refusing to overwrite nonempty directory: {output}")

    split = pd.read_csv(split_path, header=None, names=["cell_id", "split"])
    with h5py.File(h5ad, "r") as handle:
        cell_ids = decode(handle["obs/_index"][:])
        if len(cell_ids) != len(split) or not np.array_equal(
            cell_ids, split.cell_id.astype(str).to_numpy()
        ):
            raise ValidationCacheFailure("split rows no longer align to H5 observations")
        condition = categorical(handle["obs/condition"])
        cell_type = categorical(handle["obs/cell_type"])
        treatment = categorical(handle["obs/treatment"])
        split_values = split["split"].astype(str).to_numpy()
        primary = np.isin(cell_type, sorted(PRIMARY_CELL_TYPES)) & np.isin(
            treatment, sorted(PRIMARY_TREATMENTS)
        )
        validation = split_values == "val"
        control_mask = validation & primary & (condition == "control")
        task_mask = (
            validation
            & primary
            & (condition != "control")
            & np.asarray(["+" not in value for value in condition])
        )

        control_rows = np.flatnonzero(control_mask)
        control_table = pd.DataFrame(
            {
                "source_row": control_rows,
                "cell_id": cell_ids[control_rows],
                "cell_type": cell_type[control_rows],
                "treatment": treatment[control_rows],
            }
        )
        chosen_blocks = []
        candidate_counts = {}
        for (cell, state), block in control_table.groupby(
            ["cell_type", "treatment"], sort=True
        ):
            candidate_counts[f"{cell}|{state}"] = len(block)
            block = block.assign(_hash=block.cell_id.map(hash_key)).sort_values(
                ["_hash", "cell_id"], kind="mergesort"
            )
            chosen_blocks.append(block.head(1000).drop(columns="_hash"))
        chosen = (
            pd.concat(chosen_blocks, ignore_index=True)
            .sort_values("source_row", kind="mergesort")
            .reset_index(drop=True)
        )
        if len(control_table) != EXPECTED_CONTROL_CANDIDATES or len(chosen) != EXPECTED_CONTROL_SELECTED:
            raise ValidationCacheFailure(
                f"validation control counts changed: {len(control_table)}, {len(chosen)}"
            )
        if len(chosen.groupby(["cell_type", "treatment"])) != 12:
            raise ValidationCacheFailure("validation controls do not cover 12 contexts")

        task_source_rows = np.flatnonzero(task_mask)
        task_cells = pd.DataFrame(
            {
                "source_row": task_source_rows,
                "cell_type": cell_type[task_source_rows],
                "treatment": treatment[task_source_rows],
                "condition": condition[task_source_rows],
            }
        )
        task_table = (
            task_cells.groupby(
                ["cell_type", "treatment", "condition"], sort=True, observed=True
            )
            .size()
            .rename("n_truth_cells")
            .reset_index()
        )
        task_table["task_id"] = (
            task_table.cell_type.astype(str)
            + "|"
            + task_table.treatment.astype(str)
            + "::"
            + task_table.condition.astype(str)
        )
        if (
            len(task_table) != EXPECTED_TASKS
            or len(task_cells) != EXPECTED_TASK_CELLS
            or task_table.task_id.nunique() != EXPECTED_TASKS
            or len(task_table[["cell_type", "treatment"]].drop_duplicates()) != 12
        ):
            raise ValidationCacheFailure(
                f"validation task contract changed: tasks={len(task_table)}, cells={len(task_cells)}"
            )

        matrix = handle["X"]
        if matrix.attrs.get("encoding-type") != "csr_matrix":
            raise ValidationCacheFailure("Jiang24 X is no longer CSR")
        shape = tuple(map(int, matrix.attrs["shape"]))
        if shape != (1_628_476, 15_473):
            raise ValidationCacheFailure(f"Jiang24 matrix shape changed: {shape}")
        source_indptr = matrix["indptr"][:].astype(np.int64)
        source_data = matrix["data"]
        source_indices = matrix["indices"]
        gene_names = decode(handle["var/_index"][:])

        control_data_blocks: list[np.ndarray] = []
        control_index_blocks: list[np.ndarray] = []
        control_indptr = np.zeros(len(chosen) + 1, dtype=np.int64)
        for output_row, source_row in enumerate(chosen.source_row.to_numpy(int)):
            start, stop = int(source_indptr[source_row]), int(source_indptr[source_row + 1])
            row_data = source_data[start:stop].astype(np.float32, copy=False)
            row_indices = source_indices[start:stop].astype(np.int32, copy=False)
            control_data_blocks.append(row_data)
            control_index_blocks.append(row_indices)
            control_indptr[output_row + 1] = control_indptr[output_row] + len(row_data)
        control_data = np.concatenate(control_data_blocks)
        control_indices = np.concatenate(control_index_blocks)

        truth_centroids = np.zeros((len(task_table), len(gene_names)), dtype=np.float32)
        truth_rows_read = 0
        number_of_runs = 0
        for task_index, task in enumerate(task_table.itertuples(index=False)):
            rows_for_task = task_cells.loc[
                task_cells.cell_type.eq(task.cell_type)
                & task_cells.treatment.eq(task.treatment)
                & task_cells.condition.eq(task.condition),
                "source_row",
            ].to_numpy(np.int64)
            accumulator = np.zeros(len(gene_names), dtype=np.float64)
            for row_start, row_stop in consecutive_runs(rows_for_task):
                number_of_runs += 1
                value_start = int(source_indptr[row_start])
                value_stop = int(source_indptr[row_stop])
                values = source_data[value_start:value_stop].astype(np.float64, copy=False)
                indices = source_indices[value_start:value_stop].astype(np.int64, copy=False)
                accumulator += np.bincount(
                    indices, weights=values, minlength=len(gene_names)
                )
            truth_centroids[task_index] = (accumulator / len(rows_for_task)).astype(np.float32)
            truth_rows_read += len(rows_for_task)

    if truth_rows_read != EXPECTED_TASK_CELLS or not np.isfinite(truth_centroids).all():
        raise ValidationCacheFailure("validation truth centroid construction failed")
    output.mkdir(parents=True, exist_ok=True)
    control_cache_path = output / "E208_VALIDATION_CONTROL_CACHE.npz"
    temporary_cache = output / ".E208_VALIDATION_CONTROL_CACHE.npz.tmp"
    with temporary_cache.open("wb") as handle:
        np.savez(
            handle,
            data=control_data,
            indices=control_indices,
            indptr=control_indptr,
            shape=np.asarray([len(chosen), len(gene_names)], dtype=np.int64),
            gene_names=gene_names,
        )
    os.replace(temporary_cache, control_cache_path)
    chosen["cache_row"] = np.arange(len(chosen), dtype=int)
    chosen = chosen[["cache_row", "source_row", "cell_id", "cell_type", "treatment"]]
    rows_path = output / "E208_VALIDATION_CONTROL_ROWS.csv"
    tasks_path = output / "E208_VALIDATION_TASKS.csv"
    truth_path = output / "E208_VALIDATION_TRUTH_CENTROIDS.npy"
    atomic_csv(rows_path, chosen)
    atomic_csv(tasks_path, task_table)
    atomic_npy(truth_path, truth_centroids)

    selected_counts = {
        f"{cell}|{state}": int(count)
        for (cell, state), count in chosen.groupby(
            ["cell_type", "treatment"], sort=True
        ).size().items()
    }
    status = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_VALIDATION_COMPETENCE_CACHE",
        "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_h5ad": str(h5ad),
        "source_h5ad_bytes": h5ad.stat().st_size,
        "source_split": str(split_path),
        "source_split_sha256": EXPECTED_SPLIT_SHA256,
        "selection_prefix": SELECTION_PREFIX,
        "selection_rule": "all validation controls if <=1000 else 1000 smallest SHA256(cell_id)",
        "n_tasks": len(task_table),
        "n_contexts": 12,
        "n_task_truth_cells": truth_rows_read,
        "n_candidate_validation_controls": len(control_table),
        "n_selected_validation_controls": len(chosen),
        "candidate_counts": candidate_counts,
        "selected_counts": selected_counts,
        "n_genes": len(gene_names),
        "truth_source_row_runs": number_of_runs,
        "control_cache": {
            "path": str(control_cache_path),
            "bytes": control_cache_path.stat().st_size,
            "sha256": sha256_file(control_cache_path),
        },
        "control_rows": {
            "path": str(rows_path),
            "bytes": rows_path.stat().st_size,
            "sha256": sha256_file(rows_path),
        },
        "task_manifest": {
            "path": str(tasks_path),
            "bytes": tasks_path.stat().st_size,
            "sha256": sha256_file(tasks_path),
        },
        "truth_centroids": {
            "path": str(truth_path),
            "bytes": truth_path.stat().st_size,
            "sha256": sha256_file(truth_path),
        },
        "validation_expression_rows_read": truth_rows_read + len(chosen),
        "test_perturbed_expression_rows_read": 0,
        "target_truth_access": "NOT_AUTHORIZED",
    }
    atomic_json(output / "E208_VALIDATION_CACHE_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
