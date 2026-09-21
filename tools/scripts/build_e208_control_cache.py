#!/usr/bin/env python3
"""Build the frozen control-only sparse cache for E208 prediction."""

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


TARGET_CELL_TYPES = {"hap1", "ht29", "k562", "mcf7"}
TARGET_TREATMENTS = {"IFNG", "INS", "TGFB"}
SELECTION_PREFIX = "E208-control-cache-v1\0"
EXPECTED_H5_BYTES = 93_532_364_449
EXPECTED_SPLIT_SHA256 = "5af7da86a5b3994d570c0b1957d91f17cebb9f1943bac738b74fdb14b2ef5d"


class CacheFailure(RuntimeError):
    """A frozen data, selection, or sparse-matrix contract failed."""


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
        raise CacheFailure(f"invalid categorical codes: {group.name}")
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
        raise CacheFailure("Jiang24 H5 byte count changed")
    if sha256_file(split_path) != EXPECTED_SPLIT_SHA256:
        raise CacheFailure("Jiang24 split hash changed")

    split = pd.read_csv(split_path, header=None, names=["cell_id", "split"])
    with h5py.File(h5ad, "r") as handle:
        cell_ids = decode(handle["obs/_index"][:])
        if len(cell_ids) != len(split) or not np.array_equal(
            cell_ids, split.cell_id.astype(str).to_numpy()
        ):
            raise CacheFailure("split rows no longer align to H5 observation order")
        condition = categorical(handle["obs/condition"])
        cell_type = categorical(handle["obs/cell_type"])
        treatment = categorical(handle["obs/treatment"])
        split_values = split["split"].astype(str).to_numpy()
        candidate = (
            (condition == "control")
            & (split_values == "test")
            & np.isin(cell_type, sorted(TARGET_CELL_TYPES))
            & np.isin(treatment, sorted(TARGET_TREATMENTS))
        )
        candidate_indices = np.flatnonzero(candidate)
        candidate_table = pd.DataFrame(
            {
                "source_row": candidate_indices,
                "cell_id": cell_ids[candidate_indices],
                "cell_type": cell_type[candidate_indices],
                "treatment": treatment[candidate_indices],
            }
        )
        chosen_blocks = []
        candidate_counts = {}
        for (cell, state), block in candidate_table.groupby(
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
        if len(candidate_table) != 7858 or len(chosen) != 6799:
            raise CacheFailure(
                f"control counts changed: candidates={len(candidate_table)}, chosen={len(chosen)}"
            )
        if len(chosen.groupby(["cell_type", "treatment"])) != 12:
            raise CacheFailure("control cache does not contain all 12 target states")
        if not np.all(condition[chosen.source_row.to_numpy(int)] == "control"):
            raise CacheFailure("non-control row entered cache")
        if not np.all(split_values[chosen.source_row.to_numpy(int)] == "test"):
            raise CacheFailure("non-test row entered deployment cache")

        matrix = handle["X"]
        if matrix.attrs.get("encoding-type") != "csr_matrix":
            raise CacheFailure("Jiang24 X is no longer CSR")
        shape = tuple(map(int, matrix.attrs["shape"]))
        if shape != (1_628_476, 15_473):
            raise CacheFailure(f"Jiang24 matrix shape changed: {shape}")
        source_indptr = matrix["indptr"][:].astype(np.int64)
        source_data = matrix["data"]
        source_indices = matrix["indices"]
        data_blocks: list[np.ndarray] = []
        index_blocks: list[np.ndarray] = []
        output_indptr = np.zeros(len(chosen) + 1, dtype=np.int64)
        for output_row, source_row in enumerate(chosen.source_row.to_numpy(int)):
            start, stop = int(source_indptr[source_row]), int(source_indptr[source_row + 1])
            row_data = source_data[start:stop].astype(np.float32, copy=False)
            row_indices = source_indices[start:stop].astype(np.int32, copy=False)
            data_blocks.append(row_data)
            index_blocks.append(row_indices)
            output_indptr[output_row + 1] = output_indptr[output_row] + len(row_data)
        output_data = np.concatenate(data_blocks)
        output_indices = np.concatenate(index_blocks)
        gene_names = decode(handle["var/_index"][:])

    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "E208_TARGET_CONTROL_CACHE.npz"
    temporary_cache = output / ".E208_TARGET_CONTROL_CACHE.npz.tmp"
    with temporary_cache.open("wb") as file_handle:
        np.savez(
            file_handle,
            data=output_data,
            indices=output_indices,
            indptr=output_indptr,
            shape=np.asarray([len(chosen), len(gene_names)], dtype=np.int64),
            gene_names=gene_names,
        )
    os.replace(temporary_cache, cache_path)
    rows_path = output / "E208_TARGET_CONTROL_ROWS.csv"
    chosen["cache_row"] = np.arange(len(chosen), dtype=int)
    chosen = chosen[["cache_row", "source_row", "cell_id", "cell_type", "treatment"]]
    atomic_csv(rows_path, chosen)

    selected_counts = {
        f"{cell}|{state}": int(count)
        for (cell, state), count in chosen.groupby(
            ["cell_type", "treatment"], sort=True
        ).size().items()
    }
    status = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_PRETRUTH_CONTROL_CACHE",
        "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_h5ad": str(h5ad),
        "source_h5ad_bytes": h5ad.stat().st_size,
        "source_split": str(split_path),
        "source_split_sha256": EXPECTED_SPLIT_SHA256,
        "selection_prefix": SELECTION_PREFIX,
        "selection_rule": "all controls if <=1000 else 1000 smallest SHA256(cell_id)",
        "n_candidate_test_controls": len(candidate_table),
        "n_selected_test_controls": len(chosen),
        "n_contexts": len(selected_counts),
        "candidate_counts": candidate_counts,
        "selected_counts": selected_counts,
        "n_genes": len(gene_names),
        "n_nonzero": len(output_data),
        "cache": {
            "path": str(cache_path),
            "bytes": cache_path.stat().st_size,
            "sha256": sha256_file(cache_path),
        },
        "row_manifest": {
            "path": str(rows_path),
            "bytes": rows_path.stat().st_size,
            "sha256": sha256_file(rows_path),
        },
        "test_control_expression_rows_read": len(chosen),
        "test_perturbed_expression_rows_read": 0,
        "target_truth_access": "NOT_AUTHORIZED",
    }
    atomic_json(output / "E208_CONTROL_CACHE_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
