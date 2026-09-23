#!/usr/bin/env python3
"""Build train-only Jiang24 history for E245 validation genes; never read test X."""

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

from build_e208_source_effect_cache import categorical, decode, sparse_row_mean


H5_BYTES = 93_532_364_449
SPLIT_SHA256 = "5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d"
MIN_PERTURBED_CELLS = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    writer(temporary)
    os.replace(temporary, path)


def write_npy(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("h5ad", "split", "validation-tasks", "validation-cache-status", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing to overwrite E245 train-source cache")
    if args.h5ad.stat().st_size != H5_BYTES or sha256(args.split) != SPLIT_SHA256:
        raise RuntimeError("Jiang24 H5/split contract changed")
    validation_status = json.loads(args.validation_cache_status.read_text())
    if (validation_status.get("status") != "PASS" or validation_status.get("n_tasks") != 216
            or validation_status.get("test_perturbed_expression_rows_read") != 0
            or validation_status["task_manifest"]["sha256"] != sha256(args.validation_tasks)):
        raise RuntimeError("validation task manifest changed")
    tasks = pd.read_csv(args.validation_tasks)
    genes = sorted(set(tasks.condition.astype(str)))
    if len(tasks) != 216 or len(genes) != 53:
        raise RuntimeError("validation target inventory changed")
    split = pd.read_csv(args.split, header=None, names=["cell_id", "split"])
    with h5py.File(args.h5ad, "r") as h5:
        ids = decode(h5["obs/_index"][:])
        if not np.array_equal(ids, split.cell_id.astype(str).to_numpy()):
            raise RuntimeError("Jiang24 split/H5 cell IDs differ")
        condition = categorical(h5["obs/condition"])
        cells = categorical(h5["obs/cell_type"])
        treatment = categorical(h5["obs/treatment"])
        train = split.split.astype(str).to_numpy() == "train"
        control_rows = np.flatnonzero(train & (condition == "control"))
        source_rows = np.flatnonzero(train & np.isin(condition, genes))
        control = pd.DataFrame({"row": control_rows, "cell_type": cells[control_rows],
                                "treatment": treatment[control_rows]})
        source = pd.DataFrame({"row": source_rows, "cell_type": cells[source_rows],
                               "treatment": treatment[source_rows], "condition": condition[source_rows]})
        manifest = (source.groupby(["cell_type", "treatment", "condition"], sort=True).size()
                    .rename("n_perturbed_cells").reset_index())
        manifest = manifest.loc[manifest.n_perturbed_cells.ge(MIN_PERTURBED_CELLS)].reset_index(drop=True)
        manifest["source_context_id"] = manifest.cell_type.astype(str) + "|" + manifest.treatment.astype(str)
        n_source = []
        for task in tasks.itertuples(index=False):
            valid = manifest.loc[manifest.condition.eq(str(task.condition))
                                 & manifest.source_context_id.ne(f"{task.cell_type}|{task.treatment}")]
            n_source.append(len(valid))
        if sum(value >= 2 for value in n_source) != 212 or sum(value < 2 for value in n_source) != 4:
            raise RuntimeError("validation history eligibility changed")
        matrix = h5["X"]
        if matrix.attrs.get("encoding-type") != "csr_matrix" or tuple(matrix.attrs["shape"]) != (1_628_476, 15473):
            raise RuntimeError("Jiang24 expression matrix contract changed")
        indptr = matrix["indptr"][:].astype(np.int64)
        data, indices = matrix["data"], matrix["indices"]
        gene_axis = decode(h5["var/_index"][:])
        control_means = {}
        run_count = 0
        for key, block in control.groupby(["cell_type", "treatment"], sort=True):
            mean, runs = sparse_row_mean(block.row.to_numpy(np.int64), indptr, data, indices, len(gene_axis))
            control_means[(str(key[0]), str(key[1]))] = mean
            run_count += runs
        effects = np.empty((len(manifest), len(gene_axis)), dtype=np.float32)
        read_source_cells = 0
        for index, row in enumerate(manifest.itertuples(index=False)):
            selected = source.loc[source.cell_type.eq(row.cell_type) & source.treatment.eq(row.treatment)
                                  & source.condition.eq(row.condition), "row"].to_numpy(np.int64)
            mean, runs = sparse_row_mean(selected, indptr, data, indices, len(gene_axis))
            effects[index] = mean - control_means[(str(row.cell_type), str(row.treatment))]
            run_count += runs
            read_source_cells += len(selected)
    if not np.isfinite(effects).all():
        raise RuntimeError("nonfinite train-source effects")
    output.mkdir(parents=True, exist_ok=True)
    manifest.insert(0, "effect_row", np.arange(len(manifest), dtype=int))
    effect_path = output / "E245_VALIDATION_SOURCE_EFFECTS.npy"
    manifest_path = output / "E245_VALIDATION_SOURCE_MANIFEST.csv"
    genes_path = output / "E245_VALIDATION_GENE_AXIS.csv"
    atomic(effect_path, lambda p: write_npy(p, effects))
    atomic(manifest_path, lambda p: manifest.to_csv(p, index=False))
    atomic(genes_path, lambda p: pd.DataFrame({"gene_index": np.arange(len(gene_axis)),
                                               "gene_name": gene_axis}).to_csv(p, index=False))
    record = {"experiment": "E245_jiang24_quality_gated_source_transfer",
              "stage": "VALIDATION_TRAIN_ONLY_SOURCE_CACHE", "status": "PASS",
              "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "n_validation_tasks": 216, "n_eligible": 212, "n_abstain": 4,
              "n_source_effects": len(manifest), "min_source_cells": MIN_PERTURBED_CELLS,
              "train_source_expression_rows_read": read_source_cells,
              "train_control_expression_rows_read": len(control),
              "sparse_row_runs": run_count,
              "validation_perturbed_expression_rows_read": 0,
              "test_perturbed_expression_rows_read": 0,
              "validation_task_manifest_sha256": sha256(args.validation_tasks),
              "split_sha256": SPLIT_SHA256,
              "effects_sha256": sha256(effect_path), "manifest_sha256": sha256(manifest_path),
              "gene_axis_sha256": sha256(genes_path)}
    atomic(output / "E245_VALIDATION_SOURCE_CACHE_STATUS.json",
           lambda p: p.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n"))
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
