#!/usr/bin/env python3
"""Build train-only source-effect evidence for E216 before test unblinding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


EXPECTED_H5_BYTES = 93_532_364_449
EXPECTED_TASK_SHA256 = "53e3a5f363bf70a81f4074b0c001712698db8acd0852354745989db091d9d4de"
EXPECTED_SPLIT_SHA256 = "aa1261993e3456fe19db931cb9f20bd66febd4745bcd92144162c6d27afcf371"
EXPECTED_HVG_SHA256 = "daec79a6c8584fbee0104f522749ac43b900f697a26696ee59988ef3d0aea7ee"
EXPECTED_TASKS = 224
EXPECTED_HVG = 4_000


class SourceEvidenceFailure(RuntimeError):
    """The frozen source-evidence contract failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def read_csr_rows_on_gene_panel(
    h5ad_path: Path,
    positions: list[int],
    hvg_positions: np.ndarray,
    *,
    n_total_genes: int,
) -> csr_matrix:
    """Read registered CSR rows and remap them to the frozen HVG axis."""

    position_array = np.asarray(positions, dtype=np.int64)
    remap = np.full(n_total_genes, -1, dtype=np.int64)
    remap[hvg_positions] = np.arange(len(hvg_positions), dtype=np.int64)
    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    output_indptr = [0]
    with h5py.File(h5ad_path, "r") as handle:
        matrix = handle["X"]
        encoding = matrix.attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        if encoding != "csr_matrix":
            raise SourceEvidenceFailure("Jiang24 X is no longer CSR")
        starts = matrix["indptr"][position_array]
        stops = matrix["indptr"][position_array + 1]
        for start, stop in zip(starts, stops, strict=True):
            source_indices = np.asarray(matrix["indices"][int(start):int(stop)], dtype=np.int64)
            source_data = np.asarray(matrix["data"][int(start):int(stop)])
            mapped = remap[source_indices]
            keep = mapped >= 0
            data_parts.append(source_data[keep])
            index_parts.append(mapped[keep])
            output_indptr.append(output_indptr[-1] + int(keep.sum()))
    data = np.concatenate(data_parts) if data_parts else np.asarray([], dtype=np.float64)
    indices = np.concatenate(index_parts) if index_parts else np.asarray([], dtype=np.int64)
    return csr_matrix(
        (data, indices, np.asarray(output_indptr, dtype=np.int64)),
        shape=(len(positions), len(hvg_positions)),
    )


def aggregate_source_effects(
    expression: csr_matrix,
    obs: pd.DataFrame,
    task_genes: set[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return one perturbation-minus-control centroid per train source state."""

    state_columns = ["cell_type", "treatment"]
    indexed = obs.copy()
    indexed["_expression_row"] = np.arange(len(indexed), dtype=np.int64)
    controls = indexed.condition.astype(str).eq("control")
    control_centroids: dict[tuple[str, str], np.ndarray] = {}
    for state, block in indexed.loc[controls].groupby(state_columns, observed=True, sort=True):
        take = block._expression_row.to_numpy(dtype=int)
        control_centroids[tuple(str(value) for value in state)] = np.asarray(
            expression[take].mean(axis=0)
        ).ravel()

    delta_rows = []
    metadata_rows = []
    perturbed = indexed.loc[~controls & indexed.condition.astype(str).isin(task_genes)]
    group_columns = ["condition", "cell_type", "treatment"]
    for key, block in perturbed.groupby(group_columns, observed=True, sort=True):
        condition, cell_type, treatment = (str(value) for value in key)
        state = (cell_type, treatment)
        if state not in control_centroids:
            raise SourceEvidenceFailure(f"missing train control for {state}")
        take = block._expression_row.to_numpy(dtype=int)
        perturbed_centroid = np.asarray(expression[take].mean(axis=0)).ravel()
        delta_rows.append(perturbed_centroid - control_centroids[state])
        metadata_rows.append(
            {
                "condition": condition,
                "cell_type": cell_type,
                "treatment": treatment,
                "n_source_cells": int(len(take)),
            }
        )
    if not delta_rows:
        raise SourceEvidenceFailure("no train source effects were produced")
    return np.asarray(delta_rows, dtype=np.float32), pd.DataFrame(metadata_rows)


def task_source_summary(tasks: pd.DataFrame, source_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in tasks.itertuples(index=False):
        block = source_meta.loc[source_meta.condition.eq(str(task.condition))]
        block = block.loc[
            ~(
                block.cell_type.eq(str(task.cell_type))
                & block.treatment.eq(str(task.treatment))
            )
        ]
        rows.append(
            {
                "condition": str(task.condition),
                "cell_type": str(task.cell_type),
                "treatment": str(task.treatment),
                "n_source_contexts": int(len(block)),
                "n_source_cells": int(block.n_source_cells.sum()),
            }
        )
    result = pd.DataFrame(rows)
    if result.n_source_contexts.lt(2).any():
        raise SourceEvidenceFailure("a registered task has fewer than two train source states")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--resource-split", type=Path, required=True)
    parser.add_argument("--hvg", type=Path, required=True)
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    h5ad_path = args.h5ad.resolve()
    tasks_path = args.tasks.resolve()
    split_path = args.resource_split.resolve()
    hvg_path = args.hvg.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SourceEvidenceFailure(f"refusing to overwrite: {output_dir}")
    if h5ad_path.stat().st_size != EXPECTED_H5_BYTES:
        raise SourceEvidenceFailure("Jiang24 H5 identity changed")
    if sha256(tasks_path) != EXPECTED_TASK_SHA256:
        raise SourceEvidenceFailure("primary task inventory changed")
    if sha256(split_path) != EXPECTED_SPLIT_SHA256:
        raise SourceEvidenceFailure("E216 resource split changed")
    if sha256(hvg_path) != EXPECTED_HVG_SHA256:
        raise SourceEvidenceFailure("E216 HVG axis changed")

    sys.path.insert(0, str(args.perturbench_repo.resolve() / "src"))
    from perturbench.data.utils import load_dataframe_from_h5

    tasks = pd.read_csv(tasks_path)
    if len(tasks) != EXPECTED_TASKS:
        raise SourceEvidenceFailure("primary task count changed")
    hvg = pd.read_csv(hvg_path, header=None).iloc[:, 0].astype(str).tolist()
    if len(hvg) != EXPECTED_HVG:
        raise SourceEvidenceFailure("HVG count changed")
    obs = load_dataframe_from_h5(
        str(h5ad_path), "obs", ["condition", "cell_type", "treatment"]
    )
    var = load_dataframe_from_h5(str(h5ad_path), "var", [])
    split = pd.read_csv(split_path, header=None, names=["cell_barcode", "resource_split"], dtype=str)
    if not split.cell_barcode.astype(str).equals(pd.Series(obs.index.astype(str))):
        raise SourceEvidenceFailure("resource split order no longer matches H5 observations")
    obs = obs.copy()
    obs["resource_split"] = split.resource_split.to_numpy()
    task_genes = set(tasks.condition.astype(str))
    selected_mask = obs.resource_split.eq("train") & (
        obs.condition.astype(str).eq("control") | obs.condition.astype(str).isin(task_genes)
    )
    positions = np.flatnonzero(selected_mask.to_numpy()).astype(int).tolist()
    selected_obs = obs.iloc[positions].loc[:, ["condition", "cell_type", "treatment"]].reset_index(drop=True)
    gene_names = var.index.astype(str)
    hvg_positions = gene_names.get_indexer(hvg)
    if np.any(hvg_positions < 0):
        raise SourceEvidenceFailure("HVG is absent from source matrix")
    expression = read_csr_rows_on_gene_panel(
        h5ad_path,
        positions,
        hvg_positions,
        n_total_genes=len(gene_names),
    )
    deltas, source_meta = aggregate_source_effects(expression, selected_obs, task_genes)
    task_summary = task_source_summary(tasks, source_meta)

    output_dir.mkdir(parents=True)
    evidence_path = output_dir / "E216_TRAIN_SOURCE_DELTAS.npz"
    source_meta_path = output_dir / "E216_TRAIN_SOURCE_DELTA_METADATA.csv"
    task_summary_path = output_dir / "E216_TASK_SOURCE_SUPPORT.csv"
    np.savez_compressed(
        evidence_path,
        source_deltas=deltas,
        gene_names=np.asarray(hvg, dtype=str),
        condition=source_meta.condition.to_numpy(dtype=str),
        cell_type=source_meta.cell_type.to_numpy(dtype=str),
        treatment=source_meta.treatment.to_numpy(dtype=str),
        n_source_cells=source_meta.n_source_cells.to_numpy(dtype=np.int32),
    )
    atomic_text(source_meta_path, source_meta.to_csv(index=False))
    atomic_text(task_summary_path, task_summary.to_csv(index=False))
    status = {
        "experiment": "E216_jiang24_resource_bounded_confirmation",
        "stage": "D0_TRAIN_ONLY_SOURCE_EVIDENCE",
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selected_train_expression_rows": int(len(positions)),
        "source_delta_rows": int(len(source_meta)),
        "registered_tasks": int(len(task_summary)),
        "minimum_source_contexts": int(task_summary.n_source_contexts.min()),
        "genes": int(len(hvg)),
        "val_expression_rows_read": 0,
        "test_control_expression_rows_read": 0,
        "test_perturbed_expression_rows_read": 0,
        "target_truth_used_for_selection": False,
        "files": {
            evidence_path.name: {"bytes": evidence_path.stat().st_size, "sha256": sha256(evidence_path)},
            source_meta_path.name: {"bytes": source_meta_path.stat().st_size, "sha256": sha256(source_meta_path)},
            task_summary_path.name: {"bytes": task_summary_path.stat().st_size, "sha256": sha256(task_summary_path)},
        },
    }
    atomic_text(
        output_dir / "E216_SOURCE_EVIDENCE_STATUS.json",
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
