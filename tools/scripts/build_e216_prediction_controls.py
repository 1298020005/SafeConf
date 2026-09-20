#!/usr/bin/env python3
"""Freeze the small control-only prediction asset for E216.

Only rows whose condition is the unperturbed control are read from X.  The
perturbed Jiang24 test expression remains unopened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


EXPECTED_H5_BYTES = 93_532_364_449
EXPECTED_TASK_SHA256 = "53e3a5f363bf70a81f4074b0c001712698db8acd0852354745989db091d9d4de"
EXPECTED_HVG_SHA256 = "daec79a6c8584fbee0104f522749ac43b900f697a26696ee59988ef3d0aea7ee"
HASH_NAMESPACE = "E216_JIANG24_PREDICTION_CONTROL_V1"
EXPECTED_STATES = 12
EXPECTED_HVG = 4_000


class ControlAssetFailure(RuntimeError):
    """The frozen control-only asset contract failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def priority(barcode: str) -> int:
    value = hashlib.blake2b(
        f"{HASH_NAMESPACE}|{barcode}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(value, byteorder="big", signed=False)


def select_control_positions(
    obs: pd.DataFrame,
    tasks: pd.DataFrame,
    *,
    cap: int = 128,
) -> tuple[list[int], pd.DataFrame]:
    required = {"condition", "cell_type", "treatment"}
    if not required.issubset(obs.columns) or not required.issubset(tasks.columns):
        raise ControlAssetFailure("control selection metadata is incomplete")
    states = tasks.loc[:, ["cell_type", "treatment"]].drop_duplicates()
    if len(states) != EXPECTED_STATES:
        raise ControlAssetFailure(f"expected {EXPECTED_STATES} states, found {len(states)}")
    state_keys = set(states.itertuples(index=False, name=None))
    metadata = obs.loc[:, ["condition", "cell_type", "treatment"]].copy()
    metadata["position"] = np.arange(len(metadata), dtype=np.int64)
    metadata["cell_barcode"] = obs.index.astype(str)
    controls = metadata.loc[metadata.condition.astype(str).eq("control")].copy()
    controls = controls.loc[
        [
            (str(cell_type), str(treatment)) in state_keys
            for cell_type, treatment in controls[["cell_type", "treatment"]].itertuples(index=False)
        ]
    ]
    controls["priority"] = np.fromiter(
        (priority(value) for value in controls.cell_barcode),
        dtype=np.uint64,
        count=len(controls),
    )
    selected = []
    rows = []
    for (cell_type, treatment), block in controls.groupby(
        ["cell_type", "treatment"], observed=True, sort=True
    ):
        keep = block.nsmallest(min(cap, len(block)), "priority")
        if keep.empty:
            raise ControlAssetFailure(f"state has no controls: {cell_type}|{treatment}")
        selected.extend(keep.position.astype(int).tolist())
        rows.append(
            {
                "cell_type": str(cell_type),
                "treatment": str(treatment),
                "n_available_controls": int(len(block)),
                "n_selected_controls": int(len(keep)),
            }
        )
    if len(rows) != EXPECTED_STATES:
        raise ControlAssetFailure("one or more registered states lacks control cells")
    return sorted(selected), pd.DataFrame(rows)


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
    """Read scattered CSR rows without materializing the 93GB source matrix."""

    import h5py

    positions_array = np.asarray(positions, dtype=np.int64)
    remap = np.full(n_total_genes, -1, dtype=np.int64)
    remap[hvg_positions] = np.arange(len(hvg_positions), dtype=np.int64)
    data_parts = []
    index_parts = []
    output_indptr = [0]
    with h5py.File(h5ad_path, "r") as handle:
        matrix = handle["X"]
        encoding = matrix.attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        if encoding != "csr_matrix" or not {"data", "indices", "indptr"}.issubset(matrix.keys()):
            raise ControlAssetFailure("Jiang24 X is no longer the expected CSR matrix")
        starts = matrix["indptr"][positions_array]
        stops = matrix["indptr"][positions_array + 1]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--hvg", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    import anndata as ad

    args = parse_args()
    h5ad_path = args.h5ad.resolve()
    tasks_path = args.tasks.resolve()
    hvg_path = args.hvg.resolve()
    output_dir = args.output_dir.resolve()
    perturbench_src = args.perturbench_repo.resolve() / "src"
    sys.path.insert(0, str(perturbench_src))
    from perturbench.data.utils import load_dataframe_from_h5
    if output_dir.exists():
        raise ControlAssetFailure(f"refusing to overwrite: {output_dir}")
    if h5ad_path.stat().st_size != EXPECTED_H5_BYTES:
        raise ControlAssetFailure("Jiang24 H5 identity changed")
    if sha256(tasks_path) != EXPECTED_TASK_SHA256:
        raise ControlAssetFailure("E208 primary task inventory changed")
    if sha256(hvg_path) != EXPECTED_HVG_SHA256:
        raise ControlAssetFailure("E216 HVG panel changed")
    tasks = pd.read_csv(tasks_path)
    hvg = pd.read_csv(hvg_path, header=None).iloc[:, 0].astype(str).tolist()
    if len(hvg) != EXPECTED_HVG or len(set(hvg)) != EXPECTED_HVG:
        raise ControlAssetFailure("HVG panel is not the registered 4,000-gene axis")

    # Loading only three categorical columns avoids materializing unrelated
    # AnnData metadata and any expression matrix.
    source_obs = load_dataframe_from_h5(
        str(h5ad_path), "obs", ["condition", "cell_type", "treatment"]
    )
    source_var = load_dataframe_from_h5(str(h5ad_path), "var", [])
    positions, state_counts = select_control_positions(source_obs, tasks)
    source_gene_names = source_var.index.astype(str)
    hvg_positions = source_gene_names.get_indexer(hvg)
    if np.any(hvg_positions < 0):
        raise ControlAssetFailure("one or more frozen HVGs is absent from Jiang24")
    selected_obs = source_obs.iloc[positions].copy()
    selected_var = pd.DataFrame(index=pd.Index(hvg, name=source_var.index.name))
    n_total_genes = int(len(source_gene_names))
    matrix = read_csr_rows_on_gene_panel(
        h5ad_path,
        positions,
        hvg_positions,
        n_total_genes=n_total_genes,
    )
    controls = ad.AnnData(X=matrix, obs=selected_obs, var=selected_var)
    if controls.n_vars != EXPECTED_HVG or not controls.var_names.astype(str).tolist() == hvg:
        raise ControlAssetFailure("control asset gene order changed")
    if not controls.obs["condition"].astype(str).eq("control").all():
        raise ControlAssetFailure("non-control expression entered prediction asset")

    output_dir.mkdir(parents=True)
    asset_path = output_dir / "jiang24_e216_prediction_controls_hvg4000.h5ad"
    counts_path = output_dir / "E216_PREDICTION_CONTROL_COUNTS.csv"
    controls.write_h5ad(asset_path, compression="gzip")
    atomic_text(counts_path, state_counts.to_csv(index=False))
    status = {
        "experiment": "E216_jiang24_resource_bounded_confirmation",
        "stage": "D0_CONTROL_ONLY_PREDICTION_ASSET",
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_namespace": HASH_NAMESPACE,
        "states": int(len(state_counts)),
        "control_cap_per_state": 128,
        "selected_control_cells": int(controls.n_obs),
        "genes": int(controls.n_vars),
        "expression_values_read": True,
        "expression_rows_read": "selected_unperturbed_controls_only",
        "test_perturbed_expression_rows_read": 0,
        "target_truth_used_for_selection": False,
        "files": {
            asset_path.name: {"bytes": asset_path.stat().st_size, "sha256": sha256(asset_path)},
            counts_path.name: {"bytes": counts_path.stat().st_size, "sha256": sha256(counts_path)},
        },
    }
    atomic_text(
        output_dir / "E216_PREDICTION_CONTROL_STATUS.json",
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
