#!/usr/bin/env python3
"""Build one E201 prediction view with every perturbed expression row zeroed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


PUBLIC_CELL_TYPES = ("K562", "RPE1", "hepg2", "jurkat")
EXPECTED_INPUT_SHA256 = (
    "1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8"
)


class ViewFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--runtime-cache", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def hardlink_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ViewFailure(f"refusing existing runtime cache: {destination}")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)


def main() -> None:
    args = parse_args()
    source = args.source_cache.resolve()
    output = args.output_cache.resolve()
    runtime = args.runtime_cache.resolve()
    shared_runtime_cache = runtime == output
    dataset_path = source / "de_adata_test.h5ad"
    split_path = source / "splits/train_test_split.pkl"
    subgroup_path = source / "splits/subgroup.pkl"
    for path in (dataset_path, split_path, subgroup_path):
        if not path.is_file():
            raise ViewFailure(f"missing source input: {path}")
    if output.exists() or (not shared_runtime_cache and runtime.exists()):
        raise ViewFailure("prediction view output already exists")
    source_sha = sha256_file(dataset_path)
    if source_sha != EXPECTED_INPUT_SHA256:
        raise ViewFailure("official H5AD hash mismatch")

    source_data = ad.read_h5ad(dataset_path, backed="r")
    try:
        source_obs = source_data.obs.copy()
        source_var = source_data.var.copy()
        public_mask = (
            source_obs.cell_line.astype(str).isin(PUBLIC_CELL_TYPES).to_numpy()
        )
        global_indices = np.flatnonzero(public_mask)
        selected_obs = source_obs.iloc[global_indices].copy()
        selected_control = selected_obs.control.astype(bool).to_numpy()
        local_control_indices = np.flatnonzero(selected_control)
        global_control_indices = global_indices[local_control_indices]
        control_matrix = sparse.csr_matrix(source_data.X[global_control_indices])
    finally:
        source_data.file.close()

    control_coo = control_matrix.tocoo(copy=False)
    prediction_x = sparse.csr_matrix(
        (
            control_coo.data,
            (local_control_indices[control_coo.row], control_coo.col),
        ),
        shape=(len(selected_obs), len(source_var)),
        dtype=np.float32,
    )
    prediction_x.sort_indices()
    n_control_nonzero = int(control_matrix.nnz)
    n_total_nonzero = int(prediction_x.nnz)
    if n_total_nonzero != n_control_nonzero:
        raise ViewFailure("non-control values entered blind prediction matrix")
    treatment_matrix = prediction_x[~selected_control]
    if treatment_matrix.nnz != 0:
        raise ViewFailure("perturbed expression survived blind prediction view")

    output.mkdir(parents=True)
    (output / "splits").mkdir()
    output_dataset = output / "de_adata_test.h5ad"
    prediction_data = ad.AnnData(
        X=prediction_x,
        obs=selected_obs,
        var=source_var,
    )
    prediction_data.uns.clear()
    prediction_data.write_h5ad(output_dataset, compression="gzip")
    shutil.copy2(split_path, output / "splits/train_test_split.pkl")
    shutil.copy2(subgroup_path, output / "splits/subgroup.pkl")
    del prediction_data, prediction_x, treatment_matrix, control_matrix

    audit = ad.read_h5ad(output_dataset, backed="r")
    try:
        audit_obs = audit.obs.copy()
        audit_control = audit_obs.control.astype(bool).to_numpy()
        if audit.shape != (len(selected_obs), len(source_var)):
            raise ViewFailure("written prediction view shape changed")
        if set(audit_obs.cell_line.astype(str)) != set(PUBLIC_CELL_TYPES):
            raise ViewFailure("prediction view cellular contexts changed")
        if list(audit.uns.keys()):
            raise ViewFailure("prediction view contains result metadata")
        for start in range(0, len(audit_obs), 25_000):
            stop = min(start + 25_000, len(audit_obs))
            local_treatment = np.flatnonzero(~audit_control[start:stop]) + start
            if len(local_treatment) and sparse.csr_matrix(audit.X[local_treatment]).nnz:
                raise ViewFailure("written perturbed expression is not all zero")
    finally:
        audit.file.close()

    files = []
    for role, path in (
        ("blind_prediction_h5ad", output_dataset),
        ("condition_split", output / "splits/train_test_split.pkl"),
        ("subgroup_labels", output / "splits/subgroup.pkl"),
    ):
        files.append(
            {
                "role": role,
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "experiment": "E201_txpert_multitarget_retraining",
        "status": "BLIND_PREDICTION_VIEW_READY",
        "created_at": now(),
        "source_h5ad": "DATA/txpert_official_20260802/cache/K562_cross_cell_lines/de_adata_test.h5ad",
        "source_h5ad_bytes": dataset_path.stat().st_size,
        "source_h5ad_sha256": source_sha,
        "public_cell_types": list(PUBLIC_CELL_TYPES),
        "n_rows": int(len(selected_obs)),
        "n_genes": int(len(source_var)),
        "n_controls": int(selected_control.sum()),
        "n_perturbed_rows": int((~selected_control).sum()),
        "control_matrix_nonzero_values": n_control_nonzero,
        "perturbed_matrix_nonzero_values": 0,
        "uns_keys": [],
        "excluded_k562_adamson_rows": int((~public_mask).sum()),
        "runtime_cache_mode": (
            "same_path_via_parent_symlink"
            if shared_runtime_cache
            else "hardlinked_tree"
        ),
        "files": files,
    }
    write_json(output / "E201_BLIND_PREDICTION_VIEW_MANIFEST.json", manifest)
    if not shared_runtime_cache:
        hardlink_tree(output, runtime)
        for path in output.rglob("*"):
            if path.is_file():
                counterpart = runtime / path.relative_to(output)
                if path.stat().st_ino != counterpart.stat().st_ino:
                    raise ViewFailure(f"runtime cache is not hardlinked: {path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
