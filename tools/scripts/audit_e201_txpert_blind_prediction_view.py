#!/usr/bin/env python3
"""Independently audit the E201 zero-expression prediction view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


PUBLIC_CELL_TYPES = ("K562", "RPE1", "hepg2", "jurkat")
SOURCE_SHA256 = "1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8"
VIEW_SHA256 = "85f93d1b29ded34d9dcece9ecdba1ef722a3f14aeedbfbe740eed9f045fbe486"
MANIFEST_SHA256 = "27448df0378aab32e1a9fd22bf20c18c90089816cee6c28b9710cd2d6f812e7d"


class AuditFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--view-cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def order_hash(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    source_path = args.source_cache.resolve() / "de_adata_test.h5ad"
    view_path = args.view_cache.resolve() / "de_adata_test.h5ad"
    manifest_path = (
        args.view_cache.resolve() / "E201_BLIND_PREDICTION_VIEW_MANIFEST.json"
    )
    output_path = args.output_json.resolve()
    if output_path.exists():
        raise AuditFailure(f"refusing to overwrite audit: {output_path}")
    expected_files = {
        source_path: SOURCE_SHA256,
        view_path: VIEW_SHA256,
        manifest_path: MANIFEST_SHA256,
    }
    for path, expected_sha in expected_files.items():
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise AuditFailure(f"input hash mismatch: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = ad.read_h5ad(source_path, backed="r")
    view = ad.read_h5ad(view_path, backed="r")
    try:
        source_obs = source.obs.copy()
        public_global = np.flatnonzero(
            source_obs.cell_line.astype(str).isin(PUBLIC_CELL_TYPES).to_numpy()
        )
        expected_obs = source_obs.iloc[public_global].copy()
        actual_obs = view.obs.copy()
        if source.shape != (632_488, 3_352) or view.shape != (581_172, 3_352):
            raise AuditFailure("source or view shape changed")
        pd.testing.assert_frame_equal(expected_obs, actual_obs, check_names=True)
        pd.testing.assert_frame_equal(source.var, view.var, check_names=True)
        if list(view.uns.keys()):
            raise AuditFailure("blind prediction view uns is not empty")

        is_control = actual_obs.control.astype(bool).to_numpy()
        local_controls = np.flatnonzero(is_control)
        global_controls = public_global[local_controls]
        local_treatments = np.flatnonzero(~is_control)
        control_max_abs_delta = 0.0
        control_mismatch_values = 0
        for start in range(0, len(local_controls), 5_000):
            local = local_controls[start : start + 5_000]
            global_rows = global_controls[start : start + 5_000]
            source_block = sparse.csr_matrix(source.X[global_rows])
            view_block = sparse.csr_matrix(view.X[local])
            difference = source_block - view_block
            if difference.nnz:
                control_mismatch_values += int(difference.nnz)
                control_max_abs_delta = max(
                    control_max_abs_delta,
                    float(np.max(np.abs(difference.data))),
                )
        treatment_nonzero_values = 0
        for start in range(0, len(local_treatments), 25_000):
            local = local_treatments[start : start + 25_000]
            treatment_nonzero_values += int(sparse.csr_matrix(view.X[local]).nnz)
    finally:
        source.file.close()
        view.file.close()

    gates = {
        "manifest_status": manifest.get("status") == "BLIND_PREDICTION_VIEW_READY",
        "source_obs_public_subset_exact": True,
        "var_exact": True,
        "control_expression_exact": control_mismatch_values == 0,
        "perturbed_expression_all_zero": treatment_nonzero_values == 0,
        "source_perturbed_expression_not_opened": True,
        "uns_empty": True,
        "manifest_counts": int(manifest.get("n_rows", -1)) == 581_172
        and int(manifest.get("n_controls", -1)) == 39_165
        and int(manifest.get("n_perturbed_rows", -1)) == 542_007,
    }
    if not all(gates.values()):
        raise AuditFailure(f"blind prediction audit gates failed: {gates}")
    payload = {
        "experiment": "E201_txpert_multitarget_retraining",
        "stage": "BLIND_PREDICTION_VIEW_INDEPENDENT_AUDIT",
        "status": "PASS",
        "audited_at": now(),
        "source_h5ad_sha256": SOURCE_SHA256,
        "view_h5ad_sha256": VIEW_SHA256,
        "view_manifest_sha256": MANIFEST_SHA256,
        "n_rows": 581_172,
        "n_genes": 3_352,
        "n_control_rows_compared": int(len(local_controls)),
        "n_perturbed_rows_checked": int(len(local_treatments)),
        "control_mismatch_values": control_mismatch_values,
        "control_max_abs_delta": control_max_abs_delta,
        "perturbed_expression_nonzero_values": treatment_nonzero_values,
        "source_perturbed_expression_rows_opened": 0,
        "obs_order_sha256": order_hash(actual_obs.index.astype(str)),
        "condition_order_sha256": order_hash(actual_obs.condition_name.astype(str)),
        "gates": gates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
