#!/usr/bin/env python3
"""E163: validation-only futility diagnostic for Wessels raw log probability.

The runner consumes only the E161 development container and E162 attempt-002
validation artifacts.  It never imports the raw Wessels H5AD, test labels, test
expression, or E162b test-label artifacts.  E163 is diagnostic and cannot turn
the failed E162 prediction arm into a confirmatory result.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr


EXPERIMENT = "E163_wessels_validation_raw_futility"
SCHEMA = "safeconf_e163_validation_raw_futility_v1"
DATE_TAG = "20260715"
TIMEZONE = ZoneInfo("Asia/Shanghai")
SEEDS = (3407, 3408, 3409)
MAIN_SEED = 3407
N_TASKS = 24
N_PCA = 10
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 3407
MIN_VALID_BOOTSTRAP = 9_500
ESTIMABILITY_STD = 1e-12
PCA_TRANSFORM_ATOL = 2e-5
SELECTED_GENE_SHA256 = "5fbe6a1d80d163a63576552f2bc74cfd9416e65e706877cefe4ad05b2fb3a2cf"

CONTRACT_REL = (
    "docs/实验结果/E163_wessels_validation_raw_futility_20260715/"
    "ANALYSIS_CONTRACT.md"
)
RUNNER_REL = "tools/scripts/run_e163_wessels_validation_raw_futility.py"
OUTPUT_REL = "docs/实验结果/E163_wessels_validation_raw_futility_20260715"

DEV_ROOT = Path("/home/yyf/data/safeconf_e161_prescribe/wessels_e160")
FORBIDDEN_RAW_WESSELS = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "WesselsSatija2023.h5ad"
)


@dataclass(frozen=True)
class InputSpec:
    role: str
    location: str
    expected_sha256: str
    kind: str


INPUT_SPECS = (
    InputSpec(
        "E161_interface",
        "/home/yyf/data/safeconf_e161_prescribe/wessels_e160/E161_E162_INTERFACE.json",
        "b681160e2f88c500fb3004bc9fb3fa5400cccbb9e8d537bde8db043e11254ed7",
        "external_small",
    ),
    InputSpec(
        "E161_development_h5ad",
        "/home/yyf/data/safeconf_e161_prescribe/wessels_e160/perturb_processed.h5ad",
        "2921f1c8fa7e6415725380e319a5092993e725ec5cb596f225af19811e82fd40",
        "development_expression_container",
    ),
    InputSpec(
        "E161_train_only_PCA10",
        "/home/yyf/data/safeconf_e161_prescribe/wessels_e160/TRAIN_ONLY_PCA_MODEL.npz",
        "b1be7cfe03300d0c8352f7265f65496baaa4454edecf95bcd421961126e6a12f",
        "external_small",
    ),
    InputSpec(
        "E161_train_control_prior",
        "/home/yyf/data/safeconf_e161_prescribe/wessels_e160/TRAIN_ONLY_CONTROL_PRIOR.npz",
        "1e53d388a895feae2f24aa74a99a80287ec63fbc8511cc46890b331aff05acd6",
        "external_small",
    ),
    InputSpec(
        "E161_selected_gene_axis",
        "/home/yyf/data/safeconf_e161_prescribe/wessels_e160/SELECTED_GENE_AXIS.txt",
        SELECTED_GENE_SHA256,
        "external_small",
    ),
    InputSpec(
        "E161_release_status",
        "docs/实验结果/E161_wessels_trainval_preprocess_20260714/release/RUN_STATUS.json",
        "9c1601bef3bac70e5b9b85cf523bd3d000d36e0c3f84f2f35e57fa6300e5213f",
        "repo_small",
    ),
    InputSpec(
        "E161_asset_manifest",
        "docs/实验结果/E161_wessels_trainval_preprocess_20260714/release/tables/"
        "E161_ASSET_MANIFEST.csv",
        "c755ba6c70705187a45b32f0bfb95b05687a64860daa565dcdf5ad9b90a6f54d",
        "repo_small",
    ),
    InputSpec(
        "E162_attempt_002_status",
        "docs/实验结果/E162_wessels_prescribe_native_20260714/attempt_002/"
        "RUN_STATUS.json",
        "497eb2156effa3bf5f11392006f490ed9435b081ee5e64b2231e8bbb41df2e5c",
        "repo_small",
    ),
    InputSpec(
        "E162_validation_scores_seed3407",
        "docs/实验结果/E162_wessels_prescribe_native_20260714/attempt_002/locked/"
        "E162_VALIDATION_LABEL_ONLY_SCORES_SEED3407.csv",
        "3eee6168bf640c6c1f241d3cf765596d79b8b7b300ebba8324b9cb0838f02a0c",
        "repo_small",
    ),
    InputSpec(
        "E162_validation_scores_seed3408",
        "docs/实验结果/E162_wessels_prescribe_native_20260714/attempt_002/locked/"
        "E162_VALIDATION_LABEL_ONLY_SCORES_SEED3408.csv",
        "d6637337ca54604fec96f8029ab0b915c2d23b97b5a0fdfedcab5e8e8b20e058",
        "repo_small",
    ),
    InputSpec(
        "E162_validation_scores_seed3409",
        "docs/实验结果/E162_wessels_prescribe_native_20260714/attempt_002/locked/"
        "E162_VALIDATION_LABEL_ONLY_SCORES_SEED3409.csv",
        "6817e63169c5c9be6dcb9133c4eb3c085ea4c589fefdb379941c60471e582f52",
        "repo_small",
    ),
    InputSpec(
        "E162_validation_gate_seed3407",
        "docs/实验结果/E162_wessels_prescribe_native_20260714/attempt_002/locked/"
        "E162_VALIDATION_NONDEGENERACY_GATE_SEED3407.json",
        "bc4ecee7ef9b5a43a6a1575875e4a3b86fbca8a87c156b127b8e3ca9de60cb7e",
        "repo_small",
    ),
    InputSpec(
        "E162_validation_gate_seed3408",
        "docs/实验结果/E162_wessels_prescribe_native_20260714/attempt_002/locked/"
        "E162_VALIDATION_NONDEGENERACY_GATE_SEED3408.json",
        "cff7e48ec882cf66a7569359198ee54ca455b4d9653ed4fbb2b48f2a47dbef5f",
        "repo_small",
    ),
    InputSpec(
        "E162_validation_gate_seed3409",
        "docs/实验结果/E162_wessels_prescribe_native_20260714/attempt_002/locked/"
        "E162_VALIDATION_NONDEGENERACY_GATE_SEED3409.json",
        "68e28065f03ac7abe6ff12babb6f60cf99ba5c0dec64de247e867a229c3cd0c4",
        "repo_small",
    ),
)

ENDPOINTS = (
    ("pearson_effect_accuracy", "positive", "primary_PCA10_inverse_transform"),
    ("rmse_effect_error", "negative", "secondary_PCA10_inverse_transform"),
    (
        "raw_pearson_effect_accuracy_sensitivity",
        "positive",
        "mandatory_raw_selected_gene_sensitivity",
    ),
    (
        "raw_rmse_effect_error_sensitivity",
        "negative",
        "mandatory_raw_selected_gene_sensitivity",
    ),
)


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8", float_format="%.17g").encode(
        "utf-8"
    )


def fsync_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def git_output(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo)


def git_source_gate(repo: Path) -> tuple[str, pd.DataFrame]:
    head = git_output(repo, "rev-parse", "HEAD").decode("ascii").strip()
    rows: list[dict[str, Any]] = []
    for relative in (RUNNER_REL, CONTRACT_REL):
        worktree_path = repo / relative
        if not worktree_path.is_file() or worktree_path.is_symlink():
            raise RuntimeError(f"Missing or symlinked frozen source: {relative}")
        try:
            blob = git_output(repo, "show", f"HEAD:{relative}")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Frozen source is not committed at HEAD; commit before formal run: {relative}"
            ) from exc
        working = worktree_path.read_bytes()
        if working != blob:
            raise RuntimeError(f"Frozen source differs from HEAD blob: {relative}")
        rows.append(
            {
                "git_head": head,
                "relative_path": relative,
                "worktree_sha256": sha256_bytes(working),
                "head_blob_sha256": sha256_bytes(blob),
                "byte_identical": True,
            }
        )
    return head, pd.DataFrame(rows)


def resolve_input(repo: Path, spec: InputSpec) -> Path:
    path = Path(spec.location) if Path(spec.location).is_absolute() else repo / spec.location
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing or symlinked allowlisted input: {spec.role}: {path}")
    resolved = path.resolve(strict=True)
    if resolved == FORBIDDEN_RAW_WESSELS.resolve(strict=False):
        raise RuntimeError("Raw Wessels H5AD is forbidden in E163")
    if path.suffix.lower() in {".h5ad", ".h5", ".loom"} and spec.role != "E161_development_h5ad":
        raise RuntimeError(f"Unapproved expression container: {path}")
    return path


def load_and_hash_inputs(
    repo: Path,
) -> tuple[dict[str, bytes], dict[str, Path], pd.DataFrame, dict[str, Any]]:
    payloads: dict[str, bytes] = {}
    paths: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    dev_stat: dict[str, Any] = {}
    for spec in INPUT_SPECS:
        path = resolve_input(repo, spec)
        before = path.stat()
        observed = sha256_file(path)
        after = path.stat()
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if not identity:
            raise RuntimeError(f"Input changed while hashing: {spec.role}")
        if observed != spec.expected_sha256:
            raise RuntimeError(
                f"SHA256 mismatch for {spec.role}: expected {spec.expected_sha256}, got {observed}"
            )
        paths[spec.role] = path
        if spec.kind != "development_expression_container":
            payloads[spec.role] = path.read_bytes()
        else:
            dev_stat = {
                "device": int(after.st_dev),
                "inode": int(after.st_ino),
                "bytes": int(after.st_size),
                "mtime_ns": int(after.st_mtime_ns),
                "sha256": observed,
            }
        rows.append(
            {
                "role": spec.role,
                "display_path": spec.location,
                "kind": spec.kind,
                "bytes": int(after.st_size),
                "expected_sha256": spec.expected_sha256,
                "observed_sha256": observed,
                "identity_stable_during_hash": identity,
                "read_scope": (
                    "E161_train_validation_only_expression"
                    if spec.kind == "development_expression_container"
                    else "fixed_small_artifact"
                ),
            }
        )
    return payloads, paths, pd.DataFrame(rows), dev_stat


def json_payload(payloads: dict[str, bytes], role: str) -> dict[str, Any]:
    value = json.loads(payloads[role].decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {role}")
    return value


def csv_payload(payloads: dict[str, bytes], role: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payloads[role]))


def parse_locked_bool(series: pd.Series, *, column: str) -> np.ndarray:
    mapping = {True: False, False: False}
    del mapping  # prevent implicit Python truth coercion below
    values: list[bool] = []
    for value in series.tolist():
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
        elif str(value).strip().lower() == "true":
            values.append(True)
        elif str(value).strip().lower() == "false":
            values.append(False)
        else:
            raise RuntimeError(f"Invalid boolean in {column}: {value!r}")
    return np.asarray(values, dtype=bool)


def validate_upstream_status(payloads: dict[str, bytes]) -> None:
    e161 = json_payload(payloads, "E161_release_status")
    if e161.get("phase") != "complete_preprocessing_and_dev_graphs_no_training_no_test_X_access":
        raise RuntimeError("E161 phase changed")
    required_e161 = {
        "train_conditions": 72,
        "validation_conditions": 24,
        "test_conditions": 48,
        "train_cells": 11779,
        "validation_cells": 5102,
        "development_cell_graphs": 16881,
        "test_graphs": 0,
        "test_X_rows_indexed": False,
        "test_X_rows_materialized": False,
        "test_X_rows_transformed": False,
        "test_endpoint_computed": False,
    }
    for key, expected in required_e161.items():
        if e161.get(key) != expected:
            raise RuntimeError(f"E161 status field changed: {key}")

    e162 = json_payload(payloads, "E162_attempt_002_status")
    if e162.get("phase") != "failed_main_validation_nondegeneracy_gate_no_test_label_query":
        raise RuntimeError("E162 attempt-002 phase changed")
    required_e162 = {
        "main_seed": MAIN_SEED,
        "test_label_queries_started": False,
        "raw_h5ad_opened": False,
        "test_X_accessed": False,
        "test_truth_accessed": False,
        "test_endpoint_computed": False,
        "n_test_graphs": 0,
    }
    for key, expected in required_e162.items():
        if e162.get(key) != expected:
            raise RuntimeError(f"E162 status field changed: {key}")
    if tuple(e162.get("seeds", [])) != SEEDS:
        raise RuntimeError("E162 seed order changed")

    interface = json_payload(payloads, "E161_interface")
    if interface.get("schema") != "safeconf_e161_to_e162_v2":
        raise RuntimeError("E161 interface schema changed")
    if interface.get("n_selected_genes") != 2023:
        raise RuntimeError("E161 selected-gene count changed")
    if interface.get("selected_gene_order_sha256") != SELECTED_GENE_SHA256:
        raise RuntimeError("E161 selected-gene hash changed")
    if interface.get("development_graphs") != {"train": 11779, "val": 5102, "test": 0}:
        raise RuntimeError("E161 development graph counts changed")
    if interface.get("test_X_rows_indexed_materialized_or_transformed") is not False:
        raise RuntimeError("E161 interface no longer preserves the test seal")


def validate_scores_and_gates(
    payloads: dict[str, bytes],
) -> tuple[dict[int, pd.DataFrame], list[str], pd.DataFrame]:
    score_tables: dict[int, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    expected_columns = {
        "seed",
        "seed_role",
        "split",
        "condition",
        "query_has_test_expression",
        "query_has_y",
        "query_has_y_pca",
        "raw_log_prob",
        "selected_gene_order_sha256",
        *(f"predicted_pca_{index}" for index in range(N_PCA)),
    }
    canonical_order: Optional[list[str]] = None
    for seed in SEEDS:
        table = csv_payload(payloads, f"E162_validation_scores_seed{seed}")
        missing = sorted(expected_columns - set(table.columns))
        if missing:
            raise RuntimeError(f"Seed {seed} score table missing columns: {missing}")
        if len(table) != N_TASKS or table["condition"].nunique() != N_TASKS:
            raise RuntimeError(f"Seed {seed} does not contain exactly 24 unique validation tasks")
        if set(table["seed"].astype(int)) != {seed} or set(table["split"].astype(str)) != {
            "validation"
        }:
            raise RuntimeError(f"Seed {seed} identity/split changed")
        for column in ("query_has_test_expression", "query_has_y", "query_has_y_pca"):
            if parse_locked_bool(table[column], column=column).any():
                raise RuntimeError(f"Seed {seed} score row contains forbidden truth field: {column}")
        if set(table["selected_gene_order_sha256"].astype(str)) != {SELECTED_GENE_SHA256}:
            raise RuntimeError(f"Seed {seed} selected-gene hash changed")
        conditions = table["condition"].astype(str).tolist()
        for condition in conditions:
            genes = condition.split("+")
            if len(genes) != 2 or genes != sorted(genes) or any(g != g.upper() for g in genes):
                raise RuntimeError(f"Non-canonical validation pair: {condition}")
        if canonical_order is None:
            canonical_order = conditions
        elif conditions != canonical_order:
            raise RuntimeError(f"Seed {seed} validation row order changed")
        numeric_columns = ["raw_log_prob", *(f"predicted_pca_{i}" for i in range(N_PCA))]
        numeric = table[numeric_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"Seed {seed} score table contains non-finite values")

        gate = json_payload(payloads, f"E162_validation_gate_seed{seed}")
        if gate.get("n_rows") != N_TASKS:
            raise RuntimeError(f"Seed {seed} upstream gate row count changed")
        if gate.get("raw_log_prob_all_finite") is not True:
            raise RuntimeError(f"Seed {seed} raw score is not finite")
        if gate.get("raw_log_prob_exact_unique") != N_TASKS:
            raise RuntimeError(f"Seed {seed} raw score exact uniqueness changed")
        if gate.get("prediction_exact_unique_vectors") != 1 or gate.get("passed") is not False:
            raise RuntimeError(f"Seed {seed} upstream prediction-collapse record changed")
        raw = table["raw_log_prob"].to_numpy(dtype=np.float64)
        prediction = table[[f"predicted_pca_{i}" for i in range(N_PCA)]].to_numpy(
            dtype=np.float64
        )
        raw_unique = int(np.unique(raw).size)
        prediction_unique = int(np.unique(prediction, axis=0).shape[0])
        if raw_unique != N_TASKS or prediction_unique != 1:
            raise RuntimeError(f"Seed {seed} locked table disagrees with upstream gate")
        audit_rows.append(
            {
                "seed": seed,
                "n_tasks": len(table),
                "raw_log_prob_exact_unique": raw_unique,
                "raw_log_prob_sample_std_ddof1": float(np.std(raw, ddof=1)),
                "prediction_exact_unique_vectors": prediction_unique,
                "upstream_full_nondegeneracy_gate_passed": False,
                "E163_role": "raw_score_only_validation_futility_diagnostic",
            }
        )
        score_tables[seed] = table.copy()
    assert canonical_order is not None
    return score_tables, canonical_order, pd.DataFrame(audit_rows)


def dense_mean(matrix: Any) -> np.ndarray:
    value = matrix.mean(axis=0)
    if sparse.issparse(value):
        value = value.toarray()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def load_development_truth(
    dev_path: Path,
    expected_stat: dict[str, Any],
    conditions: list[str],
    selected_genes: list[str],
    pca_mean: np.ndarray,
    components: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    import anndata as ad

    before = dev_path.stat()
    observed_identity = {
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "bytes": int(before.st_size),
        "mtime_ns": int(before.st_mtime_ns),
    }
    for key in ("device", "inode", "bytes", "mtime_ns"):
        if observed_identity[key] != expected_stat[key]:
            raise RuntimeError("E161 development H5AD identity changed after hash gate")

    adata = ad.read_h5ad(dev_path, backed="r")
    access_rows: list[dict[str, Any]] = [
        {
            "container": "E161 development H5AD",
            "path": str(dev_path),
            "backed_mode": "r",
            "n_obs_container": int(adata.n_obs),
            "n_vars_container": int(adata.n_vars),
            "train_rows_present": int((adata.obs["e161_split"].astype(str) == "train").sum()),
            "validation_rows_present": int(
                (adata.obs["e161_split"].astype(str) == "val").sum()
            ),
            "test_rows_present": int((adata.obs["e161_split"].astype(str) == "test").sum()),
            "rows_materialized": 0,
            "X_columns_materialized": 0,
            "access_role": "metadata_gate_before_validation_subset",
        }
    ]
    try:
        if (int(adata.n_obs), int(adata.n_vars)) != (16881, 2023):
            raise RuntimeError("E161 development H5AD shape changed")
        required_obs = {"condition", "e161_split"}
        if not required_obs.issubset(adata.obs.columns):
            raise RuntimeError("E161 development H5AD is missing split/condition metadata")
        split_values = set(adata.obs["e161_split"].astype(str))
        if split_values != {"train", "val"}:
            raise RuntimeError(f"Development H5AD unexpectedly contains split(s): {split_values}")
        var_names = list(map(str, adata.var_names.tolist()))
        if var_names != selected_genes:
            raise RuntimeError("Development H5AD var order differs from selected-gene axis")
        if "X_pca" not in adata.obsm or tuple(adata.obsm["X_pca"].shape) != (16881, N_PCA):
            raise RuntimeError("E161 development H5AD PCA10 coordinates changed")

        val_mask = adata.obs["e161_split"].astype(str).to_numpy() == "val"
        val_positions = np.flatnonzero(val_mask)
        if len(val_positions) != 5102:
            raise RuntimeError("E161 validation cell count changed")
        val_conditions = adata.obs.iloc[val_positions]["condition"].astype(str)
        if set(val_conditions) != set(conditions) or val_conditions.nunique() != N_TASKS:
            raise RuntimeError("E161 validation condition set differs from locked score tables")
        validation = adata[val_positions, :].to_memory()
        access_rows.append(
            {
                "container": "E161 development H5AD",
                "path": str(dev_path),
                "backed_mode": "r",
                "n_obs_container": int(adata.n_obs),
                "n_vars_container": int(adata.n_vars),
                "train_rows_present": 11779,
                "validation_rows_present": 5102,
                "test_rows_present": 0,
                "rows_materialized": int(validation.n_obs),
                "X_columns_materialized": int(validation.n_vars),
                "access_role": "validation_rows_only_truth_materialization",
            }
        )
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    after = dev_path.stat()
    after_identity = {
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
    }
    if after_identity != observed_identity:
        raise RuntimeError("E161 development H5AD changed during validation read")

    truth: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    validation_obs = validation.obs["condition"].astype(str).to_numpy()
    validation_pca = np.asarray(validation.obsm["X_pca"], dtype=np.float64)
    for condition in conditions:
        mask = validation_obs == condition
        n_cells = int(mask.sum())
        if n_cells <= 0:
            raise RuntimeError(f"No E161 validation cells for {condition}")
        raw_mean = dense_mean(validation.X[mask, :])
        z_mean = validation_pca[mask, :].mean(axis=0)
        z_recomputed = (raw_mean - pca_mean) @ components.T
        max_delta = float(np.max(np.abs(z_mean - z_recomputed)))
        if not np.isfinite(max_delta) or max_delta > PCA_TRANSFORM_ATOL:
            raise RuntimeError(
                f"PCA mean transform audit failed for {condition}: {max_delta}"
            )
        truth[condition] = {
            "n_cells": n_cells,
            "raw_mean": raw_mean,
            "pca_mean": z_mean,
        }
        genes = condition.split("+")
        audit_rows.append(
            {
                "condition": condition,
                "gene_a": genes[0],
                "gene_b": genes[1],
                "n_validation_cells": n_cells,
                "pca10_mean_recompute_max_abs_delta": max_delta,
                "raw_selected_gene_mean_all_finite": bool(np.isfinite(raw_mean).all()),
                "pca10_mean_all_finite": bool(np.isfinite(z_mean).all()),
            }
        )
    return truth, pd.DataFrame(audit_rows), pd.DataFrame(access_rows)


def pearson_vector(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if (
        len(left) != len(right)
        or len(left) < 2
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.std(left, ddof=1) <= ESTIMABILITY_STD
        or np.std(right, ddof=1) <= ESTIMABILITY_STD
    ):
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def build_task_metrics(
    score_tables: dict[int, pd.DataFrame],
    conditions: list[str],
    truth: dict[str, dict[str, Any]],
    pca_mean: np.ndarray,
    components: np.ndarray,
    control: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    truth_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for condition in conditions:
        record = truth[condition]
        pca_effect = record["pca_mean"] @ components + pca_mean - control
        raw_effect = record["raw_mean"] - control
        if not np.isfinite(pca_effect).all() or not np.isfinite(raw_effect).all():
            raise RuntimeError(f"Non-finite validation truth effect: {condition}")
        truth_cache[condition] = (pca_effect, raw_effect)
        truth_rows.append(
            {
                "condition": condition,
                "n_validation_cells": record["n_cells"],
                "pca10_truth_effect_rms": float(np.sqrt(np.mean(pca_effect**2))),
                "raw_selected_truth_effect_rms": float(np.sqrt(np.mean(raw_effect**2))),
                "pca10_vs_raw_truth_pearson": pearson_vector(pca_effect, raw_effect),
                "raw_truth_is_mandatory_sensitivity": True,
            }
        )

    for seed in SEEDS:
        table = score_tables[seed]
        for _, score_row in table.iterrows():
            condition = str(score_row["condition"])
            predicted_pca = np.asarray(
                [score_row[f"predicted_pca_{index}"] for index in range(N_PCA)],
                dtype=np.float64,
            )
            predicted_effect = predicted_pca @ components + pca_mean - control
            pca_effect, raw_effect = truth_cache[condition]
            pca_accuracy = pearson_vector(predicted_effect, pca_effect)
            raw_accuracy = pearson_vector(predicted_effect, raw_effect)
            pca_rmse = float(np.sqrt(np.mean((predicted_effect - pca_effect) ** 2)))
            raw_rmse = float(np.sqrt(np.mean((predicted_effect - raw_effect) ** 2)))
            gene_a, gene_b = condition.split("+")
            rows.append(
                {
                    "seed": seed,
                    "seed_role": "main" if seed == MAIN_SEED else "training_sensitivity",
                    "split": "validation",
                    "condition": condition,
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "n_validation_cells": truth[condition]["n_cells"],
                    "raw_log_prob": float(score_row["raw_log_prob"]),
                    "raw_log_prob_orientation": "higher_is_more_confident",
                    "pearson_effect_accuracy": pca_accuracy,
                    "rmse_effect_error": pca_rmse,
                    "raw_pearson_effect_accuracy_sensitivity": raw_accuracy,
                    "raw_rmse_effect_error_sensitivity": raw_rmse,
                    "primary_truth": "train_only_PCA10_inverse_transform_effect",
                    "raw_selected_truth_sensitivity_mandatory": True,
                    "test_label_queried": False,
                    "test_X_accessed": False,
                    "test_truth_accessed": False,
                }
            )
    frame = pd.DataFrame(rows)
    metric_columns = [endpoint for endpoint, _, _ in ENDPOINTS]
    if len(frame) != N_TASKS * len(SEEDS) or not np.isfinite(
        frame[["raw_log_prob", *metric_columns]].to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError("E163 task metrics are incomplete or non-finite")
    return frame, pd.DataFrame(truth_rows)


def association_record(
    score: np.ndarray,
    endpoint: np.ndarray,
    *,
    seed: int,
    endpoint_name: str,
    expected_relation: str,
    role: str,
) -> dict[str, Any]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    endpoint = np.asarray(endpoint, dtype=np.float64).reshape(-1)
    score_finite = bool(np.isfinite(score).all())
    endpoint_finite = bool(np.isfinite(endpoint).all())
    score_unique = int(np.unique(score).size) if score_finite else 0
    endpoint_unique = int(np.unique(endpoint).size) if endpoint_finite else 0
    score_std = float(np.std(score, ddof=1)) if score_finite and len(score) > 1 else None
    endpoint_std = (
        float(np.std(endpoint, ddof=1)) if endpoint_finite and len(endpoint) > 1 else None
    )
    failure_codes: list[str] = []
    if len(score) != N_TASKS or len(endpoint) != N_TASKS:
        failure_codes.append("wrong_task_count")
    if not score_finite:
        failure_codes.append("nonfinite_score")
    if not endpoint_finite:
        failure_codes.append("nonfinite_endpoint")
    if score_unique < 2:
        failure_codes.append("constant_score")
    if endpoint_unique < 2:
        failure_codes.append("constant_endpoint")
    if score_std is None or score_std <= ESTIMABILITY_STD:
        failure_codes.append("score_sd_below_threshold")
    if endpoint_std is None or endpoint_std <= ESTIMABILITY_STD:
        failure_codes.append("endpoint_sd_below_threshold")
    estimable = not failure_codes
    rho: Optional[float] = None
    pvalue: Optional[float] = None
    if estimable:
        result = spearmanr(score, endpoint)
        if np.isfinite(result.statistic) and np.isfinite(result.pvalue):
            rho = float(result.statistic)
            pvalue = float(result.pvalue)
        else:
            estimable = False
            failure_codes.append("nonfinite_spearman")
    return {
        "seed": seed,
        "seed_role": "main" if seed == MAIN_SEED else "training_sensitivity",
        "score": "raw_log_prob",
        "score_orientation": "higher_is_more_confident",
        "endpoint": endpoint_name,
        "endpoint_role": role,
        "expected_relation": expected_relation,
        "n_tasks": len(score),
        "score_all_finite": score_finite,
        "endpoint_all_finite": endpoint_finite,
        "score_exact_unique": score_unique,
        "endpoint_exact_unique": endpoint_unique,
        "score_sample_std_ddof1": score_std,
        "endpoint_sample_std_ddof1": endpoint_std,
        "estimable": estimable,
        "spearman_rho": rho,
        "two_sided_asymptotic_pvalue_descriptive": pvalue,
        "failure_code": "" if estimable else ";".join(failure_codes),
        "participates_in_authorization_gate": endpoint_name == "pearson_effect_accuracy",
    }


def build_associations(task_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        subset = task_metrics.loc[task_metrics["seed"] == seed].copy()
        if subset["condition"].nunique() != N_TASKS:
            raise RuntimeError(f"Seed {seed} task metric condition count changed")
        score = subset["raw_log_prob"].to_numpy(dtype=np.float64)
        for endpoint, expected, role in ENDPOINTS:
            rows.append(
                association_record(
                    score,
                    subset[endpoint].to_numpy(dtype=np.float64),
                    seed=seed,
                    endpoint_name=endpoint,
                    expected_relation=expected,
                    role=role,
                )
            )
    return pd.DataFrame(rows)


def fast_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if (
        len(left) < 2
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.unique(left).size < 2
        or np.unique(right).size < 2
    ):
        return float("nan")
    ranked_left = rankdata(left, method="average")
    ranked_right = rankdata(right, method="average")
    if np.std(ranked_left, ddof=1) <= 0 or np.std(ranked_right, ddof=1) <= 0:
        return float("nan")
    return float(np.corrcoef(ranked_left, ranked_right)[0, 1])


def ci_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    valid_n = int(len(finite))
    if valid_n < MIN_VALID_BOOTSTRAP:
        return {
            "valid_replicates": valid_n,
            "invalid_replicates": int(len(values) - valid_n),
            "bootstrap_mean": None,
            "bootstrap_median": None,
            "ci95_low": None,
            "ci95_high": None,
            "ci_status": "NA_fewer_than_9500_valid_replicates",
        }
    low, high = np.quantile(finite, [0.025, 0.975], method="linear")
    return {
        "valid_replicates": valid_n,
        "invalid_replicates": int(len(values) - valid_n),
        "bootstrap_mean": float(np.mean(finite)),
        "bootstrap_median": float(np.median(finite)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci_status": "estimable",
    }


def task_bootstrap(
    task_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed = {
        seed: task_metrics.loc[task_metrics["seed"] == seed]
        .sort_values("condition")
        .reset_index(drop=True)
        for seed in SEEDS
    }
    orders = [frame["condition"].tolist() for frame in by_seed.values()]
    if any(order != orders[0] for order in orders[1:]):
        raise RuntimeError("Task bootstrap seed condition order differs")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, N_TASKS, size=(N_BOOTSTRAP, N_TASKS), endpoint=False)
    replicate_rows: list[dict[str, Any]] = []
    storage: dict[tuple[int, str], np.ndarray] = {
        (seed, endpoint): np.full(N_BOOTSTRAP, np.nan, dtype=np.float64)
        for seed in SEEDS
        for endpoint, _, _ in ENDPOINTS
    }
    for replicate, indices in enumerate(draws):
        row: dict[str, Any] = {
            "replicate": replicate,
            "resample_index_sha256": sha256_bytes(indices.astype("<i8").tobytes()),
            "resample_size": len(indices),
        }
        for seed in SEEDS:
            frame = by_seed[seed]
            score = frame["raw_log_prob"].to_numpy(dtype=np.float64)[indices]
            for endpoint, _, _ in ENDPOINTS:
                rho = fast_spearman(
                    score, frame[endpoint].to_numpy(dtype=np.float64)[indices]
                )
                storage[(seed, endpoint)][replicate] = rho
                row[f"rho_seed{seed}__{endpoint}"] = rho if np.isfinite(rho) else None
        replicate_rows.append(row)
    summary_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for endpoint, expected, role in ENDPOINTS:
            frame = by_seed[seed]
            point = fast_spearman(
                frame["raw_log_prob"].to_numpy(dtype=np.float64),
                frame[endpoint].to_numpy(dtype=np.float64),
            )
            summary_rows.append(
                {
                    "bootstrap_type": "task",
                    "seed": seed,
                    "endpoint": endpoint,
                    "endpoint_role": role,
                    "expected_relation": expected,
                    "point_spearman_rho": point,
                    "rng": "numpy.random.default_rng",
                    "rng_seed": BOOTSTRAP_SEED,
                    "replicates": N_BOOTSTRAP,
                    "resample_unit": "validation_task",
                    "resample_size": N_TASKS,
                    **ci_summary(storage[(seed, endpoint)]),
                    "participates_in_authorization_gate": False,
                }
            )
    return pd.DataFrame(replicate_rows), pd.DataFrame(summary_rows)


def component_genes(task_metrics: pd.DataFrame) -> list[str]:
    main = task_metrics.loc[task_metrics["seed"] == MAIN_SEED]
    return sorted(set(main["gene_a"].astype(str)) | set(main["gene_b"].astype(str)))


def gene_cluster_bootstrap(
    task_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed = {
        seed: task_metrics.loc[task_metrics["seed"] == seed]
        .sort_values("condition")
        .reset_index(drop=True)
        for seed in SEEDS
    }
    genes = component_genes(task_metrics)
    gene_to_indices = {
        gene: np.flatnonzero(
            (by_seed[MAIN_SEED]["gene_a"].to_numpy(str) == gene)
            | (by_seed[MAIN_SEED]["gene_b"].to_numpy(str) == gene)
        )
        for gene in genes
    }
    if any(len(indices) == 0 for indices in gene_to_indices.values()):
        raise RuntimeError("Empty component-gene cluster")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    storage = {
        seed: np.full(N_BOOTSTRAP, np.nan, dtype=np.float64) for seed in SEEDS
    }
    rows: list[dict[str, Any]] = []
    for replicate in range(N_BOOTSTRAP):
        drawn_positions = rng.integers(0, len(genes), size=len(genes), endpoint=False)
        drawn_genes = [genes[index] for index in drawn_positions]
        indices = np.concatenate([gene_to_indices[gene] for gene in drawn_genes])
        row: dict[str, Any] = {
            "replicate": replicate,
            "gene_draw_sha256": sha256_text("\n".join(drawn_genes) + "\n"),
            "n_unique_component_genes": len(genes),
            "drawn_gene_count": len(drawn_genes),
            "task_multiset_size": len(indices),
        }
        for seed in SEEDS:
            frame = by_seed[seed]
            rho = fast_spearman(
                frame["raw_log_prob"].to_numpy(dtype=np.float64)[indices],
                frame["pearson_effect_accuracy"].to_numpy(dtype=np.float64)[indices],
            )
            storage[seed][replicate] = rho
            row[f"rho_seed{seed}__pearson_effect_accuracy"] = (
                rho if np.isfinite(rho) else None
            )
        rows.append(row)
    summary_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        frame = by_seed[seed]
        point = fast_spearman(
            frame["raw_log_prob"].to_numpy(dtype=np.float64),
            frame["pearson_effect_accuracy"].to_numpy(dtype=np.float64),
        )
        summary_rows.append(
            {
                "bootstrap_type": "component_gene_cluster",
                "seed": seed,
                "endpoint": "pearson_effect_accuracy",
                "point_spearman_rho": point,
                "rng": "numpy.random.default_rng",
                "rng_seed": BOOTSTRAP_SEED,
                "replicates": N_BOOTSTRAP,
                "n_unique_component_genes": len(genes),
                "resample_algorithm": (
                    "sample K genes with replacement; append every task containing each draw"
                ),
                **ci_summary(storage[seed]),
                "participates_in_authorization_gate": False,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def leave_one_gene_out(
    task_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = component_genes(task_metrics)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        frame = task_metrics.loc[task_metrics["seed"] == seed].copy()
        for gene in genes:
            removed = (frame["gene_a"].astype(str) == gene) | (
                frame["gene_b"].astype(str) == gene
            )
            remaining = frame.loc[~removed]
            record = association_record(
                remaining["raw_log_prob"].to_numpy(dtype=np.float64),
                remaining["pearson_effect_accuracy"].to_numpy(dtype=np.float64),
                seed=seed,
                endpoint_name="pearson_effect_accuracy",
                expected_relation="positive",
                role="LOGO_dependency_sensitivity",
            )
            # association_record expects 24 for the formal endpoint; LOGO has fewer by design.
            logo_failures = [
                value
                for value in str(record["failure_code"]).split(";")
                if value and value != "wrong_task_count"
            ]
            estimable = not logo_failures
            rho: Optional[float] = None
            if estimable:
                value = fast_spearman(
                    remaining["raw_log_prob"].to_numpy(dtype=np.float64),
                    remaining["pearson_effect_accuracy"].to_numpy(dtype=np.float64),
                )
                if np.isfinite(value):
                    rho = value
                else:
                    estimable = False
                    logo_failures.append("nonfinite_spearman")
            rows.append(
                {
                    "seed": seed,
                    "removed_gene": gene,
                    "removed_task_count": int(removed.sum()),
                    "remaining_task_count": int((~removed).sum()),
                    "spearman_rho": rho,
                    "estimable": estimable,
                    "failure_code": "" if estimable else ";".join(logo_failures),
                    "participates_in_authorization_gate": False,
                }
            )
    table = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for seed in SEEDS:
        values = table.loc[
            (table["seed"] == seed) & table["estimable"], "spearman_rho"
        ].to_numpy(dtype=np.float64)
        summaries.append(
            {
                "seed": seed,
                "n_component_genes": len(genes),
                "n_estimable_LOGO_rhos": int(len(values)),
                "rho_min": float(np.min(values)) if len(values) else None,
                "rho_median": float(np.median(values)) if len(values) else None,
                "rho_max": float(np.max(values)) if len(values) else None,
                "positive_fraction": float(np.mean(values > 0)) if len(values) else None,
                "participates_in_authorization_gate": False,
            }
        )
    return table, pd.DataFrame(summaries)


def build_gate(associations: pd.DataFrame) -> dict[str, Any]:
    main_rows = associations.loc[
        associations["endpoint"] == "pearson_effect_accuracy"
    ].copy()
    if set(main_rows["seed"].astype(int)) != set(SEEDS) or len(main_rows) != len(SEEDS):
        raise RuntimeError("Missing primary association seed")
    estimable = {
        int(row.seed): bool(row.estimable) for row in main_rows.itertuples(index=False)
    }
    rhos = {
        int(row.seed): (float(row.spearman_rho) if bool(row.estimable) else None)
        for row in main_rows.itertuples(index=False)
    }
    all_estimable = all(estimable.values())
    main_positive = bool(all_estimable and rhos[MAIN_SEED] is not None and rhos[MAIN_SEED] > 0)
    positive_seed_count = int(
        sum(value is not None and value > 0 for value in rhos.values())
    )
    at_least_two_positive = bool(all_estimable and positive_seed_count >= 2)
    authorize = bool(all_estimable and main_positive and at_least_two_positive)
    return {
        "schema": "safeconf_e163_authorization_gate_v1",
        "analysis_role": "validation_informed_futility_diagnostic_not_external_confirmation",
        "score": "raw_log_prob",
        "score_orientation": "higher_is_more_confident",
        "primary_accuracy": "PCA10_inverse_transform_own_model_Pearson",
        "main_seed": MAIN_SEED,
        "seed_primary_estimable": {str(seed): estimable[seed] for seed in SEEDS},
        "seed_primary_spearman_rho": {str(seed): rhos[seed] for seed in SEEDS},
        "condition_1_all_three_primary_associations_estimable": all_estimable,
        "condition_2_main_seed_rho_strictly_positive": main_positive,
        "condition_3_at_least_two_of_three_seed_rhos_strictly_positive": at_least_two_positive,
        "positive_seed_count": positive_seed_count,
        "authorize_future_test_label_lock": authorize,
        "decision": (
            "allow_new_test_label_only_preregistration"
            if authorize
            else "stop_raw_score_path"
        ),
        "bootstrap_cluster_LOGO_or_raw_sensitivity_alter_gate": False,
        "test_label_query_authorized_inside_E163": False,
        "test_label_queried": False,
        "test_X_accessed": False,
        "test_truth_accessed": False,
        "interpretation_boundary": (
            "Passing permits only a separately committed future test-label-only contract; "
            "it is not model validation or confirmatory success."
        ),
    }


def render_report(
    associations: pd.DataFrame,
    task_bootstrap_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    logo_summary: pd.DataFrame,
    gate: dict[str, Any],
) -> str:
    primary = associations.loc[
        associations["endpoint"] == "pearson_effect_accuracy",
        ["seed", "spearman_rho", "estimable"],
    ].sort_values("seed")
    raw_pearson = associations.loc[
        associations["endpoint"] == "raw_pearson_effect_accuracy_sensitivity",
        ["seed", "spearman_rho", "estimable"],
    ].sort_values("seed")
    raw_rmse = associations.loc[
        associations["endpoint"] == "raw_rmse_effect_error_sensitivity",
        ["seed", "spearman_rho", "estimable"],
    ].sort_values("seed")
    boot = task_bootstrap_summary.loc[
        task_bootstrap_summary["endpoint"] == "pearson_effect_accuracy",
        ["seed", "ci95_low", "ci95_high", "valid_replicates"],
    ].sort_values("seed")
    cluster = cluster_summary[["seed", "ci95_low", "ci95_high", "valid_replicates"]].sort_values(
        "seed"
    )
    logo = logo_summary.sort_values("seed")

    def fmt(value: Any) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "NA"
        return f"{float(value):.4f}"

    lines = [
        "# E163 Wessels validation-only raw-score futility diagnostic",
        "",
        f"运行决策：`{gate['decision']}`。",
        "",
        "E163 只使用 E161 train/validation development 资产和 E162 已锁定的 validation 分数。"
        "Wessels test label、test expression、test truth 与 raw Wessels H5AD 均未访问。",
        "",
        "## 主要关联",
        "",
        "| seed | raw_log_prob → PCA10 own-model Pearson accuracy rho | estimable | task bootstrap 95% CI | component-gene bootstrap 95% CI | LOGO min / median / max |",
        "|---:|---:|:---:|:---:|:---:|:---:|",
    ]
    for seed in SEEDS:
        p = primary.loc[primary["seed"] == seed].iloc[0]
        b = boot.loc[boot["seed"] == seed].iloc[0]
        c = cluster.loc[cluster["seed"] == seed].iloc[0]
        l = logo.loc[logo["seed"] == seed].iloc[0]
        lines.append(
            f"| {seed} | {fmt(p['spearman_rho'])} | {bool(p['estimable'])} | "
            f"[{fmt(b['ci95_low'])}, {fmt(b['ci95_high'])}] | "
            f"[{fmt(c['ci95_low'])}, {fmt(c['ci95_high'])}] | "
            f"{fmt(l['rho_min'])} / {fmt(l['rho_median'])} / {fmt(l['rho_max'])} |"
        )
    lines.extend(
        [
            "",
            "Authorization gate 固定要求三个主要关联均可估计、3407 rho>0、且至少两个种子 rho>0。"
            f"本次 gate 为 `{gate['authorize_future_test_label_lock']}`。bootstrap、cluster bootstrap 和 LOGO 只描述不确定性及共享组分依赖，不参与 gate。",
            "",
            "## 强制 raw selected-gene truth 敏感性",
            "",
            "| seed | raw_log_prob → raw-truth Pearson rho | estimable | raw_log_prob → raw-truth RMSE rho | estimable |",
            "|---:|---:|:---:|---:|:---:|",
        ]
    )
    for seed in SEEDS:
        pearson_row = raw_pearson.loc[raw_pearson["seed"] == seed].iloc[0]
        rmse_row = raw_rmse.loc[raw_rmse["seed"] == seed].iloc[0]
        lines.append(
            f"| {seed} | {fmt(pearson_row['spearman_rho'])} | "
            f"{bool(pearson_row['estimable'])} | {fmt(rmse_row['spearman_rho'])} | "
            f"{bool(rmse_row['estimable'])} |"
        )
    lines.extend(
        [
            "",
            "raw truth 结果是预先强制的敏感性。它与 PCA10 主真值方向不一致时也必须保留，不能据此替换主 endpoint。",
            "",
            "## 结论边界",
            "",
            "E163 是看见 E162 prediction collapse 和 raw-score variation 后进行的 validation-informed 去留诊断。"
            "即使 gate 通过，也只允许另行提交一个新的 test-label-only 预注册步骤；不能写成外部验证、确认成功、模型预测非退化或 SafeConf 已在 Wessels 得到支持。",
            "",
        ]
    )
    return "\n".join(lines)


def build_status(
    *,
    git_head: str,
    started_at: str,
    completed_at: str,
    dev_stat: dict[str, Any],
    gate: dict[str, Any],
    input_manifest: pd.DataFrame,
    source_gate: pd.DataFrame,
    access_ledger: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "analysis_timing": "post_E162_training_validation_informed_pre_test_unseal",
        "phase": "complete_validation_only_futility_diagnostic_no_test_label_or_X_access",
        "started_at": started_at,
        "completed_at": completed_at,
        "git_head": git_head,
        "main_seed": MAIN_SEED,
        "training_sensitivity_seeds": [3408, 3409],
        "n_validation_tasks": N_TASKS,
        "n_validation_cells_materialized": 5102,
        "primary_score": "raw_log_prob_higher_is_more_confident",
        "primary_accuracy": "PCA10_inverse_transform_own_model_Pearson",
        "secondary_accuracy": "PCA10_inverse_transform_own_model_RMSE",
        "raw_selected_gene_truth_sensitivity_mandatory_and_reported": True,
        "authorization_gate": gate,
        "task_bootstrap": {
            "rng": "numpy.random.default_rng",
            "seed": BOOTSTRAP_SEED,
            "replicates": N_BOOTSTRAP,
            "resample_size": N_TASKS,
            "minimum_valid_replicates": MIN_VALID_BOOTSTRAP,
            "alters_authorization_gate": False,
        },
        "component_gene_cluster_bootstrap": {
            "rng": "numpy.random.default_rng",
            "seed": BOOTSTRAP_SEED,
            "replicates": N_BOOTSTRAP,
            "alters_authorization_gate": False,
        },
        "leave_one_gene_out_alters_authorization_gate": False,
        "E161_development_h5ad_integrity": dev_stat,
        "allowlisted_input_count": int(len(input_manifest)),
        "all_input_hashes_match": bool(
            (input_manifest["expected_sha256"] == input_manifest["observed_sha256"]).all()
        ),
        "all_frozen_sources_match_HEAD": bool(source_gate["byte_identical"].all()),
        "development_h5ad_opened": True,
        "raw_Wessels_h5ad_opened": False,
        "E160_test_condition_list_opened": False,
        "E162b_test_label_artifact_opened": False,
        "test_label_queried": False,
        "test_X_accessed": False,
        "test_truth_accessed": False,
        "test_endpoint_computed": False,
        "access_ledger_test_rows_present_or_materialized": int(
            access_ledger["test_rows_present"].max()
        ),
        "scientific_claim_boundary": (
            "validation-only futility/authorization diagnostic; never external or confirmatory evidence"
        ),
    }


def build_downstream_interface(
    gate: dict[str, Any],
    git_head: str,
    core_artifacts: dict[str, bytes],
) -> dict[str, Any]:
    """Minimal immutable hand-off; downstream still needs its own committed contract."""
    handoff_paths = {
        "authorization_gate": "E163_AUTHORIZATION_GATE.json",
        "run_status": "RUN_STATUS.json",
        "task_metrics": "tables/E163_TASK_METRICS.csv",
        "associations": "tables/E163_ASSOCIATIONS.csv",
        "git_source_gate": "manifests/E163_GIT_SOURCE_GATE.csv",
        "input_manifest": "manifests/E163_INPUT_MANIFEST.csv",
        "development_access_ledger": "manifests/E163_DEV_H5AD_ACCESS_LEDGER.csv",
    }
    integrity = {
        role: {
            "release_relative_path": relative,
            "bytes": len(core_artifacts[relative]),
            "sha256": sha256_bytes(core_artifacts[relative]),
        }
        for role, relative in handoff_paths.items()
    }
    return {
        "schema": "safeconf_e163_to_e164_v1",
        "experiment": EXPERIMENT,
        "phase": "validation_only_futility_diagnostic_complete_test_remains_sealed",
        "git_head_at_E163_execution": git_head,
        "authorization_gate_sha256": sha256_bytes(json_bytes(gate)),
        "validation_gate_passed": bool(gate["authorize_future_test_label_lock"]),
        "authorize_future_test_label_lock": bool(
            gate["authorize_future_test_label_lock"]
        ),
        "decision": gate["decision"],
        "main_seed": MAIN_SEED,
        "primary_score": "raw_log_prob",
        "score_orientation": "higher_is_more_confident",
        "primary_accuracy": "PCA10_inverse_transform_own_model_Pearson",
        "seed_primary_spearman_rho": gate["seed_primary_spearman_rho"],
        "downstream_permission_if_authorized": (
            "write_and_commit_a_new_test_label_only_contract; no test truth access"
        ),
        "downstream_permission_if_not_authorized": "none_stop_raw_score_path",
        "E163_is_external_or_confirmatory_evidence": False,
        "artifact_integrity": integrity,
        "release_manifest_relative_path": "RESULTS_SHA256.csv",
        "test_label_queried": False,
        "test_X_accessed": False,
        "test_truth_accessed": False,
    }


def verify_release(output_root: Path) -> None:
    release = output_root / "release"
    manifest_path = release / "RESULTS_SHA256.csv"
    if not manifest_path.is_file():
        raise RuntimeError("Missing E163 release manifest")
    manifest = pd.read_csv(manifest_path)
    expected_columns = {"relative_path", "bytes", "sha256"}
    if set(manifest.columns) != expected_columns:
        raise RuntimeError("E163 manifest schema changed")
    observed_files = sorted(
        str(path.relative_to(release))
        for path in release.rglob("*")
        if path.is_file() and path.name != "RESULTS_SHA256.csv"
    )
    if observed_files != sorted(manifest["relative_path"].astype(str).tolist()):
        raise RuntimeError("E163 release file allowlist differs from manifest")
    for row in manifest.itertuples(index=False):
        path = release / str(row.relative_path)
        if path.is_symlink() or path.stat().st_size != int(row.bytes):
            raise RuntimeError(f"E163 release size/symlink verification failed: {row.relative_path}")
        if sha256_file(path) != str(row.sha256):
            raise RuntimeError(f"E163 release hash verification failed: {row.relative_path}")
    print(f"PASS: verified {len(manifest)} E163 release artifacts")


def run_formal(repo: Path) -> None:
    output_root = repo / OUTPUT_REL
    release = output_root / "release"
    staging = output_root / ".release.staging"
    if release.exists():
        raise RuntimeError("E163 release already exists; use --verify, never overwrite")
    if staging.exists():
        raise RuntimeError("E163 staging exists; inspect it before any rerun")
    output_root.mkdir(parents=True, exist_ok=True)

    started_at = now_iso()
    git_head, source_gate = git_source_gate(repo)
    payloads, paths, input_manifest, dev_stat = load_and_hash_inputs(repo)
    validate_upstream_status(payloads)
    score_tables, conditions, score_audit = validate_scores_and_gates(payloads)

    selected_genes = payloads["E161_selected_gene_axis"].decode("utf-8").splitlines()
    if (
        len(selected_genes) != 2023
        or len(set(selected_genes)) != 2023
        or sha256_text("\n".join(selected_genes) + "\n") != SELECTED_GENE_SHA256
    ):
        raise RuntimeError("Selected-gene axis content changed")
    with np.load(io.BytesIO(payloads["E161_train_only_PCA10"]), allow_pickle=False) as pca:
        if set(pca.files) != {
            "model_genes",
            "raw_gene_indices",
            "mean",
            "components",
            "explained_variance",
            "explained_variance_ratio",
        }:
            raise RuntimeError("E161 PCA archive schema changed")
        model_genes = list(map(str, pca["model_genes"].tolist()))
        pca_mean = np.asarray(pca["mean"], dtype=np.float64)
        components = np.asarray(pca["components"], dtype=np.float64)
    if model_genes != selected_genes or pca_mean.shape != (2023,) or components.shape != (
        N_PCA,
        2023,
    ):
        raise RuntimeError("E161 PCA gene order or shape changed")
    with np.load(io.BytesIO(payloads["E161_train_control_prior"]), allow_pickle=False) as prior:
        if set(prior.files) != {
            "control_gene_mean",
            "control_pca_mean",
            "control_pca_cov",
            "n_train_controls",
        }:
            raise RuntimeError("E161 control-prior archive schema changed")
        control = np.asarray(prior["control_gene_mean"], dtype=np.float64)
        n_train_controls = int(np.asarray(prior["n_train_controls"]).reshape(-1)[0])
    if control.shape != (2023,) or n_train_controls != 424 or not np.isfinite(control).all():
        raise RuntimeError("E161 train-control prior changed")

    truth, pca_audit, access_ledger = load_development_truth(
        paths["E161_development_h5ad"],
        dev_stat,
        conditions,
        selected_genes,
        pca_mean,
        components,
    )
    task_metrics, truth_summary = build_task_metrics(
        score_tables, conditions, truth, pca_mean, components, control
    )
    associations = build_associations(task_metrics)
    task_replicates, task_summary = task_bootstrap(task_metrics)
    cluster_replicates, cluster_summary = gene_cluster_bootstrap(task_metrics)
    logo, logo_summary = leave_one_gene_out(task_metrics)
    gate = build_gate(associations)
    completed_at = now_iso()
    report = render_report(associations, task_summary, cluster_summary, logo_summary, gate)
    status = build_status(
        git_head=git_head,
        started_at=started_at,
        completed_at=completed_at,
        dev_stat=dev_stat,
        gate=gate,
        input_manifest=input_manifest,
        source_gate=source_gate,
        access_ledger=access_ledger,
    )
    core_payload_map: dict[str, bytes] = {
        "README_先看这个.md": report.encode("utf-8"),
        "RUN_STATUS.json": json_bytes(status),
        "E163_AUTHORIZATION_GATE.json": json_bytes(gate),
        "manifests/E163_GIT_SOURCE_GATE.csv": csv_bytes(source_gate),
        "manifests/E163_INPUT_MANIFEST.csv": csv_bytes(input_manifest),
        "manifests/E163_DEV_H5AD_ACCESS_LEDGER.csv": csv_bytes(access_ledger),
        "tables/E163_UPSTREAM_SCORE_AUDIT.csv": csv_bytes(score_audit),
        "tables/E163_PCA_TRANSFORM_AUDIT.csv": csv_bytes(pca_audit),
        "tables/E163_TRUTH_SUMMARY.csv": csv_bytes(truth_summary),
        "tables/E163_TASK_METRICS.csv": csv_bytes(task_metrics),
        "tables/E163_ASSOCIATIONS.csv": csv_bytes(associations),
        "tables/E163_TASK_BOOTSTRAP_REPLICATES.csv": csv_bytes(task_replicates),
        "tables/E163_TASK_BOOTSTRAP_SUMMARY.csv": csv_bytes(task_summary),
        "tables/E163_COMPONENT_GENE_BOOTSTRAP_REPLICATES.csv": csv_bytes(
            cluster_replicates
        ),
        "tables/E163_COMPONENT_GENE_BOOTSTRAP_SUMMARY.csv": csv_bytes(cluster_summary),
        "tables/E163_LEAVE_ONE_GENE_OUT.csv": csv_bytes(logo),
        "tables/E163_LEAVE_ONE_GENE_OUT_SUMMARY.csv": csv_bytes(logo_summary),
    }
    downstream_interface = build_downstream_interface(
        gate, git_head, core_payload_map
    )
    payload_map = {
        **core_payload_map,
        "E163_E164_INTERFACE.json": json_bytes(downstream_interface),
    }
    staging.mkdir(parents=False, exist_ok=False)
    for relative, payload in payload_map.items():
        fsync_write(staging / relative, payload)

    observed = sorted(
        str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file()
    )
    if observed != sorted(payload_map):
        raise RuntimeError("E163 staging allowlist changed")
    if any(path.is_symlink() for path in staging.rglob("*")):
        raise RuntimeError("E163 staging contains a symlink")
    manifest_rows = [
        {
            "relative_path": relative,
            "bytes": len(payload_map[relative]),
            "sha256": sha256_bytes(payload_map[relative]),
        }
        for relative in sorted(payload_map)
    ]
    fsync_write(staging / "RESULTS_SHA256.csv", csv_bytes(pd.DataFrame(manifest_rows)))
    for directory in sorted(
        [path for path in staging.rglob("*") if path.is_dir()], reverse=True
    ):
        fsync_directory(directory)
    fsync_directory(staging)
    fsync_directory(output_root)
    os.replace(staging, release)
    fsync_directory(output_root)
    verify_release(output_root)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing atomic E163 release without recomputing analysis",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    output_root = repo / OUTPUT_REL
    try:
        if args.verify:
            verify_release(output_root)
        else:
            run_formal(repo)
        return 0
    except Exception as exc:
        print(f"E163 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
