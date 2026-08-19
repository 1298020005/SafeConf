#!/usr/bin/env python3
"""Freeze Wessels train-only simple predictors before E163 test unsealing.

``preflight`` verifies committed inputs and opaque asset identities.  It never
opens the E161 development H5AD.  ``formal`` reads H5AD metadata through h5py
and materializes only the leading 11,779 train rows of the CSR ``X`` dataset.
Validation expression, test expression, raw Wessels data, graph caches and E162
model outputs are outside this runner's input boundary.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import traceback
import uuid
from datetime import datetime
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E162b_wessels_label_only_baselines_20260715"
CONTRACT = OUT / "ANALYSIS_CONTRACT.md"
STAGING = OUT / ".release.staging"
RELEASE = OUT / "release"
FAILURES = OUT / "failures"

E160 = ROOT / "docs/实验结果/E160_wessels_combination_contract_20260714"
E160_STATUS = E160 / "freeze/RUN_STATUS.json"
E160_SPLIT = E160 / "freeze/manifests/E160_set2conditions.json"
E161 = ROOT / "docs/实验结果/E161_wessels_trainval_preprocess_20260714"
E161_RELEASE = E161 / "release"
E161_STATUS = E161_RELEASE / "RUN_STATUS.json"
E161_REPO_MANIFEST = E161_RELEASE / "tables/E161_ASSET_MANIFEST.csv"
E161_ASSET = Path("/home/yyf/data/safeconf_e161_prescribe/wessels_e160")
E161_ASSET_MANIFEST = E161_ASSET / "ASSET_MANIFEST.csv"
E161_INTERFACE = E161_ASSET / "E161_E162_INTERFACE.json"

EXPECTED_PYTHON = Path("/home/yyf/.conda/envs/prescribe_env/bin/python")
EXPECTED_PYTHON_VERSION = "3.9.25"
EXPECTED_DISTRIBUTIONS = {
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "scipy": "1.13.1",
    "anndata": "0.10.8",
    "h5py": "3.14.0",
}

SEED = 3407
N_TRAIN = 11_779
N_VALIDATION = 5_102
N_DEV = 16_881
N_CONTROL = 424
N_SELECTED = 2_023
N_PCA = 10
N_TRAIN_CONDITIONS = 72
N_TRAIN_SINGLES = 27
N_TRAIN_PAIRS = 44
N_VALIDATION_CONDITIONS = 24
N_TEST = 48
RANDOM_PREFIX = "E162b|Wessels|random-confidence|3407\t"

BASELINE_ORDER = (
    "control",
    "cell_weighted_perturbed_mean",
    "matching_single_mean",
    "single_additive",
)
EXPECTED_RELEASE_FILES = frozenset(
    {
        ".E162b_TRANSACTION.json",
        "RUN_STATUS.json",
        "README_先看这个.md",
        "RESULTS_SHA256.csv",
        "E162b_E163_INTERFACE.json",
        "reports/E162b_REPORT.md",
        "profiles/E162b_TEST_POST_PROFILES.csv.gz",
        "profiles/E162b_TEST_EFFECT_PROFILES.csv.gz",
        "profiles/E162b_TEST_PCA10_COORDINATES.csv",
        "tables/E162b_TRAIN_REFERENCE_GENE_STATS.csv.gz",
        "tables/E162b_TRAIN_REFERENCE_PCA10_STATS.csv",
        "tables/E162b_TEST_TASKS.csv",
        "tables/E162b_RISK_BASELINES_WIDE.csv",
        "tables/E162b_RISK_BASELINES_LONG.csv",
        "tables/E162b_BASELINE_AUDIT.csv",
        "tables/E162b_X_ACCESS_LEDGER.csv",
        "tables/E162b_SOURCE_HASHES.csv",
        "tables/E162b_RUNTIME_ENVIRONMENT.csv",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "formal"), default="preflight")
    return parser.parse_args()


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path, role: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{role} must be a regular non-symlink file: {path}")


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def git_blob_gate(path: Path, head: str) -> dict[str, Any]:
    regular_file(path, "committed input")
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    try:
        blob = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Required input is not committed at HEAD: {relative}") from exc
    working = path.read_bytes()
    if working != blob:
        raise RuntimeError(f"Working file differs from HEAD blob: {relative}")
    return {
        "path": relative,
        "bytes": len(working),
        "sha256": hashlib.sha256(working).hexdigest(),
        "matches_git_head_blob": True,
    }


def runtime_gate() -> list[dict[str, Any]]:
    observed_executable = Path(sys.executable).resolve()
    if observed_executable != EXPECTED_PYTHON.resolve():
        raise RuntimeError(
            f"E162b requires {EXPECTED_PYTHON}; observed {observed_executable}"
        )
    observed_python = platform.python_version()
    if observed_python != EXPECTED_PYTHON_VERSION:
        raise RuntimeError(
            f"Python changed: {observed_python} != {EXPECTED_PYTHON_VERSION}"
        )
    rows: list[dict[str, Any]] = [
        {
            "component": "python",
            "expected_version": EXPECTED_PYTHON_VERSION,
            "observed_version": observed_python,
            "executable": str(observed_executable),
            "gate_passed": True,
        }
    ]
    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        observed = distribution_version(distribution)
        if observed != expected:
            raise RuntimeError(
                f"Dependency changed: {distribution} {observed} != {expected}"
            )
        rows.append(
            {
                "component": distribution,
                "expected_version": expected,
                "observed_version": observed,
                "executable": str(observed_executable),
                "gate_passed": True,
            }
        )
    return rows


def split_pair(condition: str) -> tuple[str, str]:
    parts = condition.split("+")
    if len(parts) != 2 or "ctrl" in parts or any(not part for part in parts):
        raise RuntimeError(f"Expected a canonical two-gene pair: {condition}")
    if parts != sorted(parts) or parts[0] == parts[1]:
        raise RuntimeError(f"Non-canonical pair ordering: {condition}")
    return parts[0], parts[1]


def validate_e160(head: str) -> dict[str, Any]:
    source_records = {
        "E160_status": git_blob_gate(E160_STATUS, head),
        "E160_split": git_blob_gate(E160_SPLIT, head),
    }
    status = json.loads(E160_STATUS.read_text(encoding="utf-8"))
    if status.get("phase") != "requirements_frozen_test_expression_unopened":
        raise RuntimeError("E160 is not the frozen, unopened-test contract")
    if status.get("raw_X_values_indexed_or_materialized") is not False:
        raise RuntimeError("E160 test-expression boundary changed")
    for relative, expected in status.get("artifact_sha256", {}).items():
        path = E160 / relative
        regular_file(path, f"E160 artifact {relative}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"E160 artifact hash mismatch: {relative}")
        git_blob_gate(path, head)

    frozen = json.loads(E160_SPLIT.read_text(encoding="utf-8"))
    splits = {
        role: [str(value) for value in frozen[role]]
        for role in ("train", "val", "test")
    }
    expected_counts = {
        "train": N_TRAIN_CONDITIONS,
        "val": N_VALIDATION_CONDITIONS,
        "test": N_TEST,
    }
    for role, expected in expected_counts.items():
        if len(splits[role]) != expected or len(set(splits[role])) != expected:
            raise RuntimeError(f"E160 {role} label count changed")
    split_sets = {role: set(values) for role, values in splits.items()}
    if any(
        split_sets[left] & split_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError("E160 condition splits overlap")
    for condition in splits["test"]:
        split_pair(condition)

    train_singles = [
        condition
        for condition in splits["train"]
        if condition != "ctrl" and "ctrl" in condition.split("+")
    ]
    train_pairs = [
        condition for condition in splits["train"] if "ctrl" not in condition.split("+")
    ]
    if (
        "ctrl" not in splits["train"]
        or len(train_singles) != N_TRAIN_SINGLES
        or len(train_pairs) != N_TRAIN_PAIRS
    ):
        raise RuntimeError("E160 train control/single/pair composition changed")
    single_by_gene: dict[str, str] = {}
    for condition in train_singles:
        parts = condition.split("+")
        if len(parts) != 2 or parts[1] != "ctrl" or parts[0] in single_by_gene:
            raise RuntimeError(f"Malformed train singleton: {condition}")
        single_by_gene[parts[0]] = condition
    missing = sorted(
        {
            gene
            for condition in splits["test"]
            for gene in split_pair(condition)
            if gene not in single_by_gene
        }
    )
    if missing:
        raise RuntimeError(f"Test components lack train singleton support: {missing}")
    return {
        "status": status,
        "splits": splits,
        "single_by_gene": single_by_gene,
        "train_pairs": train_pairs,
        "source_records": source_records,
    }


def validate_e161(head: str) -> dict[str, Any]:
    if E161_ASSET.is_symlink() or not E161_ASSET.is_dir():
        raise RuntimeError("E161 data root must be a regular non-symlink directory")
    source_records = {
        "E161_release_status": git_blob_gate(E161_STATUS, head),
        "E161_repo_asset_manifest": git_blob_gate(E161_REPO_MANIFEST, head),
    }
    status = json.loads(E161_STATUS.read_text(encoding="utf-8"))
    required_status = {
        "phase": "complete_preprocessing_and_dev_graphs_no_training_no_test_X_access",
        "train_conditions": N_TRAIN_CONDITIONS,
        "validation_conditions": N_VALIDATION_CONDITIONS,
        "test_conditions": N_TEST,
        "train_cells": N_TRAIN,
        "validation_cells": N_VALIDATION,
        "test_graphs": 0,
        "test_X_rows_indexed": False,
        "test_X_rows_materialized": False,
        "test_X_rows_transformed": False,
        "excluded_X_rows_indexed": False,
        "model_training_started": False,
        "predictions_generated": False,
        "test_endpoint_computed": False,
    }
    for key, expected in required_status.items():
        if status.get(key) != expected:
            raise RuntimeError(f"E161 status invariant changed: {key}")
    if Path(str(status.get("data_root"))) != E161_ASSET:
        raise RuntimeError("E161 data root changed")
    for relative, expected in status.get("artifact_sha256", {}).items():
        path = E161_RELEASE / relative
        regular_file(path, f"E161 release artifact {relative}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"E161 release artifact hash mismatch: {relative}")
        git_blob_gate(path, head)

    regular_file(E161_ASSET_MANIFEST, "E161 data asset manifest")
    if sha256_file(E161_ASSET_MANIFEST) != status.get("data_asset_manifest_sha256"):
        raise RuntimeError("E161 external asset manifest hash changed")
    if E161_ASSET_MANIFEST.read_bytes() != E161_REPO_MANIFEST.read_bytes():
        raise RuntimeError("E161 external and repository manifests differ")
    manifest = pd.read_csv(E161_ASSET_MANIFEST)
    if set(manifest.columns) != {"relative_path", "bytes", "sha256"}:
        raise RuntimeError("E161 asset manifest schema changed")
    manifest_rows: dict[str, dict[str, Any]] = {}
    for row in manifest.to_dict(orient="records"):
        relative = str(row["relative_path"])
        path = E161_ASSET / relative
        try:
            path.resolve().relative_to(E161_ASSET.resolve())
        except ValueError as exc:
            raise RuntimeError(f"E161 manifest path escapes the data root: {relative}") from exc
        regular_file(path, f"E161 data asset {relative}")
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != str(row["sha256"]):
            raise RuntimeError(f"E161 data asset identity changed: {relative}")
        manifest_rows[relative] = row
    if len(manifest_rows) != len(manifest):
        raise RuntimeError("E161 asset manifest contains duplicate paths")

    interface = json.loads(E161_INTERFACE.read_text(encoding="utf-8"))
    if interface.get("schema") != "safeconf_e161_to_e162_v2":
        raise RuntimeError("E161 interface is not v2")
    required_interface = {
        "data_root": str(E161_ASSET),
        "n_selected_genes": N_SELECTED,
        "split_conditions": {
            "train": N_TRAIN_CONDITIONS,
            "val": N_VALIDATION_CONDITIONS,
            "test": N_TEST,
        },
        "development_graphs": {"train": N_TRAIN, "val": N_VALIDATION, "test": 0},
        "test_X_rows_indexed_materialized_or_transformed": False,
        "engineered_construct_X_columns_indexed_or_materialized": False,
        "guide_barcode_X_columns_indexed_or_materialized": False,
        "excluded_X_columns_indexed_or_materialized": False,
    }
    for key, expected in required_interface.items():
        if interface.get(key) != expected:
            raise RuntimeError(f"E161 interface invariant changed: {key}")
    for relative, expected in interface.get("asset_sha256", {}).items():
        if relative not in manifest_rows or str(manifest_rows[relative]["sha256"]) != expected:
            raise RuntimeError(f"E161 interface/manifest mismatch: {relative}")

    required_paths = {
        "dev_h5ad": "perturb_processed.h5ad",
        "pca_model": "TRAIN_ONLY_PCA_MODEL.npz",
        "control_prior": "TRAIN_ONLY_CONTROL_PRIOR.npz",
        "selected_gene_axis": "SELECTED_GENE_AXIS.txt",
    }
    for key, relative in required_paths.items():
        if interface.get("paths", {}).get(key) != relative:
            raise RuntimeError(f"E161 required path changed: {key}")

    genes = (E161_ASSET / "SELECTED_GENE_AXIS.txt").read_text(encoding="utf-8").splitlines()
    if len(genes) != N_SELECTED or len(set(genes)) != N_SELECTED:
        raise RuntimeError("E161 selected gene axis changed")
    gene_axis_bytes = ("\n".join(genes) + "\n").encode("utf-8")
    if hashlib.sha256(gene_axis_bytes).hexdigest() != interface.get(
        "selected_gene_order_sha256"
    ):
        raise RuntimeError("E161 selected gene-order hash changed")

    with np.load(E161_ASSET / "TRAIN_ONLY_PCA_MODEL.npz", allow_pickle=False) as model:
        required_keys = {
            "model_genes",
            "raw_gene_indices",
            "mean",
            "components",
            "explained_variance",
            "explained_variance_ratio",
        }
        if set(model.files) != required_keys:
            raise RuntimeError("E161 PCA NPZ schema changed")
        model_genes = model["model_genes"].astype(str).tolist()
        pca_mean = np.asarray(model["mean"], dtype=np.float64)
        components = np.asarray(model["components"], dtype=np.float64)
        raw_indices = np.asarray(model["raw_gene_indices"], dtype=np.int64)
    if model_genes != genes or pca_mean.shape != (N_SELECTED,) or components.shape != (
        N_PCA,
        N_SELECTED,
    ):
        raise RuntimeError("E161 PCA axis/shape changed")
    if raw_indices.shape != (N_SELECTED,) or len(np.unique(raw_indices)) != N_SELECTED:
        raise RuntimeError("E161 PCA raw indices changed")
    if not np.isfinite(pca_mean).all() or not np.isfinite(components).all():
        raise RuntimeError("E161 PCA model contains non-finite values")

    with np.load(E161_ASSET / "TRAIN_ONLY_CONTROL_PRIOR.npz", allow_pickle=False) as prior:
        required_prior = {
            "control_gene_mean",
            "control_pca_mean",
            "control_pca_cov",
            "n_train_controls",
        }
        if set(prior.files) != required_prior:
            raise RuntimeError("E161 control prior schema changed")
        control_gene_mean = np.asarray(prior["control_gene_mean"], dtype=np.float64)
        n_train_controls = int(np.asarray(prior["n_train_controls"]).reshape(-1)[0])
    if control_gene_mean.shape != (N_SELECTED,) or n_train_controls != N_CONTROL:
        raise RuntimeError("E161 control prior shape/count changed")
    if not np.isfinite(control_gene_mean).all():
        raise RuntimeError("E161 control mean is non-finite")

    source_records.update(
        {
            "E161_external_asset_manifest": {
                "path": str(E161_ASSET_MANIFEST),
                "bytes": E161_ASSET_MANIFEST.stat().st_size,
                "sha256": sha256_file(E161_ASSET_MANIFEST),
                "matches_git_head_blob": False,
            },
            "E161_interface": {
                "path": str(E161_INTERFACE),
                "bytes": E161_INTERFACE.stat().st_size,
                "sha256": sha256_file(E161_INTERFACE),
                "matches_git_head_blob": False,
            },
        }
    )
    return {
        "status": status,
        "interface": interface,
        "manifest": manifest,
        "genes": genes,
        "pca_mean": pca_mean,
        "components": components,
        "control_prior_mean": control_gene_mean,
        "source_records": source_records,
    }


def common_preflight() -> dict[str, Any]:
    runtime = runtime_gate()
    head = git_head()
    committed = {
        "E162b_runner": git_blob_gate(RUNNER, head),
        "E162b_contract": git_blob_gate(CONTRACT, head),
    }
    e160 = validate_e160(head)
    e161 = validate_e161(head)
    return {
        "git_head": head,
        "runtime": runtime,
        "committed": committed,
        "e160": e160,
        "e161": e161,
    }


def decode_h5_strings(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in flat],
        dtype=object,
    )


def read_h5_column(group: h5py.Group, name: str) -> np.ndarray:
    node = group[name]
    if isinstance(node, h5py.Dataset):
        return decode_h5_strings(node[...])
    if not isinstance(node, h5py.Group) or "codes" not in node or "categories" not in node:
        raise RuntimeError(f"Unsupported H5AD column encoding: {group.name}/{name}")
    categories = decode_h5_strings(node["categories"][...])
    codes = np.asarray(node["codes"][...], dtype=np.int64)
    if np.any(codes < 0) or np.any(codes >= len(categories)):
        raise RuntimeError(f"Missing/invalid category code in {group.name}/{name}")
    return categories[codes]


def read_h5_index(group: h5py.Group) -> np.ndarray:
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    return read_h5_column(group, str(key))


def load_train_expression_only(
    dev_path: Path, expected_genes: list[str]
) -> tuple[sp.csr_matrix, np.ndarray, dict[str, Any]]:
    """Read metadata and only the contiguous train prefix of E161 CSR X."""

    before = dev_path.stat()
    with h5py.File(dev_path, "r") as handle:
        if "obs" not in handle or "var" not in handle or "X" not in handle:
            raise RuntimeError("Malformed E161 development H5AD")
        roles = read_h5_column(handle["obs"], "e161_split")
        conditions = read_h5_column(handle["obs"], "condition")
        genes = read_h5_index(handle["var"]).astype(str).tolist()
        if len(roles) != N_DEV or len(conditions) != N_DEV:
            raise RuntimeError("E161 development metadata row count changed")
        train_rows = np.flatnonzero(roles == "train")
        validation_rows = np.flatnonzero(roles == "val")
        if not np.array_equal(train_rows, np.arange(N_TRAIN, dtype=np.int64)):
            raise RuntimeError("Train rows are not the frozen contiguous H5AD prefix")
        if not np.array_equal(
            validation_rows, np.arange(N_TRAIN, N_DEV, dtype=np.int64)
        ):
            raise RuntimeError("Validation rows are not the frozen suffix")
        if genes != expected_genes:
            raise RuntimeError("Development H5AD var order differs from E161 selected axis")

        x_node = handle["X"]
        if isinstance(x_node, h5py.Dataset):
            if tuple(x_node.shape) != (N_DEV, N_SELECTED):
                raise RuntimeError("Dense development X shape changed")
            train_x = sp.csr_matrix(np.asarray(x_node[:N_TRAIN, :], dtype=np.float64))
            x_encoding = "dense_prefix_rows_only"
        elif isinstance(x_node, h5py.Group):
            encoding = x_node.attrs.get("encoding-type", "")
            if isinstance(encoding, bytes):
                encoding = encoding.decode("utf-8")
            shape = tuple(int(value) for value in x_node.attrs.get("shape", ()))
            if encoding != "csr_matrix" or shape != (N_DEV, N_SELECTED):
                raise RuntimeError(
                    f"Expected E161 CSR X {(N_DEV, N_SELECTED)}, found {encoding} {shape}"
                )
            indptr = np.asarray(x_node["indptr"][: N_TRAIN + 1], dtype=np.int64)
            if indptr.shape != (N_TRAIN + 1,) or indptr[0] != 0 or np.any(np.diff(indptr) < 0):
                raise RuntimeError("Malformed train-prefix CSR indptr")
            stop = int(indptr[-1])
            data = np.asarray(x_node["data"][:stop], dtype=np.float64)
            indices = np.asarray(x_node["indices"][:stop], dtype=np.int64)
            if len(data) != stop or len(indices) != stop:
                raise RuntimeError("Truncated train-prefix CSR payload")
            train_x = sp.csr_matrix(
                (data, indices, indptr), shape=(N_TRAIN, N_SELECTED), dtype=np.float64
            )
            x_encoding = "csr_prefix_rows_only"
        else:
            raise RuntimeError("Unsupported E161 X storage")
    after = dev_path.stat()
    identity = lambda stat_result: (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeError("E161 development H5AD changed during train read")
    if train_x.shape != (N_TRAIN, N_SELECTED) or not np.isfinite(train_x.data).all():
        raise RuntimeError("Train expression is malformed or non-finite")
    return train_x, conditions[:N_TRAIN].astype(str), {
        "storage": x_encoding,
        "train_rows_indexed": N_TRAIN,
        "validation_rows_indexed": 0,
        "test_rows_indexed": 0,
    }


def mean_and_sample_variance(matrix: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(matrix.shape[0])
    if n_rows < 2:
        raise RuntimeError("A frozen train reference group has fewer than two cells")
    total = np.asarray(matrix.sum(axis=0), dtype=np.float64).reshape(-1)
    total_square = np.asarray(matrix.power(2).sum(axis=0), dtype=np.float64).reshape(-1)
    mean = total / n_rows
    centered_ss = total_square - (total * total) / n_rows
    tolerance = 1e-8 * np.maximum(1.0, total_square)
    if np.any(centered_ss < -tolerance):
        raise RuntimeError("Numerically invalid negative sum of squares")
    variance = np.maximum(centered_ss, 0.0) / (n_rows - 1)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise RuntimeError("Non-finite train reference statistic")
    return mean, variance


def dense_mean_and_sample_variance(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.shape[0] < 2:
        raise RuntimeError("A PCA reference group has fewer than two cells")
    mean = np.mean(matrix, axis=0, dtype=np.float64)
    variance = np.var(matrix, axis=0, ddof=1, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise RuntimeError("Non-finite PCA reference statistic")
    return mean, variance


def hash_random_confidence(condition: str) -> tuple[float, str]:
    digest = hashlib.sha256((RANDOM_PREFIX + condition).encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(1 << 64), digest.hex()


def build_train_statistics(
    train_x: sp.csr_matrix,
    train_conditions: np.ndarray,
    e160: dict[str, Any],
    pca_mean: np.ndarray,
    components: np.ndarray,
) -> dict[str, Any]:
    observed = set(train_conditions.tolist())
    if observed != set(e160["splits"]["train"]):
        raise RuntimeError("Development train condition membership differs from E160")
    counts = pd.Series(train_conditions).value_counts().to_dict()
    if int(counts.get("ctrl", 0)) != N_CONTROL:
        raise RuntimeError("Train control cell count changed")
    if any(int(counts.get(condition, 0)) < 2 for condition in e160["splits"]["train"]):
        raise RuntimeError("A train condition has fewer than two cells")

    train_pca = np.asarray(train_x @ components.T, dtype=np.float64)
    train_pca -= np.asarray(pca_mean @ components.T, dtype=np.float64).reshape(1, -1)
    if train_pca.shape != (N_TRAIN, N_PCA) or not np.isfinite(train_pca).all():
        raise RuntimeError("Train-only PCA transform failed")

    group_specs: list[tuple[str, str, np.ndarray]] = [
        ("control", "ctrl", train_conditions == "ctrl"),
        (
            "cell_weighted_noncontrol",
            "ALL_NONCONTROL_TRAIN_CELLS",
            train_conditions != "ctrl",
        ),
    ]
    for gene, condition in e160["single_by_gene"].items():
        group_specs.append(("singleton", condition, train_conditions == condition))

    group_stats: dict[str, dict[str, Any]] = {}
    gene_rows: list[pd.DataFrame] = []
    pca_rows: list[dict[str, Any]] = []
    for role, group, mask in group_specs:
        indices = np.flatnonzero(mask)
        gene_mean, gene_variance = mean_and_sample_variance(train_x[indices])
        pca_group_mean, pca_variance = dense_mean_and_sample_variance(train_pca[indices])
        key = group
        group_stats[key] = {
            "role": role,
            "n_cells": int(len(indices)),
            "gene_mean": gene_mean,
            "gene_variance": gene_variance,
            "pca_mean": pca_group_mean,
            "pca_variance": pca_variance,
        }
        gene_rows.append(
            pd.DataFrame(
                {
                    "group_role": role,
                    "group": group,
                    "n_cells": int(len(indices)),
                    "gene_index_zero_based": np.arange(N_SELECTED, dtype=np.int64),
                    "mean": gene_mean,
                    "sample_variance_ddof1": gene_variance,
                }
            )
        )
        for coordinate in range(N_PCA):
            pca_rows.append(
                {
                    "group_role": role,
                    "group": group,
                    "n_cells": int(len(indices)),
                    "pca_coordinate": coordinate + 1,
                    "mean": pca_group_mean[coordinate],
                    "sample_variance_ddof1": pca_variance[coordinate],
                }
            )
    return {
        "groups": group_stats,
        "gene_table": pd.concat(gene_rows, ignore_index=True),
        "pca_table": pd.DataFrame(pca_rows),
        "condition_cell_counts": {key: int(value) for key, value in counts.items()},
    }


def build_predictions_and_risks(
    e160: dict[str, Any],
    stats: dict[str, Any],
    pca_mean: np.ndarray,
    components: np.ndarray,
) -> dict[str, Any]:
    groups = stats["groups"]
    mu0 = groups["ctrl"]["gene_mean"]
    mu_perturbed = groups["ALL_NONCONTROL_TRAIN_CELLS"]["gene_mean"]
    var0_gene = groups["ctrl"]["gene_variance"]
    var0_pca = groups["ctrl"]["pca_variance"]
    n0 = groups["ctrl"]["n_cells"]

    pair_degree: dict[str, int] = {gene: 0 for gene in e160["single_by_gene"]}
    for condition in e160["train_pairs"]:
        left, right = split_pair(condition)
        pair_degree[left] = pair_degree.get(left, 0) + 1
        pair_degree[right] = pair_degree.get(right, 0) + 1

    test_labels = e160["splits"]["test"]
    posts = np.empty((len(BASELINE_ORDER), N_TEST, N_SELECTED), dtype=np.float64)
    effects = np.empty_like(posts)
    post_pca = np.empty((len(BASELINE_ORDER), N_TEST, N_PCA), dtype=np.float64)
    effect_pca = np.empty_like(post_pca)
    tasks: list[dict[str, Any]] = []
    risks_wide: list[dict[str, Any]] = []
    risks_long: list[dict[str, Any]] = []
    ctrl_pca = (mu0 - pca_mean) @ components.T
    matching_post_formula_max_delta = 0.0

    for test_index, condition in enumerate(test_labels):
        gene_a, gene_b = split_pair(condition)
        cond_a = e160["single_by_gene"][gene_a]
        cond_b = e160["single_by_gene"][gene_b]
        group_a = groups[cond_a]
        group_b = groups[cond_b]
        mu_a = group_a["gene_mean"]
        mu_b = group_b["gene_mean"]
        delta_sum = (mu_a - mu0) + (mu_b - mu0)
        matching_effect = 0.5 * delta_sum
        additive_effect = 2.0 * matching_effect
        matching_post = mu0 + matching_effect
        matching_post_formula_max_delta = max(
            matching_post_formula_max_delta,
            float(np.max(np.abs(matching_post - 0.5 * (mu_a + mu_b)))),
        )
        baseline_posts = (
            mu0,
            mu_perturbed,
            matching_post,
            mu0 + additive_effect,
        )
        baseline_effects = (
            np.zeros(N_SELECTED, dtype=np.float64),
            mu_perturbed - mu0,
            matching_effect,
            additive_effect,
        )
        for baseline_index, (post, effect) in enumerate(
            zip(baseline_posts, baseline_effects)
        ):
            posts[baseline_index, test_index] = post
            effects[baseline_index, test_index] = effect
            post_coordinate = (post - pca_mean) @ components.T
            post_pca[baseline_index, test_index] = post_coordinate
            effect_pca[baseline_index, test_index] = post_coordinate - ctrl_pca

        n_a = int(group_a["n_cells"])
        n_b = int(group_b["n_cells"])
        degree_a = int(pair_degree[gene_a])
        degree_b = int(pair_degree[gene_b])
        gene_se = float(
            np.sqrt(
                np.mean(
                    group_a["gene_variance"] / (4.0 * n_a)
                    + group_b["gene_variance"] / (4.0 * n_b)
                    + var0_gene / n0
                )
            )
        )
        pca_se = float(
            np.sqrt(
                np.mean(
                    group_a["pca_variance"] / (4.0 * n_a)
                    + group_b["pca_variance"] / (4.0 * n_b)
                    + var0_pca / n0
                )
            )
        )
        matching_rms = float(np.sqrt(np.mean(matching_effect * matching_effect)))
        additive_rms = float(np.sqrt(np.mean(additive_effect * additive_effect)))
        random_confidence, random_digest = hash_random_confidence(condition)
        confidence_values = {
            "min_single_cell_count_confidence": float(np.log1p(min(n_a, n_b))),
            "min_train_pair_degree_confidence": float(
                np.log1p(min(degree_a, degree_b))
            ),
            "matching_se_pca10_confidence": -pca_se,
            "matching_se_gene_confidence": -gene_se,
            "matching_magnitude_confidence": -matching_rms,
            "hash_random_confidence": random_confidence,
            "constant_confidence": 0.0,
            "exact_pair_support_confidence": 0.0,
        }
        task_row = {
            "test_index": test_index,
            "condition": condition,
            "gene_a": gene_a,
            "gene_b": gene_b,
            "train_single_condition_a": cond_a,
            "train_single_condition_b": cond_b,
            "train_single_cells_a": n_a,
            "train_single_cells_b": n_b,
            "train_pair_degree_a": degree_a,
            "train_pair_degree_b": degree_b,
            "exact_train_pair_support": 0,
            "both_components_have_train_singletons": True,
            "test_input_role": "canonical_condition_string_only",
        }
        tasks.append(task_row)
        wide = {
            **task_row,
            "matching_effect_rms": matching_rms,
            "additive_effect_rms": additive_rms,
            "matching_se_pca10_raw": pca_se,
            "matching_se_gene_raw": gene_se,
            "hash_random_sha256": random_digest,
            **confidence_values,
            "raw_orientation": "matching_rms:higher_effect_size;se:higher_uncertainty",
            "analysis_orientation": "higher_expected_accuracy",
            "direction_frozen_before_E163": True,
        }
        risks_wide.append(wide)

        long_specs = (
            (
                "min_single_cell_count_confidence",
                float(min(n_a, n_b)),
                confidence_values["min_single_cell_count_confidence"],
                "higher_singleton_cell_support",
                "estimable",
            ),
            (
                "min_train_pair_degree_confidence",
                float(min(degree_a, degree_b)),
                confidence_values["min_train_pair_degree_confidence"],
                "higher_train_pair_degree",
                "estimable",
            ),
            (
                "matching_se_pca10_confidence",
                pca_se,
                confidence_values["matching_se_pca10_confidence"],
                "higher_uncertainty",
                "estimable",
            ),
            (
                "matching_se_gene_confidence",
                gene_se,
                confidence_values["matching_se_gene_confidence"],
                "higher_uncertainty",
                "estimable_sensitivity",
            ),
            (
                "matching_magnitude_confidence",
                matching_rms,
                confidence_values["matching_magnitude_confidence"],
                "higher_effect_size",
                "estimable",
            ),
            (
                "hash_random_confidence",
                random_confidence,
                random_confidence,
                "random_no_semantic_direction",
                "estimable_negative_control",
            ),
            (
                "constant_confidence",
                0.0,
                0.0,
                "constant",
                "constant_non_estimable",
            ),
            (
                "exact_pair_support_confidence",
                0.0,
                0.0,
                "higher_exact_pair_support",
                "constant_non_estimable_by_design",
            ),
        )
        for name, raw_value, score, raw_orientation, estimability in long_specs:
            risks_long.append(
                {
                    "test_index": test_index,
                    "condition": condition,
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "score_name": name,
                    "raw_value": raw_value,
                    "analysis_score": score,
                    "raw_orientation": raw_orientation,
                    "analysis_orientation": "higher_expected_accuracy",
                    "estimability_before_truth": estimability,
                    "direction_frozen_before_E163": True,
                }
            )

    if not np.isfinite(posts).all() or not np.isfinite(effects).all():
        raise RuntimeError("A simple baseline profile is non-finite")
    if not np.isfinite(post_pca).all() or not np.isfinite(effect_pca).all():
        raise RuntimeError("A simple baseline PCA coordinate is non-finite")
    if np.max(np.abs(effects[0])) != 0.0:
        raise RuntimeError("Control effect is not exactly zero")
    additive_delta = float(np.max(np.abs(effects[3] - 2.0 * effects[2])))
    if additive_delta != 0.0:
        raise RuntimeError("Additive effect is not exactly twice matching effect")
    unique_matching = int(np.unique(posts[2], axis=0).shape[0])
    unique_additive = int(np.unique(posts[3], axis=0).shape[0])
    if unique_matching < 24 or unique_additive < 24:
        raise RuntimeError("Matching/additive prediction non-degeneracy gate failed")
    risk_wide_frame = pd.DataFrame(risks_wide)
    additive_rms_delta = float(
        np.max(
            np.abs(
                risk_wide_frame["additive_effect_rms"].to_numpy(dtype=float)
                - 2.0
                * risk_wide_frame["matching_effect_rms"].to_numpy(dtype=float)
            )
        )
    )
    if additive_rms_delta > 1e-12:
        raise RuntimeError("Additive effect RMS is not twice matching effect RMS")
    random_unique = int(
        risk_wide_frame["hash_random_confidence"].nunique(dropna=False)
    )
    if random_unique < 24:
        raise RuntimeError("Hash-random negative control has an unexpected collision")

    return {
        "posts": posts,
        "effects": effects,
        "post_pca": post_pca,
        "effect_pca": effect_pca,
        "tasks": pd.DataFrame(tasks),
        "risk_wide": risk_wide_frame,
        "risk_long": pd.DataFrame(risks_long),
        "audit_values": {
            "control_effect_max_abs": float(np.max(np.abs(effects[0]))),
            "additive_minus_2x_matching_effect_max_abs": additive_delta,
            "matching_post_minus_single_mean_max_abs": matching_post_formula_max_delta,
            "additive_rms_minus_2x_matching_rms_max_abs": additive_rms_delta,
            "matching_unique_post_vectors": unique_matching,
            "additive_unique_post_vectors": unique_additive,
            "hash_random_unique_values": random_unique,
            "post_profile_shape": list(posts.shape),
            "effect_profile_shape": list(effects.shape),
            "post_pca_shape": list(post_pca.shape),
            "effect_pca_shape": list(effect_pca.shape),
        },
    }


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    payload = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode(
        "utf-8"
    )
    atomic_write(path, payload)


def atomic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gz_handle:
            with io.TextIOWrapper(gz_handle, encoding="utf-8", newline="") as text_handle:
                frame.to_csv(
                    text_handle,
                    index=False,
                    float_format="%.17g",
                    lineterminator="\n",
                )
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
    temporary.replace(path)


def fsync_tree(root: Path) -> None:
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.is_symlink():
            raise RuntimeError(f"Symlink rejected in E162b staging: {path}")
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    directories = sorted(
        [root, *(value for value in root.rglob("*") if value.is_dir())],
        key=lambda value: len(value.parts),
        reverse=True,
    )
    for directory in directories:
        descriptor = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def profile_frame(
    values: np.ndarray, test_labels: list[str], genes: list[str]
) -> pd.DataFrame:
    if values.shape != (len(BASELINE_ORDER), N_TEST, N_SELECTED):
        raise RuntimeError("Profile tensor shape changed")
    flat = values.reshape(len(BASELINE_ORDER) * N_TEST, N_SELECTED)
    frame = pd.DataFrame(flat, columns=genes)
    frame.insert(
        0,
        "condition",
        [condition for _baseline in BASELINE_ORDER for condition in test_labels],
    )
    frame.insert(
        0,
        "baseline",
        [baseline for baseline in BASELINE_ORDER for _condition in test_labels],
    )
    return frame


def pca_frame(
    post: np.ndarray, effect: np.ndarray, test_labels: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for baseline_index, baseline in enumerate(BASELINE_ORDER):
        for test_index, condition in enumerate(test_labels):
            row: dict[str, Any] = {
                "baseline": baseline,
                "condition": condition,
                "test_index": test_index,
            }
            row.update(
                {f"post_PC{coordinate + 1}": post[baseline_index, test_index, coordinate] for coordinate in range(N_PCA)}
            )
            row.update(
                {f"effect_PC{coordinate + 1}": effect[baseline_index, test_index, coordinate] for coordinate in range(N_PCA)}
            )
            rows.append(row)
    return pd.DataFrame(rows)


def source_hash_table(common: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    combined: dict[str, dict[str, Any]] = {}
    combined.update(common["committed"])
    combined.update(common["e160"]["source_records"])
    combined.update(common["e161"]["source_records"])
    required_external = {
        "E161_development_h5ad": E161_ASSET / "perturb_processed.h5ad",
        "E161_selected_gene_axis": E161_ASSET / "SELECTED_GENE_AXIS.txt",
        "E161_PCA_model": E161_ASSET / "TRAIN_ONLY_PCA_MODEL.npz",
        "E161_control_prior": E161_ASSET / "TRAIN_ONLY_CONTROL_PRIOR.npz",
    }
    for role, path in required_external.items():
        combined[role] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "matches_git_head_blob": False,
        }
    for role, record in combined.items():
        rows.append(
            {
                "source_role": role,
                "path": record["path"],
                "bytes": int(record["bytes"]),
                "sha256": record["sha256"],
                "matches_git_head_blob": bool(record["matches_git_head_blob"]),
                "git_head": common["git_head"],
            }
        )
    return pd.DataFrame(rows).sort_values("source_role").reset_index(drop=True)


def build_audit_table(
    predictions: dict[str, Any], control_delta: float
) -> pd.DataFrame:
    values = predictions["audit_values"]
    rows = [
        ("test_tasks", N_TEST, N_TEST, True),
        ("baselines", len(BASELINE_ORDER), 4, True),
        ("selected_genes", N_SELECTED, N_SELECTED, True),
        ("pca_coordinates", N_PCA, N_PCA, True),
        ("control_prior_recomputed_max_abs_delta", control_delta, "<=5e-6", control_delta <= 5e-6),
        ("control_effect_max_abs", values["control_effect_max_abs"], 0.0, values["control_effect_max_abs"] == 0.0),
        (
            "additive_minus_2x_matching_effect_max_abs",
            values["additive_minus_2x_matching_effect_max_abs"],
            0.0,
            values["additive_minus_2x_matching_effect_max_abs"] == 0.0,
        ),
        (
            "matching_post_minus_single_mean_max_abs",
            values["matching_post_minus_single_mean_max_abs"],
            "<=1e-12",
            values["matching_post_minus_single_mean_max_abs"] <= 1e-12,
        ),
        (
            "additive_rms_minus_2x_matching_rms_max_abs",
            values["additive_rms_minus_2x_matching_rms_max_abs"],
            "<=1e-12",
            values["additive_rms_minus_2x_matching_rms_max_abs"] <= 1e-12,
        ),
        (
            "matching_unique_post_vectors",
            values["matching_unique_post_vectors"],
            ">=24",
            values["matching_unique_post_vectors"] >= 24,
        ),
        (
            "additive_unique_post_vectors",
            values["additive_unique_post_vectors"],
            ">=24",
            values["additive_unique_post_vectors"] >= 24,
        ),
        (
            "hash_random_unique_values",
            values["hash_random_unique_values"],
            ">=24",
            values["hash_random_unique_values"] >= 24,
        ),
        ("train_X_rows_indexed", N_TRAIN, N_TRAIN, True),
        ("validation_X_rows_indexed", 0, 0, True),
        ("test_X_rows_indexed", 0, 0, True),
        ("raw_file_opened", False, False, True),
        ("test_truth_or_error_used", False, False, True),
    ]
    return pd.DataFrame(rows, columns=["check", "observed", "required", "gate_passed"])


def write_release(
    common: dict[str, Any],
    stats: dict[str, Any],
    predictions: dict[str, Any],
    access: dict[str, Any],
    control_delta: float,
    transaction_id: str,
    started_at: str,
) -> dict[str, Any]:
    test_labels = common["e160"]["splits"]["test"]
    genes = common["e161"]["genes"]
    transaction = {
        "schema": "e162b_atomic_staging_v1",
        "transaction_id": transaction_id,
        "experiment": "E162b_wessels_label_only_baselines",
        "target": str(RELEASE),
        "planned_phase": "complete_pretest_label_only_baselines_no_val_or_test_X",
        "created_at": started_at,
    }
    atomic_write(STAGING / ".E162b_TRANSACTION.json", json_bytes(transaction))

    post_frame = profile_frame(predictions["posts"], test_labels, genes)
    effect_frame = profile_frame(predictions["effects"], test_labels, genes)
    coordinates = pca_frame(
        predictions["post_pca"], predictions["effect_pca"], test_labels
    )
    gene_stats = stats["gene_table"].copy()
    gene_stats.insert(
        4,
        "gene",
        [
            genes[index]
            for index in gene_stats["gene_index_zero_based"].to_numpy(dtype=int)
        ],
    )
    audit = build_audit_table(predictions, control_delta)
    if not audit["gate_passed"].astype(bool).all():
        raise RuntimeError("E162b final baseline audit failed")

    atomic_gzip_csv(
        post_frame, STAGING / "profiles/E162b_TEST_POST_PROFILES.csv.gz"
    )
    atomic_gzip_csv(
        effect_frame, STAGING / "profiles/E162b_TEST_EFFECT_PROFILES.csv.gz"
    )
    atomic_csv(coordinates, STAGING / "profiles/E162b_TEST_PCA10_COORDINATES.csv")
    atomic_gzip_csv(
        gene_stats, STAGING / "tables/E162b_TRAIN_REFERENCE_GENE_STATS.csv.gz"
    )
    atomic_csv(
        stats["pca_table"], STAGING / "tables/E162b_TRAIN_REFERENCE_PCA10_STATS.csv"
    )
    atomic_csv(predictions["tasks"], STAGING / "tables/E162b_TEST_TASKS.csv")
    atomic_csv(
        predictions["risk_wide"], STAGING / "tables/E162b_RISK_BASELINES_WIDE.csv"
    )
    atomic_csv(
        predictions["risk_long"], STAGING / "tables/E162b_RISK_BASELINES_LONG.csv"
    )
    atomic_csv(audit, STAGING / "tables/E162b_BASELINE_AUDIT.csv")
    atomic_csv(pd.DataFrame(access["ledger_rows"]), STAGING / "tables/E162b_X_ACCESS_LEDGER.csv")
    atomic_csv(
        source_hash_table(common), STAGING / "tables/E162b_SOURCE_HASHES.csv"
    )
    atomic_csv(
        pd.DataFrame(common["runtime"]),
        STAGING / "tables/E162b_RUNTIME_ENVIRONMENT.csv",
    )

    report = f"""# E162b Wessels 解封前简单基线

- 48 个 test inputs 仅为 E160 canonical condition 字符串；
- 训练统计只来自 E161 的 {N_TRAIN:,} 个 train cells；
- validation/test expression 访问均为 0；
- 四个冻结预测器：control、cell-weighted perturbed mean、matching single mean、single additive；
- matching unique profiles：{predictions['audit_values']['matching_unique_post_vectors']}；
- additive unique profiles：{predictions['audit_values']['additive_unique_post_vectors']}；
- additive effect 与 2×matching effect 最大差：{predictions['audit_values']['additive_minus_2x_matching_effect_max_abs']:.17g}；
- additive effect RMS 与 2×matching effect RMS 最大差：{predictions['audit_values']['additive_rms_minus_2x_matching_rms_max_abs']:.17g}；
- 风险列已统一为 higher expected accuracy；magnitude 与 SE 的原始正值和固定负号同时保留；
- PRESCRIBE predicted magnitude 不在本 runner 计算，后续从 E162 原值合并并固定用负 RMS 作为 confidence。

本阶段没有 test truth、effect、error 或评价指标，不能据此报告模型优劣。
"""
    atomic_write(STAGING / "reports/E162b_REPORT.md", report.encode("utf-8"))
    atomic_write(
        STAGING / "README_先看这个.md",
        (
            "# E162b\n\n先看 `reports/E162b_REPORT.md`，再核对任务、风险、profile、访问账本和接口。\n"
        ).encode("utf-8"),
    )

    scientific_paths = sorted(
        path
        for path in STAGING.rglob("*")
        if path.is_file() and path.name not in {"RUN_STATUS.json", "RESULTS_SHA256.csv"}
    )
    scientific_hashes = {
        path.relative_to(STAGING).as_posix(): sha256_file(path)
        for path in scientific_paths
    }
    interface = {
        "schema": "safeconf_e162b_to_e163_v1",
        "experiment": "E162b_wessels_label_only_baselines",
        "git_head": common["git_head"],
        "transaction_id": transaction_id,
        "source_interface_schema": common["e161"]["interface"]["schema"],
        "source_interface_sha256": sha256_file(E161_INTERFACE),
        "e160_test_split_sha256": sha256_file(E160_SPLIT),
        "test_label_source": "E160_set2conditions.json::test canonical strings only",
        "n_test_labels": N_TEST,
        "test_label_order": test_labels,
        "n_selected_genes": N_SELECTED,
        "selected_gene_order_sha256": common["e161"]["interface"][
            "selected_gene_order_sha256"
        ],
        "n_pca_coordinates": N_PCA,
        "baseline_order": list(BASELINE_ORDER),
        "profile_shape": [len(BASELINE_ORDER), N_TEST, N_SELECTED],
        "pca_shape": [len(BASELINE_ORDER), N_TEST, N_PCA],
        "risk_analysis_orientation": "higher_expected_accuracy",
        "primary_se_score": "matching_se_pca10_confidence",
        "magnitude_policy": {
            "raw": "matching_effect_rms; higher_effect_size",
            "analysis": "matching_magnitude_confidence=-matching_effect_rms; higher_expected_accuracy",
            "direction_frozen_before_E163": True,
        },
        "prescribe_extension": {
            "computed_here": False,
            "required_raw": "PRESCRIBE predicted-effect RMS from E162",
            "required_analysis": "negative PRESCRIBE predicted-effect RMS",
            "direction_frozen_before_E163": True,
        },
        "access_boundary": {
            "raw_file_opened": False,
            "train_X_rows_indexed_materialized_transformed": N_TRAIN,
            "validation_X_rows_indexed_materialized_transformed": 0,
            "test_X_rows_indexed_materialized_transformed": 0,
            "excluded_X_rows_indexed_materialized_transformed": 0,
            "test_cell_count_truth_effect_error_or_DE_used": False,
        },
        "paths": {
            "post_profiles": "profiles/E162b_TEST_POST_PROFILES.csv.gz",
            "effect_profiles": "profiles/E162b_TEST_EFFECT_PROFILES.csv.gz",
            "pca_coordinates": "profiles/E162b_TEST_PCA10_COORDINATES.csv",
            "tasks": "tables/E162b_TEST_TASKS.csv",
            "risk_wide": "tables/E162b_RISK_BASELINES_WIDE.csv",
            "risk_long": "tables/E162b_RISK_BASELINES_LONG.csv",
            "gene_stats": "tables/E162b_TRAIN_REFERENCE_GENE_STATS.csv.gz",
            "pca_stats": "tables/E162b_TRAIN_REFERENCE_PCA10_STATS.csv",
            "access_ledger": "tables/E162b_X_ACCESS_LEDGER.csv",
        },
        "artifact_sha256_before_interface": scientific_hashes,
    }
    atomic_write(STAGING / "E162b_E163_INTERFACE.json", json_bytes(interface))

    result_paths = sorted(
        path
        for path in STAGING.rglob("*")
        if path.is_file() and path.name not in {"RUN_STATUS.json", "RESULTS_SHA256.csv"}
    )
    manifest_rows = [
        {
            "relative_path": path.relative_to(STAGING).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in result_paths
    ]
    manifest = pd.DataFrame(manifest_rows)
    atomic_csv(manifest, STAGING / "RESULTS_SHA256.csv")

    completed_at = now_text()
    status = {
        "experiment": "E162b_wessels_label_only_baselines",
        "phase": "complete_pretest_label_only_baselines_no_val_or_test_X",
        "transaction_id": transaction_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "git_head": common["git_head"],
        "n_train_X_rows_indexed": N_TRAIN,
        "n_validation_X_rows_indexed": 0,
        "n_test_X_rows_indexed": 0,
        "n_raw_X_rows_indexed": 0,
        "raw_file_opened": False,
        "test_inputs": "48 canonical condition strings only",
        "test_cell_count_truth_effect_error_or_DE_used": False,
        "n_test_tasks": N_TEST,
        "n_baselines": len(BASELINE_ORDER),
        "n_selected_genes": N_SELECTED,
        "n_pca_coordinates": N_PCA,
        "profile_shape": [len(BASELINE_ORDER), N_TEST, N_SELECTED],
        "pca_shape": [len(BASELINE_ORDER), N_TEST, N_PCA],
        "matching_unique_post_vectors": predictions["audit_values"][
            "matching_unique_post_vectors"
        ],
        "additive_unique_post_vectors": predictions["audit_values"][
            "additive_unique_post_vectors"
        ],
        "control_effect_max_abs": predictions["audit_values"][
            "control_effect_max_abs"
        ],
        "additive_minus_2x_matching_effect_max_abs": predictions["audit_values"][
            "additive_minus_2x_matching_effect_max_abs"
        ],
        "matching_post_minus_single_mean_max_abs": predictions["audit_values"][
            "matching_post_minus_single_mean_max_abs"
        ],
        "additive_rms_minus_2x_matching_rms_max_abs": predictions[
            "audit_values"
        ]["additive_rms_minus_2x_matching_rms_max_abs"],
        "hash_random_unique_values": predictions["audit_values"][
            "hash_random_unique_values"
        ],
        "control_prior_recomputed_max_abs_delta": control_delta,
        "risk_analysis_orientation": "higher_expected_accuracy",
        "prescribe_predicted_magnitude_computed_here": False,
        "results_manifest_sha256": sha256_file(STAGING / "RESULTS_SHA256.csv"),
        "artifact_sha256": {
            row["relative_path"]: row["sha256"] for row in manifest_rows
        },
    }
    atomic_write(STAGING / "RUN_STATUS.json", json_bytes(status))

    observed = {
        path.relative_to(STAGING).as_posix()
        for path in STAGING.rglob("*")
        if path.is_file()
    }
    if observed != EXPECTED_RELEASE_FILES:
        raise RuntimeError(
            f"E162b release allowlist mismatch: {sorted(observed ^ EXPECTED_RELEASE_FILES)}"
        )
    if any(path.is_symlink() for path in STAGING.rglob("*")):
        raise RuntimeError("Symlink rejected in E162b release staging")
    for row in manifest_rows:
        path = STAGING / row["relative_path"]
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Post-write hash mismatch: {row['relative_path']}")
    fsync_tree(STAGING)
    STAGING.replace(RELEASE)
    parent_descriptor = os.open(str(OUT), os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return status


def record_failure(error: BaseException) -> None:
    FAILURES.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    path = FAILURES / f"E162b_FAILURE_{stamp}.json"
    payload = {
        "experiment": "E162b_wessels_label_only_baselines",
        "failed_at": now_text(),
        "error_type": type(error).__name__,
        "error": repr(error),
        "traceback": traceback.format_exc(),
        "staging_preserved": STAGING.exists(),
        "release_exists": RELEASE.exists(),
    }
    atomic_write(path, json_bytes(payload))


def formal(common: dict[str, Any]) -> dict[str, Any]:
    if RELEASE.exists():
        raise FileExistsError("E162b release already exists; refusing overwrite")
    if STAGING.exists():
        raise FileExistsError("E162b staging exists; audit it before retrying")
    STAGING.mkdir(parents=False, exist_ok=False)
    started_at = now_text()
    transaction_id = uuid.uuid4().hex

    dev_path = E161_ASSET / "perturb_processed.h5ad"
    train_x, train_conditions, read_audit = load_train_expression_only(
        dev_path, common["e161"]["genes"]
    )
    stats = build_train_statistics(
        train_x,
        train_conditions,
        common["e160"],
        common["e161"]["pca_mean"],
        common["e161"]["components"],
    )
    recomputed_control = stats["groups"]["ctrl"]["gene_mean"]
    control_delta = float(
        np.max(np.abs(recomputed_control - common["e161"]["control_prior_mean"]))
    )
    if control_delta > 5e-6:
        raise RuntimeError(
            f"Recomputed train control mean differs from E161 prior: {control_delta}"
        )
    predictions = build_predictions_and_risks(
        common["e160"],
        stats,
        common["e161"]["pca_mean"],
        common["e161"]["components"],
    )
    ledger_rows = [
        {
            "source": "E161 development H5AD",
            "role": "metadata",
            "rows_indexed": N_DEV,
            "X_rows_indexed": 0,
            "X_rows_materialized": 0,
            "X_rows_transformed": 0,
            "detail": "obs role/condition and var names only",
        },
        {
            "source": "E161 development H5AD",
            "role": "train",
            "rows_indexed": N_TRAIN,
            "X_rows_indexed": read_audit["train_rows_indexed"],
            "X_rows_materialized": N_TRAIN,
            "X_rows_transformed": N_TRAIN,
            "detail": read_audit["storage"],
        },
        {
            "source": "E161 development H5AD",
            "role": "validation",
            "rows_indexed": N_VALIDATION,
            "X_rows_indexed": 0,
            "X_rows_materialized": 0,
            "X_rows_transformed": 0,
            "detail": "metadata role verified; X/layers/obsm unopened",
        },
        {
            "source": "E160 frozen split JSON",
            "role": "test",
            "rows_indexed": N_TEST,
            "X_rows_indexed": 0,
            "X_rows_materialized": 0,
            "X_rows_transformed": 0,
            "detail": "48 canonical condition strings only; no cell count/truth/effect/error/DE",
        },
        {
            "source": "Wessels raw",
            "role": "raw/test/excluded",
            "rows_indexed": 0,
            "X_rows_indexed": 0,
            "X_rows_materialized": 0,
            "X_rows_transformed": 0,
            "detail": "file not opened by E162b",
        },
    ]
    access = {"ledger_rows": ledger_rows}
    return write_release(
        common,
        stats,
        predictions,
        access,
        control_delta,
        transaction_id,
        started_at,
    )


def main() -> None:
    args = parse_args()
    try:
        common = common_preflight()
        summary = {
            "experiment": "E162b_wessels_label_only_baselines",
            "mode": args.mode,
            "git_head": common["git_head"],
            "runtime_gate": "passed",
            "committed_input_gate": "passed",
            "e160_gate": "passed",
            "e161_gate": "passed",
            "development_h5ad_opened": False,
            "raw_file_opened": False,
            "validation_X_rows_indexed": 0,
            "test_X_rows_indexed": 0,
        }
        if args.mode == "preflight":
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return
        status = formal(common)
        print(json.dumps(status, ensure_ascii=False, indent=2))
    except BaseException as error:
        if args.mode == "formal":
            record_failure(error)
        raise


if __name__ == "__main__":
    main()
