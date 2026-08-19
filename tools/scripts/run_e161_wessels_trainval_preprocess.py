#!/usr/bin/env python3
"""E161: leakage-safe Wessels train/validation preprocessing for PRESCRIBE.

``preflight`` is metadata-only: it validates Git/E160/raw identity and opens the
AnnData container backed read-only for ``obs``, ``var_names`` and shape.  It
does not index ``X``.  ``formal`` is the only branch allowed to call
``read_allowed_expression``; that helper rejects every split except train/val
and checks the requested raw row indices against the frozen test/excluded sets
before touching ``X``.

The formal output contains development (train+validation) expression and cell
graphs only.  No model is instantiated or trained and no test score, truth,
effect, prediction or error is computed here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import pickle
import platform
import shutil
import stat
import subprocess
import sys
import traceback
import uuid
from datetime import datetime
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.utils.extmath import safe_sparse_dot


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
RAW = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "WesselsSatija2023.h5ad"
)
PRESCRIBE = Path("/home/yyf/archive/external/PRESCRIBE")
SCPERTURB = Path("/home/yyf/archive/external/scPerturb")
E160 = ROOT / "docs/实验结果/E160_wessels_combination_contract_20260714"
E160_FREEZE = E160 / "freeze"
CONTRACT = (
    ROOT
    / "docs/实验结果/E161_wessels_trainval_preprocess_20260714/"
    "ANALYSIS_CONTRACT.md"
)
AMENDMENT = CONTRACT.parent / "PREFLIGHT_FAILURE_AND_AMENDMENT_20260715.md"
SECOND_AMENDMENT = CONTRACT.parent / "ENDOGENOUS_AXIS_AMENDMENT_20260715.md"
E161A = ROOT / "docs/实验结果/E161a_wessels_feature_boundary_audit_20260715"
E161A_STATUS = E161A / "RUN_STATUS.json"
E161A_MANIFEST = E161A / "RESULTS_SHA256.csv"
OUT = CONTRACT.parent
RELEASE = OUT / "release"
REPO_STAGING = OUT / ".release.staging"
ASSET_PARENT = Path("/home/yyf/data/safeconf_e161_prescribe")
ASSET_STAGING = ASSET_PARENT / ".wessels_e160.staging"
ASSET_FINAL = ASSET_PARENT / "wessels_e160"
PRESCRIBE_LINK = PRESCRIBE / "data/wessels_e160"
TRANSACTION_JOURNAL = OUT / ".E161_TRANSACTION.json"

EXPECTED_PYTHON_EXECUTABLE = Path(
    "/home/yyf/.conda/envs/prescribe_env/bin/python"
)
EXPECTED_PYTHON_VERSION = "3.9.25"
EXPECTED_DISTRIBUTIONS = {
    "anndata": "0.10.8",
    "scanpy": "1.10.3",
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "scipy": "1.13.1",
    "scikit-learn": "1.6.1",
    "scikit-misc": "0.3.1",
    "h5py": "3.14.0",
    "torch": "2.1.2+cu118",
    "torch-geometric": "2.6.1",
}

SEED = 3407
TARGET_SUM = 10_000.0
N_HVG = 2_000
N_PCA = 10
EXPECTED_RAW_SHAPE = (30_707, 21_052)
EXPECTED_ENDOGENOUS_FEATURES = 20_631
EXPECTED_ENGINEERED_FEATURES = 8
EXPECTED_GUIDE_BARCODE_FEATURES = 413
EXPECTED_EXCLUDED_FEATURES = 421
EXPECTED_FULL_FEATURE_AXIS_SHA256 = "dea725a87c973ca15590b08b309df3a926dc0233391cb2df76518c847229e780"
EXPECTED_ENDOGENOUS_FEATURE_AXIS_SHA256 = "dbed3dad178ea500b01625abf5121c9ee17bdd501b87d2fcdede0b6bade654e7"
EXPECTED_ENGINEERED_FEATURE_AXIS_SHA256 = "103c2df8585646aa6dccde85866353889a699420b5536157b8babbd9b9aec554"
EXPECTED_GUIDE_BARCODE_FEATURE_AXIS_SHA256 = "9088328f4ac6b2a1b109c254f0068504d25618478383fcbb3f43be8e59dd06d2"
EXPECTED_EXCLUDED_FEATURE_AXIS_SHA256 = "e6e54ba5c0f63d62b599754ab3866da7cdf8194be4dfefd46dabc7d6a73e8116"
EXPECTED_RAW_BYTES = 219_393_529
EXPECTED_RAW_MD5 = "6897bfdcda928a678208fecf4eeb282e"
EXPECTED_RAW_SHA256 = "5da0485aed81b23bda57b4a7b4510a394682d54911416db89b4846ff6dd34732"
EXPECTED_SPLIT_CONDITIONS = {"train": 72, "val": 24, "test": 48}
EXPECTED_SPLIT_CELLS = {"train": 11_779, "val": 5_102, "test": 9_902, "excluded": 3_924}
EXPECTED_DEV_CELLS = 16_881
EXPECTED_PERT_GENES = 27
EXPECTED_PRESCRIBE_COMMIT = "6f7264a205aaff654a9594863c5c10b656f88ebe"
EXPECTED_SCPERTURB_COMMIT = "b69f72a070a92bcbaf41e7f9897b11598109ab48"
LOCKED_SOURCE_SHA256 = {
    "gears_init": "3cdc747e61b16e073873d7f5ccb4f7c872d921c5355f2615049de09f279233ee",
    "gears_pertdata": "d7316bc19fc70d78c78d0dabf126df161ae861a6620d85a0d16aeaeee27ba59c",
    "gears_utils": "89d1da79df60d14d929aed05ec904b4ef2664855abe89b5bce5112b88f80395a",
    "gears_data_utils": "7043e80d4280cd81ec2ff6c78609235f407a4cd3dc3f337f3e53804e16c537fc",
    "prescribe_data_adapter": "f5247ceb8cd5e8a5782c74d1d4e17350dfbed2e2c911f56fb4bf9f69344acc77",
    "prescribe_dataloader": "e8ac66674935ecd32d0d029169160f97abdfe6361c36ce9ff089598390e30362",
    "gene2go_all": "f145c5e84a53048d87942a417d870a4f2d8db50200b96e492b358c13aba8c771",
    "scgpt_embedding": "9a5be69676bc09fbf996ae7be1d4faa09c9f32abbf733f33fc130153829ad8ce",
}
SOURCE_PATHS = {
    "gears_init": PRESCRIBE / "gears/__init__.py",
    "gears_pertdata": PRESCRIBE / "gears/pertdata.py",
    "gears_utils": PRESCRIBE / "gears/utils.py",
    "gears_data_utils": PRESCRIBE / "gears/data_utils.py",
    "prescribe_data_adapter": PRESCRIBE / "src/data/pertdata.py",
    "prescribe_dataloader": PRESCRIBE / "src/data/dataloader.py",
    "gene2go_all": PRESCRIBE / "data/gene2go_all.pkl",
    "scgpt_embedding": PRESCRIBE / "scLLM_weights/scGPT/embedding.pkl",
}
SCPERTURB_SOURCE_PATHS = {
    "scperturb_wessels_notebook": (
        SCPERTURB / "dataset_processing/notebooks/WesselsSatija2023.ipynb"
    ),
    "scperturb_qc_utils": SCPERTURB / "utils.py",
}
LOCKED_SCPERTURB_SOURCE_SHA256 = {
    "scperturb_wessels_notebook": "b7ce4d66890831210d20b0bcc865b8eb27f84326a7176d5b179b19c00480e3d1",
    "scperturb_qc_utils": "5647f6ddeaad80a8bd596928e767f60406bd7fb959a9966c192247ae19015975",
}
STAGING_SENTINEL = ".E161_STAGING.json"
REPO_ALLOWLIST = frozenset(
    {
        "RUN_STATUS.json",
        "README_先看这个.md",
        "reports/E161_REPORT.md",
        "tables/E161_X_ACCESS_LEDGER.csv",
        "tables/E161_NORMALIZATION_AUDIT.csv",
        "tables/E161_GENE_AXIS_AUDIT.csv",
        "tables/E161_CONDITION_EDISTANCE.csv",
        "tables/E161_SPLIT_AND_LEAKAGE_AUDIT.csv",
        "tables/E161_GRAPH_AUDIT.csv",
        "tables/E161_SOURCE_HASHES.csv",
        "tables/E161_RUNTIME_ENVIRONMENT.csv",
        "tables/E161_ASSET_MANIFEST.csv",
    }
)

DATA_SCIENTIFIC_ALLOWLIST = frozenset(
    {
        "perturb_processed.h5ad",
        f"set2conditions_{SEED}.pkl",
        f"frozen_pert_gene_set_{SEED}.pkl",
        "data_pyg/cell_graphs.pkl",
        "data_pyg/mean.npy",
        "data_pyg/cov.npy",
        "TRAIN_ONLY_PCA_MODEL.npz",
        "TRAIN_ONLY_CONTROL_PRIOR.npz",
        "ENDOGENOUS_GENE_AXIS.txt",
        "FULL_RAW_FEATURE_AXIS.txt",
        "ENGINEERED_CONSTRUCT_FEATURE_AXIS.txt",
        "GUIDE_BARCODE_FEATURE_AXIS.txt",
        "EXCLUDED_FEATURE_AXIS.txt",
        "SELECTED_GENE_AXIS.txt",
        "train_only_edistance_labels.csv",
        "train_only_gene_axis_audit.csv",
        "E161_E162_INTERFACE.json",
        "ASSET_MANIFEST.csv",
    }
)
DATA_OPERATIONAL_ALLOWLIST = DATA_SCIENTIFIC_ALLOWLIST | {STAGING_SENTINEL}
DATA_STAGING_TEMP_ALLOWLIST = DATA_OPERATIONAL_ALLOWLIST | {
    "perturb_processed.tmp.h5ad",
    f"set2conditions_{SEED}.pkl.tmp",
    f"frozen_pert_gene_set_{SEED}.pkl.tmp",
    "data_pyg/mean.npy.tmp",
    "data_pyg/cov.npy.tmp",
    "TRAIN_ONLY_PCA_MODEL.npz.tmp",
    "TRAIN_ONLY_CONTROL_PRIOR.npz.tmp",
    "E161_E162_INTERFACE.json.tmp",
    "ASSET_MANIFEST.csv.tmp",
    "ENDOGENOUS_GENE_AXIS.txt.tmp",
    "FULL_RAW_FEATURE_AXIS.txt.tmp",
    "ENGINEERED_CONSTRUCT_FEATURE_AXIS.txt.tmp",
    "GUIDE_BARCODE_FEATURE_AXIS.txt.tmp",
    "EXCLUDED_FEATURE_AXIS.txt.tmp",
    "SELECTED_GENE_AXIS.txt.tmp",
}
DATA_DIRECTORY_ALLOWLIST = frozenset({"data_pyg"})
REPO_OPERATIONAL_ALLOWLIST = REPO_ALLOWLIST | {STAGING_SENTINEL}
REPO_STAGING_TEMP_ALLOWLIST = REPO_OPERATIONAL_ALLOWLIST | {
    f"{relative}.tmp" for relative in REPO_ALLOWLIST
}
REPO_DIRECTORY_ALLOWLIST = frozenset({"reports", "tables"})


def runtime_gate() -> list[dict[str, str]]:
    observed_executable = Path(sys.executable)
    if observed_executable != EXPECTED_PYTHON_EXECUTABLE:
        raise RuntimeError(
            "E161 must run with the frozen PRESCRIBE interpreter: "
            f"{EXPECTED_PYTHON_EXECUTABLE}; observed {observed_executable}"
        )
    observed_python = platform.python_version()
    if observed_python != EXPECTED_PYTHON_VERSION:
        raise RuntimeError(
            f"E161 Python changed: {observed_python} != {EXPECTED_PYTHON_VERSION}"
        )
    rows = [
        {
            "component": "python",
            "expected_version": EXPECTED_PYTHON_VERSION,
            "observed_version": observed_python,
            "executable": str(observed_executable),
            "gate_passed": "true",
        }
    ]
    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        observed = distribution_version(distribution)
        if observed != expected:
            raise RuntimeError(
                f"E161 dependency changed: {distribution} {observed} != {expected}"
            )
        rows.append(
            {
                "component": distribution,
                "expected_version": expected,
                "observed_version": observed,
                "executable": str(observed_executable),
                "gate_passed": "true",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "formal"), default="preflight")
    parser.add_argument("--recover-staging", action="store_true")
    return parser.parse_args()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_raw_once(path: Path) -> dict[str, Any]:
    # Path.stat(follow_symlinks=...) is not available on every Python 3.9
    # build used by PRESCRIBE; lstat() is the portable no-follow spelling.
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise RuntimeError("Wessels raw source must be a regular non-symlink file")
    md5 = hashlib.md5()  # noqa: S324 - fixed public dataset identity, not security
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError("Raw identity changed between stat and open")
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    after = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(after):
        raise RuntimeError("Raw identity changed while hashing")
    result = {
        "md5": md5.hexdigest(),
        "sha256": sha.hexdigest(),
        "bytes": int(after.st_size),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "mtime_ns": int(after.st_mtime_ns),
    }
    if result["bytes"] != EXPECTED_RAW_BYTES:
        raise RuntimeError(f"Unexpected Wessels byte size: {result['bytes']}")
    if result["md5"] != EXPECTED_RAW_MD5 or result["sha256"] != EXPECTED_RAW_SHA256:
        raise RuntimeError("Wessels raw checksum changed")
    return result


def git_blob_gate(path: Path, head: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Required committed input is not a regular file: {path}")
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    try:
        blob = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Required input is not committed at HEAD: {relative}") from exc
    working = path.read_bytes()
    if blob != working:
        raise RuntimeError(f"Working file differs from Git HEAD blob: {relative}")
    return {
        "path": relative,
        "bytes": len(working),
        "sha256": hashlib.sha256(working).hexdigest(),
        "matches_git_head_blob": True,
    }


def validate_e160(head: str) -> tuple[dict[str, Any], pd.DataFrame, dict[str, list[str]]]:
    status_path = E160_FREEZE / "RUN_STATUS.json"
    git_blob_gate(status_path, head)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("phase") != "requirements_frozen_test_expression_unopened":
        raise RuntimeError("E160 is not frozen with test expression unopened")
    if tuple(status.get("dataset_shape", [])) != EXPECTED_RAW_SHAPE:
        raise RuntimeError("E160 dataset shape changed")
    if status.get("raw_X_values_indexed_or_materialized") is not False:
        raise RuntimeError("E160 no-X invariant changed")
    integrity = status.get("raw_integrity", {})
    if (
        integrity.get("md5") != EXPECTED_RAW_MD5
        or integrity.get("sha256") != EXPECTED_RAW_SHA256
        or integrity.get("bytes") != EXPECTED_RAW_BYTES
    ):
        raise RuntimeError("E160 raw identity changed")
    for relative, expected in status.get("artifact_sha256", {}).items():
        path = E160 / relative
        if sha256_file(path) != expected:
            raise RuntimeError(f"E160 artifact hash mismatch: {relative}")
        git_blob_gate(path, head)
    git_blob_gate(E160 / "ANALYSIS_CONTRACT.md", head)
    git_blob_gate(E160 / "SCGPT_PERTURBATION_VOCABULARY.txt", head)

    audit = pd.read_csv(E160_FREEZE / "manifests/E160_CONDITION_AUDIT.csv")
    split_json = json.loads(
        (E160_FREEZE / "manifests/E160_set2conditions.json").read_text(encoding="utf-8")
    )
    split_dict = {key: [str(x) for x in split_json[key]] for key in ("train", "val", "test")}
    for role, expected in EXPECTED_SPLIT_CONDITIONS.items():
        if len(split_dict[role]) != expected or len(set(split_dict[role])) != expected:
            raise RuntimeError(f"E160 {role} condition count changed")
    sets = {key: set(value) for key, value in split_dict.items()}
    if (sets["train"] & sets["val"]) or (sets["train"] & sets["test"]) or (sets["val"] & sets["test"]):
        raise RuntimeError("E160 split overlap")
    return status, audit, split_dict


def validate_e161a(head: str) -> dict[str, Any]:
    required = [
        E161A / "ANALYSIS_CONTRACT.md",
        E161A_STATUS,
        E161A / "CANDIDATE_BOUNDARY_AUDIT.csv",
        E161A / "ENGINEERED_CONSTRUCT_COUNTS.csv",
        E161A / "ACCESS_LEDGER.json",
        E161A / "REPORT.md",
        E161A_MANIFEST,
    ]
    for path in required:
        git_blob_gate(path, head)
    status = json.loads(E161A_STATUS.read_text(encoding="utf-8"))
    if (
        status.get("experiment") != "E161a"
        or status.get("status") != "completed_diagnostic"
        or status.get("candidate_boundaries") != list(range(20_631, 20_640))
        or status.get("exact_matching_boundaries") != []
        or status.get("unique_exact_boundary") is not None
        or status.get("train_cells_read") != 11_779
        or status.get("validation_expression_rows_read") != 0
        or status.get("test_expression_rows_read") != 0
        or status.get("excluded_expression_rows_read") != 0
        or status.get("guide_or_barcode_columns_read") != 0
    ):
        raise RuntimeError("E161a boundary-audit status changed")
    candidates = pd.read_csv(E161A / "CANDIDATE_BOUNDARY_AUDIT.csv")
    if (
        candidates["prefix_features"].astype(int).tolist()
        != list(range(20_631, 20_640))
        or candidates["exact_match_all_train_cells"].astype(bool).any()
        or int(candidates.loc[candidates["prefix_features"].eq(20_631), "mismatched_train_cells"].iloc[0])
        != 10_215
        or int(candidates.loc[candidates["prefix_features"].eq(20_636), "mismatched_train_cells"].iloc[0])
        != 591
    ):
        raise RuntimeError("E161a candidate-boundary results changed")
    ledger = json.loads((E161A / "ACCESS_LEDGER.json").read_text(encoding="utf-8"))
    if (
        ledger.get("split") != "train"
        or ledger.get("rows_indexed") != 11_779
        or ledger.get("columns_indexed") != 20_639
        or ledger.get("validation_rows_indexed") != 0
        or ledger.get("test_rows_indexed") != 0
        or ledger.get("excluded_rows_indexed") != 0
        or ledger.get("guide_or_barcode_columns_indexed") != 0
    ):
        raise RuntimeError("E161a access ledger changed")
    manifest = pd.read_csv(E161A_MANIFEST)
    if list(manifest.columns) != ["path", "bytes", "sha256"]:
        raise RuntimeError("Malformed E161a result manifest")
    for row in manifest.itertuples(index=False):
        path = ROOT / str(row.path)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(row.bytes)
            or sha256_file(path) != str(row.sha256)
        ):
            raise RuntimeError(f"E161a manifest mismatch: {path}")
        git_blob_gate(path, head)
    return status


def source_gate() -> list[dict[str, Any]]:
    prescribe_commit = subprocess.check_output(
        ["git", "-C", str(PRESCRIBE), "rev-parse", "HEAD"], text=True
    ).strip()
    if prescribe_commit != EXPECTED_PRESCRIBE_COMMIT:
        raise RuntimeError(f"Unexpected PRESCRIBE commit: {prescribe_commit}")
    rows = []
    for role, path in SOURCE_PATHS.items():
        observed = sha256_file(path)
        if observed != LOCKED_SOURCE_SHA256[role]:
            raise RuntimeError(f"Locked source changed: {role}: {observed}")
        rows.append(
            {
                "source_role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": observed,
                "source_git_commit": prescribe_commit,
            }
        )
    scperturb_commit = subprocess.check_output(
        ["git", "-C", str(SCPERTURB), "rev-parse", "HEAD"], text=True
    ).strip()
    if scperturb_commit != EXPECTED_SCPERTURB_COMMIT:
        raise RuntimeError(f"Unexpected scPerturb commit: {scperturb_commit}")
    for role, path in SCPERTURB_SOURCE_PATHS.items():
        observed = sha256_file(path)
        if observed != LOCKED_SCPERTURB_SOURCE_SHA256[role]:
            raise RuntimeError(f"Locked scPerturb source changed: {role}: {observed}")
        rows.append(
            {
                "source_role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": observed,
                "source_git_commit": scperturb_commit,
            }
        )
    return rows


def metadata_preflight() -> dict[str, Any]:
    runtime_rows = runtime_gate()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    committed = {
        "runner": git_blob_gate(RUNNER, head),
        "contract": git_blob_gate(CONTRACT, head),
        "preflight_amendment": git_blob_gate(AMENDMENT, head),
        "endogenous_axis_amendment": git_blob_gate(SECOND_AMENDMENT, head),
    }
    e160_status, condition_audit, split_dict = validate_e160(head)
    e161a_status = validate_e161a(head)
    sources = source_gate()

    raw_stat = RAW.lstat()
    if RAW.is_symlink() or not stat.S_ISREG(raw_stat.st_mode):
        raise RuntimeError("Wessels raw must be a regular non-symlink file")
    locked_identity = e160_status["raw_integrity"]
    observed_identity = {
        "device": int(raw_stat.st_dev),
        "inode": int(raw_stat.st_ino),
        "bytes": int(raw_stat.st_size),
        "mtime_ns": int(raw_stat.st_mtime_ns),
    }
    for key, observed in observed_identity.items():
        if observed != int(locked_identity[key]):
            raise RuntimeError(f"Raw identity differs from E160 at {key}")

    raw = ad.read_h5ad(RAW, backed="r")
    try:
        if raw.shape != EXPECTED_RAW_SHAPE:
            raise RuntimeError(f"Unexpected Wessels shape: {raw.shape}")
        required_obs = {
            "perturbation",
            "nperts",
            "Guide.Class",
            "cell_line",
            "celltype",
            "ncounts",
        }
        if not required_obs.issubset(raw.obs.columns):
            raise RuntimeError(f"Wessels obs fields changed: {sorted(required_obs-set(raw.obs.columns))}")
        obs = raw.obs[
            ["perturbation", "nperts", "Guide.Class", "cell_line", "celltype", "ncounts"]
        ].copy()
        raw_var_names = raw.var_names.astype(str).to_numpy(copy=True)
    finally:
        raw.file.close()
    raw_to_canonical = condition_audit.set_index("raw_condition")[
        "canonical_condition"
    ].astype(str).to_dict()
    if not obs.index.is_unique:
        raise RuntimeError("Raw Wessels obs names are no longer unique")
    canonical = obs["perturbation"].astype(str).map(raw_to_canonical)
    if canonical.isna().any():
        raise RuntimeError("A Wessels raw condition lacks the E160 canonical mapping")
    split_lookup = {
        condition: role for role, values in split_dict.items() for condition in values
    }
    roles = canonical.map(split_lookup).fillna("excluded")
    # ncounts is expression-derived upstream metadata and is needed only for
    # non-binding train/validation provenance.  Keep sealed roles redacted.
    obs.loc[roles.isin(["test", "excluded"]), "ncounts"] = np.nan
    observed_cells = roles.value_counts().to_dict()
    for role, expected in EXPECTED_SPLIT_CELLS.items():
        if int(observed_cells.get(role, 0)) != expected:
            raise RuntimeError(
                f"Metadata cell count mismatch for {role}: {observed_cells.get(role, 0)} != {expected}"
            )
    if (
        len(raw_var_names) != EXPECTED_RAW_SHAPE[1]
        or len(set(raw_var_names)) != len(raw_var_names)
    ):
        raise RuntimeError("Raw Wessels feature axis missing or duplicated")
    raw_axis_sha = hashlib.sha256(
        ("\n".join(raw_var_names.astype(str)) + "\n").encode("utf-8")
    ).hexdigest()
    if raw_axis_sha != EXPECTED_FULL_FEATURE_AXIS_SHA256:
        raise RuntimeError("Raw Wessels full feature axis changed")
    var_names = raw_var_names[:EXPECTED_ENDOGENOUS_FEATURES].copy()
    engineered_var_names = raw_var_names[
        EXPECTED_ENDOGENOUS_FEATURES:
        EXPECTED_ENDOGENOUS_FEATURES + EXPECTED_ENGINEERED_FEATURES
    ].copy()
    guide_barcode_var_names = raw_var_names[
        EXPECTED_ENDOGENOUS_FEATURES + EXPECTED_ENGINEERED_FEATURES:
    ].copy()
    excluded_var_names = raw_var_names[EXPECTED_ENDOGENOUS_FEATURES:].copy()
    if len(engineered_var_names) != EXPECTED_ENGINEERED_FEATURES:
        raise RuntimeError("Wessels engineered-construct feature count changed")
    if len(guide_barcode_var_names) != EXPECTED_GUIDE_BARCODE_FEATURES:
        raise RuntimeError("Wessels guide/barcode feature count changed")
    if len(excluded_var_names) != EXPECTED_EXCLUDED_FEATURES:
        raise RuntimeError("Wessels total excluded feature count changed")
    endogenous_axis_sha = hashlib.sha256(
        ("\n".join(var_names.astype(str)) + "\n").encode("utf-8")
    ).hexdigest()
    engineered_axis_sha = hashlib.sha256(
        ("\n".join(engineered_var_names.astype(str)) + "\n").encode("utf-8")
    ).hexdigest()
    guide_barcode_axis_sha = hashlib.sha256(
        ("\n".join(guide_barcode_var_names.astype(str)) + "\n").encode("utf-8")
    ).hexdigest()
    excluded_axis_sha = hashlib.sha256(
        ("\n".join(excluded_var_names.astype(str)) + "\n").encode("utf-8")
    ).hexdigest()
    if endogenous_axis_sha != EXPECTED_ENDOGENOUS_FEATURE_AXIS_SHA256:
        raise RuntimeError("Wessels endogenous feature axis changed")
    if engineered_axis_sha != EXPECTED_ENGINEERED_FEATURE_AXIS_SHA256:
        raise RuntimeError("Wessels engineered-construct axis changed")
    if guide_barcode_axis_sha != EXPECTED_GUIDE_BARCODE_FEATURE_AXIS_SHA256:
        raise RuntimeError("Wessels guide/barcode feature axis changed")
    if excluded_axis_sha != EXPECTED_EXCLUDED_FEATURE_AXIS_SHA256:
        raise RuntimeError("Wessels excluded feature axis changed")
    if (
        var_names[0] != "OR4F5"
        or var_names[-1] != "AC213203.1"
        or engineered_var_names.tolist()
        != ["eGFP", "Blast", "Cas9", "Puro", "Cas13d", "AsCas12a", "MeCP2", "KRAB"]
        or guide_barcode_var_names[0] != "ATXN7L3_g1:NT_g2"
        or guide_barcode_var_names[-1] != "CD71-Mpknot"
        or excluded_var_names[0] != "eGFP"
        or excluded_var_names[-1] != "CD71-Mpknot"
    ):
        raise RuntimeError("Wessels endogenous/excluded boundary sentinels changed")

    split_rows = {
        role: np.flatnonzero(roles.to_numpy() == role).astype(np.int64)
        for role in ("train", "val", "test", "excluded")
    }
    for left in split_rows:
        for right in split_rows:
            if left < right and np.intersect1d(split_rows[left], split_rows[right]).size:
                raise RuntimeError(f"Row-index overlap: {left}/{right}")
    train_singles = condition_audit.loc[
        condition_audit["split"].eq("train")
        & condition_audit["n_perturbation_genes"].eq(1)
        & condition_audit["model_compatible"].eq(True),  # noqa: E712
        "perturbation_genes",
    ].astype(str).tolist()
    if len(train_singles) != EXPECTED_PERT_GENES or len(set(train_singles)) != EXPECTED_PERT_GENES:
        raise RuntimeError("Expected exactly 27 train-single perturbation genes")
    if not set(train_singles).issubset(set(var_names)):
        raise RuntimeError("A forced train-single perturbation gene is absent from raw var")
    test_components = {
        gene for condition in split_dict["test"] for gene in condition.split("+")
    }
    if not test_components.issubset(set(train_singles)):
        raise RuntimeError("A test-pair component is not represented by a train single")

    return {
        "git_head": head,
        "committed_input_gate": committed,
        "e160_status": e160_status,
        "e161a_status": e161a_status,
        "condition_audit": condition_audit,
        "split_dict": split_dict,
        "obs": obs,
        "canonical": canonical,
        "roles": roles,
        "split_rows": split_rows,
        "var_names": var_names,
        "raw_var_names": raw_var_names,
        "engineered_var_names": engineered_var_names,
        "guide_barcode_var_names": guide_barcode_var_names,
        "excluded_var_names": excluded_var_names,
        "forced_genes": sorted(train_singles),
        "source_rows": sources,
        "runtime_rows": runtime_rows,
        "raw_identity": observed_identity,
        "raw_metadata_opened_backed_read_only": True,
        "test_and_excluded_obs_ncounts_redacted": True,
        "raw_X_rows_indexed_or_materialized": False,
    }


def read_allowed_expression(
    raw: ad.AnnData,
    role: str,
    row_indices: np.ndarray,
    frozen: dict[str, Any],
    ledger: list[dict[str, Any]],
) -> sp.csr_matrix:
    """Only function in E161 permitted to index the backed raw X matrix."""
    if role not in {"train", "val"}:
        raise RuntimeError(f"Expression access forbidden for split={role}")
    rows = np.asarray(row_indices, dtype=np.int64)
    if not len(rows) or not np.all(rows[:-1] < rows[1:]):
        raise RuntimeError(f"{role} row indices must be sorted, unique, and non-empty")
    if not np.array_equal(rows, np.asarray(frozen["split_rows"][role], dtype=np.int64)):
        raise RuntimeError(f"{role} expression request differs from the frozen role rows")
    for forbidden in ("test", "excluded"):
        if np.intersect1d(rows, frozen["split_rows"][forbidden]).size:
            raise RuntimeError(f"{role} expression request intersects {forbidden} rows")
    # Only endogenous genes are materialized.  The eight engineered constructs
    # and 413 guide/barcode columns are excluded before normalization/HVG/PCA.
    matrix = raw.X[rows, :EXPECTED_ENDOGENOUS_FEATURES]
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    matrix = matrix.tocsr()
    if matrix.shape != (len(rows), EXPECTED_ENDOGENOUS_FEATURES):
        raise RuntimeError(f"Unexpected {role} endogenous-axis shape: {matrix.shape}")
    if not np.issubdtype(matrix.dtype, np.integer):
        raise RuntimeError("Wessels X is no longer integer raw counts")
    if matrix.nnz and matrix.data.min() < 0:
        raise RuntimeError(f"Negative raw count in {role}")
    ledger.append(
        {
            "phase": f"semantic_{role}_endogenous_axis_read",
            "split": role,
            "rows_indexed": len(rows),
            "columns_indexed": matrix.shape[1],
            "engineered_construct_columns_indexed": 0,
            "guide_barcode_columns_indexed": 0,
            "total_excluded_columns_indexed": 0,
            "row_index_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
            "test_row_intersection": 0,
            "excluded_row_intersection": 0,
            "X_indexed": True,
            "X_materialized": True,
            "X_transformed": False,
            "rows_transformed": 0,
        }
    )
    return matrix


def fit_gene_axis(
    train_full: sp.csr_matrix, var_names: np.ndarray, forced_genes: list[str]
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    detected = np.asarray((train_full > 0).sum(axis=0)).reshape(-1).astype(int)
    total = np.asarray(train_full.sum(axis=0)).reshape(-1).astype(np.int64)
    hvg_data = ad.AnnData(
        X=train_full,
        var=pd.DataFrame(index=pd.Index(var_names.astype(str), name="gene")),
    )
    sc.pp.highly_variable_genes(
        hvg_data, n_top_genes=N_HVG, flavor="seurat_v3", subset=False
    )
    hvg = hvg_data.var["highly_variable"].to_numpy(bool)
    if int(hvg.sum()) != N_HVG:
        raise RuntimeError("Train-only seurat_v3 did not return exactly 2,000 HVGs")
    forced = np.isin(var_names.astype(str), forced_genes)
    selected = hvg | forced
    selected_indices = np.flatnonzero(selected).astype(np.int64)
    if not N_HVG <= len(selected_indices) <= N_HVG + EXPECTED_PERT_GENES:
        raise RuntimeError("Selected axis is outside the preregistered 2,000..2,027 range")
    selected_genes = var_names[selected_indices].astype(str).tolist()
    if not set(forced_genes).issubset(selected_genes):
        raise RuntimeError("A forced train-single gene is missing from the selected axis")
    axis = pd.DataFrame(
        {
            "raw_gene_index": np.arange(len(var_names), dtype=int),
            "gene": var_names.astype(str),
            "train_detected_cells": detected,
            "train_total_counts": total,
            "train_seurat_v3_hvg": hvg,
            "forced_train_single_gene": forced,
            "selected_model_gene": selected,
        }
    )
    for column in ("highly_variable_rank", "means", "variances", "variances_norm"):
        axis[f"train_hvg_{column}"] = (
            hvg_data.var[column].to_numpy()
            if column in hvg_data.var
            else np.full(len(axis), np.nan)
        )
    selected_lookup = {gene: index for index, gene in enumerate(selected_genes)}
    axis["selected_gene_index"] = axis["gene"].map(selected_lookup).fillna(-1).astype(int)
    return axis, selected_indices, selected_genes


def normalize_with_endogenous_library(
    endogenous_counts: sp.csr_matrix,
    selected_indices: np.ndarray,
    role: str,
    expected_obs_ncounts: np.ndarray | None = None,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, dict[str, Any]]:
    library = np.asarray(endogenous_counts.sum(axis=1)).reshape(-1).astype(np.float64)
    if not np.isfinite(library).all() or np.any(library <= 0):
        raise RuntimeError(f"{role} has zero/non-finite endogenous library size")
    metadata_max_abs_delta = np.nan
    metadata_min_delta = np.nan
    metadata_max_delta = np.nan
    metadata_sum_delta = np.nan
    metadata_mismatched_cells = np.nan
    if expected_obs_ncounts is not None:
        expected_library = np.asarray(expected_obs_ncounts, dtype=np.float64)
        if expected_library.shape != library.shape or not np.isfinite(expected_library).all():
            raise RuntimeError(f"Malformed {role} obs[ncounts] metadata")
        metadata_delta = library - expected_library
        metadata_max_abs_delta = float(np.max(np.abs(metadata_delta)))
        metadata_min_delta = float(np.min(metadata_delta))
        metadata_max_delta = float(np.max(metadata_delta))
        metadata_sum_delta = float(np.sum(metadata_delta))
        metadata_mismatched_cells = int(np.count_nonzero(metadata_delta))
    selected_counts = endogenous_counts[:, selected_indices].tocsr()
    selected_sum = np.asarray(selected_counts.sum(axis=1)).reshape(-1).astype(float)
    scale = TARGET_SUM / library
    normalized64 = sp.diags(scale, format="csr") @ selected_counts.astype(np.float64)
    np.log1p(normalized64.data, out=normalized64.data)
    normalized64.eliminate_zeros()
    normalized = normalized64.astype(np.float32)
    if not np.isfinite(normalized.data).all():
        raise RuntimeError(f"Non-finite {role} normalized values")
    coo = selected_counts.tocoo()
    sample_n = min(coo.nnz, 10_000)
    max_error = 0.0
    if sample_n:
        take = np.linspace(0, coo.nnz - 1, sample_n, dtype=int)
        rows, cols = coo.row[take], coo.col[take]
        expected = np.log1p(coo.data[take].astype(float) * scale[rows])
        observed = np.asarray(normalized[rows, cols]).reshape(-1).astype(float)
        max_error = float(np.max(np.abs(expected - observed)))
    if max_error > 1e-6:
        raise RuntimeError(f"{role} normalization formula mismatch: {max_error}")
    audit = {
        "split": role,
        "n_cells": len(library),
        "denominator_endogenous_feature_count": EXPECTED_ENDOGENOUS_FEATURES,
        "excluded_engineered_construct_feature_count": EXPECTED_ENGINEERED_FEATURES,
        "excluded_guide_or_barcode_feature_count": EXPECTED_GUIDE_BARCODE_FEATURES,
        "excluded_total_feature_count": EXPECTED_EXCLUDED_FEATURES,
        "selected_gene_count": len(selected_indices),
        "library_min": float(library.min()),
        "library_median": float(np.median(library)),
        "library_max": float(library.max()),
        "selected_fraction_min": float(np.min(selected_sum / library)),
        "selected_fraction_median": float(np.median(selected_sum / library)),
        "selected_fraction_max": float(np.max(selected_sum / library)),
        "formula_check_n": sample_n,
        "formula_max_abs_error": max_error,
        "obs_ncounts_max_abs_delta": metadata_max_abs_delta,
        "endogenous_minus_obs_ncounts_min": metadata_min_delta,
        "endogenous_minus_obs_ncounts_max": metadata_max_delta,
        "endogenous_minus_obs_ncounts_sum": metadata_sum_delta,
        "obs_ncounts_mismatched_cells": metadata_mismatched_cells,
        "obs_ncounts_is_binding_gate": False,
        "formula": "log1p(selected_count*10000/full_20631_endogenous_library)",
    }
    return normalized, selected_counts, library, audit


def fixed_pca_transform(
    matrix: sp.csr_matrix, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    transformed = safe_sparse_dot(matrix, components.T, dense_output=True)
    transformed -= np.asarray(mean @ components.T, dtype=transformed.dtype)
    return np.asarray(transformed, dtype=np.float32)


def within_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        raise RuntimeError("E-distance condition has fewer than two train cells")
    mean = values.mean(axis=0)
    return float(
        2.0 / (n - 1)
        * (float(np.square(values).sum()) - n * float(np.dot(mean, mean)))
    )


def fit_train_edistance(
    train_pca: np.ndarray, conditions: np.ndarray, split_dict: dict[str, list[str]]
) -> pd.DataFrame:
    ctrl = np.asarray(train_pca[conditions == "ctrl"], dtype=np.float64)
    if len(ctrl) != 424:
        raise RuntimeError(f"Expected 424 train controls, found {len(ctrl)}")
    ctrl_mean = ctrl.mean(axis=0)
    ctrl_mean_sqnorm = float(np.square(ctrl).sum(axis=1).mean())
    sigma_ctrl = within_sigma(ctrl)
    rows = []
    for condition in split_dict["train"]:
        values = np.asarray(train_pca[conditions == condition], dtype=np.float64)
        if condition == "ctrl":
            delta, sigma, edistance = sigma_ctrl, sigma_ctrl, 0.0
        else:
            mean = values.mean(axis=0)
            delta = (
                ctrl_mean_sqnorm
                + float(np.square(values).sum(axis=1).mean())
                - 2.0 * float(np.dot(ctrl_mean, mean))
            )
            sigma = within_sigma(values)
            edistance = 2.0 * delta - sigma_ctrl - sigma
        rows.append(
            {
                "condition": condition,
                "n_train_cells": len(values),
                "y_d_delta_to_control": delta,
                "y_s_sigma_within": sigma,
                "sigma_control": sigma_ctrl,
                "y_n_edistance": edistance,
                "fit_scope": "all_train_cells_unequal_n_moment_formula",
            }
        )
    result = pd.DataFrame(rows)
    numeric = result[
        ["y_d_delta_to_control", "y_s_sigma_within", "sigma_control", "y_n_edistance"]
    ].to_numpy(float)
    if len(result) != 72 or not np.isfinite(numeric).all():
        raise RuntimeError("Malformed train E-distance table")
    return result


def make_dev_obs(
    frozen: dict[str, Any], role: str, library: np.ndarray
) -> pd.DataFrame:
    indices = frozen["split_rows"][role]
    source = frozen["obs"].iloc[indices]
    canonical = frozen["canonical"].iloc[indices].astype(str).to_numpy()
    obs = pd.DataFrame(index=source.index.copy())
    obs["raw_perturbation"] = source["perturbation"].astype(str).to_numpy()
    obs["perturbation"] = canonical
    obs["condition"] = canonical
    obs["e161_split"] = role
    obs["cell_line"] = source["cell_line"].astype(str).to_numpy()
    obs["cell_type"] = source["celltype"].astype(str).to_numpy()
    obs["nperts"] = pd.to_numeric(source["nperts"], errors="raise").to_numpy(dtype=int)
    obs["Guide.Class"] = source["Guide.Class"].astype(str).to_numpy()
    obs["e161_raw_row_index"] = indices
    obs["e161_endogenous_library_size_20631"] = library.astype(np.int64)
    obs["condition_name"] = obs["cell_line"] + "_" + obs["condition"]
    return obs


def build_development(
    frozen: dict[str, Any],
    axis: pd.DataFrame,
    selected_indices: np.ndarray,
    selected_genes: list[str],
    train_norm: sp.csr_matrix,
    val_norm: sp.csr_matrix,
    train_selected_counts: sp.csr_matrix,
    val_selected_counts: sp.csr_matrix,
    train_library: np.ndarray,
    val_library: np.ndarray,
    train_pca: np.ndarray,
    val_pca: np.ndarray,
    pca: PCA,
    edistance: pd.DataFrame,
) -> tuple[ad.AnnData, np.ndarray, np.ndarray, np.ndarray]:
    train_obs = make_dev_obs(frozen, "train", train_library)
    val_obs = make_dev_obs(frozen, "val", val_library)
    obs = pd.concat([train_obs, val_obs], axis=0)
    for column in (
        "raw_perturbation",
        "perturbation",
        "condition",
        "e161_split",
        "cell_line",
        "cell_type",
        "Guide.Class",
        "condition_name",
    ):
        obs[column] = pd.Categorical(obs[column])

    selected_axis = axis.iloc[selected_indices].reset_index(drop=True)
    var = pd.DataFrame(index=pd.Index(selected_genes, name="gene_symbol"))
    var["gene_name"] = selected_genes
    var["raw_gene_index"] = selected_axis["raw_gene_index"].to_numpy(dtype=int)
    var["e161_train_detected_cells"] = selected_axis["train_detected_cells"].to_numpy(dtype=int)
    var["e161_train_total_counts"] = selected_axis["train_total_counts"].to_numpy(dtype=np.int64)
    var["highly_variable"] = selected_axis["train_seurat_v3_hvg"].to_numpy(dtype=bool)
    var["e161_forced_perturbation_gene"] = selected_axis[
        "forced_train_single_gene"
    ].to_numpy(dtype=bool)
    for source, target in (
        ("train_hvg_highly_variable_rank", "highly_variable_rank"),
        ("train_hvg_means", "means"),
        ("train_hvg_variances", "variances"),
        ("train_hvg_variances_norm", "variances_norm"),
    ):
        var[target] = selected_axis[source].to_numpy()

    development = ad.AnnData(
        X=sp.vstack([train_norm, val_norm], format="csr"), obs=obs, var=var
    )
    development.layers["counts"] = sp.vstack(
        [train_selected_counts, val_selected_counts], format="csr"
    )
    development.obsm["X_pca"] = np.vstack([train_pca, val_pca]).astype(np.float32)
    development.uns["pca_mean"] = np.asarray(pca.mean_, dtype=np.float32)
    development.uns["pca_components"] = np.asarray(pca.components_, dtype=np.float32)
    development.uns["processed"] = True
    development.uns["log1p"] = {"base": None, "target_sum": TARGET_SUM}
    development.uns["hvg"] = {
        "flavor": "seurat_v3",
        "n_top_genes": N_HVG,
        "fit_scope": "train_only_full_20631_endogenous_raw_count_axis",
        "excluded_engineered_construct_features": EXPECTED_ENGINEERED_FEATURES,
        "excluded_guide_or_barcode_features": EXPECTED_GUIDE_BARCODE_FEATURES,
        "forced_train_single_genes": EXPECTED_PERT_GENES,
    }
    e_lookup = edistance.set_index("condition")
    train_set = set(frozen["split_dict"]["train"])
    dev_conditions = frozen["split_dict"]["train"] + frozen["split_dict"]["val"]
    for key, column in (
        ("y_d", "y_d_delta_to_control"),
        ("y_s", "y_s_sigma_within"),
        ("y_n", "y_n_edistance"),
    ):
        development.uns[key] = {
            condition: (
                float(e_lookup.loc[condition, column])
                if condition in train_set
                else float("nan")
            )
            for condition in dev_conditions
        }
    ranked = axis.loc[axis["selected_model_gene"]].copy()
    ranked["rank_sort"] = ranked["train_hvg_highly_variable_rank"].fillna(np.inf)
    callback_order = ranked.sort_values(["rank_sort", "raw_gene_index"])["gene"].to_numpy(str)
    development.uns["rank_genes_groups_cov_all"] = {
        name: callback_order
        for name in development.obs["condition_name"].cat.categories.astype(str)
    }
    development.uns["e161_callback_policy"] = (
        "train-HVG-order compatibility placeholder; not condition-specific DE"
    )
    development.uns["e161_provenance"] = {
        "normalization": "full_20631_endogenous_library_then_selected_target10000_log1p",
        "gene_selection": "train_only_seurat_v3_top2000_union_27_train_single_genes",
        "pca": "train_only_randomized_PCA10_seed3407",
        "edistance": "all_train_cells_unequal_n_moment_formula_no_balancing",
        "test_or_excluded_X_rows_indexed_materialized_transformed": False,
        "raw_sha256": EXPECTED_RAW_SHA256,
        "full_feature_axis_sha256": EXPECTED_FULL_FEATURE_AXIS_SHA256,
        "endogenous_feature_axis_sha256": EXPECTED_ENDOGENOUS_FEATURE_AXIS_SHA256,
        "engineered_construct_feature_axis_sha256": EXPECTED_ENGINEERED_FEATURE_AXIS_SHA256,
        "guide_barcode_feature_axis_sha256": EXPECTED_GUIDE_BARCODE_FEATURE_AXIS_SHA256,
        "excluded_feature_axis_sha256": EXPECTED_EXCLUDED_FEATURE_AXIS_SHA256,
        "selected_gene_order_sha256": hashlib.sha256(
            ("\n".join(selected_genes) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    if development.n_obs != EXPECTED_DEV_CELLS:
        raise RuntimeError("Development AnnData cell count mismatch")
    observed_conditions = set(development.obs["condition"].astype(str))
    if observed_conditions != set(dev_conditions):
        raise RuntimeError("Development AnnData condition membership mismatch")
    if observed_conditions & set(frozen["split_dict"]["test"]):
        raise RuntimeError("Test condition appears in development AnnData")
    sealed_obs_names = set(
        frozen["obs"].index[
            np.concatenate(
                [frozen["split_rows"]["test"], frozen["split_rows"]["excluded"]]
            )
        ].astype(str)
    )
    if set(development.obs_names.astype(str)) & sealed_obs_names:
        raise RuntimeError("A sealed test/excluded obs name appears in development AnnData")
    for key in ("y_d", "y_s", "y_n"):
        if set(development.uns[key]) != observed_conditions:
            raise RuntimeError(f"Development {key} keys differ from development conditions")
    expected_callback_keys = set(
        development.obs["condition_name"].cat.categories.astype(str)
    )
    if set(development.uns["rank_genes_groups_cov_all"]) != expected_callback_keys:
        raise RuntimeError("Development compatibility callback keys changed")

    ctrl_mask = development.obs["condition"].astype(str).eq("ctrl").to_numpy()
    if int(ctrl_mask.sum()) != 424:
        raise RuntimeError("Development control cell count mismatch")
    control_gene_mean = np.asarray(development.X[ctrl_mask].mean(axis=0)).reshape(-1).astype(np.float32)
    control_pca = np.asarray(development.obsm["X_pca"][ctrl_mask], dtype=np.float64)
    control_pca_mean = control_pca.mean(axis=0)
    control_pca_cov = np.cov(control_pca, rowvar=False)
    if (
        control_gene_mean.shape != (development.n_vars,)
        or control_pca_mean.shape != (N_PCA,)
        or control_pca_cov.shape != (N_PCA, N_PCA)
        or not np.isfinite(control_gene_mean).all()
        or not np.isfinite(control_pca_mean).all()
        or not np.isfinite(control_pca_cov).all()
    ):
        raise RuntimeError("Malformed train-control gene/PCA prior")
    if not np.allclose(control_pca_cov, control_pca_cov.T, atol=1e-10):
        raise RuntimeError("Train-control PCA covariance is not symmetric")
    try:
        np.linalg.cholesky(control_pca_cov)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Train-control PCA covariance is not positive definite") from exc
    return development, control_gene_mean, control_pca_mean, control_pca_cov


def atomic_pickle(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=4)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_numpy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_data_assets(
    development: ad.AnnData,
    frozen: dict[str, Any],
    journal: dict[str, Any],
    axis: pd.DataFrame,
    selected_indices: np.ndarray,
    selected_genes: list[str],
    pca: PCA,
    control_gene_mean: np.ndarray,
    control_pca_mean: np.ndarray,
    control_pca_cov: np.ndarray,
    edistance: pd.DataFrame,
) -> None:
    ASSET_STAGING.mkdir(parents=True, exist_ok=False)
    write_transaction_sentinel(
        ASSET_STAGING, journal, "asset", ASSET_FINAL
    )
    (ASSET_STAGING / "data_pyg").mkdir()
    # Keep the terminal suffix .h5ad so anndata selects its HDF5 writer.
    temporary_h5ad = ASSET_STAGING / "perturb_processed.tmp.h5ad"
    development.write_h5ad(temporary_h5ad, compression="gzip")
    temporary_h5ad.replace(ASSET_STAGING / "perturb_processed.h5ad")
    atomic_pickle(ASSET_STAGING / f"set2conditions_{SEED}.pkl", frozen["split_dict"])
    atomic_pickle(
        ASSET_STAGING / f"frozen_pert_gene_set_{SEED}.pkl", frozen["forced_genes"]
    )
    atomic_numpy(ASSET_STAGING / "data_pyg/mean.npy", control_pca_mean)
    atomic_numpy(ASSET_STAGING / "data_pyg/cov.npy", control_pca_cov)
    atomic_npz(
        ASSET_STAGING / "TRAIN_ONLY_PCA_MODEL.npz",
        model_genes=np.asarray(selected_genes, dtype=str),
        raw_gene_indices=selected_indices,
        mean=np.asarray(pca.mean_, dtype=np.float32),
        components=np.asarray(pca.components_, dtype=np.float32),
        explained_variance=np.asarray(pca.explained_variance_, dtype=np.float32),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
    )
    atomic_npz(
        ASSET_STAGING / "TRAIN_ONLY_CONTROL_PRIOR.npz",
        control_gene_mean=control_gene_mean,
        control_pca_mean=control_pca_mean,
        control_pca_cov=control_pca_cov,
        n_train_controls=np.asarray([424], dtype=np.int64),
    )
    write_atomic_payload(
        ASSET_STAGING / "ENDOGENOUS_GENE_AXIS.txt",
        ("\n".join(frozen["var_names"].astype(str)) + "\n").encode("utf-8"),
    )
    write_atomic_payload(
        ASSET_STAGING / "FULL_RAW_FEATURE_AXIS.txt",
        ("\n".join(frozen["raw_var_names"].astype(str)) + "\n").encode("utf-8"),
    )
    write_atomic_payload(
        ASSET_STAGING / "ENGINEERED_CONSTRUCT_FEATURE_AXIS.txt",
        ("\n".join(frozen["engineered_var_names"].astype(str)) + "\n").encode("utf-8"),
    )
    write_atomic_payload(
        ASSET_STAGING / "GUIDE_BARCODE_FEATURE_AXIS.txt",
        ("\n".join(frozen["guide_barcode_var_names"].astype(str)) + "\n").encode("utf-8"),
    )
    write_atomic_payload(
        ASSET_STAGING / "EXCLUDED_FEATURE_AXIS.txt",
        ("\n".join(frozen["excluded_var_names"].astype(str)) + "\n").encode("utf-8"),
    )
    write_atomic_payload(
        ASSET_STAGING / "SELECTED_GENE_AXIS.txt",
        ("\n".join(selected_genes) + "\n").encode("utf-8"),
    )
    edistance.to_csv(ASSET_STAGING / "train_only_edistance_labels.csv", index=False)
    axis.to_csv(ASSET_STAGING / "train_only_gene_axis_audit.csv", index=False)


def build_and_audit_graphs(
    development: ad.AnnData,
    frozen: dict[str, Any],
    control_gene_mean: np.ndarray,
) -> pd.DataFrame:
    with (PRESCRIBE / "scLLM_weights/scGPT/embedding.pkl").open("rb") as handle:
        embedding = pickle.load(handle)
    for gene in frozen["forced_genes"]:
        if gene not in embedding or not np.isfinite(np.asarray(embedding[gene])).all():
            raise RuntimeError(f"Invalid frozen scGPT embedding for {gene}")
    del embedding

    old_cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        os.chdir(PRESCRIBE)
        sys.path.insert(0, str(PRESCRIBE))
        from gears import PertData  # noqa: PLC0415

        imported_source = Path(inspect.getfile(PertData)).resolve()
        if imported_source != SOURCE_PATHS["gears_pertdata"].resolve():
            raise RuntimeError(f"Unexpected imported PertData source: {imported_source}")
        if sha256_file(imported_source) != LOCKED_SOURCE_SHA256["gears_pertdata"]:
            raise RuntimeError("Imported PertData source hash changed after preflight")

        pert_data = PertData(
            str(PRESCRIBE / "data") + os.sep,
            gene_set_path=str(ASSET_STAGING / f"frozen_pert_gene_set_{SEED}.pkl"),
            default_pert_graph=False,
        )
        for gene in frozen["forced_genes"]:
            pert_data.gene2go.setdefault(gene, set())
        pert_data.load(data_path=str(ASSET_STAGING))
        pert_data.prepare_split(
            split="custom",
            seed=SEED,
            split_dict_path=str(ASSET_STAGING / f"set2conditions_{SEED}.pkl"),
        )
        pert_data.dataset_name = "wessels_e160"

        expected_keys = set(frozen["split_dict"]["train"]) | set(
            frozen["split_dict"]["val"]
        )
        graph_keys = set(pert_data.dataset_processed)
        if graph_keys != expected_keys:
            raise RuntimeError(
                "Development graph keys changed: "
                f"missing={sorted(expected_keys-graph_keys)}, extra={sorted(graph_keys-expected_keys)}"
            )
        if graph_keys & set(frozen["split_dict"]["test"]):
            raise RuntimeError("A test condition appears in the graph cache")
        split_lookup = {
            condition: role
            for role in ("train", "val")
            for condition in frozen["split_dict"][role]
        }
        rows = []
        total_graphs = 0
        for condition in sorted(expected_keys):
            role = split_lookup[condition]
            graphs = pert_data.dataset_processed[condition]
            condition_mask = development.obs["condition"].astype(str).eq(condition).to_numpy()
            truth = development[condition_mask]
            if len(graphs) != truth.n_obs:
                raise RuntimeError(f"Graph count mismatch for {condition}")
            max_x_delta = 0.0
            max_y_delta = 0.0
            max_pca_delta = 0.0
            bad = 0
            for index, graph in enumerate(graphs):
                x = graph.x.detach().cpu().numpy().reshape(-1)
                y = graph.y.detach().cpu().numpy().reshape(-1)
                y_pca = graph.y_pca.detach().cpu().numpy().reshape(-1)
                expected_y = np.asarray(truth.X[index].toarray()).reshape(-1)
                expected_pca = np.asarray(truth.obsm["X_pca"][index]).reshape(-1)
                if (
                    x.shape != (development.n_vars,)
                    or y.shape != (development.n_vars,)
                    or y_pca.shape != (N_PCA,)
                    or not np.isfinite(x).all()
                    or not np.isfinite(y).all()
                    or not np.isfinite(y_pca).all()
                    or str(graph.pert) != condition
                ):
                    bad += 1
                    continue
                max_x_delta = max(max_x_delta, float(np.max(np.abs(x - control_gene_mean))))
                max_y_delta = max(max_y_delta, float(np.max(np.abs(y - expected_y))))
                max_pca_delta = max(
                    max_pca_delta, float(np.max(np.abs(y_pca - expected_pca)))
                )
                y_n = graph.y_n.detach().cpu().numpy().reshape(-1)
                y_d = graph.y_d.detach().cpu().numpy().reshape(-1)
                y_s = graph.y_s.detach().cpu().numpy().reshape(-1)
                labels = np.concatenate([y_n, y_d, y_s])
                if (role == "train" and not np.isfinite(labels).all()) or (
                    role == "val" and not np.isnan(labels).all()
                ):
                    bad += 1
                pert_idx = np.asarray(graph.pert_idx, dtype=int).reshape(-1)
                expected_perts = [x for x in condition.split("+") if x != "ctrl"]
                expected_n_perts = len(expected_perts)
                if condition == "ctrl":
                    if pert_idx.tolist() != [-1]:
                        bad += 1
                elif len(pert_idx) != expected_n_perts or np.any(pert_idx < 0):
                    bad += 1
                elif pert_idx.max(initial=-1) >= len(pert_data.pert_names):
                    bad += 1
                elif np.asarray(pert_data.pert_names)[pert_idx].astype(str).tolist() != expected_perts:
                    bad += 1
                de_idx = np.asarray(graph.de_idx, dtype=int).reshape(-1)
                de_valid = (
                    len(de_idx) == 20
                    and (
                        np.all(de_idx == -1)
                        if condition == "ctrl"
                        else (np.all(de_idx >= 0) and np.all(de_idx < development.n_vars))
                    )
                )
                if not de_valid:
                    bad += 1
            if bad or max(max_x_delta, max_y_delta, max_pca_delta) > 1e-6:
                raise RuntimeError(f"Exhaustive graph audit failed for {condition}")
            rows.append(
                {
                    "condition": condition,
                    "split": role,
                    "n_graphs": len(graphs),
                    "n_h5ad_cells": truth.n_obs,
                    "max_abs_x_minus_control": max_x_delta,
                    "max_abs_y_minus_h5ad": max_y_delta,
                    "max_abs_y_pca_minus_h5ad": max_pca_delta,
                    "bad_graphs": bad,
                }
            )
            total_graphs += len(graphs)
        if total_graphs != EXPECTED_DEV_CELLS:
            raise RuntimeError(f"Expected 16,881 development graphs, found {total_graphs}")
        frame = pd.DataFrame(rows)
        counts = frame.groupby("split")["n_graphs"].sum().to_dict()
        if counts != {"train": 11_779, "val": 5_102}:
            raise RuntimeError(f"Graph split counts changed: {counts}")
        del pert_data
        gc.collect()
        return frame
    finally:
        sys.path[:] = old_path
        os.chdir(old_cwd)


def write_atomic_payload(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(root: Path) -> None:
    """Durably flush a staging tree before its atomic directory rename."""
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Cannot fsync non-directory staging root: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in files):
        raise RuntimeError(f"Symlink rejected inside staging tree: {root}")
    for path in files:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    directories = sorted(
        [root, *(path for path in root.rglob("*") if path.is_dir())],
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        fsync_directory(directory)


def build_asset_manifest(
    selected_genes: list[str], frozen: dict[str, Any], graph_audit: pd.DataFrame
) -> pd.DataFrame:
    expected_before_interface = {
        "perturb_processed.h5ad",
        f"set2conditions_{SEED}.pkl",
        f"frozen_pert_gene_set_{SEED}.pkl",
        "data_pyg/cell_graphs.pkl",
        "data_pyg/mean.npy",
        "data_pyg/cov.npy",
        "TRAIN_ONLY_PCA_MODEL.npz",
        "TRAIN_ONLY_CONTROL_PRIOR.npz",
        "ENDOGENOUS_GENE_AXIS.txt",
        "FULL_RAW_FEATURE_AXIS.txt",
        "ENGINEERED_CONSTRUCT_FEATURE_AXIS.txt",
        "GUIDE_BARCODE_FEATURE_AXIS.txt",
        "EXCLUDED_FEATURE_AXIS.txt",
        "SELECTED_GENE_AXIS.txt",
        "train_only_edistance_labels.csv",
        "train_only_gene_axis_audit.csv",
    }
    observed_before_interface = {
        path.relative_to(ASSET_STAGING).as_posix()
        for path in ASSET_STAGING.rglob("*")
        if path.is_file() and path.name != STAGING_SENTINEL
    }
    if observed_before_interface != expected_before_interface:
        raise RuntimeError(
            f"Data asset allowlist mismatch: {observed_before_interface ^ expected_before_interface}"
        )
    base_paths = sorted(
        path
        for path in ASSET_STAGING.rglob("*")
        if path.is_file()
        and path.name
        not in {STAGING_SENTINEL, "ASSET_MANIFEST.csv", "E161_E162_INTERFACE.json"}
    )
    if any(path.is_symlink() for path in base_paths):
        raise RuntimeError("Symlink rejected inside E161 data staging")
    base_hashes = {
        path.relative_to(ASSET_STAGING).as_posix(): sha256_file(path) for path in base_paths
    }
    interface = {
        "schema": "safeconf_e161_to_e162_v2",
        "data_name": "wessels_e160",
        "data_root": str(ASSET_FINAL),
        "seed": SEED,
        "n_selected_genes": len(selected_genes),
        "selected_gene_order_sha256": hashlib.sha256(
            ("\n".join(selected_genes) + "\n").encode("utf-8")
        ).hexdigest(),
        "endogenous_feature_order_sha256": hashlib.sha256(
            ("\n".join(frozen["var_names"].astype(str)) + "\n").encode("utf-8")
        ).hexdigest(),
        "engineered_construct_feature_order_sha256": hashlib.sha256(
            ("\n".join(frozen["engineered_var_names"].astype(str)) + "\n").encode("utf-8")
        ).hexdigest(),
        "guide_barcode_feature_order_sha256": hashlib.sha256(
            ("\n".join(frozen["guide_barcode_var_names"].astype(str)) + "\n").encode("utf-8")
        ).hexdigest(),
        "full_raw_feature_order_sha256": hashlib.sha256(
            ("\n".join(frozen["raw_var_names"].astype(str)) + "\n").encode("utf-8")
        ).hexdigest(),
        "excluded_feature_order_sha256": hashlib.sha256(
            ("\n".join(frozen["excluded_var_names"].astype(str)) + "\n").encode("utf-8")
        ).hexdigest(),
        "n_endogenous_features": EXPECTED_ENDOGENOUS_FEATURES,
        "n_engineered_construct_features": EXPECTED_ENGINEERED_FEATURES,
        "n_guide_barcode_features": EXPECTED_GUIDE_BARCODE_FEATURES,
        "n_excluded_features": EXPECTED_EXCLUDED_FEATURES,
        "split_conditions": {role: len(frozen["split_dict"][role]) for role in ("train", "val", "test")},
        "development_graphs": {"train": 11_779, "val": 5_102, "test": 0},
        "graph_keys": int(len(graph_audit)),
        "normalization": "full_20631_endogenous_library_then_selected_target10000_log1p",
        "obs_ncounts_policy": "non_binding_upstream_metadata_audit_only",
        "endogenous_axis_amendment_sha256": sha256_file(SECOND_AMENDMENT),
        "e161a_status_sha256": sha256_file(E161A_STATUS),
        "e161a_manifest_sha256": sha256_file(E161A_MANIFEST),
        "test_X_rows_indexed_materialized_or_transformed": False,
        "test_and_excluded_obs_ncounts_redacted": True,
        "engineered_construct_X_columns_indexed_or_materialized": False,
        "guide_barcode_X_columns_indexed_or_materialized": False,
        "excluded_X_columns_indexed_or_materialized": False,
        "custom_adapter": "upstream PertData.load(data_path)+prepare_split(custom); never LoadData",
        "label_only_x": "TRAIN_ONLY_CONTROL_PRIOR.npz::control_gene_mean",
        "paths": {
            "dev_h5ad": "perturb_processed.h5ad",
            "split_pickle": f"set2conditions_{SEED}.pkl",
            "graph_cache": "data_pyg/cell_graphs.pkl",
            "pca_model": "TRAIN_ONLY_PCA_MODEL.npz",
            "control_prior": "TRAIN_ONLY_CONTROL_PRIOR.npz",
            "selected_gene_axis": "SELECTED_GENE_AXIS.txt",
            "endogenous_gene_axis": "ENDOGENOUS_GENE_AXIS.txt",
            "engineered_construct_axis": "ENGINEERED_CONSTRUCT_FEATURE_AXIS.txt",
            "guide_barcode_axis": "GUIDE_BARCODE_FEATURE_AXIS.txt",
            "excluded_feature_axis": "EXCLUDED_FEATURE_AXIS.txt",
        },
        "asset_sha256": base_hashes,
    }
    write_atomic_payload(
        ASSET_STAGING / "E161_E162_INTERFACE.json", json_bytes(interface)
    )
    paths = sorted(
        path
        for path in ASSET_STAGING.rglob("*")
        if path.is_file()
        and path.name not in {STAGING_SENTINEL, "ASSET_MANIFEST.csv"}
    )
    manifest = pd.DataFrame(
        [
            {
                "relative_path": path.relative_to(ASSET_STAGING).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ]
    )
    write_atomic_payload(
        ASSET_STAGING / "ASSET_MANIFEST.csv", manifest.to_csv(index=False).encode("utf-8")
    )
    return manifest


def audit_tree_allowlist(
    root: Path,
    allowed_files: frozenset[str] | set[str],
    allowed_directories: frozenset[str] | set[str],
    *,
    require_exact_files: bool,
    require_exact_directories: bool,
) -> tuple[set[str], set[str]]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Transaction tree is not a regular directory: {root}")
    files: set[str] = set()
    directories: set[str] = set()
    for child in root.rglob("*"):
        if child.is_symlink():
            raise RuntimeError(f"Symlink rejected in transaction tree: {child}")
        relative = child.relative_to(root).as_posix()
        if child.is_file():
            files.add(relative)
        elif child.is_dir():
            directories.add(relative)
        else:
            raise RuntimeError(f"Non-file/non-directory rejected in transaction tree: {child}")
    allowed_file_set = set(allowed_files)
    allowed_directory_set = set(allowed_directories)
    if not files.issubset(allowed_file_set):
        raise RuntimeError(f"Unexpected transaction files in {root}: {files-allowed_file_set}")
    if not directories.issubset(allowed_directory_set):
        raise RuntimeError(
            f"Unexpected transaction directories in {root}: "
            f"{directories-allowed_directory_set}"
        )
    if require_exact_files and files != allowed_file_set:
        raise RuntimeError(f"Incomplete transaction files in {root}: {files ^ allowed_file_set}")
    if require_exact_directories and directories != allowed_directory_set:
        raise RuntimeError(
            f"Incomplete transaction directories in {root}: "
            f"{directories ^ allowed_directory_set}"
        )
    return files, directories


def transaction_roots() -> dict[str, str]:
    return {
        "asset_staging": str(ASSET_STAGING),
        "asset_final": str(ASSET_FINAL),
        "prescribe_link": str(PRESCRIBE_LINK),
        "repo_staging": str(REPO_STAGING),
        "repo_release": str(RELEASE),
    }


def write_transaction_journal(journal: dict[str, Any]) -> None:
    journal["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_atomic_payload(TRANSACTION_JOURNAL, json_bytes(journal))
    fsync_directory(OUT)


def load_transaction_journal() -> dict[str, Any]:
    if (
        not TRANSACTION_JOURNAL.is_file()
        or TRANSACTION_JOURNAL.is_symlink()
    ):
        raise RuntimeError(f"E161 transaction journal missing or unsafe: {TRANSACTION_JOURNAL}")
    journal = json.loads(TRANSACTION_JOURNAL.read_text(encoding="utf-8"))
    if (
        journal.get("schema") != "safeconf_e161_transaction_v1"
        or journal.get("experiment") != "E161"
        or journal.get("roots") != transaction_roots()
        or not isinstance(journal.get("transaction_id"), str)
        or len(journal["transaction_id"]) != 32
    ):
        raise RuntimeError("E161 transaction journal identity mismatch")
    return journal


def update_transaction_phase(
    journal: dict[str, Any], phase: str, **details: Any
) -> dict[str, Any]:
    allowed_transitions = {
        "building": {"ready_to_publish", "rolled_back"},
        "ready_to_publish": {"asset_published", "complete"},
        "asset_published": {"link_published", "complete"},
        "link_published": {"complete"},
        "complete": {"complete"},
        "rolled_back": {"rolled_back"},
    }
    current = str(journal.get("phase"))
    if phase not in allowed_transitions.get(current, set()):
        raise RuntimeError(f"Illegal E161 transaction transition: {current} -> {phase}")
    journal = dict(journal)
    journal["phase"] = phase
    journal.update(details)
    history = list(journal.get("phase_history", []))
    history.append(
        {
            "phase": phase,
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    journal["phase_history"] = history
    write_transaction_journal(journal)
    return journal


def begin_transaction(
    frozen: dict[str, Any], raw_hash: dict[str, Any]
) -> dict[str, Any]:
    previous: dict[str, Any] | None = None
    if TRANSACTION_JOURNAL.exists() or TRANSACTION_JOURNAL.is_symlink():
        previous = load_transaction_journal()
        if previous.get("phase") != "rolled_back":
            raise RuntimeError(
                "An active/completed E161 transaction journal exists; "
                "use --recover-staging for an interrupted transaction"
            )
    journal = {
        "schema": "safeconf_e161_transaction_v1",
        "experiment": "E161",
        "transaction_id": uuid.uuid4().hex,
        "phase": "building",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "phase_history": [
            {
                "phase": "building",
                "at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        ],
        "roots": transaction_roots(),
        "git_head": frozen["git_head"],
        "runner_sha256": sha256_file(RUNNER),
        "contract_sha256": sha256_file(CONTRACT),
        "preflight_amendment_sha256": sha256_file(AMENDMENT),
        "endogenous_axis_amendment_sha256": sha256_file(SECOND_AMENDMENT),
        "e161a_status_sha256": sha256_file(E161A_STATUS),
        "e161a_manifest_sha256": sha256_file(E161A_MANIFEST),
        "raw_sha256": raw_hash["sha256"],
        "runtime_environment": {
            row["component"]: row["observed_version"]
            for row in frozen["runtime_rows"]
        },
        "previous_rolled_back_transaction": (
            {
                "transaction_id": previous["transaction_id"],
                "updated_at": previous.get("updated_at"),
            }
            if previous is not None
            else None
        ),
    }
    write_transaction_journal(journal)
    return journal


def sentinel_payload(
    journal: dict[str, Any], role: str, target: Path
) -> dict[str, Any]:
    return {
        "schema": "safeconf_e161_transaction_sentinel_v1",
        "experiment": "E161",
        "transaction_id": journal["transaction_id"],
        "role": role,
        "target": str(target),
        "journal": str(TRANSACTION_JOURNAL),
    }


def write_transaction_sentinel(
    root: Path, journal: dict[str, Any], role: str, target: Path
) -> None:
    write_atomic_payload(
        root / STAGING_SENTINEL,
        json_bytes(sentinel_payload(journal, role, target)),
    )
    fsync_directory(root)


def validate_transaction_sentinel(
    root: Path, journal: dict[str, Any], role: str, target: Path
) -> None:
    sentinel = root / STAGING_SENTINEL
    if not sentinel.is_file() or sentinel.is_symlink():
        raise RuntimeError(f"Transaction sentinel missing or unsafe: {sentinel}")
    observed = json.loads(sentinel.read_text(encoding="utf-8"))
    if observed != sentinel_payload(journal, role, target):
        raise RuntimeError(f"Transaction sentinel identity mismatch: {sentinel}")


def validate_data_tree(root: Path, journal: dict[str, Any]) -> None:
    audit_tree_allowlist(
        root,
        DATA_OPERATIONAL_ALLOWLIST,
        DATA_DIRECTORY_ALLOWLIST,
        require_exact_files=True,
        require_exact_directories=True,
    )
    validate_transaction_sentinel(root, journal, "asset", ASSET_FINAL)
    manifest_path = root / "ASSET_MANIFEST.csv"
    manifest = pd.read_csv(manifest_path)
    if list(manifest.columns) != ["relative_path", "bytes", "sha256"]:
        raise RuntimeError("Malformed E161 asset manifest schema")
    expected_manifest_paths = set(DATA_SCIENTIFIC_ALLOWLIST) - {"ASSET_MANIFEST.csv"}
    observed_manifest_paths = set(manifest["relative_path"].astype(str))
    if observed_manifest_paths != expected_manifest_paths or manifest["relative_path"].duplicated().any():
        raise RuntimeError("E161 asset manifest path set mismatch")
    for row in manifest.itertuples(index=False):
        path = root / str(row.relative_path)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Unsafe/missing E161 manifest asset: {path}")
        if path.stat().st_size != int(row.bytes) or sha256_file(path) != str(row.sha256):
            raise RuntimeError(f"E161 manifest asset hash mismatch: {path}")
    expected_manifest_hash = journal.get("asset_manifest_sha256")
    if expected_manifest_hash and sha256_file(manifest_path) != expected_manifest_hash:
        raise RuntimeError("E161 asset manifest differs from transaction journal")


def validate_repo_tree(root: Path, journal: dict[str, Any]) -> None:
    audit_tree_allowlist(
        root,
        REPO_OPERATIONAL_ALLOWLIST,
        REPO_DIRECTORY_ALLOWLIST,
        require_exact_files=True,
        require_exact_directories=True,
    )
    validate_transaction_sentinel(root, journal, "repo", RELEASE)
    status_path = root / "RUN_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("transaction_id") != journal["transaction_id"]:
        raise RuntimeError("E161 release status transaction id mismatch")
    artifact_hashes = status.get("artifact_sha256", {})
    expected_artifacts = set(REPO_ALLOWLIST) - {"RUN_STATUS.json"}
    if set(artifact_hashes) != expected_artifacts:
        raise RuntimeError("E161 release artifact allowlist mismatch")
    for relative, expected in artifact_hashes.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"E161 release artifact hash mismatch: {relative}")
    expected_status_hash = journal.get("repo_status_sha256")
    if expected_status_hash and sha256_file(status_path) != expected_status_hash:
        raise RuntimeError("E161 release status differs from transaction journal")


def strict_remove_staging(
    root: Path,
    journal: dict[str, Any],
    role: str,
    target: Path,
    allowed_files: frozenset[str] | set[str],
    allowed_directories: frozenset[str] | set[str],
) -> None:
    if not root.exists() and not root.is_symlink():
        return
    validate_transaction_sentinel(root, journal, role, target)
    audit_tree_allowlist(
        root,
        allowed_files,
        allowed_directories,
        require_exact_files=False,
        require_exact_directories=False,
    )
    shutil.rmtree(root)
    fsync_directory(root.parent)


def publication_gate(
    frozen: dict[str, Any], raw_hash: dict[str, Any]
) -> None:
    runtime_gate()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != frozen["git_head"]:
        raise RuntimeError("Git HEAD changed during E161 formal execution")
    git_blob_gate(RUNNER, head)
    git_blob_gate(CONTRACT, head)
    git_blob_gate(AMENDMENT, head)
    git_blob_gate(SECOND_AMENDMENT, head)
    validate_e160(head)
    validate_e161a(head)
    source_gate()
    observed_raw = hash_raw_once(RAW)
    if observed_raw != raw_hash:
        raise RuntimeError("Raw identity/hash changed before E161 publication")


def ensure_prescribe_link() -> None:
    if PRESCRIBE_LINK.exists() or PRESCRIBE_LINK.is_symlink():
        if (
            not PRESCRIBE_LINK.is_symlink()
            or PRESCRIBE_LINK.resolve() != ASSET_FINAL.resolve()
        ):
            raise RuntimeError(f"Unexpected PRESCRIBE data link target: {PRESCRIBE_LINK}")
        return
    PRESCRIBE_LINK.symlink_to(ASSET_FINAL, target_is_directory=True)
    fsync_directory(PRESCRIBE_LINK.parent)


def recover_transaction(
    frozen: dict[str, Any], raw_hash: dict[str, Any]
) -> str:
    journal = load_transaction_journal()
    phase = str(journal.get("phase"))
    if journal.get("git_head") != frozen["git_head"] or journal.get("raw_sha256") != raw_hash["sha256"]:
        raise RuntimeError("Current Git/raw identity differs from interrupted E161 transaction")

    asset_stage = ASSET_STAGING.exists() or ASSET_STAGING.is_symlink()
    asset_final = ASSET_FINAL.exists() or ASSET_FINAL.is_symlink()
    repo_stage = REPO_STAGING.exists() or REPO_STAGING.is_symlink()
    repo_final = RELEASE.exists() or RELEASE.is_symlink()
    if asset_stage and asset_final:
        raise RuntimeError("Both E161 asset staging and final exist")
    if repo_stage and repo_final:
        raise RuntimeError("Both E161 repo staging and release exist")

    if phase in {"ready_to_publish", "asset_published", "link_published", "complete"}:
        publication_gate(frozen, raw_hash)
        asset_root = ASSET_FINAL if asset_final else ASSET_STAGING
        repo_root = RELEASE if repo_final else REPO_STAGING
        if not (asset_root.exists() and repo_root.exists()):
            raise RuntimeError("Publishable E161 transaction is missing a complete tree")
        validate_data_tree(asset_root, journal)
        validate_repo_tree(repo_root, journal)
        if asset_root == ASSET_STAGING:
            fsync_tree(ASSET_STAGING)
            ASSET_STAGING.rename(ASSET_FINAL)
            fsync_directory(ASSET_PARENT)
        journal = update_transaction_phase(
            journal,
            "asset_published" if phase == "ready_to_publish" else phase,
        ) if phase == "ready_to_publish" else journal
        ensure_prescribe_link()
        if journal["phase"] == "asset_published":
            journal = update_transaction_phase(journal, "link_published")
        if repo_root == REPO_STAGING:
            fsync_tree(REPO_STAGING)
            REPO_STAGING.rename(RELEASE)
            fsync_directory(OUT)
        validate_data_tree(ASSET_FINAL, journal)
        validate_repo_tree(RELEASE, journal)
        if journal["phase"] != "complete":
            journal = update_transaction_phase(journal, "complete")
        return "complete"

    if phase == "building":
        if asset_final or repo_final:
            raise RuntimeError("Refusing rollback because a final E161 tree exists")
        if PRESCRIBE_LINK.exists() or PRESCRIBE_LINK.is_symlink():
            if (
                not PRESCRIBE_LINK.is_symlink()
                or PRESCRIBE_LINK.resolve() != ASSET_FINAL.resolve()
            ):
                raise RuntimeError(
                    f"Unexpected PRESCRIBE link during E161 rollback: {PRESCRIBE_LINK}"
                )
            PRESCRIBE_LINK.unlink()
            fsync_directory(PRESCRIBE_LINK.parent)
        strict_remove_staging(
            ASSET_STAGING,
            journal,
            "asset",
            ASSET_FINAL,
            DATA_STAGING_TEMP_ALLOWLIST,
            DATA_DIRECTORY_ALLOWLIST,
        )
        strict_remove_staging(
            REPO_STAGING,
            journal,
            "repo",
            RELEASE,
            REPO_STAGING_TEMP_ALLOWLIST,
            REPO_DIRECTORY_ALLOWLIST,
        )
        update_transaction_phase(journal, "rolled_back")
        return "rolled_back"
    if phase == "rolled_back":
        if (
            asset_stage
            or asset_final
            or repo_stage
            or repo_final
            or PRESCRIBE_LINK.exists()
            or PRESCRIBE_LINK.is_symlink()
        ):
            raise RuntimeError("Rolled-back E161 transaction still has filesystem artifacts")
        return "rolled_back"
    raise RuntimeError(f"Unknown E161 transaction phase: {phase}")


def publish_repo_release(
    frozen: dict[str, Any],
    journal: dict[str, Any],
    raw_hash: dict[str, Any],
    access_ledger: pd.DataFrame,
    normalization_audit: pd.DataFrame,
    axis: pd.DataFrame,
    edistance: pd.DataFrame,
    graph_audit: pd.DataFrame,
    asset_manifest: pd.DataFrame,
    started_at: str,
) -> None:
    REPO_STAGING.mkdir(parents=True, exist_ok=False)
    write_transaction_sentinel(
        REPO_STAGING, journal, "repo", RELEASE
    )
    (REPO_STAGING / "tables").mkdir()
    (REPO_STAGING / "reports").mkdir()

    normalization_audit.to_csv(
        REPO_STAGING / "tables/E161_NORMALIZATION_AUDIT.csv", index=False
    )
    axis.to_csv(REPO_STAGING / "tables/E161_GENE_AXIS_AUDIT.csv", index=False)
    edistance.to_csv(
        REPO_STAGING / "tables/E161_CONDITION_EDISTANCE.csv", index=False
    )
    graph_audit.to_csv(REPO_STAGING / "tables/E161_GRAPH_AUDIT.csv", index=False)
    access_ledger.to_csv(
        REPO_STAGING / "tables/E161_X_ACCESS_LEDGER.csv", index=False
    )
    asset_manifest.to_csv(
        REPO_STAGING / "tables/E161_ASSET_MANIFEST.csv", index=False
    )
    source_rows = list(frozen["source_rows"])
    source_rows.extend(
        [
            {
                "source_role": "E161_runner",
                "path": str(RUNNER),
                "bytes": RUNNER.stat().st_size,
                "sha256": sha256_file(RUNNER),
                "source_git_commit": frozen["git_head"],
            },
            {
                "source_role": "E161_contract",
                "path": str(CONTRACT),
                "bytes": CONTRACT.stat().st_size,
                "sha256": sha256_file(CONTRACT),
                "source_git_commit": frozen["git_head"],
            },
            {
                "source_role": "E161_preflight_amendment",
                "path": str(AMENDMENT),
                "bytes": AMENDMENT.stat().st_size,
                "sha256": sha256_file(AMENDMENT),
                "source_git_commit": frozen["git_head"],
            },
            {
                "source_role": "E161_endogenous_axis_amendment",
                "path": str(SECOND_AMENDMENT),
                "bytes": SECOND_AMENDMENT.stat().st_size,
                "sha256": sha256_file(SECOND_AMENDMENT),
                "source_git_commit": frozen["git_head"],
            },
            {
                "source_role": "E161a_run_status",
                "path": str(E161A_STATUS),
                "bytes": E161A_STATUS.stat().st_size,
                "sha256": sha256_file(E161A_STATUS),
                "source_git_commit": frozen["git_head"],
            },
            {
                "source_role": "E161a_results_manifest",
                "path": str(E161A_MANIFEST),
                "bytes": E161A_MANIFEST.stat().st_size,
                "sha256": sha256_file(E161A_MANIFEST),
                "source_git_commit": frozen["git_head"],
            },
            {
                "source_role": "Wessels_raw",
                "path": str(RAW),
                "bytes": raw_hash["bytes"],
                "sha256": raw_hash["sha256"],
                "source_git_commit": "",
            },
            {
                "source_role": "E160_run_status",
                "path": str(E160_FREEZE / "RUN_STATUS.json"),
                "bytes": (E160_FREEZE / "RUN_STATUS.json").stat().st_size,
                "sha256": sha256_file(E160_FREEZE / "RUN_STATUS.json"),
                "source_git_commit": frozen["git_head"],
            },
            {
                "source_role": "E160_condition_audit",
                "path": str(E160_FREEZE / "manifests/E160_CONDITION_AUDIT.csv"),
                "bytes": (E160_FREEZE / "manifests/E160_CONDITION_AUDIT.csv").stat().st_size,
                "sha256": sha256_file(E160_FREEZE / "manifests/E160_CONDITION_AUDIT.csv"),
                "source_git_commit": frozen["git_head"],
            },
            {
                "source_role": "E160_split_json",
                "path": str(E160_FREEZE / "manifests/E160_set2conditions.json"),
                "bytes": (E160_FREEZE / "manifests/E160_set2conditions.json").stat().st_size,
                "sha256": sha256_file(E160_FREEZE / "manifests/E160_set2conditions.json"),
                "source_git_commit": frozen["git_head"],
            },
        ]
    )
    pd.DataFrame(source_rows).to_csv(
        REPO_STAGING / "tables/E161_SOURCE_HASHES.csv", index=False
    )
    pd.DataFrame(frozen["runtime_rows"]).to_csv(
        REPO_STAGING / "tables/E161_RUNTIME_ENVIRONMENT.csv", index=False
    )
    split_rows = []
    for role in ("train", "val", "test", "excluded"):
        split_rows.append(
            {
                "split": role,
                "n_conditions": (
                    len(frozen["split_dict"][role]) if role != "excluded" else np.nan
                ),
                "n_cells_obs_metadata": len(frozen["split_rows"][role]),
                "semantic_X_rows_indexed": (
                    len(frozen["split_rows"][role]) if role in {"train", "val"} else 0
                ),
                "in_development_h5ad": role in {"train", "val"},
                "in_graph_cache": role in {"train", "val"},
                "test_or_excluded_leakage": False,
            }
        )
    pd.DataFrame(split_rows).to_csv(
        REPO_STAGING / "tables/E161_SPLIT_AND_LEAKAGE_AUDIT.csv", index=False
    )
    report = f"""# E161 Wessels train/validation 预处理报告

- train: 72 conditions / 11,779 cells;
- validation: 24 conditions / 5,102 cells;
- test: 48 conditions / 9,902 cells，expression 未索引、未物化、未转换；
- 容器共 21,052 features；8 个实验构造和 413 个 guide/barcode 列均未读取；
- 归一化分母：每个细胞前 20,631 个内源基因的 raw-count library；
- 原始 `obs[ncounts]` 是上游 `nCount_RNA` 元数据，只作差值审计，不作为分母硬门控；
- feature: train-only seurat_v3 top-2,000 与 27 个 train-single genes 并集；
- PCA/control prior/E-distance: 全部 train cells 拟合；
- E-distance: unequal-n moment formula，不做 15-cell 平衡；
- development graphs: 96 conditions / 16,881 graphs，test graph = 0；
- 本阶段未训练模型，未产生 test prediction 或 endpoint。
"""
    (REPO_STAGING / "reports/E161_REPORT.md").write_text(report, encoding="utf-8")
    (REPO_STAGING / "README_先看这个.md").write_text(
        "# E161\n\n先阅读 `reports/E161_REPORT.md`，再核对 tables 与数据资产 manifest。\n",
        encoding="utf-8",
    )

    observed_without_status = {
        path.relative_to(REPO_STAGING).as_posix()
        for path in REPO_STAGING.rglob("*")
        if path.is_file() and path.name != STAGING_SENTINEL
    }
    expected_without_status = set(REPO_ALLOWLIST) - {"RUN_STATUS.json"}
    if observed_without_status != expected_without_status:
        raise RuntimeError(
            f"Repo release allowlist mismatch: {observed_without_status ^ expected_without_status}"
        )
    artifact_hashes = {
        relative: sha256_file(REPO_STAGING / relative)
        for relative in sorted(observed_without_status)
    }
    status = {
        "experiment": "E161_wessels_trainval_preprocess",
        "transaction_id": journal["transaction_id"],
        "phase": "complete_preprocessing_and_dev_graphs_no_training_no_test_X_access",
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_head": frozen["git_head"],
        "raw_integrity": raw_hash,
        "normalization": "full_20631_endogenous_library_then_selected_target10000_log1p",
        "obs_ncounts_policy": "non_binding_upstream_metadata_audit_only",
        "endogenous_feature_count": EXPECTED_ENDOGENOUS_FEATURES,
        "engineered_construct_feature_count": EXPECTED_ENGINEERED_FEATURES,
        "guide_barcode_feature_count": EXPECTED_GUIDE_BARCODE_FEATURES,
        "excluded_feature_count": EXPECTED_EXCLUDED_FEATURES,
        "train_conditions": 72,
        "validation_conditions": 24,
        "test_conditions": 48,
        "train_cells": 11_779,
        "validation_cells": 5_102,
        "test_cells_metadata_only": 9_902,
        "development_graph_conditions": 96,
        "development_cell_graphs": 16_881,
        "test_graphs": 0,
        "test_X_rows_indexed": False,
        "test_X_rows_materialized": False,
        "test_X_rows_transformed": False,
        "excluded_X_rows_indexed": False,
        "test_and_excluded_obs_ncounts_redacted": True,
        "engineered_construct_X_columns_indexed": False,
        "guide_barcode_X_columns_indexed": False,
        "excluded_X_columns_indexed": False,
        "model_training_started": False,
        "predictions_generated": False,
        "test_endpoint_computed": False,
        "data_root": str(ASSET_FINAL),
        "data_asset_manifest_sha256": sha256_file(ASSET_STAGING / "ASSET_MANIFEST.csv"),
        "artifact_sha256": artifact_hashes,
    }
    write_atomic_payload(REPO_STAGING / "RUN_STATUS.json", json_bytes(status))


def formal(recover: bool) -> None:
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    frozen = metadata_preflight()
    raw_hash = hash_raw_once(RAW)
    if raw_hash["sha256"] != frozen["e160_status"]["raw_integrity"]["sha256"]:
        raise RuntimeError("Formal raw hash differs from E160")
    if recover:
        recovered = recover_transaction(frozen, raw_hash)
        if recovered == "complete":
            print(
                json.dumps(
                    json.loads((RELEASE / "RUN_STATUS.json").read_text()),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        if recovered != "rolled_back":
            raise RuntimeError(f"Unexpected E161 recovery result: {recovered}")
    elif TRANSACTION_JOURNAL.exists() or TRANSACTION_JOURNAL.is_symlink():
        journal_state = load_transaction_journal().get("phase")
        if journal_state != "rolled_back":
            raise FileExistsError(
                "An E161 transaction already exists "
                f"(phase={journal_state}); use --recover-staging"
            )

    if ASSET_FINAL.exists() or ASSET_FINAL.is_symlink() or RELEASE.exists() or RELEASE.is_symlink():
        raise FileExistsError("E161 final asset/release already exists; refusing overwrite")
    if ASSET_STAGING.exists() or ASSET_STAGING.is_symlink() or REPO_STAGING.exists() or REPO_STAGING.is_symlink():
        raise FileExistsError("E161 staging exists; audit it or use --recover-staging")
    if PRESCRIBE_LINK.exists() or PRESCRIBE_LINK.is_symlink():
        raise FileExistsError(
            f"PRESCRIBE data path is occupied before a new E161 transaction: {PRESCRIBE_LINK}"
        )
    journal = begin_transaction(frozen, raw_hash)

    ledger: list[dict[str, Any]] = [
        {
            "phase": "opaque_raw_file_identity_hash",
            "split": "all_file_bytes_not_semantic_X_access",
            "rows_indexed": 0,
            "columns_indexed": 0,
            "row_index_sha256": "",
            "test_row_intersection": 0,
            "excluded_row_intersection": 0,
            "X_indexed": False,
            "X_materialized": False,
            "X_transformed": False,
            "rows_transformed": 0,
        }
    ]
    raw = ad.read_h5ad(RAW, backed="r")
    try:
        train_full = read_allowed_expression(
            raw,
            "train",
            frozen["split_rows"]["train"],
            frozen,
            ledger,
        )
        axis, selected_indices, selected_genes = fit_gene_axis(
            train_full, frozen["var_names"], frozen["forced_genes"]
        )
        train_norm, train_selected_counts, train_library, train_norm_audit = (
            normalize_with_endogenous_library(
                train_full,
                selected_indices,
                "train",
                pd.to_numeric(
                    frozen["obs"].iloc[frozen["split_rows"]["train"]]["ncounts"],
                    errors="raise",
                ).to_numpy(),
            )
        )
        dense = train_norm.toarray().astype(np.float32, copy=False)
        pca = PCA(
            n_components=N_PCA,
            svd_solver="randomized",
            random_state=SEED,
            whiten=False,
        )
        train_pca = pca.fit_transform(dense).astype(np.float32)
        del dense
        train_conditions = frozen["canonical"].iloc[
            frozen["split_rows"]["train"]
        ].astype(str).to_numpy()
        edistance = fit_train_edistance(
            train_pca, train_conditions, frozen["split_dict"]
        )

        # Validation expression is not opened until every train-fitted object above
        # has already been fixed in memory.
        val_full = read_allowed_expression(
            raw,
            "val",
            frozen["split_rows"]["val"],
            frozen,
            ledger,
        )
        val_norm, val_selected_counts, val_library, val_norm_audit = (
            normalize_with_endogenous_library(
                val_full,
                selected_indices,
                "val",
                pd.to_numeric(
                    frozen["obs"].iloc[frozen["split_rows"]["val"]]["ncounts"],
                    errors="raise",
                ).to_numpy(),
            )
        )
        val_pca = fixed_pca_transform(
            val_norm,
            np.asarray(pca.mean_, dtype=np.float32),
            np.asarray(pca.components_, dtype=np.float32),
        )
    finally:
        raw.file.close()
    if hash_raw_once(RAW) != raw_hash:
        raise RuntimeError("Raw identity/hash changed after train/validation reads")
    ledger.extend(
        [
            {
                "phase": "semantic_train_fixed_transform",
                "split": "train",
                "rows_indexed": 0,
                "columns_indexed": len(selected_genes),
                "row_index_sha256": "",
                "test_row_intersection": 0,
                "excluded_row_intersection": 0,
                "X_indexed": False,
                "X_materialized": False,
                "X_transformed": True,
                "rows_transformed": 11_779,
            },
            {
                "phase": "semantic_val_fixed_transform",
                "split": "val",
                "rows_indexed": 0,
                "columns_indexed": len(selected_genes),
                "row_index_sha256": "",
                "test_row_intersection": 0,
                "excluded_row_intersection": 0,
                "X_indexed": False,
                "X_materialized": False,
                "X_transformed": True,
                "rows_transformed": 5_102,
            },
            {
                "phase": "sealed_test_zero_access",
                "split": "test",
                "rows_indexed": 0,
                "columns_indexed": 0,
                "row_index_sha256": "",
                "test_row_intersection": 0,
                "excluded_row_intersection": 0,
                "X_indexed": False,
                "X_materialized": False,
                "X_transformed": False,
                "rows_transformed": 0,
            },
            {
                "phase": "excluded_zero_access",
                "split": "excluded",
                "rows_indexed": 0,
                "columns_indexed": 0,
                "row_index_sha256": "",
                "test_row_intersection": 0,
                "excluded_row_intersection": 0,
                "X_indexed": False,
                "X_materialized": False,
                "X_transformed": False,
                "rows_transformed": 0,
            },
        ]
    )
    development, control_gene_mean, control_pca_mean, control_pca_cov = build_development(
        frozen,
        axis,
        selected_indices,
        selected_genes,
        train_norm,
        val_norm,
        train_selected_counts,
        val_selected_counts,
        train_library,
        val_library,
        train_pca,
        val_pca,
        pca,
        edistance,
    )
    write_data_assets(
        development,
        frozen,
        journal,
        axis,
        selected_indices,
        selected_genes,
        pca,
        control_gene_mean,
        control_pca_mean,
        control_pca_cov,
        edistance,
    )
    graph_audit = build_and_audit_graphs(
        development, frozen, control_gene_mean
    )
    asset_manifest = build_asset_manifest(selected_genes, frozen, graph_audit)
    ledger_frame = pd.DataFrame(ledger)
    for column in (
        "engineered_construct_columns_indexed",
        "guide_barcode_columns_indexed",
        "total_excluded_columns_indexed",
    ):
        ledger_frame[column] = ledger_frame.get(column, 0).fillna(0).astype(int)
        if int(ledger_frame[column].sum()) != 0:
            raise RuntimeError(f"An excluded Wessels feature was indexed: {column}")
    publish_repo_release(
        frozen,
        journal,
        raw_hash,
        ledger_frame,
        pd.DataFrame([train_norm_audit, val_norm_audit]),
        axis,
        edistance,
        graph_audit,
        asset_manifest,
        started_at,
    )

    publication_gate(frozen, raw_hash)
    journal = update_transaction_phase(
        journal,
        "ready_to_publish",
        asset_manifest_sha256=sha256_file(ASSET_STAGING / "ASSET_MANIFEST.csv"),
        repo_status_sha256=sha256_file(REPO_STAGING / "RUN_STATUS.json"),
    )
    validate_data_tree(ASSET_STAGING, journal)
    validate_repo_tree(REPO_STAGING, journal)
    fsync_tree(ASSET_STAGING)
    fsync_tree(REPO_STAGING)

    # Sentinels deliberately move with each directory.  They remain in the
    # published trees as transaction identity/provenance, eliminating the
    # unsafe unlink-before-rename crash window.
    ASSET_STAGING.rename(ASSET_FINAL)
    fsync_directory(ASSET_PARENT)
    journal = update_transaction_phase(journal, "asset_published")
    ensure_prescribe_link()
    journal = update_transaction_phase(journal, "link_published")
    REPO_STAGING.rename(RELEASE)
    fsync_directory(OUT)
    validate_data_tree(ASSET_FINAL, journal)
    validate_repo_tree(RELEASE, journal)
    journal = update_transaction_phase(
        journal,
        "complete",
        asset_final_sha256_manifest=sha256_file(ASSET_FINAL / "ASSET_MANIFEST.csv"),
        repo_final_status_sha256=sha256_file(RELEASE / "RUN_STATUS.json"),
    )
    print(json.dumps(json.loads((RELEASE / "RUN_STATUS.json").read_text()), ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "preflight":
        if args.recover_staging:
            raise RuntimeError("--recover-staging is only valid with --mode formal")
        frozen = metadata_preflight()
        print(
            json.dumps(
                {
                    "mode": "preflight",
                    "phase": "metadata_and_contracts_passed_no_X_access",
                    "git_head": frozen["git_head"],
                    "shape": list(EXPECTED_RAW_SHAPE),
                    "endogenous_features": EXPECTED_ENDOGENOUS_FEATURES,
                    "excluded_engineered_construct_features": EXPECTED_ENGINEERED_FEATURES,
                    "excluded_guide_or_barcode_features": EXPECTED_GUIDE_BARCODE_FEATURES,
                    "excluded_total_features": EXPECTED_EXCLUDED_FEATURES,
                    "split_cells": {
                        role: len(frozen["split_rows"][role])
                        for role in ("train", "val", "test", "excluded")
                    },
                    "raw_X_rows_indexed_or_materialized": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    try:
        formal(args.recover_staging)
    except Exception as exc:
        failures = OUT / "failures"
        failures.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        transaction = None
        if TRANSACTION_JOURNAL.is_file() and not TRANSACTION_JOURNAL.is_symlink():
            try:
                transaction = load_transaction_journal()
            except Exception:
                transaction = None
        write_atomic_payload(
            failures / f"E161_FAILURE_{stamp}.json",
            json_bytes(
                {
                    "phase": "failed_preserve_staging_no_overwrite",
                    "failed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "git_head": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                    ).strip(),
                    "transaction_id": (
                        transaction.get("transaction_id") if transaction else None
                    ),
                    "transaction_phase": (
                        transaction.get("phase") if transaction else None
                    ),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "asset_staging": str(ASSET_STAGING),
                    "repo_staging": str(REPO_STAGING),
                }
            ),
        )
        raise


if __name__ == "__main__":
    main()
