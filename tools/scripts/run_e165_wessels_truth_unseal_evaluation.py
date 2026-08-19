#!/usr/bin/env python3
"""E165: one-time Wessels test-truth unseal and frozen evaluation.

Preflight never opens the raw H5AD.  Formal execution writes the irreversible
unseal event before hashing or semantically opening that file.  The only raw-X
slice is the 9,902 frozen test rows by the first 20,631 endogenous columns.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import importlib.metadata
import json
import math
import os
import stat
import subprocess
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
RAW = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "WesselsSatija2023.h5ad"
)
E160 = ROOT / "docs/实验结果/E160_wessels_combination_contract_20260714"
E161 = ROOT / "docs/实验结果/E161_wessels_trainval_preprocess_20260714"
E161_DATA = Path("/home/yyf/data/safeconf_e161_prescribe/wessels_e160")
E162 = ROOT / "docs/实验结果/E162_wessels_prescribe_native_20260714"
E162_ATTEMPT = E162 / "attempt_002"
E162B = ROOT / "docs/实验结果/E162b_wessels_label_only_baselines_20260715"
E163 = ROOT / "docs/实验结果/E163_wessels_validation_raw_futility_20260715"
E164 = ROOT / "docs/实验结果/E164_wessels_pretruth_lock_20260715"
OUT = ROOT / "docs/实验结果/E165_wessels_truth_unseal_evaluation_20260715"
CONTRACT = OUT / "ANALYSIS_CONTRACT.md"
EVENT = OUT / "TEST_TRUTH_UNSEAL_EVENT.json"
STAGING = OUT / ".release.staging"
RELEASE = OUT / "release"
FAILURES = OUT / "failures"

E160_STATUS = E160 / "freeze/RUN_STATUS.json"
E160_SPLIT = E160 / "freeze/manifests/E160_set2conditions.json"
E160_CONDITION_AUDIT = E160 / "freeze/manifests/E160_CONDITION_AUDIT.csv"
E161_STATUS = E161 / "release/RUN_STATUS.json"
E161_MANIFEST = E161 / "release/tables/E161_ASSET_MANIFEST.csv"
E161_INTERFACE = E161_DATA / "E161_E162_INTERFACE.json"
E162_STATUS = E162_ATTEMPT / "RUN_STATUS.json"
E162B_STATUS = E162B / "release/RUN_STATUS.json"
E162B_INTERFACE = E162B / "release/E162b_E163_INTERFACE.json"
E162B_MANIFEST = E162B / "release/RESULTS_SHA256.csv"
E163_STATUS = E163 / "release/RUN_STATUS.json"
E163_GATE = E163 / "release/E163_AUTHORIZATION_GATE.json"
E163_INTERFACE = E163 / "release/E163_E164_INTERFACE.json"
E164_STATUS = E164 / "release/RUN_STATUS.json"
E164_INTERFACE = E164 / "release/E164_E165_INTERFACE.json"
E164_MANIFEST = E164 / "release/RESULTS_SHA256.csv"
E164_QUERY_EVENT = E164 / "TEST_LABEL_QUERY_EVENT.json"

EXPECTED_PYTHON = Path("/home/yyf/.conda/envs/prescribe_env/bin/python")
EXPECTED_RAW_SHA256 = "5da0485aed81b23bda57b4a7b4510a394682d54911416db89b4846ff6dd34732"
EXPECTED_RAW_MD5 = "6897bfdcda928a678208fecf4eeb282e"
EXPECTED_RAW_BYTES = 219_393_529
EXPECTED_RAW_SHAPE = (30_707, 21_052)
N_ENDOGENOUS = 20_631
N_EXCLUDED = 421
N_SELECTED = 2_023
N_PCA = 10
N_TEST_ROWS = 9_902
N_TEST_TASKS = 48
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 3407
MIN_VALID_BOOTSTRAPS = 9_500
MAIN_SEED = 3407
SEEDS = (3407, 3408, 3409)
BASELINE_ORDER = (
    "control_no_change",
    "cell_weighted_perturbed_mean",
    "condition_balanced_perturbed_mean",
    "matching_single_mean",
    "single_additive",
)
RISK_SCORE_COLUMNS = (
    "min_single_cell_count_confidence",
    "min_train_pair_degree_confidence",
    "matching_se_pca10_confidence",
    "matching_se_gene_confidence",
    "matching_magnitude_confidence",
    "hash_random_confidence",
    "constant_confidence",
    "exact_pair_support_confidence",
)
COVERAGES = tuple(np.round(np.arange(0.50, 1.001, 0.05), 2).tolist())
EXPECTED_VERSIONS = {
    "anndata": "0.10.8",
    "h5py": "3.14.0",
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "scipy": "1.13.1",
}
UPSTREAM_DOIS = (
    "10.1038/s41587-025-02777-8",
    "10.64898/2026.04.20.719650",
    "10.1038/s41587-026-03113-4",
)
SYSTEMA_CODE_COMMIT = "aaf5b5353993b48b78543f2f93b3e18ca65df515"

ALLOWLIST = {
    ".E165_TRANSACTION.json",
    "RUN_STATUS.json",
    "README_先看这个.md",
    "RESULTS_SHA256.csv",
    "E165_E166_INTERFACE.json",
    "reports/E165_REPORT.md",
    "figures/E165_SUMMARY_WHITE.svg",
    "profiles/E165_TEST_TRUTH_PROFILES.csv.gz",
    "tables/E165_TEST_TRUTH_TASKS.csv",
    "tables/E165_TEST_TRUTH_PCA10.csv",
    "tables/E165_PREDICTOR_TASK_METRICS.csv.gz",
    "tables/E165_PREDICTOR_SUMMARY.csv",
    "tables/E165_BASELINE_HIERARCHY.csv",
    "tables/E165_HYPOTHESIS_TESTS.csv",
    "tables/E165_NATIVE_SCORE_ASSOCIATIONS.csv",
    "tables/E165_BOOTSTRAP_REPLICATES.csv.gz",
    "tables/E165_BOOTSTRAP_SUMMARY.csv",
    "tables/E165_LOGO.csv",
    "tables/E165_SECONDARY_PREDICTION_CONTRASTS.csv",
    "tables/E165_COVERAGE_CURVES.csv",
    "tables/E165_AURC_ERROR_CAPTURE.csv",
    "tables/E165_SPLIT_HALF_REFERENCE.csv",
    "tables/E165_CENTROID_ACCURACY.csv",
    "tables/E165_X_ACCESS_LEDGER.csv",
    "tables/E165_INPUT_HASHES.csv",
    "tables/E165_RUNTIME_ENVIRONMENT.csv",
}


class IntegrityFailure(RuntimeError):
    """Input, access-boundary, or atomic-publication failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - locked provenance, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    def json_safe(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): json_safe(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_safe(val) for val in item]
        if isinstance(item, np.ndarray):
            return [json_safe(val) for val in item.tolist()]
        if isinstance(item, (np.bool_, bool)):
            return bool(item)
        if isinstance(item, (np.integer, int)) and not isinstance(item, bool):
            return int(item)
        if isinstance(item, (np.floating, float)):
            number = float(item)
            return number if math.isfinite(number) else None
        return item

    return (
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegrityFailure(f"Expected JSON object: {path}")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes, *, overwrite: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(payload)
    if path.exists() and not overwrite:
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise IntegrityFailure(f"Immutable artifact differs: {path}")
        return digest
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if overwrite:
        temporary.replace(path)
    else:
        # link(2) is an atomic no-replace publication primitive: unlike
        # exists()+rename it cannot overwrite a winner in a concurrent run.
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            temporary.unlink(missing_ok=True)
            raise
        temporary.unlink()
    fsync_directory(path.parent)
    return digest


def atomic_json(path: Path, value: dict[str, Any], *, overwrite: bool = False) -> str:
    return atomic_bytes(path, canonical_json_bytes(value), overwrite=overwrite)


def atomic_csv(path: Path, frame: pd.DataFrame) -> str:
    return atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def atomic_gzip_csv(path: Path, frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False).encode("utf-8")
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    return atomic_bytes(path, compressed)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_blob_gate(path: Path, head: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    relative = path.relative_to(ROOT).as_posix()
    try:
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"Required input is not committed at HEAD: {relative}") from exc
    observed = sha256_file(path)
    expected = sha256_bytes(committed)
    if observed != expected:
        raise IntegrityFailure(f"Working file differs from HEAD blob: {relative}")
    return {"path": relative, "sha256": observed, "bytes": path.stat().st_size}


def runtime_gate() -> list[dict[str, Any]]:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise IntegrityFailure(f"Expected {EXPECTED_PYTHON}, found {sys.executable}")
    if sys.version_info[:3] != (3, 9, 25):
        raise IntegrityFailure(f"Expected Python 3.9.25, found {sys.version.split()[0]}")
    rows = [{"component": "python", "expected": "3.9.25", "observed": sys.version.split()[0]}]
    for package, expected in EXPECTED_VERSIONS.items():
        observed = importlib.metadata.version(package)
        if observed != expected:
            raise IntegrityFailure(f"{package} version changed: {observed} != {expected}")
        rows.append({"component": package, "expected": expected, "observed": observed})
    return rows


def verify_status_artifacts(base: Path, status: dict[str, Any], head: str) -> list[dict[str, Any]]:
    mapping = status.get("artifact_sha256", {})
    if not isinstance(mapping, dict) or not mapping:
        raise IntegrityFailure(f"Status lacks artifact_sha256: {base}")
    rows = []
    for relative, expected in sorted(mapping.items()):
        path = base / str(relative)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != str(expected):
            raise IntegrityFailure(f"Upstream artifact mismatch: {path}")
        if path.is_relative_to(ROOT):
            git_blob_gate(path, head)
        rows.append({"path": str(path), "sha256": str(expected), "bytes": path.stat().st_size})
    return rows


def verify_interface_artifacts(
    release: Path,
    interface: dict[str, Any],
    *,
    hash_field: str,
) -> list[dict[str, Any]]:
    mapping = interface.get(hash_field)
    if not isinstance(mapping, dict) or not mapping:
        raise IntegrityFailure(f"Interface lacks exact {hash_field} mapping: {release}")
    rows = []
    for relative, expected in sorted(mapping.items()):
        path = release / str(relative)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != str(expected):
            raise IntegrityFailure(f"Interface artifact mismatch: {path}")
        rows.append({"path": str(path), "sha256": str(expected), "bytes": path.stat().st_size})
    return rows


def raw_stat_gate(e160_status: dict[str, Any]) -> dict[str, int]:
    observed = RAW.lstat()
    if RAW.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise IntegrityFailure("Wessels raw must be a regular non-symlink file")
    identity = {
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "bytes": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
    }
    expected = e160_status.get("raw_integrity", {})
    for key, value in identity.items():
        if int(expected.get(key, -1)) != value:
            raise IntegrityFailure(f"Raw stat identity changed at {key}")
    if identity["bytes"] != EXPECTED_RAW_BYTES:
        raise IntegrityFailure("Raw byte size changed")
    return identity


def strict_bool_scalar(value: Any, *, context: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise IntegrityFailure(f"Non-boolean value at {context}: {value!r}")


def strict_bool_series(series: pd.Series, *, context: str) -> np.ndarray:
    return np.asarray(
        [strict_bool_scalar(value, context=f"{context}[{index}]") for index, value in enumerate(series)],
        dtype=bool,
    )


def audit_e160_test_design(
    e160_status: dict[str, Any], split: dict[str, Any]
) -> dict[str, Any]:
    """Prove the frozen 48-task/9,902-cell design before any raw open."""
    if tuple(e160_status.get("dataset_shape", ())) != EXPECTED_RAW_SHAPE:
        raise IntegrityFailure("E160 frozen raw shape changed")
    audit = pd.read_csv(E160_CONDITION_AUDIT)
    required = {
        "raw_condition", "canonical_condition", "perturbation_genes",
        "n_perturbation_genes", "n_cells_obs_metadata", "model_compatible",
        "eligible_pair_min75", "split",
    }
    if not required.issubset(audit.columns):
        raise IntegrityFailure(f"E160 condition audit lacks {sorted(required-set(audit.columns))}")
    if (
        audit["canonical_condition"].astype(str).duplicated().any()
        or audit["raw_condition"].astype(str).duplicated().any()
    ):
        raise IntegrityFailure("E160 raw/canonical condition audit is not one-to-one")
    split_values = audit["split"].astype(str)
    for role, expected_n in (("train", 72), ("val", 24), ("test", 48)):
        locked = list(map(str, split[role]))
        audited = audit.loc[split_values.eq(role), "canonical_condition"].astype(str).tolist()
        if len(audited) != expected_n or set(audited) != set(locked):
            raise IntegrityFailure(f"E160 {role} audit disagrees with split lock")
    test = audit.loc[split_values.eq("test")].copy()
    cells = pd.to_numeric(test["n_cells_obs_metadata"], errors="raise").to_numpy(np.int64)
    if len(test) != N_TEST_TASKS or int(cells.sum()) != N_TEST_ROWS or np.any(cells <= 0):
        raise IntegrityFailure("E160 test metadata is not exactly 48 tasks / 9,902 cells")
    n_components = pd.to_numeric(test["n_perturbation_genes"], errors="raise").to_numpy(np.int64)
    if not np.all(n_components == 2):
        raise IntegrityFailure("E160 test contains a non-double perturbation")
    if not strict_bool_series(test["model_compatible"], context="E160.model_compatible").all():
        raise IntegrityFailure("E160 test contains a model-incompatible pair")
    if not strict_bool_series(test["eligible_pair_min75"], context="E160.eligible_pair_min75").all():
        raise IntegrityFailure("E160 test contains a pair below its frozen eligibility gate")
    train_set = set(map(str, split["train"]))
    test_set = set(map(str, split["test"]))
    if train_set & test_set:
        raise IntegrityFailure("E160 exact test combination appears in train")
    component_support: dict[str, str] = {}
    for row in test.itertuples(index=False):
        condition = str(row.canonical_condition)
        components = condition.split("+")
        audit_components = str(row.perturbation_genes).split(";")
        if (
            len(components) != 2 or "ctrl" in components or components[0] == components[1]
            or set(components) != set(audit_components)
        ):
            raise IntegrityFailure(f"Invalid E160 two-component encoding: {condition}")
        for component in components:
            singleton = f"{component}+ctrl"
            if singleton not in train_set:
                raise IntegrityFailure(f"E160 test component lacks train singleton: {singleton}")
            component_support[component] = singleton
    cell_count_map = dict(sorted(zip(test["canonical_condition"].astype(str), cells.tolist())))
    return {
        "audit_sha256": sha256_file(E160_CONDITION_AUDIT),
        "dataset_shape": list(EXPECTED_RAW_SHAPE),
        "test_tasks": N_TEST_TASKS,
        "test_cells_metadata": N_TEST_ROWS,
        "all_test_tasks_are_two_component": True,
        "exact_test_combinations_absent_train": True,
        "all_component_singletons_present_train": True,
        "n_unique_test_components": len(component_support),
        "test_cell_counts_by_condition": cell_count_map,
        "test_condition_set_sha256": axis_hash(sorted(test_set)),
        "test_cell_count_by_condition_sha256": sha256_bytes(
            canonical_json_bytes(cell_count_map)
        ),
    }


def metadata_preflight() -> dict[str, Any]:
    """Verify every upstream lock without opening or hashing RAW."""
    runtime = runtime_gate()
    head = git_head()
    committed_paths = [
        RUNNER, CONTRACT,
        E160 / "ANALYSIS_CONTRACT.md", E160_STATUS, E160_SPLIT, E160_CONDITION_AUDIT,
        E161 / "ANALYSIS_CONTRACT.md", E161_STATUS, E161_MANIFEST,
        E162 / "ANALYSIS_CONTRACT.md", E162_STATUS,
        E162B / "ANALYSIS_CONTRACT.md", E162B_STATUS, E162B_INTERFACE,
        E162B_MANIFEST,
        E163 / "ANALYSIS_CONTRACT.md", E163_STATUS, E163_GATE, E163_INTERFACE,
        E164 / "ANALYSIS_CONTRACT.md", E164_STATUS, E164_INTERFACE,
        E164_MANIFEST, E164_QUERY_EVENT,
    ]
    committed = [git_blob_gate(path, head) for path in committed_paths]

    e160 = load_json(E160_STATUS)
    if e160.get("phase") != "requirements_frozen_test_expression_unopened":
        raise IntegrityFailure("E160 freeze phase changed")
    if e160.get("raw_X_values_indexed_or_materialized") is not False:
        raise IntegrityFailure("E160 does not certify sealed raw X")
    raw_identity = e160.get("raw_integrity", {})
    if (
        raw_identity.get("sha256") != EXPECTED_RAW_SHA256
        or raw_identity.get("md5") != EXPECTED_RAW_MD5
        or int(raw_identity.get("bytes", -1)) != EXPECTED_RAW_BYTES
    ):
        raise IntegrityFailure("E160 raw cryptographic identity changed")
    split = load_json(E160_SPLIT)
    if set(split) != {"train", "val", "test"} or {k: len(split[k]) for k in split} != {
        "train": 72, "val": 24, "test": 48
    }:
        raise IntegrityFailure("E160 split changed")
    test_order = list(map(str, split["test"]))
    if len(set(test_order)) != N_TEST_TASKS:
        raise IntegrityFailure("E160 test tasks are duplicated")
    e160_design = audit_e160_test_design(e160, split)

    e161_status = load_json(E161_STATUS)
    if e161_status.get("phase") != "complete_preprocessing_and_dev_graphs_no_training_no_test_X_access":
        raise IntegrityFailure("E161 release phase changed")
    if any(e161_status.get(key) is not False for key in (
        "test_X_rows_indexed", "test_X_rows_materialized", "test_X_rows_transformed",
        "test_endpoint_computed",
    )):
        raise IntegrityFailure("E161 no-test boundary changed")
    e161_interface = load_json(E161_INTERFACE)
    if e161_interface.get("schema") != "safeconf_e161_to_e162_v2":
        raise IntegrityFailure("E161 interface schema changed")
    if (
        e161_interface.get("n_endogenous_features") != N_ENDOGENOUS
        or e161_interface.get("n_excluded_features") != N_EXCLUDED
        or e161_interface.get("n_selected_genes") != N_SELECTED
        or e161_interface.get("development_graphs", {}).get("test") != 0
    ):
        raise IntegrityFailure("E161 feature/access boundary changed")
    e161_asset_rows = verify_interface_artifacts(
        E161_DATA, e161_interface, hash_field="asset_sha256"
    )

    e162 = load_json(E162_STATUS)
    if e162.get("phase") != "failed_main_validation_nondegeneracy_gate_no_test_label_query":
        raise IntegrityFailure("E162 failure phase changed")
    if any(e162.get(key) is not False for key in (
        "raw_h5ad_opened", "test_X_accessed", "test_truth_accessed",
        "test_endpoint_computed", "test_label_queries_started",
    )):
        raise IntegrityFailure("E162 no-test failure boundary changed")

    e162b_status = load_json(E162B_STATUS)
    e162b_interface = load_json(E162B_INTERFACE)
    if e162b_status.get("phase") != "complete_pretest_label_only_baselines_no_val_or_test_X":
        raise IntegrityFailure("E162b release phase changed")
    if e162b_interface.get("schema") != "safeconf_e162b_to_e163_v1":
        raise IntegrityFailure("E162b interface schema changed")
    if e162b_interface.get("test_label_order") != test_order:
        raise IntegrityFailure("E162b test order changed")
    e162b_rows = verify_status_artifacts(E162B / "release", e162b_status, head)

    e163_status = load_json(E163_STATUS)
    e163_gate = load_json(E163_GATE)
    e163_interface = load_json(E163_INTERFACE)
    if e163_status.get("phase") != "complete_validation_only_futility_diagnostic_no_test_label_or_X_access":
        raise IntegrityFailure("E163 release phase changed")
    if e163_gate.get("schema") != "safeconf_e163_authorization_gate_v1":
        raise IntegrityFailure("E163 gate schema changed")
    if e163_interface.get("schema") != "safeconf_e163_to_e164_v1":
        raise IntegrityFailure("E163 interface schema changed")
    if e163_interface.get("phase") != "validation_only_futility_diagnostic_complete_test_remains_sealed":
        raise IntegrityFailure("E163 interface phase changed")
    e163_authorized = strict_bool_scalar(
        e163_gate.get("authorize_future_test_label_lock"),
        context="E163 gate authorize_future_test_label_lock",
    )
    if e163_authorized != strict_bool_scalar(
        e163_interface.get("authorize_future_test_label_lock"),
        context="E163 interface authorize_future_test_label_lock",
    ):
        raise IntegrityFailure("E163 authorization fields disagree")
    if any(e163_interface.get(key) is not False for key in (
        "test_label_queried", "test_X_accessed", "test_truth_accessed",
    )):
        raise IntegrityFailure("E163 test boundary changed")
    if any(e163_status.get(key) is not False for key in (
        "raw_Wessels_h5ad_opened", "test_label_queried", "test_X_accessed",
        "test_truth_accessed", "test_endpoint_computed",
    )):
        raise IntegrityFailure("E163 status does not certify the sealed test boundary")
    e163_rows = []
    integrity = e163_interface.get("artifact_integrity", {})
    if not isinstance(integrity, dict) or not integrity:
        raise IntegrityFailure("E163 interface lacks artifact integrity records")
    for role, record in sorted(integrity.items()):
        path = E163 / "release" / str(record.get("release_relative_path", ""))
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != str(record.get("sha256"))
        ):
            raise IntegrityFailure(f"E163 interface artifact mismatch: {role}")
        git_blob_gate(path, head)
        e163_rows.append({"path": str(path), "sha256": str(record["sha256"]), "bytes": path.stat().st_size})
    e163_manifest = E163 / "release" / str(e163_interface.get("release_manifest_relative_path", ""))
    git_blob_gate(e163_manifest, head)

    e164_status = load_json(E164_STATUS)
    e164_interface = load_json(E164_INTERFACE)
    if e164_interface.get("schema") != "safeconf_e164_to_e165_v1":
        raise IntegrityFailure("E164 interface schema changed")
    allowed_e164_phases = {
        "complete_dual_arm_pretruth_lock_no_test_X_or_truth",
        "complete_baseline_arm_pretruth_lock_prescribe_not_authorized",
        "complete_baseline_arm_pretruth_lock_prescribe_raw_gate_failed",
    }
    if e164_status.get("phase") not in allowed_e164_phases:
        raise IntegrityFailure("E164 status is not a frozen terminal pretruth phase")
    if e164_interface.get("phase") != e164_status.get("phase"):
        raise IntegrityFailure("E164 status/interface terminal phases disagree")
    if strict_bool_scalar(
        e164_interface.get("baseline_arm_authorized"),
        context="E164 interface baseline_arm_authorized",
    ) is not True:
        raise IntegrityFailure("E164 baseline arm is not authorized")
    if strict_bool_scalar(
        e164_status.get("baseline_arm_authorized"),
        context="E164 status baseline_arm_authorized",
    ) is not True:
        raise IntegrityFailure("E164 status does not authorize the baseline arm")
    prescribe_authorized = strict_bool_scalar(
        e164_interface.get("prescribe_arm_authorized"),
        context="E164 interface prescribe_arm_authorized",
    )
    if prescribe_authorized != strict_bool_scalar(
        e164_status.get("prescribe_arm_authorized"),
        context="E164 status prescribe_arm_authorized",
    ):
        raise IntegrityFailure("E164 status/interface PRESCRIBE authorization disagree")
    if prescribe_authorized and not e163_authorized:
        raise IntegrityFailure("E164 PRESCRIBE arm contradicts the E163 gate")
    status_main_gate = e164_status.get("main_raw_gate_passed")
    interface_main_gate = e164_interface.get("main_raw_gate_passed")
    if status_main_gate is None or interface_main_gate is None:
        if status_main_gate is not None or interface_main_gate is not None:
            raise IntegrityFailure("E164 status/interface main raw-gate nullability disagrees")
        main_gate_passed: bool | None = None
    else:
        main_gate_passed = strict_bool_scalar(status_main_gate, context="E164 status main_raw_gate_passed")
        if main_gate_passed != strict_bool_scalar(
            interface_main_gate, context="E164 interface main_raw_gate_passed"
        ):
            raise IntegrityFailure("E164 status/interface main raw gate disagree")
    status_raw_gates = e164_status.get("raw_score_gates")
    interface_raw_gates = e164_interface.get("raw_score_gates")
    if status_raw_gates != interface_raw_gates:
        raise IntegrityFailure("E164 status/interface raw-score gate records disagree")
    if e163_authorized and main_gate_passed is None:
        raise IntegrityFailure("E164 omitted the main raw gate despite E163 authorization")
    expected_prescribe_authorization = bool(e163_authorized and main_gate_passed is True)
    if prescribe_authorized != expected_prescribe_authorization:
        raise IntegrityFailure("E164 PRESCRIBE authorization does not equal E163 gate AND seed3407 gate")
    if prescribe_authorized:
        if e164_status.get("phase") != "complete_dual_arm_pretruth_lock_no_test_X_or_truth":
            raise IntegrityFailure("E164 dual-arm authorization has the wrong terminal phase")
        if not isinstance(interface_raw_gates, dict) or set(interface_raw_gates) != {str(x) for x in SEEDS}:
            raise IntegrityFailure("E164 dual-arm release lacks the three recorded raw-score gates")
        main_record = interface_raw_gates[str(MAIN_SEED)]
        if (
            strict_bool_scalar(main_record.get("passed"), context="E164 seed3407 passed") is not True
            or int(main_record.get("n_rows", -1)) != N_TEST_TASKS
            or strict_bool_scalar(
                main_record.get("raw_log_prob_all_finite"), context="E164 seed3407 finite"
            ) is not True
            or int(main_record.get("raw_log_prob_exact_unique", -1)) < 24
            or float(main_record.get("raw_log_prob_sample_std_ddof1", math.nan)) <= 1e-6
        ):
            raise IntegrityFailure("E164 seed3407 blocking raw-score gate is inconsistent")
    elif e163_authorized:
        if e164_status.get("phase") != "complete_baseline_arm_pretruth_lock_prescribe_raw_gate_failed":
            raise IntegrityFailure("E164 failed main raw gate has the wrong terminal phase")
    elif e164_status.get("phase") != "complete_baseline_arm_pretruth_lock_prescribe_not_authorized":
        raise IntegrityFailure("E164 E163-closed arm has the wrong terminal phase")
    if e164_interface.get("test_label_order") != test_order:
        raise IntegrityFailure("E164 test order changed")
    if tuple(e164_interface.get("baseline_order", [])) != BASELINE_ORDER:
        raise IntegrityFailure("E164 baseline order changed")
    paths = e164_interface.get("paths", {})
    if not isinstance(paths, dict) or not {"baseline_post_profiles", "risk_wide"}.issubset(paths):
        raise IntegrityFailure("E164 baseline paths are incomplete")
    prescribe_paths = paths.get("prescribe_scores", {})
    if prescribe_authorized:
        if not isinstance(prescribe_paths, dict) or set(prescribe_paths) != {str(x) for x in SEEDS}:
            raise IntegrityFailure("Authorized E164 PRESCRIBE score paths are incomplete")
    elif prescribe_paths not in ({}, None):
        raise IntegrityFailure("Closed E164 PRESCRIBE arm must not expose score paths")
    e164_rows = verify_interface_artifacts(
        E164 / "release", e164_interface, hash_field="artifact_sha256"
    )
    e164_status_rows = verify_status_artifacts(E164 / "release", e164_status, head)
    if sha256_file(E164_MANIFEST) != str(e164_status.get("results_manifest_sha256")):
        raise IntegrityFailure("E164 results manifest differs from status")
    interface_access = e164_interface.get("access_boundary")
    if not isinstance(interface_access, dict):
        raise IntegrityFailure("E164 interface lacks its access boundary")
    if (
        interface_access.get("raw_Wessels_opened") is not False
        or int(interface_access.get("test_X_rows_indexed_materialized_transformed", -1)) != 0
        or interface_access.get("test_truth_effect_error_DE_or_endpoint_used") is not False
    ):
        raise IntegrityFailure("E164 pretruth boundary changed")
    if any(e164_status.get(key) is not False for key in (
        "raw_Wessels_opened", "test_X_accessed", "test_truth_accessed", "test_endpoint_computed",
    )) or any(int(e164_status.get(key, -1)) != 0 for key in (
        "validation_X_rows_indexed_materialized_transformed",
        "test_X_rows_indexed_materialized_transformed",
        "excluded_X_rows_indexed_materialized_transformed",
    )):
        raise IntegrityFailure("E164 status does not certify the pretruth boundary")
    query_record = e164_interface.get("query_event")
    if not isinstance(query_record, dict):
        raise IntegrityFailure("E164 interface lacks its query-event record")
    if e163_authorized:
        if (
            query_record.get("created") is not True
            or query_record.get("path") != "../TEST_LABEL_QUERY_EVENT.json"
            or sha256_file(E164_QUERY_EVENT) != str(query_record.get("sha256"))
            or sha256_file(E164_QUERY_EVENT)
            != str(e164_status.get("test_label_query_event_sha256"))
            or e164_status.get("test_label_queries_started") is not True
            or int(e164_status.get("test_label_strings_forwarded", -1)) != N_TEST_TASKS
        ):
            raise IntegrityFailure("E164 irreversible label-query record is inconsistent")
    elif query_record.get("created") is not False:
        raise IntegrityFailure("E164 created a query event despite a closed E163 arm")

    raw_stat = raw_stat_gate(e160)
    input_rows = (
        committed + e161_asset_rows + e162b_rows + e163_rows
        + e164_rows + e164_status_rows
    )
    fingerprint_view = {
        "git_head": head,
        "committed": committed,
        "e160_status_sha256": sha256_file(E160_STATUS),
        "e160_split_sha256": sha256_file(E160_SPLIT),
        "e160_condition_audit_sha256": sha256_file(E160_CONDITION_AUDIT),
        "e160_test_design": e160_design,
        "e161_status_sha256": sha256_file(E161_STATUS),
        "e161_interface_sha256": sha256_file(E161_INTERFACE),
        "e162_failure_status_sha256": sha256_file(E162_STATUS),
        "e162b_interface_sha256": sha256_file(E162B_INTERFACE),
        "e163_gate_sha256": sha256_file(E163_GATE),
        "e163_interface_sha256": sha256_file(E163_INTERFACE),
        "e164_interface_sha256": sha256_file(E164_INTERFACE),
        "raw_stat": raw_stat,
        "test_order": test_order,
        "baseline_arm_authorized": True,
        "prescribe_arm_authorized": prescribe_authorized,
        "e164_main_raw_gate_passed": main_gate_passed,
    }
    fingerprint = sha256_bytes(canonical_json_bytes(fingerprint_view))
    return {
        **fingerprint_view,
        "gate_fingerprint_sha256": fingerprint,
        "runtime": runtime,
        "input_rows": input_rows,
        "e160_status": e160,
        "e161_interface": e161_interface,
        "e163_gate": e163_gate,
        "e164_interface": e164_interface,
        "e164_paths": paths,
        "raw_h5ad_opened": False,
        "raw_file_hashed": False,
        "test_truth_accessed": False,
    }


def event_lock(preflight: dict[str, Any]) -> dict[str, Any]:
    e161_assets = preflight["e161_interface"]["asset_sha256"]
    return {
        "git_head": preflight["git_head"],
        "gate_fingerprint_sha256": preflight["gate_fingerprint_sha256"],
        "runner_sha256": sha256_file(RUNNER),
        "contract_sha256": sha256_file(CONTRACT),
        "e160_split_sha256": preflight["e160_split_sha256"],
        "e160_condition_audit_sha256": preflight["e160_condition_audit_sha256"],
        "e160_test_design": preflight["e160_test_design"],
        "e161_interface_sha256": preflight["e161_interface_sha256"],
        "e161_selected_axis_sha256": e161_assets["SELECTED_GENE_AXIS.txt"],
        "e161_endogenous_axis_sha256": e161_assets["ENDOGENOUS_GENE_AXIS.txt"],
        "e161_pca_model_sha256": e161_assets["TRAIN_ONLY_PCA_MODEL.npz"],
        "e161_control_prior_sha256": e161_assets["TRAIN_ONLY_CONTROL_PRIOR.npz"],
        "e162_failure_status_sha256": preflight["e162_failure_status_sha256"],
        "e162b_interface_sha256": preflight["e162b_interface_sha256"],
        "e163_gate_sha256": preflight["e163_gate_sha256"],
        "e164_interface_sha256": preflight["e164_interface_sha256"],
        "raw_stat": preflight["raw_stat"],
        "raw_expected_sha256": EXPECTED_RAW_SHA256,
        "test_condition_order": preflight["test_order"],
        "expected_test_rows": N_TEST_ROWS,
        "endogenous_columns": N_ENDOGENOUS,
        "endogenous_column_indices_sha256": sha256_bytes(
            np.arange(N_ENDOGENOUS, dtype=np.int64).tobytes()
        ),
        "excluded_columns": N_EXCLUDED,
        "selected_genes": N_SELECTED,
        "baseline_arm_authorized": True,
        "prescribe_arm_authorized": preflight["prescribe_arm_authorized"],
    }


def verify_unseal_event_lock(
    preflight: dict[str, Any], *, transaction_id: str | None = None
) -> dict[str, Any]:
    if EVENT.is_symlink() or not EVENT.is_file():
        raise IntegrityFailure("Irreversible unseal event is absent or not a regular file")
    observed = load_json(EVENT)
    if observed.get("schema") != "safeconf_e165_irreversible_truth_unseal_v1":
        raise IntegrityFailure("Existing unseal event schema changed")
    if observed.get("lock") != event_lock(preflight) or observed.get("irreversible") is not True:
        raise IntegrityFailure("Existing unseal event belongs to a different frozen run")
    if transaction_id is not None and observed.get("transaction_id") != transaction_id:
        raise IntegrityFailure("Unseal event transaction changed")
    return observed


def write_or_verify_unseal_event(preflight: dict[str, Any]) -> dict[str, Any]:
    """The first persistent formal action; it precedes every RAW file open."""
    locked = event_lock(preflight)
    if EVENT.exists():
        return verify_unseal_event_lock(preflight)
    if STAGING.exists() or RELEASE.exists():
        raise IntegrityFailure("E165 output exists without the irreversible unseal event")
    event = {
        "schema": "safeconf_e165_irreversible_truth_unseal_v1",
        "event": "Wessels test truth unsealed",
        "irreversible": True,
        "created_before_any_raw_file_open": True,
        "unsealed_at": now(),
        "transaction_id": uuid.uuid4().hex,
        "lock": locked,
    }
    try:
        atomic_json(EVENT, event)
    except FileExistsError as exc:
        # A concurrent writer won the no-replace link.  Never silently join a
        # concurrently running truth-unseal transaction.
        verify_unseal_event_lock(preflight)
        raise IntegrityFailure("Concurrent E165 unseal event won; refusing a second formal run") from exc
    return event


def axis_hash(values: list[str]) -> str:
    return sha256_bytes(("\n".join(values) + "\n").encode("utf-8"))


def load_frozen_pca_control(preflight: dict[str, Any]) -> dict[str, Any]:
    interface = preflight["e161_interface"]
    paths = interface["paths"]
    selected_path = E161_DATA / paths["selected_gene_axis"]
    endogenous_path = E161_DATA / paths["endogenous_gene_axis"]
    excluded_path = E161_DATA / paths["excluded_feature_axis"]
    selected = selected_path.read_text(encoding="utf-8").splitlines()
    endogenous = endogenous_path.read_text(encoding="utf-8").splitlines()
    excluded = excluded_path.read_text(encoding="utf-8").splitlines()
    if len(selected) != N_SELECTED or axis_hash(selected) != interface["selected_gene_order_sha256"]:
        raise IntegrityFailure("E161 selected axis changed")
    if len(endogenous) != N_ENDOGENOUS or axis_hash(endogenous) != interface["endogenous_feature_order_sha256"]:
        raise IntegrityFailure("E161 endogenous axis changed")
    if len(excluded) != N_EXCLUDED or axis_hash(excluded) != interface["excluded_feature_order_sha256"]:
        raise IntegrityFailure("E161 excluded axis changed")
    with np.load(E161_DATA / paths["pca_model"], allow_pickle=False) as archive:
        genes = np.asarray(archive["model_genes"]).astype(str).tolist()
        indices = np.asarray(archive["raw_gene_indices"], dtype=np.int64)
        mean = np.asarray(archive["mean"], dtype=np.float64)
        components = np.asarray(archive["components"], dtype=np.float64)
    with np.load(E161_DATA / paths["control_prior"], allow_pickle=False) as archive:
        control = np.asarray(archive["control_gene_mean"], dtype=np.float64)
        control_pca = np.asarray(archive["control_pca_mean"], dtype=np.float64)
    if genes != selected or indices.shape != (N_SELECTED,) or not np.all(indices[:-1] < indices[1:]):
        raise IntegrityFailure("E161 PCA selected axis/index order changed")
    if indices.min() < 0 or indices.max() >= N_ENDOGENOUS:
        raise IntegrityFailure("A selected index enters an excluded feature column")
    if mean.shape != (N_SELECTED,) or components.shape != (N_PCA, N_SELECTED):
        raise IntegrityFailure("E161 PCA shapes changed")
    if control.shape != (N_SELECTED,) or control_pca.shape != (N_PCA,):
        raise IntegrityFailure("E161 control shapes changed")
    arrays = (indices, mean, components, control, control_pca)
    if not all(np.isfinite(value).all() for value in arrays):
        raise IntegrityFailure("Non-finite frozen PCA/control asset")
    transformed_control = (control - mean) @ components.T
    control_pca_recompute_max_abs_delta = float(
        np.max(np.abs(transformed_control - control_pca))
    )
    # E161 stored per-cell PCA coordinates as float32 before averaging the
    # controls; recomputing from the float64 gene mean is therefore close but
    # not bit-identical (frozen observed maximum is about 2.15e-4).
    if control_pca_recompute_max_abs_delta > 5e-4:
        raise IntegrityFailure("Frozen control PCA and gene mean disagree")
    return {
        "selected": selected,
        "endogenous": endogenous,
        "excluded": excluded,
        "selected_indices": indices,
        "pca_mean": mean,
        "components": components,
        "control": control,
        "control_pca": control_pca,
        "control_pca_recompute_max_abs_delta": control_pca_recompute_max_abs_delta,
    }


def read_test_truth_once(
    preflight: dict[str, Any], frozen: dict[str, Any], event: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Open RAW after the event and perform the sole semantic X slice."""
    import anndata as ad

    # This second comparison is immediately adjacent to the raw hash/open.
    # Existence alone is insufficient after recovery or concurrent activity.
    verify_unseal_event_lock(preflight, transaction_id=str(event["transaction_id"]))
    before_stat = raw_stat_gate(preflight["e160_status"])
    if before_stat != event["lock"]["raw_stat"]:
        raise IntegrityFailure("Raw stat identity differs from the unseal event")
    observed_sha = sha256_file(RAW)
    observed_md5 = md5_file(RAW)
    if observed_sha != EXPECTED_RAW_SHA256 or observed_md5 != EXPECTED_RAW_MD5:
        raise IntegrityFailure("Raw bytes changed after the unseal event")
    ledger: list[dict[str, Any]] = [{
        "phase": "post_event_opaque_raw_hash",
        "split": "opaque_file_bytes",
        "rows_indexed": 0,
        "columns_indexed": 0,
        "excluded_rows_indexed": 0,
        "excluded_columns_indexed": 0,
        "X_materialized": False,
        "X_transformed": False,
        "sha256": observed_sha,
        "md5": observed_md5,
    }]
    audit = pd.read_csv(E160_CONDITION_AUDIT)
    if not {"raw_condition", "canonical_condition"}.issubset(audit.columns):
        raise IntegrityFailure("E160 condition audit schema changed")
    mapper = audit.set_index("raw_condition")["canonical_condition"].astype(str).to_dict()
    split_json = load_json(E160_SPLIT)
    split_lookup = {
        str(condition): role
        for role in ("train", "val", "test")
        for condition in split_json[role]
    }

    raw = ad.read_h5ad(RAW, backed="r")
    try:
        if raw.shape != EXPECTED_RAW_SHAPE:
            raise IntegrityFailure(f"Raw shape changed: {raw.shape}")
        if "perturbation" not in raw.obs.columns or not raw.obs_names.is_unique:
            raise IntegrityFailure("Raw condition/obs_name metadata changed")
        var_names = raw.var_names.astype(str).tolist()
        if var_names[:N_ENDOGENOUS] != frozen["endogenous"]:
            raise IntegrityFailure("Raw endogenous prefix differs from E161")
        if var_names[N_ENDOGENOUS:] != frozen["excluded"]:
            raise IntegrityFailure("Raw 421-column exclusion boundary differs from E161")
        canonical = raw.obs["perturbation"].astype(str).map(mapper)
        if canonical.isna().any():
            raise IntegrityFailure("Raw condition lacks E160 canonical mapping")
        roles = canonical.map(split_lookup).fillna("excluded")
        test_rows = np.flatnonzero(roles.to_numpy() == "test").astype(np.int64)
        if (
            test_rows.shape != (N_TEST_ROWS,)
            or len(np.unique(test_rows)) != N_TEST_ROWS
            or not np.all(test_rows[:-1] < test_rows[1:])
        ):
            raise IntegrityFailure("Raw test-row membership changed")
        if set(canonical.iloc[test_rows].astype(str)) != set(preflight["test_order"]):
            raise IntegrityFailure("Raw test condition set changed")
        observed_counts = canonical.iloc[test_rows].astype(str).value_counts().to_dict()
        if observed_counts != preflight["e160_test_design"]["test_cell_counts_by_condition"]:
            raise IntegrityFailure("Raw test counts by condition differ from E160 metadata audit")
        forbidden_rows = np.flatnonzero(roles.to_numpy() != "test").astype(np.int64)
        if np.intersect1d(test_rows, forbidden_rows).size:
            raise IntegrityFailure("Test row request intersects a forbidden split")
        # Sole permitted raw-X materialization in E165.
        counts = raw.X[test_rows, :N_ENDOGENOUS]
        if not sp.issparse(counts):
            counts = sp.csr_matrix(counts)
        counts = counts.tocsr()
        obs_names = raw.obs_names[test_rows].astype(str).to_numpy(copy=True)
        conditions = canonical.iloc[test_rows].astype(str).to_numpy(copy=True)
    finally:
        raw.file.close()
    after_stat = raw_stat_gate(preflight["e160_status"])
    if after_stat != before_stat:
        raise IntegrityFailure("Raw file identity changed during the one-time truth read")
    if counts.shape != (N_TEST_ROWS, N_ENDOGENOUS):
        raise IntegrityFailure("Test count slice shape changed")
    if not np.issubdtype(counts.dtype, np.integer) or (counts.nnz and counts.data.min() < 0):
        raise IntegrityFailure("Raw test X is no longer nonnegative integer counts")
    libraries = np.asarray(counts.sum(axis=1), dtype=np.float64).reshape(-1)
    if libraries.shape != (N_TEST_ROWS,) or not np.isfinite(libraries).all() or np.any(libraries <= 0):
        raise IntegrityFailure("Invalid test endogenous library")
    normalized = counts.astype(np.float64).multiply((10_000.0 / libraries)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    selected = normalized[:, frozen["selected_indices"]].toarray().astype(np.float64, copy=False)
    if selected.shape != (N_TEST_ROWS, N_SELECTED) or not np.isfinite(selected).all():
        raise IntegrityFailure("Invalid selected test truth matrix")
    ledger.append({
        "phase": "semantic_test_endogenous_read_and_fixed_transform",
        "split": "test",
        "rows_indexed": N_TEST_ROWS,
        "columns_indexed": N_ENDOGENOUS,
        "selected_columns_materialized_after_normalization": N_SELECTED,
        "train_rows_indexed": 0,
        "validation_rows_indexed": 0,
        "excluded_rows_indexed": 0,
        "engineered_construct_columns_indexed": 0,
        "guide_barcode_columns_indexed": 0,
        "excluded_columns_indexed": 0,
        "row_index_sha256": sha256_bytes(test_rows.tobytes()),
        "selected_index_sha256": sha256_bytes(frozen["selected_indices"].tobytes()),
        "endogenous_column_indices_sha256": sha256_bytes(
            np.arange(N_ENDOGENOUS, dtype=np.int64).tobytes()
        ),
        "X_materialized": True,
        "X_transformed": True,
        "rows_transformed": N_TEST_ROWS,
        "normalization": "log1p(10000*count/endogenous_library)",
    })
    return selected, conditions, obs_names, ledger


def task_truth(
    cell_matrix: np.ndarray,
    conditions: np.ndarray,
    order: list[str],
    frozen: dict[str, Any],
) -> dict[str, np.ndarray]:
    posts = []
    counts = []
    for condition in order:
        mask = conditions == condition
        if not mask.any():
            raise IntegrityFailure(f"No truth cells for {condition}")
        posts.append(cell_matrix[mask].mean(axis=0))
        counts.append(int(mask.sum()))
    post = np.asarray(posts, dtype=np.float64)
    n_cells = np.asarray(counts, dtype=np.int64)
    if n_cells.sum() != N_TEST_ROWS:
        raise IntegrityFailure("Task truth cell counts do not sum to 9,902")
    pca = (post - frozen["pca_mean"]) @ frozen["components"].T
    pca_reconstructed_post = pca @ frozen["components"] + frozen["pca_mean"]
    return {
        "post": post,
        "raw_effect": post - frozen["control"][None, :],
        "pca": pca,
        "pca_post": pca_reconstructed_post,
        "pca_effect": pca_reconstructed_post - frozen["control"][None, :],
        "n_cells": n_cells,
    }


def release_path(release: Path, relative: str) -> Path:
    candidate = Path(str(relative))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise IntegrityFailure(f"Unsafe interface-relative path: {relative}")
    unresolved = release / candidate
    if unresolved.is_symlink():
        raise IntegrityFailure(f"Interface path is a symlink: {relative}")
    resolved = unresolved.resolve()
    if not resolved.is_relative_to(release.resolve()):
        raise IntegrityFailure(f"Interface path escapes release: {relative}")
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return resolved


def load_locked_predictions(
    preflight: dict[str, Any], frozen: dict[str, Any]
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[int, pd.DataFrame]]:
    interface = preflight["e164_interface"]
    paths = preflight["e164_paths"]
    release = E164 / "release"
    profiles_path = release_path(release, paths["baseline_post_profiles"])
    risk_path = release_path(release, paths["risk_wide"])
    profiles = pd.read_csv(profiles_path)
    if profiles.columns[:2].tolist() != ["baseline", "condition"]:
        raise IntegrityFailure("E164 baseline profile leading columns changed")
    if profiles.columns[2:].astype(str).tolist() != frozen["selected"]:
        raise IntegrityFailure("E164 profile gene axis changed")
    expected_pairs = [
        (baseline, condition)
        for baseline in BASELINE_ORDER
        for condition in preflight["test_order"]
    ]
    observed_pairs = list(zip(profiles["baseline"].astype(str), profiles["condition"].astype(str)))
    if observed_pairs != expected_pairs:
        raise IntegrityFailure("E164 baseline/profile row order changed")
    values = profiles.iloc[:, 2:].to_numpy(dtype=np.float64)
    if values.shape != (len(BASELINE_ORDER) * N_TEST_TASKS, N_SELECTED) or not np.isfinite(values).all():
        raise IntegrityFailure("E164 baseline profiles are malformed")
    baseline_profiles = {
        baseline: values[index * N_TEST_TASKS:(index + 1) * N_TEST_TASKS]
        for index, baseline in enumerate(BASELINE_ORDER)
    }
    condition_balanced = baseline_profiles["condition_balanced_perturbed_mean"]
    if np.max(np.abs(condition_balanced - condition_balanced[0])) > 1e-12:
        raise IntegrityFailure("Condition-balanced perturbed reference is not task invariant")

    risk = pd.read_csv(risk_path)
    if "condition" not in risk or risk["condition"].astype(str).tolist() != preflight["test_order"]:
        raise IntegrityFailure("E164 risk table task order changed")
    missing_risk = set(RISK_SCORE_COLUMNS) - set(risk.columns)
    if missing_risk:
        raise IntegrityFailure(f"E164 risk table lacks frozen columns: {sorted(missing_risk)}")
    for column in RISK_SCORE_COLUMNS:
        risk[column] = pd.to_numeric(risk[column], errors="raise")

    native: dict[int, pd.DataFrame] = {}
    if preflight["prescribe_arm_authorized"]:
        for seed in SEEDS:
            path = release_path(release, paths["prescribe_scores"][str(seed)])
            frame = pd.read_csv(path)
            required = {
                "condition", "raw_log_prob", "query_has_test_expression",
                "query_has_y", "query_has_y_pca", "selected_gene_order_sha256",
                *{f"predicted_pca_{index}" for index in range(N_PCA)},
            }
            if not required.issubset(frame.columns) or len(frame) != N_TEST_TASKS:
                raise IntegrityFailure(f"E164 PRESCRIBE table is malformed for seed {seed}")
            if frame["condition"].astype(str).tolist() != preflight["test_order"]:
                raise IntegrityFailure(f"E164 PRESCRIBE task order changed for seed {seed}")
            for flag in ("query_has_test_expression", "query_has_y", "query_has_y_pca"):
                if strict_bool_series(frame[flag], context=f"E164 seed{seed}.{flag}").any():
                    raise IntegrityFailure(f"E164 label-only query contains truth flag: {flag}")
            if set(frame["selected_gene_order_sha256"].astype(str)) != {
                preflight["e161_interface"]["selected_gene_order_sha256"]
            }:
                raise IntegrityFailure(f"E164 PRESCRIBE gene hash changed for seed {seed}")
            numeric = [
                "raw_log_prob", "official_combined_confidence", "predicted_magnitude_rms",
                *[f"predicted_pca_{index}" for index in range(N_PCA)],
            ]
            if not set(numeric).issubset(frame.columns):
                raise IntegrityFailure(f"E164 PRESCRIBE score baselines are missing for seed {seed}")
            frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
            if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
                raise IntegrityFailure(f"Non-finite E164 PRESCRIBE values for seed {seed}")
            raw = frame["raw_log_prob"].to_numpy(float)
            observed_gate = {
                "n_rows": len(frame),
                "raw_log_prob_all_finite": bool(np.isfinite(raw).all()),
                "raw_log_prob_exact_unique": int(np.unique(raw).size),
                "raw_log_prob_sample_std_ddof1": float(np.std(raw, ddof=1)),
            }
            recorded_gate = interface["raw_score_gates"][str(seed)]
            for key in ("n_rows", "raw_log_prob_all_finite", "raw_log_prob_exact_unique"):
                recorded = recorded_gate.get(key)
                if key == "raw_log_prob_all_finite":
                    recorded = strict_bool_scalar(recorded, context=f"E164 seed{seed} gate finite")
                if observed_gate[key] != recorded:
                    raise IntegrityFailure(f"E164 seed{seed} observed score disagrees with gate at {key}")
            if not math.isclose(
                observed_gate["raw_log_prob_sample_std_ddof1"],
                float(recorded_gate.get("raw_log_prob_sample_std_ddof1", math.nan)),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise IntegrityFailure(f"E164 seed{seed} observed score std disagrees with gate")
            # Only the pre-registered main seed is authorization-changing.
            # Sensitivity seeds remain reportable even if constant/degenerate.
            if seed == MAIN_SEED and (
                observed_gate["raw_log_prob_exact_unique"] < 24
                or observed_gate["raw_log_prob_sample_std_ddof1"] <= 1e-6
                or strict_bool_scalar(recorded_gate.get("passed"), context="E164 seed3407 gate") is not True
            ):
                raise IntegrityFailure("E164 main raw-score gate failed after authorization")
            native[seed] = frame
    return baseline_profiles, risk, native


def safe_pearson(left: np.ndarray, right: np.ndarray) -> tuple[float, str]:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        return math.nan, "shape_or_nonfinite"
    if np.std(left, ddof=1) <= 1e-12 or np.std(right, ddof=1) <= 1e-12:
        return math.nan, "constant_vector"
    return float(np.corrcoef(left, right)[0, 1]), ""


def safe_cosine(left: np.ndarray, right: np.ndarray) -> tuple[float, str]:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return math.nan, "shape_or_nonfinite"
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return math.nan, "zero_norm"
    return float(np.dot(left, right) / denominator), ""


def direction_accuracy(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return math.nan
    return float(np.mean((left * right) > 0.0))


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta))) if np.isfinite(delta).all() else math.nan


def metric_set(prediction: np.ndarray, truth: np.ndarray, indices: np.ndarray | None = None) -> dict[str, Any]:
    if indices is not None:
        prediction = prediction[indices]
        truth = truth[indices]
    pearson, pearson_reason = safe_pearson(prediction, truth)
    cosine, cosine_reason = safe_cosine(prediction, truth)
    return {
        "pearson": pearson,
        "pearson_na_reason": pearson_reason,
        "cosine": cosine,
        "cosine_na_reason": cosine_reason,
        "direction_accuracy": direction_accuracy(prediction, truth),
        "mse": rmse(prediction, truth) ** 2,
        "rmse": rmse(prediction, truth),
    }


def centroid_accuracy(
    predicted_post: np.ndarray, truth_posts: np.ndarray, index: int
) -> tuple[float, float]:
    """Return official Systema rank fraction and strict nearest-centroid hit.

    Systema's released implementation counts the fraction of competing truth
    centroids whose Euclidean distance is strictly larger than the distance to
    the correct centroid.  E164 also froze the stricter nearest-centroid hit;
    both are retained, and exact ties are misses in both definitions.
    """
    distances = np.sqrt(np.sum((truth_posts - predicted_post[None, :]) ** 2, axis=1))
    correct = distances[index]
    other = np.delete(distances, index)
    return float(np.mean(correct < other)), float(correct < np.min(other))


def evaluate_predictors(
    preflight: dict[str, Any],
    frozen: dict[str, Any],
    truth: dict[str, np.ndarray],
    baseline_profiles: dict[str, np.ndarray],
    native: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    predictor_posts: dict[str, np.ndarray] = dict(baseline_profiles)
    predictor_meta: dict[str, tuple[str, float]] = {
        baseline: ("baseline", math.nan) for baseline in BASELINE_ORDER
    }
    for seed, frame in native.items():
        coordinates = frame[[f"predicted_pca_{index}" for index in range(N_PCA)]].to_numpy(float)
        post = coordinates @ frozen["components"] + frozen["pca_mean"]
        predictor = f"prescribe_seed{seed}"
        predictor_posts[predictor] = post
        predictor_meta[predictor] = ("prescribe_main" if seed == MAIN_SEED else "prescribe_sensitivity", float(seed))
    systema_reference = baseline_profiles["condition_balanced_perturbed_mean"][0]
    rows: list[dict[str, Any]] = []
    centroid_rows: list[dict[str, Any]] = []
    for predictor, posts in predictor_posts.items():
        if posts.shape != truth["post"].shape or not np.isfinite(posts).all():
            raise IntegrityFailure(f"Malformed predictor post matrix: {predictor}")
        raw_effects = posts - frozen["control"][None, :]
        pca_coordinates = (posts - frozen["pca_mean"]) @ frozen["components"].T
        pca_posts = pca_coordinates @ frozen["components"] + frozen["pca_mean"]
        pca_effects = pca_posts - frozen["control"][None, :]
        systema_effects = posts - systema_reference[None, :]
        family, seed = predictor_meta[predictor]
        for index, condition in enumerate(preflight["test_order"]):
            pca_top20 = np.argsort(-np.abs(truth["pca_effect"][index]), kind="stable")[:20]
            raw_top20 = np.argsort(-np.abs(truth["raw_effect"][index]), kind="stable")[:20]
            pca_metrics = metric_set(pca_effects[index], truth["pca_effect"][index])
            raw_metrics = metric_set(raw_effects[index], truth["raw_effect"][index])
            pca20 = metric_set(pca_effects[index], truth["pca_effect"][index], pca_top20)
            raw20 = metric_set(raw_effects[index], truth["raw_effect"][index], raw_top20)
            systema = metric_set(systema_effects[index], truth["post"][index] - systema_reference)
            centroid, nearest_hit = centroid_accuracy(
                posts[index], truth["post"], index
            )
            row: dict[str, Any] = {
                "predictor": predictor,
                "family": family,
                "seed": seed,
                "condition": condition,
                "n_truth_cells": int(truth["n_cells"][index]),
                "pca_top20_gene_indices": ";".join(map(str, pca_top20.tolist())),
                "raw_top20_gene_indices": ";".join(map(str, raw_top20.tolist())),
                "systema_centroid_accuracy": centroid,
                "systema_nearest_centroid_hit": nearest_hit,
            }
            for prefix, values in (
                ("pca10", pca_metrics), ("raw", raw_metrics),
                ("pca10_top20", pca20), ("raw_top20", raw20),
                ("systema_perturbed_reference", systema),
            ):
                row.update({f"{prefix}_{key}": value for key, value in values.items()})
            rows.append(row)
            centroid_rows.append({
                "predictor": predictor,
                "family": family,
                "seed": seed,
                "condition": condition,
                "centroid_accuracy": centroid,
                "nearest_centroid_hit": nearest_hit,
                "distance_metric": "euclidean_selected_gene_post_profile",
                "ties": "miss",
                "n_competing_truth_centroids": N_TEST_TASKS - 1,
                "official_systema_definition": (
                    "fraction_of_other_truth_centroids_strictly_farther_than_correct"
                ),
            })
    return pd.DataFrame(rows), pd.DataFrame(centroid_rows), predictor_posts


def summarize_predictors(task_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column for column in task_metrics.columns
        if column not in {
            "seed", "n_truth_cells"
        } and pd.api.types.is_numeric_dtype(task_metrics[column])
    ]
    rows = []
    for (predictor, family, seed), group in task_metrics.groupby(
        ["predictor", "family", "seed"], dropna=False, sort=False
    ):
        row: dict[str, Any] = {"predictor": predictor, "family": family, "seed": seed, "n_tasks": len(group)}
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            row[f"mean_{column}"] = float(np.mean(finite)) if finite.size else math.nan
            row[f"median_{column}"] = float(np.median(finite)) if finite.size else math.nan
            row[f"n_finite_{column}"] = int(finite.size)
        rows.append(row)
    return pd.DataFrame(rows)


def safe_spearman(score: np.ndarray, endpoint: np.ndarray) -> tuple[float, str]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    endpoint = np.asarray(endpoint, dtype=np.float64).reshape(-1)
    if score.shape != endpoint.shape or score.size < 3:
        return math.nan, "shape_or_too_few_tasks"
    if not np.isfinite(score).all() or not np.isfinite(endpoint).all():
        return math.nan, "nonfinite"
    if np.unique(score).size < 2 or np.unique(endpoint).size < 2:
        return math.nan, "constant"
    if np.std(score, ddof=1) <= 1e-12 or np.std(endpoint, ddof=1) <= 1e-12:
        return math.nan, "constant"
    ranked_score = rankdata(score, method="average")
    ranked_endpoint = rankdata(endpoint, method="average")
    ranked_score = ranked_score - ranked_score.mean()
    ranked_endpoint = ranked_endpoint - ranked_endpoint.mean()
    denominator = float(
        np.sqrt(np.dot(ranked_score, ranked_score) * np.dot(ranked_endpoint, ranked_endpoint))
    )
    if denominator <= 0:
        return math.nan, "constant_rank"
    value = float(np.dot(ranked_score, ranked_endpoint) / denominator)
    return (value, "") if np.isfinite(value) else (math.nan, "nonfinite_statistic")


def safe_spearman_p_greater(score: np.ndarray, endpoint: np.ndarray) -> float:
    rho, _ = safe_spearman(score, endpoint)
    if not np.isfinite(rho):
        return math.nan
    result = spearmanr(
        np.asarray(score, dtype=np.float64),
        np.asarray(endpoint, dtype=np.float64),
        alternative="greater",
    )
    return float(result.pvalue) if np.isfinite(result.pvalue) else math.nan


def bootstrap_spearman(score: np.ndarray, endpoint: np.ndarray) -> float:
    """Spearman kernel for already shape-aligned bootstrap samples."""
    score = np.asarray(score, dtype=np.float64)
    endpoint = np.asarray(endpoint, dtype=np.float64)
    if not np.isfinite(score).all() or not np.isfinite(endpoint).all():
        return math.nan
    ranked_score = rankdata(score, method="average")
    ranked_endpoint = rankdata(endpoint, method="average")
    ranked_score -= ranked_score.mean()
    ranked_endpoint -= ranked_endpoint.mean()
    denominator = float(
        np.sqrt(np.dot(ranked_score, ranked_score) * np.dot(ranked_endpoint, ranked_endpoint))
    )
    return (
        float(np.dot(ranked_score, ranked_endpoint) / denominator)
        if denominator > 0 else math.nan
    )


def holm_adjust(values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(values)
    finite = [(index, float(value)) for index, value in enumerate(values) if np.isfinite(value)]
    finite.sort(key=lambda pair: pair[1])
    running = 0.0
    total = len(finite)
    for rank, (index, value) in enumerate(finite):
        running = max(running, (total - rank) * value)
        adjusted[index] = min(1.0, running)
    return adjusted


def bootstrap_value_summary(
    analysis_id: str, scheme: str, values: np.ndarray
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size >= MIN_VALID_BOOTSTRAPS:
        low, high = np.quantile(finite, [0.025, 0.975], method="linear")
    else:
        low, high = math.nan, math.nan
    return {
        "analysis_id": analysis_id,
        "scheme": scheme,
        "n_replicates": BOOTSTRAPS,
        "n_finite": int(finite.size),
        "bootstrap_mean": float(np.mean(finite)) if finite.size else math.nan,
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_estimable": bool(finite.size >= MIN_VALID_BOOTSTRAPS),
    }


def component_structure(order: list[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    components: set[str] = set()
    parsed = []
    for condition in order:
        genes = condition.split("+")
        if len(genes) != 2 or "ctrl" in genes or genes[0] == genes[1]:
            raise IntegrityFailure(f"Expected a two-component test pair: {condition}")
        parsed.append(genes)
        components.update(genes)
    genes = sorted(components)
    mapping = {
        gene: np.asarray(
            [index for index, pair in enumerate(parsed) if gene in pair], dtype=np.int64
        )
        for gene in genes
    }
    if any(not len(indices) for indices in mapping.values()):
        raise IntegrityFailure("Empty component cluster")
    return genes, mapping


def fixed_bootstrap_indices(order: list[str]) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    task_rng = np.random.default_rng(BOOTSTRAP_SEED)
    task_draws = task_rng.integers(0, N_TEST_TASKS, size=(BOOTSTRAPS, N_TEST_TASKS), dtype=np.int16)
    genes, mapping = component_structure(order)
    cluster_rng = np.random.default_rng(BOOTSTRAP_SEED)
    gene_draws = cluster_rng.integers(0, len(genes), size=(BOOTSTRAPS, len(genes)), dtype=np.int16)
    cluster_draws = []
    for draw in gene_draws:
        pieces = [mapping[genes[int(index)]] for index in draw]
        cluster_draws.append(np.concatenate(pieces))
    return task_draws, cluster_draws, genes


def bootstrap_statistic(
    analysis_id: str,
    scheme: str,
    draws: Any,
    statistic: Callable[[np.ndarray], float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = np.full(BOOTSTRAPS, np.nan, dtype=np.float64)
    for replicate, indices in enumerate(draws):
        try:
            value = float(statistic(np.asarray(indices, dtype=np.int64)))
        except (ValueError, FloatingPointError, IndexError):
            value = math.nan
        if np.isfinite(value):
            values[replicate] = value
    finite = values[np.isfinite(values)]
    if finite.size >= MIN_VALID_BOOTSTRAPS:
        low, high = np.quantile(finite, [0.025, 0.975], method="linear")
    else:
        low, high = math.nan, math.nan
    frame = pd.DataFrame({
        "analysis_id": analysis_id,
        "scheme": scheme,
        "replicate": np.arange(BOOTSTRAPS, dtype=int),
        "statistic": values,
        "estimable": np.isfinite(values),
    })
    summary = {
        "analysis_id": analysis_id,
        "scheme": scheme,
        "n_replicates": BOOTSTRAPS,
        "n_finite": int(finite.size),
        "bootstrap_mean": float(np.mean(finite)) if finite.size else math.nan,
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_estimable": bool(finite.size >= MIN_VALID_BOOTSTRAPS),
    }
    return frame, summary


def baseline_hierarchy(task_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, baseline in enumerate(BASELINE_ORDER):
        group = task_metrics.loc[task_metrics["predictor"].eq(baseline)]
        if len(group) != N_TEST_TASKS:
            raise IntegrityFailure(f"Missing baseline metrics: {baseline}")
        row = {
            "hierarchy_index": index,
            "predictor": baseline,
            "n_tasks": len(group),
            "mean_pca10_rmse": float(group["pca10_rmse"].mean()),
            "mean_pca10_pearson": float(group["pca10_pearson"].mean(skipna=True)),
            "mean_raw_rmse": float(group["raw_rmse"].mean()),
            "mean_raw_pearson": float(group["raw_pearson"].mean(skipna=True)),
            "mean_systema_centroid_accuracy": float(group["systema_centroid_accuracy"].mean()),
            "mean_nearest_centroid_hit": float(group["systema_nearest_centroid_hit"].mean()),
            "previous_predictor": BASELINE_ORDER[index - 1] if index else "",
            "pca10_rmse_improvement_vs_previous": math.nan,
            "raw_rmse_improvement_vs_previous": math.nan,
        }
        if index:
            previous = task_metrics.loc[task_metrics["predictor"].eq(BASELINE_ORDER[index - 1])]
            row["pca10_rmse_improvement_vs_previous"] = float(
                previous["pca10_rmse"].to_numpy().mean() - group["pca10_rmse"].to_numpy().mean()
            )
            row["raw_rmse_improvement_vs_previous"] = float(
                previous["raw_rmse"].to_numpy().mean() - group["raw_rmse"].to_numpy().mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def secondary_prediction_contrasts(task_metrics: pd.DataFrame) -> pd.DataFrame:
    endpoint_specs = (
        ("pca10_pearson", "higher_is_better"),
        ("pca10_cosine", "higher_is_better"),
        ("pca10_direction_accuracy", "higher_is_better"),
        ("pca10_rmse", "lower_is_better"),
        ("pca10_mse", "lower_is_better"),
        ("raw_pearson", "higher_is_better"),
        ("raw_cosine", "higher_is_better"),
        ("raw_direction_accuracy", "higher_is_better"),
        ("raw_rmse", "lower_is_better"),
        ("raw_mse", "lower_is_better"),
    )
    rows: list[dict[str, Any]] = []
    for index in range(1, len(BASELINE_ORDER)):
        previous_name = BASELINE_ORDER[index - 1]
        current_name = BASELINE_ORDER[index]
        previous = task_metrics.loc[task_metrics["predictor"].eq(previous_name)]
        current = task_metrics.loc[task_metrics["predictor"].eq(current_name)]
        if (
            previous["condition"].astype(str).tolist()
            != current["condition"].astype(str).tolist()
            or len(previous) != N_TEST_TASKS
        ):
            raise IntegrityFailure("Adjacent baseline task order differs")
        for endpoint, orientation in endpoint_specs:
            left = previous[endpoint].to_numpy(float)
            right = current[endpoint].to_numpy(float)
            improvement = right - left if orientation == "higher_is_better" else left - right
            finite = improvement[np.isfinite(improvement)]
            if finite.size < 3:
                pvalue = math.nan
                reason = "fewer_than_3_finite_pairs"
            elif np.all(np.abs(finite) <= 1e-15):
                pvalue = 1.0
                reason = "all_paired_differences_zero"
            else:
                result = wilcoxon(finite, alternative="two-sided", zero_method="wilcox")
                pvalue = float(result.pvalue) if np.isfinite(result.pvalue) else math.nan
                reason = ""
            rows.append({
                "previous_predictor": previous_name,
                "current_predictor": current_name,
                "endpoint": endpoint,
                "orientation": orientation,
                "n_finite_paired_tasks": int(finite.size),
                "mean_improvement_current_vs_previous": (
                    float(np.mean(finite)) if finite.size else math.nan
                ),
                "median_improvement_current_vs_previous": (
                    float(np.median(finite)) if finite.size else math.nan
                ),
                "raw_two_sided_wilcoxon_p": pvalue,
                "na_reason": reason,
                "multiplicity_family": "secondary_prediction_contrasts_holm",
            })
    adjusted = holm_adjust([float(row["raw_two_sided_wilcoxon_p"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value
        row["holm_family_size"] = len(rows)
    return pd.DataFrame(rows)


def association_endpoints(group: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "pca10_pearson_accuracy": group["pca10_pearson"].to_numpy(float),
        "pca10_cosine_accuracy": group["pca10_cosine"].to_numpy(float),
        "pca10_direction_accuracy": group["pca10_direction_accuracy"].to_numpy(float),
        "pca10_negative_rmse": -group["pca10_rmse"].to_numpy(float),
        "raw_pearson_accuracy_sensitivity": group["raw_pearson"].to_numpy(float),
        "raw_cosine_accuracy_sensitivity": group["raw_cosine"].to_numpy(float),
        "raw_direction_accuracy_sensitivity": group["raw_direction_accuracy"].to_numpy(float),
        "raw_negative_rmse_sensitivity": -group["raw_rmse"].to_numpy(float),
    }


def hypothesis_and_native_statistics(
    preflight: dict[str, Any],
    task_metrics: pd.DataFrame,
    risk: pd.DataFrame,
    native: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_draws, cluster_draws, genes = fixed_bootstrap_indices(preflight["test_order"])
    replicate_frames: list[pd.DataFrame] = []
    bootstrap_summaries: list[dict[str, Any]] = []
    hypothesis_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    logo_rows: list[dict[str, Any]] = []

    cellweighted = task_metrics.loc[
        task_metrics["predictor"].eq("cell_weighted_perturbed_mean"), "pca10_rmse"
    ].to_numpy(float)
    matching_group = task_metrics.loc[task_metrics["predictor"].eq("matching_single_mean")]
    matching_rmse = matching_group["pca10_rmse"].to_numpy(float)
    h1_delta = cellweighted - matching_rmse
    h1_point = float(np.mean(h1_delta))
    for scheme, draws in (("task", task_draws), ("component_gene_cluster", cluster_draws)):
        frame, summary = bootstrap_statistic(
            "H1_matching_vs_cellweighted_pca10_rmse", scheme, draws,
            lambda indices, delta=h1_delta: float(np.mean(delta[indices])),
        )
        replicate_frames.append(frame)
        bootstrap_summaries.append(summary)
    h1_summaries = [x for x in bootstrap_summaries if x["analysis_id"].startswith("H1_")]
    h1_passed = bool(
        h1_point > 0
        and len(h1_summaries) == 2
        and all(np.isfinite(x["ci_low"]) and x["ci_low"] > 0 for x in h1_summaries)
    )
    hypothesis_rows.append({
        "hypothesis": "H1",
        "definition": "mean(RMSE_cellweighted-RMSE_matching)>0",
        "point_estimate": h1_point,
        "task_ci_low": next(x["ci_low"] for x in h1_summaries if x["scheme"] == "task"),
        "task_ci_high": next(x["ci_high"] for x in h1_summaries if x["scheme"] == "task"),
        "cluster_ci_low": next(x["ci_low"] for x in h1_summaries if x["scheme"] == "component_gene_cluster"),
        "cluster_ci_high": next(x["ci_high"] for x in h1_summaries if x["scheme"] == "component_gene_cluster"),
        "passed": h1_passed,
        "interpretation": "positive_favors_matching",
        "confirmatory_status": "preregistered_primary_baseline_hypothesis",
        "family_alpha": 0.025,
    })

    h2_score = risk["matching_se_pca10_confidence"].to_numpy(float)
    h2_endpoint = -matching_rmse
    h2_point, h2_reason = safe_spearman(h2_score, h2_endpoint)
    for scheme, draws in (("task", task_draws), ("component_gene_cluster", cluster_draws)):
        frame, summary = bootstrap_statistic(
            "H2_matching_se_vs_negative_matching_rmse", scheme, draws,
            lambda indices, score=h2_score, endpoint=h2_endpoint: bootstrap_spearman(
                score[indices], endpoint[indices]
            ),
        )
        replicate_frames.append(frame)
        bootstrap_summaries.append(summary)
    h2_summaries = [x for x in bootstrap_summaries if x["analysis_id"].startswith("H2_")]
    h2_passed = bool(
        h1_passed and np.isfinite(h2_point) and h2_point > 0
        and all(np.isfinite(x["ci_low"]) and x["ci_low"] > 0 for x in h2_summaries)
    )
    hypothesis_rows.append({
        "hypothesis": "H2",
        "definition": "Spearman(matching_se_pca10_confidence,-matching_PCA10_RMSE)>0",
        "point_estimate": h2_point,
        "na_reason": h2_reason,
        "task_ci_low": next(x["ci_low"] for x in h2_summaries if x["scheme"] == "task"),
        "task_ci_high": next(x["ci_high"] for x in h2_summaries if x["scheme"] == "task"),
        "cluster_ci_low": next(x["ci_low"] for x in h2_summaries if x["scheme"] == "component_gene_cluster"),
        "cluster_ci_high": next(x["ci_high"] for x in h2_summaries if x["scheme"] == "component_gene_cluster"),
        "passed": h2_passed,
        "interpretation": "higher_confidence_should_track_lower_error",
        "confirmatory_status": (
            "confirmatory_H1_passed" if h1_passed else "descriptive_not_confirmatory_due_to_H1"
        ),
        "family_alpha": 0.025,
        "raw_one_sided_p": safe_spearman_p_greater(h2_score, h2_endpoint),
    })

    matching_group = task_metrics.loc[task_metrics["predictor"].eq("matching_single_mean")]
    if matching_group["condition"].astype(str).tolist() != preflight["test_order"]:
        raise IntegrityFailure("Matching metric task order changed")
    score_specs: list[dict[str, Any]] = []
    matching_magnitude_key = "matching__matching_magnitude_confidence"
    for score_name in RISK_SCORE_COLUMNS:
        score_specs.append({
            "key": f"matching__{score_name}",
            "score": score_name,
            "score_family": "matching_risk",
            "predictor": "matching_single_mean",
            "seed": math.nan,
            "seed_role": "not_applicable",
            "values": risk[score_name].to_numpy(float),
            "endpoints": association_endpoints(matching_group),
            "magnitude_key": matching_magnitude_key,
            "official_key": None,
        })
    if preflight["prescribe_arm_authorized"]:
        for seed in SEEDS:
            group = task_metrics.loc[task_metrics["predictor"].eq(f"prescribe_seed{seed}")]
            if group["condition"].astype(str).tolist() != preflight["test_order"]:
                raise IntegrityFailure(f"Native metric task order changed for seed {seed}")
            magnitude_key = f"native_seed{seed}__predicted_magnitude_confidence"
            official_key = f"native_seed{seed}__official_combined_confidence"
            native_scores = {
                "raw_log_prob": native[seed]["raw_log_prob"].to_numpy(float),
                "official_combined_confidence": native[seed]["official_combined_confidence"].to_numpy(float),
                "predicted_magnitude_confidence": -native[seed]["predicted_magnitude_rms"].to_numpy(float),
            }
            for score_name, values in native_scores.items():
                score_specs.append({
                    "key": f"native_seed{seed}__{score_name}",
                    "score": score_name,
                    "score_family": "native_score",
                    "predictor": f"prescribe_seed{seed}",
                    "seed": seed,
                    "seed_role": "main" if seed == MAIN_SEED else "training_sensitivity",
                    "values": values,
                    "endpoints": association_endpoints(group),
                    "magnitude_key": magnitude_key,
                    "official_key": official_key,
                })

    plan_by_key = {spec["key"]: spec for spec in score_specs}
    frame_by_key: dict[tuple[str, str, str], pd.DataFrame] = {}
    summary_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    point_by_key: dict[tuple[str, str], float] = {}
    for spec in score_specs:
        score = np.asarray(spec["values"], dtype=float)
        if score.shape != (N_TEST_TASKS,):
            raise IntegrityFailure(f"Risk score shape changed: {spec['key']}")
        for endpoint_name, endpoint in spec["endpoints"].items():
            analysis_id = f"risk__{spec['key']}__{endpoint_name}"
            point, reason = safe_spearman(score, endpoint)
            point_by_key[(spec["key"], endpoint_name)] = point
            local_summaries = []
            for scheme, draws in (("task", task_draws), ("component_gene_cluster", cluster_draws)):
                frame, summary = bootstrap_statistic(
                    analysis_id,
                    scheme,
                    draws,
                    lambda indices, x=score, y=endpoint: bootstrap_spearman(
                        x[indices], y[indices]
                    ),
                )
                frame["score_key"] = spec["key"]
                frame["endpoint"] = endpoint_name
                frame_by_key[(spec["key"], endpoint_name, scheme)] = frame
                summary_by_key[(spec["key"], endpoint_name, scheme)] = summary
                bootstrap_summaries.append(summary)
                local_summaries.append(summary)
            if (
                spec["key"] == "native_seed3407__raw_log_prob"
                and endpoint_name == "pca10_pearson_accuracy"
            ):
                multiplicity_family = "confirmatory_prescribe_primary_alpha_0p025"
                truth_role = "primary"
            elif spec["key"] == matching_magnitude_key.replace(
                "matching_magnitude_confidence", "matching_se_pca10_confidence"
            ) and endpoint_name == "pca10_negative_rmse":
                multiplicity_family = "fixed_sequence_H2_alpha_0p025"
                truth_role = "H2_duplicate_complete_risk_matrix"
            else:
                multiplicity_family = "secondary_risk_holm_family"
                truth_role = (
                    "raw_truth_mandatory_sensitivity"
                    if endpoint_name.startswith("raw_") else "secondary"
                )
            association_rows.append({
                "analysis_id": analysis_id,
                "score_key": spec["key"],
                "score_family": spec["score_family"],
                "predictor": spec["predictor"],
                "seed": spec["seed"],
                "seed_role": spec["seed_role"],
                "score": spec["score"],
                "endpoint": endpoint_name,
                "truth_role": truth_role,
                "multiplicity_family": multiplicity_family,
                "n_tasks": N_TEST_TASKS,
                "rho": point,
                "na_reason": reason,
                "raw_one_sided_p": safe_spearman_p_greater(score, endpoint),
                "task_ci_low": next(x["ci_low"] for x in local_summaries if x["scheme"] == "task"),
                "task_ci_high": next(x["ci_high"] for x in local_summaries if x["scheme"] == "task"),
                "cluster_ci_low": next(x["ci_low"] for x in local_summaries if x["scheme"] == "component_gene_cluster"),
                "cluster_ci_high": next(x["ci_high"] for x in local_summaries if x["scheme"] == "component_gene_cluster"),
                "magnitude_score_key": spec["magnitude_key"],
                "official_score_key": spec["official_key"] or "",
            })

            per_gene_rhos = []
            for gene in genes:
                keep = np.asarray(
                    [gene not in condition.split("+") for condition in preflight["test_order"]],
                    dtype=bool,
                )
                rho, logo_reason = safe_spearman(score[keep], endpoint[keep])
                per_gene_rhos.append(rho)
                logo_rows.append({
                    "record_type": "per_gene",
                    "analysis_id": analysis_id,
                    "score_key": spec["key"],
                    "predictor": spec["predictor"],
                    "seed": spec["seed"],
                    "endpoint": endpoint_name,
                    "removed_gene": gene,
                    "removed_tasks": int((~keep).sum()),
                    "remaining_tasks": int(keep.sum()),
                    "rho": rho,
                    "estimable": bool(np.isfinite(rho)),
                    "na_reason": logo_reason,
                })
            finite_logo = np.asarray(per_gene_rhos, dtype=float)
            finite_logo = finite_logo[np.isfinite(finite_logo)]
            logo_rows.append({
                "record_type": "summary",
                "analysis_id": analysis_id,
                "score_key": spec["key"],
                "predictor": spec["predictor"],
                "seed": spec["seed"],
                "endpoint": endpoint_name,
                "removed_gene": "__ALL_COMPONENT_GENES__",
                "removed_tasks": math.nan,
                "remaining_tasks": math.nan,
                "rho": math.nan,
                "estimable": bool(finite_logo.size),
                "na_reason": "" if finite_logo.size else "no_estimable_LOGO_rho",
                "logo_n_component_genes": len(genes),
                "logo_n_estimable": int(finite_logo.size),
                "logo_rho_min": float(np.min(finite_logo)) if finite_logo.size else math.nan,
                "logo_rho_median": float(np.median(finite_logo)) if finite_logo.size else math.nan,
                "logo_rho_max": float(np.max(finite_logo)) if finite_logo.size else math.nan,
                "logo_fraction_positive": float(np.mean(finite_logo > 0)) if finite_logo.size else math.nan,
            })

    # Attach paired delta-rho versus the frozen magnitude score (and, for
    # native scores, official confidence) using exactly the same draws.
    for row in association_rows:
        key = str(row["score_key"])
        endpoint_name = str(row["endpoint"])
        comparator_keys = {
            "magnitude": str(row["magnitude_score_key"]),
            "official": str(row["official_score_key"]),
        }
        for comparator_name, comparator_key in comparator_keys.items():
            prefix = f"delta_rho_vs_{comparator_name}"
            if not comparator_key or comparator_key not in plan_by_key:
                row[f"{comparator_name}_rho"] = math.nan
                row[prefix] = math.nan
                row[f"{prefix}_task_ci_low"] = math.nan
                row[f"{prefix}_task_ci_high"] = math.nan
                row[f"{prefix}_cluster_ci_low"] = math.nan
                row[f"{prefix}_cluster_ci_high"] = math.nan
                continue
            comparator_point = point_by_key[(comparator_key, endpoint_name)]
            row[f"{comparator_name}_rho"] = comparator_point
            row[prefix] = (
                float(row["rho"] - comparator_point)
                if np.isfinite(row["rho"]) and np.isfinite(comparator_point) else math.nan
            )
            for scheme in ("task", "component_gene_cluster"):
                frame = frame_by_key[(key, endpoint_name, scheme)]
                comparator_frame = frame_by_key[(comparator_key, endpoint_name, scheme)]
                comparator_values = comparator_frame["statistic"].to_numpy(float)
                delta_values = frame["statistic"].to_numpy(float) - comparator_values
                frame[f"{comparator_name}_statistic"] = comparator_values
                frame[prefix] = delta_values
                delta_summary = bootstrap_value_summary(
                    f"{row['analysis_id']}__{prefix}", scheme, delta_values
                )
                bootstrap_summaries.append(delta_summary)
                short = "task" if scheme == "task" else "cluster"
                row[f"{prefix}_{short}_ci_low"] = delta_summary["ci_low"]
                row[f"{prefix}_{short}_ci_high"] = delta_summary["ci_high"]

    secondary_positions = [
        index for index, row in enumerate(association_rows)
        if row["multiplicity_family"] == "secondary_risk_holm_family"
    ]
    secondary_adjusted = holm_adjust([
        float(association_rows[index]["raw_one_sided_p"]) for index in secondary_positions
    ])
    for row in association_rows:
        row["holm_adjusted_p"] = math.nan
        row["holm_family_size"] = len(secondary_positions) if (
            row["multiplicity_family"] == "secondary_risk_holm_family"
        ) else 0
    for index, adjusted in zip(secondary_positions, secondary_adjusted):
        association_rows[index]["holm_adjusted_p"] = adjusted

    if preflight["prescribe_arm_authorized"]:
        native_primary = [
            row for row in association_rows
            if row["score_key"] == "native_seed3407__raw_log_prob"
            and row["endpoint"] == "pca10_pearson_accuracy"
        ]
        if len(native_primary) != 1:
            raise IntegrityFailure("PRESCRIBE primary association is not unique")
        primary = native_primary[0]
        primary_passed = bool(
            np.isfinite(primary["rho"])
            and float(primary["rho"]) > 0
            and np.isfinite(primary["task_ci_low"])
            and float(primary["task_ci_low"]) > 0
            and np.isfinite(primary["cluster_ci_low"])
            and float(primary["cluster_ci_low"]) > 0
        )
        hypothesis_rows.append({
            "hypothesis": "P1_PRESCRIBE_RAW_SCORE",
            "definition": (
                "Spearman(seed3407 raw_log_prob, own PCA10 Pearson accuracy)>0"
            ),
            "point_estimate": primary["rho"],
            "raw_one_sided_p": primary["raw_one_sided_p"],
            "task_ci_low": primary["task_ci_low"],
            "task_ci_high": primary["task_ci_high"],
            "cluster_ci_low": primary["cluster_ci_low"],
            "cluster_ci_high": primary["cluster_ci_high"],
            "passed": primary_passed,
            "interpretation": "higher_raw_log_prob_should_track_higher_own_accuracy",
            "confirmatory_status": (
                "preregistered_test_endpoint_after_validation_informed_futility_gate"
            ),
            "family_alpha": 0.025,
        })
    else:
        hypothesis_rows.append({
            "hypothesis": "P1_PRESCRIBE_RAW_SCORE",
            "definition": (
                "Spearman(seed3407 raw_log_prob, own PCA10 Pearson accuracy)>0"
            ),
            "point_estimate": math.nan,
            "raw_one_sided_p": math.nan,
            "task_ci_low": math.nan,
            "task_ci_high": math.nan,
            "cluster_ci_low": math.nan,
            "cluster_ci_high": math.nan,
            "passed": False,
            "interpretation": "not_run_prescribe_arm_not_authorized",
            "confirmatory_status": "not_run_prescribe_arm_not_authorized",
            "family_alpha": 0.025,
        })

    if not preflight["prescribe_arm_authorized"]:
        association_rows.append({
            "analysis_id": "native_arm_not_run",
            "score_key": "native_not_run",
            "score_family": "native_score",
            "predictor": "not_run",
            "seed": math.nan,
            "seed_role": "not_run",
            "score": "all_native_scores",
            "endpoint": "all_native_endpoints",
            "truth_role": "not_run",
            "multiplicity_family": "not_run",
            "n_tasks": 0,
            "rho": math.nan,
            "na_reason": "not_run_prescribe_arm_not_authorized",
            "raw_one_sided_p": math.nan,
            "holm_adjusted_p": math.nan,
            "holm_family_size": 0,
        })
    replicate_frames.extend(frame_by_key.values())
    replicates = pd.concat(replicate_frames, ignore_index=True)
    return (
        pd.DataFrame(hypothesis_rows),
        pd.DataFrame(association_rows),
        replicates,
        pd.DataFrame(bootstrap_summaries),
        pd.DataFrame(logo_rows),
    )


def split_half_reference(
    cells: np.ndarray,
    conditions: np.ndarray,
    obs_names: np.ndarray,
    preflight: dict[str, Any],
    frozen: dict[str, Any],
    full_truth: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for task_index, condition in enumerate(preflight["test_order"]):
        cell_indices = np.flatnonzero(conditions == condition)
        if len(cell_indices) < 2:
            raise IntegrityFailure(f"Split-half requires at least two cells: {condition}")
        keyed = []
        for index in cell_indices:
            digest = hashlib.sha256(
                f"E165|Wessels|split-half|3407\t{condition}\t{obs_names[index]}".encode("utf-8")
            ).hexdigest()
            keyed.append((digest, str(obs_names[index]), int(index)))
        keyed.sort()
        if len({digest for digest, _, _ in keyed}) != len(keyed):
            raise IntegrityFailure(f"Split-half SHA collision: {condition}")
        half_a_indices = np.asarray([item[2] for rank, item in enumerate(keyed) if rank % 2 == 0], dtype=int)
        half_b_indices = np.asarray([item[2] for rank, item in enumerate(keyed) if rank % 2 == 1], dtype=int)
        if (
            not len(half_a_indices) or not len(half_b_indices)
            or np.intersect1d(half_a_indices, half_b_indices).size
            or set(np.concatenate([half_a_indices, half_b_indices])) != set(cell_indices)
        ):
            raise IntegrityFailure(f"Invalid deterministic split-half partition: {condition}")
        post_a = cells[half_a_indices].mean(axis=0)
        post_b = cells[half_b_indices].mean(axis=0)
        raw_a = post_a - frozen["control"]
        raw_b = post_b - frozen["control"]
        pca_a = (post_a - frozen["pca_mean"]) @ frozen["components"].T
        pca_b = (post_b - frozen["pca_mean"]) @ frozen["components"].T
        pca_effect_a = pca_a @ frozen["components"] + frozen["pca_mean"] - frozen["control"]
        pca_effect_b = pca_b @ frozen["components"] + frozen["pca_mean"] - frozen["control"]
        raw_top20 = np.argsort(-np.abs(full_truth["raw_effect"][task_index]), kind="stable")[:20]
        pca_top20 = np.argsort(-np.abs(full_truth["pca_effect"][task_index]), kind="stable")[:20]
        row: dict[str, Any] = {
            "condition": condition,
            "n_full_cells": len(cell_indices),
            "n_half_a": len(half_a_indices),
            "n_half_b": len(half_b_indices),
            "half_a_obs_name_sha256": sha256_bytes(
                ("\n".join(obs_names[half_a_indices].astype(str)) + "\n").encode("utf-8")
            ),
            "half_b_obs_name_sha256": sha256_bytes(
                ("\n".join(obs_names[half_b_indices].astype(str)) + "\n").encode("utf-8")
            ),
            "interpretation": "split_half_reproducibility_benchmark_reference_not_upper_bound",
        }
        too_small = len(cell_indices) < 4
        na_metrics = {
            "pearson": math.nan,
            "pearson_na_reason": "fewer_than_4_cells",
            "cosine": math.nan,
            "cosine_na_reason": "fewer_than_4_cells",
            "direction_accuracy": math.nan,
            "mse": math.nan,
            "rmse": math.nan,
        }
        for prefix, values in (
            ("raw", na_metrics if too_small else metric_set(raw_a, raw_b)),
            (
                "raw_top20",
                na_metrics if too_small else metric_set(raw_a, raw_b, raw_top20),
            ),
            (
                "pca10",
                na_metrics if too_small else metric_set(pca_effect_a, pca_effect_b),
            ),
            (
                "pca10_top20",
                na_metrics
                if too_small
                else metric_set(pca_effect_a, pca_effect_b, pca_top20),
            ),
        ):
            row.update({f"{prefix}_{key}": value for key, value in values.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_analysis(
    preflight: dict[str, Any],
    task_metrics: pd.DataFrame,
    risk: pd.DataFrame,
    native: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_specs: list[tuple[str, str, np.ndarray, str]] = []
    for score_name in RISK_SCORE_COLUMNS:
        score_specs.append((score_name, "matching_single_mean", risk[score_name].to_numpy(float), "matching_risk"))
    if preflight["prescribe_arm_authorized"]:
        for seed in SEEDS:
            score_specs.append((
                f"raw_log_prob_seed{seed}", f"prescribe_seed{seed}",
                native[seed]["raw_log_prob"].to_numpy(float), "native_raw_score",
            ))
        score_specs.extend([
            (
                "official_combined_confidence_seed3407", "prescribe_seed3407",
                native[MAIN_SEED]["official_combined_confidence"].to_numpy(float),
                "native_official_score",
            ),
            (
                "negative_predicted_magnitude_seed3407", "prescribe_seed3407",
                -native[MAIN_SEED]["predicted_magnitude_rms"].to_numpy(float),
                "native_magnitude_score",
            ),
        ])
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for score_name, predictor, score, score_family in score_specs:
        metrics = task_metrics.loc[task_metrics["predictor"].eq(predictor)]
        if len(metrics) != N_TEST_TASKS or metrics["condition"].astype(str).tolist() != preflight["test_order"]:
            raise IntegrityFailure(f"Coverage endpoint task order changed: {score_name}")
        score_estimable = bool(
            np.isfinite(score).all() and np.unique(score).size >= 2 and np.std(score, ddof=1) > 1e-12
        )
        for truth_role, error_column, accuracy_column in (
            ("pca10_primary", "pca10_rmse", "pca10_pearson"),
            ("raw_truth_sensitivity", "raw_rmse", "raw_pearson"),
        ):
            error = metrics[error_column].to_numpy(float)
            accuracy = metrics[accuracy_column].to_numpy(float)
            if not np.isfinite(error).all():
                raise IntegrityFailure(f"Coverage error is non-finite: {score_name}/{truth_role}")
            high_error_count = int(math.ceil(0.20 * N_TEST_TASKS))
            condition_tiebreak = np.asarray(preflight["test_order"], dtype=str)
            high_error_order = np.lexsort((condition_tiebreak, -error))
            high_error = set(high_error_order[:high_error_count].tolist())
            retained_errors = []
            for coverage in COVERAGES:
                retained_n = int(math.ceil(coverage * N_TEST_TASKS))
                rejected_n = N_TEST_TASKS - retained_n
                if score_estimable:
                    order = np.lexsort((condition_tiebreak, -score))
                    retained = order[:retained_n]
                    rejected = order[retained_n:]
                    mean_error = float(np.mean(error[retained]))
                    finite_accuracy = accuracy[retained][np.isfinite(accuracy[retained])]
                    mean_accuracy = float(np.mean(finite_accuracy)) if finite_accuracy.size else math.nan
                    captured = len(high_error & set(rejected.tolist()))
                    error_capture = float(captured / high_error_count) if rejected_n else math.nan
                    enrichment = (
                        float((captured / rejected_n) / (high_error_count / N_TEST_TASKS))
                        if rejected_n else math.nan
                    )
                    retained_errors.append(mean_error)
                    reason = ""
                else:
                    mean_error = mean_accuracy = error_capture = enrichment = math.nan
                    retained_errors.append(math.nan)
                    reason = "constant_or_nonfinite_score"
                curve_rows.append({
                    "score": score_name,
                    "score_family": score_family,
                    "predictor": predictor,
                    "truth_role": truth_role,
                    "coverage": coverage,
                    "retained_n": retained_n,
                    "rejected_n": rejected_n,
                    "retained_mean_error": mean_error,
                    "retained_mean_accuracy": mean_accuracy,
                    "high_error_capture_in_rejected": error_capture,
                    "rejected_high_error_enrichment": enrichment,
                    "estimable": score_estimable,
                    "na_reason": reason,
                })
            if score_estimable and np.isfinite(retained_errors).all():
                aurc = float(np.trapz(retained_errors, np.asarray(COVERAGES)) / (COVERAGES[-1] - COVERAGES[0]))
            else:
                aurc = math.nan
            selected_curve = [
                row for row in curve_rows
                if row["score"] == score_name and row["truth_role"] == truth_role
            ]
            c75 = next(row for row in selected_curve if row["coverage"] == 0.75)
            c80 = next(row for row in selected_curve if row["coverage"] == 0.8)
            summary_rows.append({
                "score": score_name,
                "score_family": score_family,
                "predictor": predictor,
                "truth_role": truth_role,
                "aurc_coverage_0p50_to_1p00": aurc,
                "error_capture_at_0p75": c75["high_error_capture_in_rejected"],
                "rejected_enrichment_at_0p75": c75["rejected_high_error_enrichment"],
                "error_capture_at_0p80": c80["high_error_capture_in_rejected"],
                "rejected_enrichment_at_0p80": c80["rejected_high_error_enrichment"],
                "score_unique_values": int(np.unique(score).size) if np.isfinite(score).all() else 0,
                "estimable": score_estimable,
                "na_reason": "" if score_estimable else "constant_or_nonfinite_score",
            })
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def truth_tables(
    preflight: dict[str, Any], frozen: dict[str, Any], truth: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks = pd.DataFrame({
        "test_index": np.arange(N_TEST_TASKS, dtype=int),
        "condition": preflight["test_order"],
        "gene_a": [value.split("+")[0] for value in preflight["test_order"]],
        "gene_b": [value.split("+")[1] for value in preflight["test_order"]],
        "n_test_cells": truth["n_cells"],
        "truth_role": "one_time_unsealed_test_truth",
    })
    profile_blocks = []
    for profile_kind, values in (
        ("raw_normalized_post", truth["post"]),
        ("raw_normalized_effect_vs_train_control", truth["raw_effect"]),
        ("pca10_reconstructed_post", truth["pca_post"]),
        ("pca10_reconstructed_effect_vs_train_control", truth["pca_effect"]),
    ):
        block = pd.DataFrame(values, columns=frozen["selected"])
        block.insert(0, "condition", preflight["test_order"])
        block.insert(0, "profile_kind", profile_kind)
        profile_blocks.append(block)
    profiles = pd.concat(profile_blocks, ignore_index=True)
    pca = pd.DataFrame(truth["pca"], columns=[f"truth_pca_{index}" for index in range(N_PCA)])
    pca_effect = truth["pca"] - frozen["control_pca"][None, :]
    for index in range(N_PCA):
        pca[f"truth_effect_pca_{index}"] = pca_effect[:, index]
    pca.insert(0, "condition", preflight["test_order"])
    return tasks, profiles, pca


def render_report(
    preflight: dict[str, Any],
    hierarchy: pd.DataFrame,
    hypotheses: pd.DataFrame,
    associations: pd.DataFrame,
    split_half: pd.DataFrame,
) -> str:
    h1 = hypotheses.loc[hypotheses["hypothesis"].eq("H1")].iloc[0]
    h2 = hypotheses.loc[hypotheses["hypothesis"].eq("H2")].iloc[0]
    p1 = hypotheses.loc[
        hypotheses["hypothesis"].eq("P1_PRESCRIBE_RAW_SCORE")
    ].iloc[0]
    native_text = "PRESCRIBE arm未获授权，因此没有查询或评价native test labels。"
    if preflight["prescribe_arm_authorized"]:
        primary = associations.loc[
            associations["seed"].eq(MAIN_SEED)
            & associations["score"].eq("raw_log_prob")
            & associations["endpoint"].eq("pca10_pearson_accuracy")
        ]
        if len(primary) == 1:
            row = primary.iloc[0]
            native_text = (
                f"PRESCRIBE主种子3407：raw_log_prob与PCA10 own-model Pearson accuracy的"
                f"Spearman rho={row['rho']:.4g}，task 95% CI "
                f"[{row['task_ci_low']:.4g}, {row['task_ci_high']:.4g}]，component-gene "
                f"95% CI [{row['cluster_ci_low']:.4g}, {row['cluster_ci_high']:.4g}]，"
                f"预注册门通过={bool(p1['passed'])}。"
            )
    hierarchy_lines = "\n".join(
        f"- {row.predictor}: mean PCA10 RMSE={row.mean_pca10_rmse:.4g}; "
        f"raw RMSE={row.mean_raw_rmse:.4g}; centroid accuracy={row.mean_systema_centroid_accuracy:.4g}"
        for row in hierarchy.itertuples(index=False)
    )
    split_mean = float(split_half["raw_pearson"].mean(skipna=True))
    return f"""# E165 Wessels一次性test truth评价报告

## 访问边界

- 不可逆事件：`../TEST_TRUTH_UNSEAL_EVENT.json`；event在任何raw文件open之前落盘。
- 唯一expression读取：9,902 test rows × 前20,631 endogenous columns。
- 421 engineered/guide/barcode columns、train/validation/excluded rows读取数均为0。
- 归一化：每cell `log1p(10000 × endogenous count / endogenous library)`，再取冻结2,023 selected axis。
- baseline arm：已执行；PRESCRIBE arm：`{preflight['prescribe_arm_authorized']}`。

## 五级baseline

{hierarchy_lines}

H1固定为 `mean(RMSE_cellweighted - RMSE_matching)`，observed={h1['point_estimate']:.4g}，通过={bool(h1['passed'])}。

H2 observed rho={h2['point_estimate']:.4g}，通过={bool(h2['passed'])}，解释层级=`{h2['confirmatory_status']}`。H1未通过时，H2只保留描述性结果。

## PRESCRIBE raw score

{native_text}

PCA10 truth是E160主要口径；raw selected-gene truth敏感性、task bootstrap、component-gene bootstrap和LOGO均强制留存。

## Systema、SBB与split-half

Systema perturbed reference使用train condition centroids等权平均的冻结profile；官方centroid accuracy是正确centroid距离严格小于其余47个距离的比例，另报告更严格的nearest-centroid hit，tie在两者中都失败。语义核对来源为Systema官方代码commit `{SYSTEMA_CODE_COMMIT}`。

split-half按condition+obs_name SHA确定性分半，平均raw-effect Pearson为{split_mean:.4g}。它是reproducibility benchmark/reference，不是upper bound：每半样本数小于完整truth。

Top20仅按truth effect绝对值选择，作为SBB signal-sensitive诊断；五级baseline hierarchy提供性能地板。除预注册H1/H2和native主终点外，Systema/SBB扩展均为contextual/descriptive。

## 方法来源

- [Systema](https://doi.org/10.1038/s41587-025-02777-8)
- [SBB principles](https://doi.org/10.64898/2026.04.20.719650)
- [TxPert](https://doi.org/10.1038/s41587-026-03113-4)

所有有利与不利结果均保留；本实验不保证期刊录用。
"""


def render_white_svg(
    hierarchy: pd.DataFrame, hypotheses: pd.DataFrame, preflight: dict[str, Any]
) -> str:
    width, height = 1200, 720
    maximum = max(float(hierarchy["mean_pca10_rmse"].max()), 1e-12)
    bars = []
    colors = ["#7f7f7f", "#9ca3af", "#5b8db8", "#2f6f9f", "#b45f4b"]
    labels = ["Control", "Cell-weighted", "Condition-balanced", "Matching", "Additive"]
    for index, row in enumerate(hierarchy.itertuples(index=False)):
        x = 110 + index * 190
        bar_height = 330 * float(row.mean_pca10_rmse) / maximum
        y = 505 - bar_height
        bars.append(
            f'<rect x="{x}" y="{y:.2f}" width="120" height="{bar_height:.2f}" fill="{colors[index]}"/>'
            f'<text x="{x + 60}" y="535" text-anchor="middle" font-size="20">{html.escape(labels[index])}</text>'
            f'<text x="{x + 60}" y="{y - 12:.2f}" text-anchor="middle" font-size="19">{row.mean_pca10_rmse:.3g}</text>'
        )
    h1 = hypotheses.loc[hypotheses["hypothesis"].eq("H1")].iloc[0]
    h2 = hypotheses.loc[hypotheses["hypothesis"].eq("H2")].iloc[0]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<style>text{{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#202124}}</style>
<text x="70" y="70" font-size="34" font-weight="700">Wessels frozen test evaluation</text>
<text x="70" y="108" font-size="20" fill="#5f6368">Seen singles → unseen double perturbations · 48 tasks · 9,902 cells</text>
<line x1="80" y1="505" x2="1080" y2="505" stroke="#333" stroke-width="1.5"/>
<text x="80" y="155" font-size="23" font-weight="700">Mean PCA10 RMSE (lower is better)</text>
{''.join(bars)}
<rect x="70" y="590" width="1060" height="82" fill="#f7f7f5" stroke="#d8d8d4"/>
<text x="95" y="624" font-size="20">H1 Δ(cell-weighted − matching) = {float(h1['point_estimate']):.4g} · passed: {str(bool(h1['passed'])).lower()}</text>
<text x="95" y="655" font-size="20">H2 ρ(SE confidence, −matching error) = {float(h2['point_estimate']):.4g} · {html.escape(str(h2['confirmatory_status']))}</text>
<text x="1110" y="700" text-anchor="end" font-size="15" fill="#6b7280">White-background scientific summary · PRESCRIBE arm: {str(preflight['prescribe_arm_authorized']).lower()}</text>
</svg>'''


def prepare_staging(event: dict[str, Any]) -> None:
    if RELEASE.exists():
        raise FileExistsError("E165 release already exists; refusing overwrite")
    if STAGING.exists():
        sentinel = STAGING / ".E165_TRANSACTION.json"
        if not sentinel.is_file() or load_json(sentinel).get("transaction_id") != event["transaction_id"]:
            raise IntegrityFailure("Orphan E165 staging belongs to another transaction")
        preserved = OUT / f".failed_staging_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        STAGING.replace(preserved)
        fsync_directory(OUT)
    STAGING.mkdir(parents=True, exist_ok=False)
    for directory in ("reports", "figures", "profiles", "tables"):
        (STAGING / directory).mkdir()
    atomic_json(STAGING / ".E165_TRANSACTION.json", {
        "schema": "safeconf_e165_release_transaction_v1",
        "transaction_id": event["transaction_id"],
        "unseal_event_sha256": sha256_file(EVENT),
        "phase": "building",
        "created_at": now(),
    })


def scientific_interface(
    preflight: dict[str, Any],
    event: dict[str, Any],
    hypotheses: pd.DataFrame,
    existing_hashes: dict[str, str],
) -> dict[str, Any]:
    h1 = hypotheses.loc[hypotheses["hypothesis"].eq("H1")].iloc[0]
    h2 = hypotheses.loc[hypotheses["hypothesis"].eq("H2")].iloc[0]
    p1 = hypotheses.loc[
        hypotheses["hypothesis"].eq("P1_PRESCRIBE_RAW_SCORE")
    ].iloc[0]
    return {
        "schema": "safeconf_e165_to_e166_v1",
        "experiment": "E165_wessels_truth_unseal_evaluation",
        "git_head": preflight["git_head"],
        "gate_fingerprint_sha256": preflight["gate_fingerprint_sha256"],
        "unseal_event_relative_path": "../TEST_TRUTH_UNSEAL_EVENT.json",
        "unseal_event_sha256": sha256_file(EVENT),
        "transaction_id": event["transaction_id"],
        "baseline_arm_evaluated": True,
        "prescribe_arm_evaluated": preflight["prescribe_arm_authorized"],
        "prescribe_arm_closed_reason": "" if preflight["prescribe_arm_authorized"] else "not_run_prescribe_arm_not_authorized",
        "test_conditions": N_TEST_TASKS,
        "test_cells": N_TEST_ROWS,
        "endogenous_columns_read": N_ENDOGENOUS,
        "selected_genes": N_SELECTED,
        "excluded_columns_read": 0,
        "train_validation_excluded_rows_read": 0,
        "normalization": "per_cell_log1p(10000*endogenous_count/endogenous_library)",
        "baseline_order": list(BASELINE_ORDER),
        "H1": {
            "definition": "mean(RMSE_cellweighted-RMSE_matching)>0",
            "point": float(h1["point_estimate"]),
            "passed": bool(h1["passed"]),
        },
        "H2": {
            "definition": "Spearman(matching_se_pca10_confidence,-matching_PCA10_RMSE)>0",
            "point": float(h2["point_estimate"]),
            "passed": bool(h2["passed"]),
            "confirmatory_status": str(h2["confirmatory_status"]),
        },
        "P1_PRESCRIBE_RAW_SCORE": {
            "definition": (
                "Spearman(seed3407 raw_log_prob, own PCA10 Pearson accuracy)>0"
            ),
            "point": float(p1["point_estimate"]),
            "passed": bool(p1["passed"]),
            "confirmatory_status": str(p1["confirmatory_status"]),
        },
        "systema": {
            "perturbed_reference": "train_condition_centroids_equal_weight",
            "centroid_accuracy_distance": "euclidean",
            "centroid_accuracy_definition": (
                "fraction_of_other_truth_centroids_strictly_farther_than_correct"
            ),
            "nearest_centroid_hit_also_reported": True,
            "tie_policy": "miss",
            "official_code_commit_semantics": SYSTEMA_CODE_COMMIT,
        },
        "split_half": "deterministic_reproducibility_benchmark_reference_not_upper_bound",
        "distribution_metrics_computed": False,
        "upstream_dois": list(UPSTREAM_DOIS),
        "paths": {
            "truth_profiles": "profiles/E165_TEST_TRUTH_PROFILES.csv.gz",
            "task_metrics": "tables/E165_PREDICTOR_TASK_METRICS.csv.gz",
            "hypotheses": "tables/E165_HYPOTHESIS_TESTS.csv",
            "native_associations": "tables/E165_NATIVE_SCORE_ASSOCIATIONS.csv",
            "secondary_prediction_contrasts": "tables/E165_SECONDARY_PREDICTION_CONTRASTS.csv",
            "bootstrap_replicates": "tables/E165_BOOTSTRAP_REPLICATES.csv.gz",
            "coverage": "tables/E165_COVERAGE_CURVES.csv",
            "split_half": "tables/E165_SPLIT_HALF_REFERENCE.csv",
            "access_ledger": "tables/E165_X_ACCESS_LEDGER.csv",
        },
        "artifact_sha256_before_interface": existing_hashes,
    }


def publish_release(
    preflight: dict[str, Any],
    event: dict[str, Any],
    runtime: list[dict[str, Any]],
    truth_tasks: pd.DataFrame,
    truth_profiles: pd.DataFrame,
    truth_pca: pd.DataFrame,
    task_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    hierarchy: pd.DataFrame,
    hypotheses: pd.DataFrame,
    associations: pd.DataFrame,
    prediction_contrasts: pd.DataFrame,
    replicates: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    logo: pd.DataFrame,
    coverage: pd.DataFrame,
    aurc: pd.DataFrame,
    split_half: pd.DataFrame,
    centroid: pd.DataFrame,
    ledger: list[dict[str, Any]],
) -> None:
    prepare_staging(event)
    atomic_bytes(STAGING / "README_先看这个.md", (
        "# E165\n\n先看 `reports/E165_REPORT.md`。test truth已不可逆解封；全部结果含失败与NA均保留。\n"
    ).encode("utf-8"))
    atomic_csv(STAGING / "tables/E165_TEST_TRUTH_TASKS.csv", truth_tasks)
    atomic_gzip_csv(STAGING / "profiles/E165_TEST_TRUTH_PROFILES.csv.gz", truth_profiles)
    atomic_csv(STAGING / "tables/E165_TEST_TRUTH_PCA10.csv", truth_pca)
    atomic_gzip_csv(STAGING / "tables/E165_PREDICTOR_TASK_METRICS.csv.gz", task_metrics)
    atomic_csv(STAGING / "tables/E165_PREDICTOR_SUMMARY.csv", summary)
    atomic_csv(STAGING / "tables/E165_BASELINE_HIERARCHY.csv", hierarchy)
    atomic_csv(STAGING / "tables/E165_HYPOTHESIS_TESTS.csv", hypotheses)
    atomic_csv(STAGING / "tables/E165_NATIVE_SCORE_ASSOCIATIONS.csv", associations)
    atomic_csv(
        STAGING / "tables/E165_SECONDARY_PREDICTION_CONTRASTS.csv",
        prediction_contrasts,
    )
    atomic_gzip_csv(STAGING / "tables/E165_BOOTSTRAP_REPLICATES.csv.gz", replicates)
    atomic_csv(STAGING / "tables/E165_BOOTSTRAP_SUMMARY.csv", bootstrap_summary)
    if logo.empty:
        logo = pd.DataFrame(columns=[
            "analysis_id", "seed", "endpoint", "removed_gene", "removed_tasks",
            "remaining_tasks", "rho", "estimable", "na_reason",
        ])
    atomic_csv(STAGING / "tables/E165_LOGO.csv", logo)
    atomic_csv(STAGING / "tables/E165_COVERAGE_CURVES.csv", coverage)
    atomic_csv(STAGING / "tables/E165_AURC_ERROR_CAPTURE.csv", aurc)
    atomic_csv(STAGING / "tables/E165_SPLIT_HALF_REFERENCE.csv", split_half)
    atomic_csv(STAGING / "tables/E165_CENTROID_ACCURACY.csv", centroid)
    atomic_csv(STAGING / "tables/E165_X_ACCESS_LEDGER.csv", pd.DataFrame(ledger))
    atomic_csv(STAGING / "tables/E165_INPUT_HASHES.csv", pd.DataFrame(preflight["input_rows"]))
    atomic_csv(STAGING / "tables/E165_RUNTIME_ENVIRONMENT.csv", pd.DataFrame(runtime))
    report = render_report(preflight, hierarchy, hypotheses, associations, split_half)
    atomic_bytes(STAGING / "reports/E165_REPORT.md", report.encode("utf-8"))
    atomic_bytes(
        STAGING / "figures/E165_SUMMARY_WHITE.svg",
        render_white_svg(hierarchy, hypotheses, preflight).encode("utf-8"),
    )

    before_interface = {
        path.relative_to(STAGING).as_posix(): sha256_file(path)
        for path in sorted(STAGING.rglob("*"))
        if path.is_file() and path.name not in {"RUN_STATUS.json", "RESULTS_SHA256.csv", "E165_E166_INTERFACE.json"}
    }
    interface = scientific_interface(preflight, event, hypotheses, before_interface)
    atomic_json(STAGING / "E165_E166_INTERFACE.json", interface)

    manifest_rows = []
    for path in sorted(STAGING.rglob("*")):
        if path.is_symlink():
            raise IntegrityFailure(f"Symlink in E165 staging: {path}")
        if path.is_file() and path.name not in {"RUN_STATUS.json", "RESULTS_SHA256.csv"}:
            manifest_rows.append({
                "relative_path": path.relative_to(STAGING).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    manifest = pd.DataFrame(manifest_rows)
    atomic_csv(STAGING / "RESULTS_SHA256.csv", manifest)
    manifest_sha = sha256_file(STAGING / "RESULTS_SHA256.csv")
    hypothesis_status = {
        str(row.hypothesis): {
            "point_estimate": (
                float(row.point_estimate)
                if np.isfinite(float(row.point_estimate)) else None
            ),
            "passed": bool(row.passed),
            "confirmatory_status": str(row.confirmatory_status),
        }
        for row in hypotheses.itertuples(index=False)
    }
    status = {
        "schema": "safeconf_e165_truth_unseal_evaluation_v1",
        "experiment": "E165_wessels_truth_unseal_evaluation",
        "phase": "complete_one_time_test_truth_evaluation",
        "completed_at": now(),
        "git_head": preflight["git_head"],
        "transaction_id": event["transaction_id"],
        "unseal_event_sha256": sha256_file(EVENT),
        "baseline_arm_evaluated": True,
        "prescribe_arm_evaluated": preflight["prescribe_arm_authorized"],
        "raw_h5ad_opened_after_event": True,
        "raw_sha256_verified_after_event": True,
        "test_X_rows_indexed_materialized_transformed": N_TEST_ROWS,
        "test_conditions": N_TEST_TASKS,
        "endogenous_columns_indexed": N_ENDOGENOUS,
        "selected_columns_after_transform": N_SELECTED,
        "train_rows_indexed": 0,
        "validation_rows_indexed": 0,
        "excluded_rows_indexed": 0,
        "engineered_guide_barcode_columns_indexed": 0,
        "distribution_metrics_computed": False,
        "hypotheses": hypothesis_status,
        "results_manifest_sha256": manifest_sha,
        "artifact_sha256": {
            row["relative_path"]: row["sha256"] for row in manifest_rows
        },
    }
    atomic_json(STAGING / "RUN_STATUS.json", status)
    observed = {
        path.relative_to(STAGING).as_posix()
        for path in STAGING.rglob("*") if path.is_file()
    }
    if observed != ALLOWLIST:
        raise IntegrityFailure(
            f"E165 release allowlist mismatch: missing={sorted(ALLOWLIST-observed)}, extra={sorted(observed-ALLOWLIST)}"
        )
    for path in STAGING.rglob("*"):
        if path.is_symlink():
            raise IntegrityFailure(f"Symlink found before publication: {path}")
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted(
        [path for path in STAGING.rglob("*") if path.is_dir()], key=lambda value: len(value.parts), reverse=True
    ):
        fsync_directory(directory)
    fsync_directory(STAGING)
    STAGING.replace(RELEASE)
    fsync_directory(OUT)
    return status


def validate_evaluation_outputs(
    preflight: dict[str, Any],
    truth: dict[str, np.ndarray],
    truth_tasks: pd.DataFrame,
    truth_profiles: pd.DataFrame,
    truth_pca: pd.DataFrame,
    task_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    hierarchy: pd.DataFrame,
    hypotheses: pd.DataFrame,
    associations: pd.DataFrame,
    prediction_contrasts: pd.DataFrame,
    replicates: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    logo: pd.DataFrame,
    coverage: pd.DataFrame,
    aurc: pd.DataFrame,
    split_half: pd.DataFrame,
    centroid: pd.DataFrame,
    ledger: list[dict[str, Any]],
) -> None:
    """Fail before publication if any frozen axis or output family is incomplete."""
    expected_predictors = list(BASELINE_ORDER)
    if preflight["prescribe_arm_authorized"]:
        expected_predictors.extend([f"prescribe_seed{seed}" for seed in SEEDS])
    if (
        truth["post"].shape != (N_TEST_TASKS, N_SELECTED)
        or truth["raw_effect"].shape != (N_TEST_TASKS, N_SELECTED)
        or truth["pca_post"].shape != (N_TEST_TASKS, N_SELECTED)
        or truth["pca_effect"].shape != (N_TEST_TASKS, N_SELECTED)
        or truth["pca"].shape != (N_TEST_TASKS, N_PCA)
        or int(np.sum(truth["n_cells"])) != N_TEST_ROWS
    ):
        raise IntegrityFailure("Frozen truth output shapes are incomplete")
    if (
        len(truth_tasks) != N_TEST_TASKS
        or truth_tasks["condition"].astype(str).tolist() != preflight["test_order"]
        or int(truth_tasks["n_test_cells"].sum()) != N_TEST_ROWS
    ):
        raise IntegrityFailure("Truth task table changed its frozen axis")
    profile_kinds = [
        "raw_normalized_post",
        "raw_normalized_effect_vs_train_control",
        "pca10_reconstructed_post",
        "pca10_reconstructed_effect_vs_train_control",
    ]
    if (
        truth_profiles.shape != (4 * N_TEST_TASKS, N_SELECTED + 2)
        or truth_profiles.columns[:2].tolist() != ["profile_kind", "condition"]
        or truth_profiles["profile_kind"].drop_duplicates().tolist() != profile_kinds
        or any(
            truth_profiles.loc[
                truth_profiles["profile_kind"].eq(kind), "condition"
            ].astype(str).tolist()
            != preflight["test_order"]
            for kind in profile_kinds
        )
    ):
        raise IntegrityFailure("Truth profile publication table is malformed")
    if (
        truth_pca.shape != (N_TEST_TASKS, 1 + 2 * N_PCA)
        or truth_pca["condition"].astype(str).tolist() != preflight["test_order"]
    ):
        raise IntegrityFailure("Truth PCA publication table is malformed")
    expected_metric_pairs = [
        (predictor, condition)
        for predictor in expected_predictors
        for condition in preflight["test_order"]
    ]
    observed_metric_pairs = list(
        zip(task_metrics["predictor"].astype(str), task_metrics["condition"].astype(str))
    )
    if observed_metric_pairs != expected_metric_pairs:
        raise IntegrityFailure("Predictor/task metric order is incomplete")
    required_metric_columns = {
        "pca10_pearson", "pca10_cosine", "pca10_direction_accuracy",
        "pca10_mse", "pca10_rmse", "raw_pearson", "raw_cosine",
        "raw_direction_accuracy", "raw_mse", "raw_rmse",
        "pca10_top20_rmse", "raw_top20_rmse",
        "systema_perturbed_reference_pearson",
        "systema_centroid_accuracy", "systema_nearest_centroid_hit",
    }
    if not required_metric_columns.issubset(task_metrics.columns):
        raise IntegrityFailure("Task metric table lacks a frozen endpoint")
    if not np.isfinite(
        task_metrics[["pca10_rmse", "raw_rmse", "systema_centroid_accuracy"]]
        .to_numpy(float)
    ).all():
        raise IntegrityFailure("A mandatory finite error/centroid endpoint is non-finite")
    if summary["predictor"].astype(str).tolist() != expected_predictors:
        raise IntegrityFailure("Predictor summary order changed")
    if hierarchy["predictor"].astype(str).tolist() != list(BASELINE_ORDER):
        raise IntegrityFailure("Baseline hierarchy order changed")
    if hypotheses["hypothesis"].astype(str).tolist() != [
        "H1", "H2", "P1_PRESCRIBE_RAW_SCORE"
    ]:
        raise IntegrityFailure("Confirmatory hypothesis table is incomplete")
    if prediction_contrasts.shape[0] != (len(BASELINE_ORDER) - 1) * 10:
        raise IntegrityFailure("Secondary prediction contrast family is incomplete")
    expected_associations = len(RISK_SCORE_COLUMNS) * 8
    if preflight["prescribe_arm_authorized"]:
        expected_associations += len(SEEDS) * 3 * 8
    else:
        expected_associations += 1
    if len(associations) != expected_associations:
        raise IntegrityFailure("Risk/native association family is incomplete")
    if replicates.empty or bootstrap_summary.empty or logo.empty:
        raise IntegrityFailure("Bootstrap or LOGO evidence is absent")
    expected_coverage_scores = len(RISK_SCORE_COLUMNS)
    if preflight["prescribe_arm_authorized"]:
        expected_coverage_scores += len(SEEDS) + 2
    if len(coverage) != expected_coverage_scores * 2 * len(COVERAGES):
        raise IntegrityFailure("Selective-coverage grid is incomplete")
    if len(aurc) != expected_coverage_scores * 2:
        raise IntegrityFailure("AURC/error-capture summary is incomplete")
    if len(split_half) != N_TEST_TASKS or len(centroid) != len(expected_metric_pairs):
        raise IntegrityFailure("Split-half or centroid output is incomplete")
    if len(ledger) != 2 or int(ledger[-1].get("rows_indexed", -1)) != N_TEST_ROWS:
        raise IntegrityFailure("Raw-X access ledger is incomplete")


def verify_published_release(status: dict[str, Any]) -> None:
    if not RELEASE.is_dir() or RELEASE.is_symlink():
        raise IntegrityFailure("Published E165 release is absent or invalid")
    observed = {
        path.relative_to(RELEASE).as_posix()
        for path in RELEASE.rglob("*") if path.is_file()
    }
    if observed != ALLOWLIST or any(path.is_symlink() for path in RELEASE.rglob("*")):
        raise IntegrityFailure("Published E165 allowlist/symlink audit failed")
    manifest = pd.read_csv(RELEASE / "RESULTS_SHA256.csv")
    if list(manifest.columns) != ["relative_path", "bytes", "sha256"]:
        raise IntegrityFailure("Published E165 manifest schema changed")
    if sha256_file(RELEASE / "RESULTS_SHA256.csv") != status["results_manifest_sha256"]:
        raise IntegrityFailure("Published E165 manifest hash differs from status")
    for row in manifest.itertuples(index=False):
        path = RELEASE / str(row.relative_path)
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != int(row.bytes)
            or sha256_file(path) != str(row.sha256)
        ):
            raise IntegrityFailure(f"Published E165 artifact mismatch: {path}")
    if load_json(RELEASE / "RUN_STATUS.json") != status:
        raise IntegrityFailure("Published E165 status differs from in-memory status")


def record_failure(error: BaseException) -> None:
    """Preserve failures only after the irreversible event exists."""
    if not EVENT.exists():
        return
    FAILURES.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    path = FAILURES / f"E165_FAILURE_{stamp}_{uuid.uuid4().hex[:8]}.json"
    payload = {
        "schema": "safeconf_e165_failure_v1",
        "experiment": "E165_wessels_truth_unseal_evaluation",
        "failed_at": now(),
        "error_type": type(error).__name__,
        "error": repr(error),
        "traceback": traceback.format_exc(),
        "unseal_event_exists": True,
        "unseal_event_sha256": sha256_file(EVENT),
        "test_truth_is_permanently_unsealed": True,
        "staging_preserved": STAGING.exists(),
        "release_exists": RELEASE.exists(),
    }
    atomic_json(path, payload)


def preflight_output(
    preflight: dict[str, Any],
    frozen: dict[str, Any],
    baseline_profiles: dict[str, np.ndarray],
    risk: pd.DataFrame,
    native: dict[int, pd.DataFrame],
) -> dict[str, Any]:
    return {
        "schema": "safeconf_e165_preflight_v1",
        "experiment": "E165_wessels_truth_unseal_evaluation",
        "phase": "all_committed_pretruth_gates_passed_raw_not_opened_or_hashed",
        "git_head": preflight["git_head"],
        "gate_fingerprint_sha256": preflight["gate_fingerprint_sha256"],
        "baseline_arm_authorized": True,
        "prescribe_arm_authorized": preflight["prescribe_arm_authorized"],
        "baseline_predictors": list(baseline_profiles),
        "risk_rows": len(risk),
        "native_seeds": sorted(native),
        "selected_genes": len(frozen["selected"]),
        "endogenous_genes": len(frozen["endogenous"]),
        "test_conditions": len(preflight["test_order"]),
        "expected_test_rows": N_TEST_ROWS,
        "raw_stat_checked": True,
        "raw_file_hashed": False,
        "raw_h5ad_opened": False,
        "test_X_accessed": False,
        "test_truth_accessed": False,
        "unseal_event_exists_before_formal": EVENT.exists(),
    }


def formal(
    preflight: dict[str, Any],
    frozen: dict[str, Any],
    baseline_profiles: dict[str, np.ndarray],
    risk: pd.DataFrame,
    native: dict[int, pd.DataFrame],
) -> dict[str, Any]:
    if RELEASE.exists():
        raise FileExistsError("E165 release already exists; refusing a second truth evaluation")
    event = write_or_verify_unseal_event(preflight)
    verify_unseal_event_lock(
        preflight, transaction_id=str(event["transaction_id"])
    )
    cells, conditions, obs_names, ledger = read_test_truth_once(
        preflight, frozen, event
    )
    truth = task_truth(cells, conditions, preflight["test_order"], frozen)
    task_metrics, centroid, _predictor_posts = evaluate_predictors(
        preflight, frozen, truth, baseline_profiles, native
    )
    summary = summarize_predictors(task_metrics)
    hierarchy = baseline_hierarchy(task_metrics)
    prediction_contrasts = secondary_prediction_contrasts(task_metrics)
    (
        hypotheses,
        associations,
        replicates,
        bootstrap_summary,
        logo,
    ) = hypothesis_and_native_statistics(preflight, task_metrics, risk, native)
    split_half = split_half_reference(
        cells, conditions, obs_names, preflight, frozen, truth
    )
    coverage, aurc = coverage_analysis(preflight, task_metrics, risk, native)
    truth_tasks, truth_profiles, truth_pca = truth_tables(
        preflight, frozen, truth
    )
    validate_evaluation_outputs(
        preflight,
        truth,
        truth_tasks,
        truth_profiles,
        truth_pca,
        task_metrics,
        summary,
        hierarchy,
        hypotheses,
        associations,
        prediction_contrasts,
        replicates,
        bootstrap_summary,
        logo,
        coverage,
        aurc,
        split_half,
        centroid,
        ledger,
    )
    status = publish_release(
        preflight,
        event,
        preflight["runtime"],
        truth_tasks,
        truth_profiles,
        truth_pca,
        task_metrics,
        summary,
        hierarchy,
        hypotheses,
        associations,
        prediction_contrasts,
        replicates,
        bootstrap_summary,
        logo,
        coverage,
        aurc,
        split_half,
        centroid,
        ledger,
    )
    verify_published_release(status)
    return status


def main() -> None:
    args = parse_args()
    try:
        preflight = metadata_preflight()
        # Parse and shape-check every non-truth asset before the irreversible
        # event.  This does not open or hash the raw Wessels H5AD.
        frozen = load_frozen_pca_control(preflight)
        baseline_profiles, risk, native = load_locked_predictions(preflight, frozen)
        result = (
            preflight_output(preflight, frozen, baseline_profiles, risk, native)
            if args.mode == "preflight"
            else formal(preflight, frozen, baseline_profiles, risk, native)
        )
    except BaseException as error:
        if args.mode == "formal":
            record_failure(error)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
