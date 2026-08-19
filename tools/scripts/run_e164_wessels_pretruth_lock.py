#!/usr/bin/env python3
"""E164: freeze Wessels baseline and optional PRESCRIBE arms before test truth.

The E162b/Systema baseline arm is independent of the E163 PRESCRIBE futility
diagnostic.  A terminal E163 release always permits the baseline arm.  Only an
E163 interface with ``validation_gate_passed=true`` permits the irreversible
48-label forward through the three E162 attempt_002 checkpoints.

Neither mode opens the raw Wessels H5AD.  ``preflight`` performs hashes and
metadata checks only.  ``formal`` reads exactly the 11,779-row train prefix of
the E161 development H5AD for the condition-balanced Systema reference.  Test,
validation and excluded expression rows remain untouched.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
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
from types import ModuleType, SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E164_wessels_pretruth_lock_20260715"
CONTRACT = OUT / "ANALYSIS_CONTRACT.md"
STAGING = OUT / ".release.staging"
RELEASE = OUT / "release"
FAILURES = OUT / "failures"
QUERY_EVENT = OUT / "TEST_LABEL_QUERY_EVENT.json"

E160 = ROOT / "docs/实验结果/E160_wessels_combination_contract_20260714"
E160_SPLIT = E160 / "freeze/manifests/E160_set2conditions.json"
E160_CONDITION_AUDIT = E160 / "freeze/manifests/E160_CONDITION_AUDIT.csv"
E161_ASSET = Path("/home/yyf/data/safeconf_e161_prescribe/wessels_e160")
E161_INTERFACE = E161_ASSET / "E161_E162_INTERFACE.json"
E161_H5AD = E161_ASSET / "perturb_processed.h5ad"
E161_PCA = E161_ASSET / "TRAIN_ONLY_PCA_MODEL.npz"
E161_CONTROL = E161_ASSET / "TRAIN_ONLY_CONTROL_PRIOR.npz"
E161_GENES = E161_ASSET / "SELECTED_GENE_AXIS.txt"

E162_RUNNER = ROOT / "tools/scripts/run_e162_wessels_prescribe_native.py"
E162_ATTEMPT = (
    ROOT
    / "docs/实验结果/E162_wessels_prescribe_native_20260714/attempt_002"
)
E162_STATUS = E162_ATTEMPT / "RUN_STATUS.json"

E162B_RELEASE = (
    ROOT
    / "docs/实验结果/E162b_wessels_label_only_baselines_20260715/release"
)
E162B_STATUS = E162B_RELEASE / "RUN_STATUS.json"
E162B_INTERFACE = E162B_RELEASE / "E162b_E163_INTERFACE.json"
E162B_MANIFEST = E162B_RELEASE / "RESULTS_SHA256.csv"

E163_RELEASE = (
    ROOT
    / "docs/实验结果/E163_wessels_validation_raw_futility_20260715/release"
)
E163_STATUS = E163_RELEASE / "RUN_STATUS.json"
E163_INTERFACE = E163_RELEASE / "E163_E164_INTERFACE.json"
E163_MANIFEST = E163_RELEASE / "RESULTS_SHA256.csv"
E163_AUTHORIZATION = E163_RELEASE / "E163_AUTHORIZATION_GATE.json"

EXPECTED_PYTHON = Path("/home/yyf/.conda/envs/prescribe_env/bin/python")
EXPECTED_PYTHON_VERSION = "3.9.25"
EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "scipy": "1.13.1",
    "h5py": "3.14.0",
}

# Frozen after the actual E163 terminal release at Git f541b5e.
EXPECTED_SHA256 = {
    "E163_status": "167e40294dede72d9b9862738f01af6666a8b43230e9e57f8c6892b6e885a01c",
    "E163_interface": "ac8eb69ce82b9b1b6f085a112860fb7e9e0b0e3eeb19b98be2c04c8e7ae016ea",
    "E163_manifest": "751c7c4cc50ed771b285d353b12a9193799b48b2e3917988c6d7afeb8bce790e",
    "E163_authorization": "e9f0ef8f924acf23d4369a9b7ca9d2f1827ef4b5219dadc1a9ee49aad028b374",
    "E162_status": "497eb2156effa3bf5f11392006f490ed9435b081ee5e64b2231e8bbb41df2e5c",
    "E162b_status": "96a0b720d1ad5349346e52b1f116f6cbe1bc28b11e104fc1ad4dd78206e8bd40",
    "E162b_interface": "56d7e34191ba23b7e2048b44f7995601769cb2804253411c3dc1d98d00680412",
    "E162b_manifest": "896fccc23b822a4026f672f0321411957bb8768200aa9552f069ea7669b25c7e",
    "E160_split": "dedd26e86b022bf5ffb86d485b3a33276c0c35cf959c3e2881034be8c636b514",
    "E160_condition_audit": "5c87b0ab01a3735a05178f8ebf48ebda0b4328d955e199cf8db85de05f3b8125",
    "E161_interface": "b681160e2f88c500fb3004bc9fb3fa5400cccbb9e8d537bde8db043e11254ed7",
    "E161_genes": "5fbe6a1d80d163a63576552f2bc74cfd9416e65e706877cefe4ad05b2fb3a2cf",
    "E161_pca": "b1be7cfe03300d0c8352f7265f65496baaa4454edecf95bcd421961126e6a12f",
    "E161_control": "1e53d388a895feae2f24aa74a99a80287ec63fbc8511cc46890b331aff05acd6",
    "E161_h5ad": "2921f1c8fa7e6415725380e319a5092993e725ec5cb596f225af19811e82fd40",
}

SEEDS = (3407, 3408, 3409)
MAIN_SEED = 3407
N_TRAIN = 11_779
N_VALIDATION = 5_102
N_DEV = 16_881
N_TEST = 48
N_SELECTED = 2_023
N_PCA = 10
N_TRAIN_CONDITIONS = 72
N_NONCONTROL_TRAIN_CONDITIONS = 71
RAW_UNIQUE_MINIMUM = 24
RAW_STD_MINIMUM = 1e-6
ESTIMABILITY_STD_MINIMUM = 1e-12

E162B_BASELINE_ORDER = (
    "control",
    "cell_weighted_perturbed_mean",
    "matching_single_mean",
    "single_additive",
)
BASELINE_ORDER = (
    "control_no_change",
    "cell_weighted_perturbed_mean",
    "condition_balanced_perturbed_mean",
    "matching_single_mean",
    "single_additive",
)

BASE_RELEASE_FILES = {
    ".E164_TRANSACTION.json",
    "RUN_STATUS.json",
    "README_先看这个.md",
    "RESULTS_SHA256.csv",
    "E164_E165_INTERFACE.json",
    "E165_EVALUATION_SPEC.json",
    "reports/E164_REPORT.md",
    "profiles/E164_BASELINE_TEST_POST_PROFILES.csv.gz",
    "profiles/E164_BASELINE_TEST_PCA10_COORDINATES.csv",
    "profiles/E164_SYSTEMA_CONDITION_BALANCED_PERTURBED_MEAN.csv.gz",
    "profiles/E164_SYSTEMA_CONDITION_BALANCED_PERTURBED_MEAN_PCA10.csv",
    "tables/E164_RISK_WIDE.csv",
    "tables/E164_SYSTEMA_REFERENCE_AUDIT.csv",
    "tables/E164_X_ACCESS_LEDGER.csv",
    "tables/E164_SOURCE_HASHES.csv",
    "tables/E164_RUNTIME_ENVIRONMENT.csv",
}
PRESCRIBE_RELEASE_FILES = {
    "tables/E164_ESTIMABILITY.csv",
    "tables/E164_DEGENERACY_AUDIT.csv",
    *{
        f"tables/E164_PRESCRIBE_TEST_LABEL_ONLY_SCORES_SEED{seed}.csv"
        for seed in SEEDS
    },
    *{
        f"tables/E164_RAW_SCORE_GATE_SEED{seed}.json" for seed in SEEDS
    },
    *{
        f"profiles/E164_PRESCRIBE_TEST_POST_PROFILES_SEED{seed}.csv.gz"
        for seed in SEEDS
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preflight", "formal"))
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser.parse_args()


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    regular_file(path, "JSON input")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def regular_file(path: Path, role: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{role} must be a regular non-symlink file: {path}")


def require_hash(path: Path, expected: str, role: str) -> dict[str, Any]:
    regular_file(path, role)
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{role} SHA256 changed: {observed} != {expected}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def git_blob_gate(path: Path, *, require: bool) -> dict[str, Any]:
    regular_file(path, "Git-gated source")
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    try:
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as exc:
        if require:
            raise RuntimeError(f"Required source is not committed at HEAD: {relative}") from exc
        return {
            "path": relative,
            "working_sha256": sha256_file(path),
            "committed": False,
            "matches_head_blob": False,
        }
    working = path.read_bytes()
    matches = working == committed
    if require and not matches:
        raise RuntimeError(f"Working source differs from HEAD blob: {relative}")
    return {
        "path": relative,
        "working_sha256": sha256_bytes(working),
        "committed_sha256": sha256_bytes(committed),
        "committed": True,
        "matches_head_blob": matches,
    }


def safe_release_path(release: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise RuntimeError(f"Unsafe release-manifest path: {relative}")
    candidate = release / value
    try:
        candidate.resolve().relative_to(release.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Release-manifest path escapes root: {relative}") from exc
    return candidate


def verify_release_manifest(release: Path, manifest_path: Path) -> dict[str, dict[str, Any]]:
    regular_file(manifest_path, "release manifest")
    frame = pd.read_csv(manifest_path)
    if set(frame.columns) != {"relative_path", "bytes", "sha256"}:
        raise RuntimeError(f"Unexpected manifest schema: {manifest_path}")
    if frame["relative_path"].astype(str).duplicated().any():
        raise RuntimeError(f"Duplicate paths in manifest: {manifest_path}")
    records: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        relative = str(row["relative_path"])
        path = safe_release_path(release, relative)
        regular_file(path, f"manifest payload {relative}")
        observed = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if (
            observed["bytes"] != int(row["bytes"])
            or observed["sha256"] != str(row["sha256"])
        ):
            raise RuntimeError(f"Manifest payload changed: {release / relative}")
        records[relative] = observed
    return records


def runtime_gate() -> list[dict[str, Any]]:
    executable = Path(sys.executable).resolve()
    if executable != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"E164 requires {EXPECTED_PYTHON}; observed {executable}")
    python = platform.python_version()
    if python != EXPECTED_PYTHON_VERSION:
        raise RuntimeError(f"Python changed: {python} != {EXPECTED_PYTHON_VERSION}")
    rows = [
        {
            "component": "python",
            "expected_version": EXPECTED_PYTHON_VERSION,
            "observed_version": python,
            "executable": str(executable),
            "gate_passed": True,
        }
    ]
    for name, expected in EXPECTED_VERSIONS.items():
        observed = distribution_version(name)
        if observed != expected:
            raise RuntimeError(f"Dependency changed: {name} {observed} != {expected}")
        rows.append(
            {
                "component": name,
                "expected_version": expected,
                "observed_version": observed,
                "executable": str(executable),
                "gate_passed": True,
            }
        )
    return rows


def verify_e163() -> dict[str, Any]:
    fixed = {
        "status": require_hash(E163_STATUS, EXPECTED_SHA256["E163_status"], "E163 status"),
        "interface": require_hash(
            E163_INTERFACE, EXPECTED_SHA256["E163_interface"], "E163 interface"
        ),
        "manifest": require_hash(
            E163_MANIFEST, EXPECTED_SHA256["E163_manifest"], "E163 manifest"
        ),
        "authorization": require_hash(
            E163_AUTHORIZATION,
            EXPECTED_SHA256["E163_authorization"],
            "E163 authorization gate",
        ),
    }
    records = verify_release_manifest(E163_RELEASE, E163_MANIFEST)
    for required in (
        "RUN_STATUS.json",
        "E163_E164_INTERFACE.json",
        "E163_AUTHORIZATION_GATE.json",
    ):
        if required not in records:
            raise RuntimeError(f"E163 manifest omits {required}")
    status = load_json(E163_STATUS)
    interface = load_json(E163_INTERFACE)
    if status.get("phase") != (
        "complete_validation_only_futility_diagnostic_no_test_label_or_X_access"
    ):
        raise RuntimeError("E163 is not in its fixed terminal validation-only phase")
    if interface.get("schema") != "safeconf_e163_to_e164_v1":
        raise RuntimeError("E163 interface schema changed")
    authorization = interface.get("validation_gate_passed")
    if not isinstance(authorization, bool):
        raise RuntimeError("E163 validation_gate_passed is not a frozen boolean")
    if interface.get("authorize_future_test_label_lock") is not authorization:
        raise RuntimeError("E163 authorization booleans disagree")
    if status.get("authorization_gate", {}).get(
        "authorize_future_test_label_lock"
    ) is not authorization:
        raise RuntimeError("E163 status/interface authorization disagree")
    false_checks = {
        "status.test_label_queried": status.get("test_label_queried"),
        "status.test_X_accessed": status.get("test_X_accessed"),
        "status.test_truth_accessed": status.get("test_truth_accessed"),
        "status.test_endpoint_computed": status.get("test_endpoint_computed"),
        "interface.test_label_queried": interface.get("test_label_queried"),
        "interface.test_X_accessed": interface.get("test_X_accessed"),
        "interface.test_truth_accessed": interface.get("test_truth_accessed"),
    }
    if any(value is not False for value in false_checks.values()):
        raise RuntimeError(f"E163 test seal changed: {false_checks}")
    return {
        "status": status,
        "interface": interface,
        "validation_gate_passed": authorization,
        "manifest_records": records,
        "fixed_hashes": fixed,
    }


def verify_e162_failure() -> dict[str, Any]:
    status_hash = require_hash(
        E162_STATUS, EXPECTED_SHA256["E162_status"], "E162 attempt_002 status"
    )
    status = load_json(E162_STATUS)
    if status.get("phase") != "failed_main_validation_nondegeneracy_gate_no_test_label_query":
        raise RuntimeError("E162 failure phase was changed or rewritten")
    fixed_false = (
        "test_label_queries_started",
        "test_X_accessed",
        "test_truth_accessed",
        "test_endpoint_computed",
    )
    if any(status.get(key) is not False for key in fixed_false):
        raise RuntimeError("E162 no-test-query failure boundary changed")
    if int(status.get("n_test_graphs", -1)) != 0:
        raise RuntimeError("E162 unexpectedly contains test graphs")
    if (E162_ATTEMPT / "TEST_LABEL_QUERY_EVENT.json").exists():
        raise RuntimeError("E162 has an unexpected test-label query event")

    gates = status.get("validation_nondegeneracy_gates")
    if not isinstance(gates, dict) or set(gates) != {str(seed) for seed in SEEDS}:
        raise RuntimeError("E162 validation gate map changed")
    checkpoints: dict[str, Any] = {}
    for seed in SEEDS:
        key = str(seed)
        gate = gates[key]
        if (
            gate.get("passed") is not False
            or gate.get("n_rows") != 24
            or gate.get("raw_log_prob_all_finite") is not True
            or gate.get("raw_log_prob_exact_unique") != 24
            or float(gate.get("raw_log_prob_sample_std_ddof1", 0.0)) <= RAW_STD_MINIMUM
            or gate.get("prediction_all_finite") is not True
            or gate.get("prediction_exact_unique_vectors") != 1
            or gate.get("prediction_any_coordinate_std_gt_1e_minus_6") is not False
            or [float(value) for value in gate.get("prediction_coordinate_sample_std_ddof1", [])]
            != [0.0] * N_PCA
        ):
            raise RuntimeError(f"E162 seed {seed} frozen validation failure changed")
        seed_dir = E162_ATTEMPT / f"seed_{seed}"
        seed_status_path = seed_dir / "RUN_STATUS.json"
        seed_status = load_json(seed_status_path)
        if (
            seed_status.get("phase") != "checkpoint_locked_before_any_label_only_forward"
            or seed_status.get("test_label_queried") is not False
            or seed_status.get("test_X_accessed") is not False
            or seed_status.get("test_truth_accessed") is not False
            or seed_status.get("validation_nondegeneracy_gate") != gate
        ):
            raise RuntimeError(f"E162 seed {seed} checkpoint status changed")
        ready_path = seed_dir / "CHECKPOINT_READY.json"
        regular_file(ready_path, f"E162 seed {seed} checkpoint-ready record")
        checkpoint_audit = seed_status.get("checkpoint_audit", {})
        if (
            not isinstance(checkpoint_audit, dict)
            or sha256_file(ready_path)
            != checkpoint_audit.get("checkpoint_ready_sha256")
        ):
            raise RuntimeError(f"E162 seed {seed} checkpoint-ready hash changed")
        locked = seed_status.get("locked_slim_checkpoint")
        if not isinstance(locked, dict):
            raise RuntimeError(f"E162 seed {seed} locked checkpoint audit is absent")
        checkpoint = Path(str(locked.get("path")))
        regular_file(checkpoint, f"E162 seed {seed} locked checkpoint")
        if (
            checkpoint.stat().st_size != int(locked.get("bytes", -1))
            or sha256_file(checkpoint) != locked.get("sha256")
        ):
            raise RuntimeError(f"E162 seed {seed} locked checkpoint changed")
        top_audit = status.get("checkpoint_audits", {}).get(key, {}).get(
            "locked_slim_checkpoint", {}
        )
        if top_audit.get("sha256") != locked.get("sha256"):
            raise RuntimeError(f"E162 seed {seed} checkpoint audits disagree")
        validation_table = (
            E162_ATTEMPT
            / f"locked/E162_VALIDATION_LABEL_ONLY_SCORES_SEED{seed}.csv"
        )
        validation_gate = (
            E162_ATTEMPT
            / f"locked/E162_VALIDATION_NONDEGENERACY_GATE_SEED{seed}.json"
        )
        if sha256_file(validation_table) != seed_status.get(
            "validation_label_only_scores_sha256"
        ):
            raise RuntimeError(f"E162 seed {seed} validation-score hash changed")
        if load_json(validation_gate) != gate:
            raise RuntimeError(f"E162 seed {seed} validation-gate artifact changed")
        checkpoints[key] = {
            "seed_status_path": str(seed_status_path),
            "seed_status_sha256": sha256_file(seed_status_path),
            "checkpoint_ready_path": str(ready_path),
            "checkpoint_ready_sha256": sha256_file(ready_path),
            "locked_checkpoint": {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
            },
            "validation_gate": gate,
        }
    return {"status": status, "status_hash": status_hash, "checkpoints": checkpoints}


def verify_e162b() -> dict[str, Any]:
    fixed = {
        "status": require_hash(
            E162B_STATUS, EXPECTED_SHA256["E162b_status"], "E162b status"
        ),
        "interface": require_hash(
            E162B_INTERFACE,
            EXPECTED_SHA256["E162b_interface"],
            "E162b interface",
        ),
        "manifest": require_hash(
            E162B_MANIFEST, EXPECTED_SHA256["E162b_manifest"], "E162b manifest"
        ),
    }
    status = load_json(E162B_STATUS)
    interface = load_json(E162B_INTERFACE)
    records = verify_release_manifest(E162B_RELEASE, E162B_MANIFEST)
    if status.get("phase") != "complete_pretest_label_only_baselines_no_val_or_test_X":
        raise RuntimeError("E162b release phase changed")
    if interface.get("schema") != "safeconf_e162b_to_e163_v1":
        raise RuntimeError("E162b interface schema changed")
    if interface.get("baseline_order") != list(E162B_BASELINE_ORDER):
        raise RuntimeError("E162b baseline order changed")
    if (
        interface.get("n_test_labels") != N_TEST
        or interface.get("n_selected_genes") != N_SELECTED
        or interface.get("n_pca_coordinates") != N_PCA
    ):
        raise RuntimeError("E162b dimensions changed")
    boundary = interface.get("access_boundary", {})
    if (
        boundary.get("raw_file_opened") is not False
        or boundary.get("validation_X_rows_indexed_materialized_transformed") != 0
        or boundary.get("test_X_rows_indexed_materialized_transformed") != 0
        or boundary.get("excluded_X_rows_indexed_materialized_transformed") != 0
        or boundary.get("test_cell_count_truth_effect_error_or_DE_used") is not False
    ):
        raise RuntimeError("E162b access boundary changed")
    status_artifacts = status.get("artifact_sha256")
    manifest_artifacts = {key: row["sha256"] for key, row in records.items()}
    if status.get("results_manifest_sha256") != EXPECTED_SHA256["E162b_manifest"]:
        raise RuntimeError("E162b status/manifest hash disagree")
    if status_artifacts != manifest_artifacts:
        raise RuntimeError("E162b status artifact map differs from manifest")
    required_paths = interface.get("paths", {})
    for key in ("post_profiles", "pca_coordinates", "tasks", "risk_wide"):
        relative = required_paths.get(key)
        if relative not in records:
            raise RuntimeError(f"E162b interface path is not manifest-bound: {key}")
    return {
        "status": status,
        "interface": interface,
        "manifest_records": records,
        "fixed_hashes": fixed,
    }


def import_e162_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("safeconf_e162_for_e164", E162_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the frozen E162 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def common_preflight(*, formal: bool) -> dict[str, Any]:
    runtime = runtime_gate()
    head = git_head()
    own_git = {
        "runner": git_blob_gate(RUNNER, require=formal),
        "contract": git_blob_gate(CONTRACT, require=formal),
    }
    e163 = verify_e163()
    e162 = verify_e162_failure()
    e162b = verify_e162b()
    require_hash(E160_SPLIT, EXPECTED_SHA256["E160_split"], "E160 split")
    require_hash(
        E160_CONDITION_AUDIT,
        EXPECTED_SHA256["E160_condition_audit"],
        "E160 condition audit",
    )
    require_hash(E161_INTERFACE, EXPECTED_SHA256["E161_interface"], "E161 interface")
    require_hash(E161_GENES, EXPECTED_SHA256["E161_genes"], "E161 gene axis")
    require_hash(E161_PCA, EXPECTED_SHA256["E161_pca"], "E161 PCA model")
    require_hash(E161_CONTROL, EXPECTED_SHA256["E161_control"], "E161 control prior")
    regular_file(E161_H5AD, "E161 development H5AD")
    if E161_H5AD.stat().st_size <= 0:
        raise RuntimeError("E161 development H5AD is empty")
    if formal and sha256_file(E161_H5AD) != EXPECTED_SHA256["E161_h5ad"]:
        raise RuntimeError("E161 development H5AD hash changed")

    e162_module = import_e162_runner()
    native_gate = e162_module.metadata_preflight(formal=formal)
    test_labels = [str(value) for value in native_gate["split"]["test"]]
    if len(test_labels) != N_TEST or len(set(test_labels)) != N_TEST:
        raise RuntimeError("E160 test-label axis changed")
    if test_labels != [str(value) for value in e162b["interface"]["test_label_order"]]:
        raise RuntimeError("E160 and E162b test-label order disagree")
    if sha256_file(E160_SPLIT) != native_gate["e160_split_sha256"]:
        raise RuntimeError("E160 split differs from the native E162 gate")
    if sha256_file(E161_INTERFACE) != native_gate["e161_interface_sha256"]:
        raise RuntimeError("E161 interface differs from the native E162 gate")
    train_single_genes = {
        condition.split("+")[0]
        for condition in native_gate["split"]["train"]
        if condition != "ctrl"
        and len(condition.split("+")) == 2
        and "ctrl" in condition.split("+")
    }
    test_components = {
        gene for condition in test_labels for gene in condition.split("+")
    }
    if len(train_single_genes) != 27 or not test_components.issubset(train_single_genes):
        raise RuntimeError("E160 seen-single/unseen-double semantics changed")
    condition_audit = pd.read_csv(E160_CONDITION_AUDIT)
    test_audit = condition_audit[
        condition_audit["split"].astype(str).eq("test")
    ].copy()
    if (
        test_audit["canonical_condition"].astype(str).tolist() != test_labels
        or len(test_audit) != N_TEST
        or test_audit["n_perturbation_genes"].astype(int).tolist() != [2] * N_TEST
        or test_audit["guide_class"].astype(str).tolist() != ["Dual"] * N_TEST
        or int(test_audit["n_cells_obs_metadata"].astype(int).sum()) != 9_902
    ):
        raise RuntimeError("E160 frozen 48-pair/9,902-cell metadata audit changed")
    return {
        "head": head,
        "runtime": runtime,
        "own_git": own_git,
        "e163": e163,
        "e162": e162,
        "e162b": e162b,
        "e162_module": e162_module,
        "native_gate": native_gate,
        "test_labels": test_labels,
        "test_label_order_sha256": sha256_bytes(
            ("\n".join(test_labels) + "\n").encode("utf-8")
        ),
        "perturbation_genes": sorted(train_single_genes),
        "frozen_test_cell_count_metadata_only": 9_902,
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
        raise RuntimeError(f"Invalid category code in {group.name}/{name}")
    return categories[codes]


def read_h5_index(group: h5py.Group) -> np.ndarray:
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    return read_h5_column(group, str(key))


def load_train_expression_only(expected_genes: list[str]) -> tuple[sp.csr_matrix, np.ndarray, dict[str, Any]]:
    """Materialize only the frozen 11,779-row train prefix of E161 CSR X."""

    before = E161_H5AD.stat()
    with h5py.File(E161_H5AD, "r") as handle:
        if not {"obs", "var", "X"}.issubset(handle.keys()):
            raise RuntimeError("Malformed E161 development H5AD")
        roles = read_h5_column(handle["obs"], "e161_split").astype(str)
        conditions = read_h5_column(handle["obs"], "condition").astype(str)
        genes = read_h5_index(handle["var"]).astype(str).tolist()
        if len(roles) != N_DEV or len(conditions) != N_DEV or genes != expected_genes:
            raise RuntimeError("E161 development metadata axis changed")
        train_rows = np.flatnonzero(roles == "train")
        validation_rows = np.flatnonzero(roles == "val")
        if not np.array_equal(train_rows, np.arange(N_TRAIN, dtype=np.int64)):
            raise RuntimeError("E161 train rows are not the frozen contiguous prefix")
        if not np.array_equal(
            validation_rows, np.arange(N_TRAIN, N_DEV, dtype=np.int64)
        ):
            raise RuntimeError("E161 validation rows are not the frozen suffix")

        node = handle["X"]
        if isinstance(node, h5py.Dataset):
            if tuple(node.shape) != (N_DEV, N_SELECTED):
                raise RuntimeError("E161 dense X shape changed")
            train_x = sp.csr_matrix(np.asarray(node[:N_TRAIN, :], dtype=np.float64))
            storage = "dense_train_prefix_only"
        elif isinstance(node, h5py.Group):
            encoding = node.attrs.get("encoding-type", "")
            if isinstance(encoding, bytes):
                encoding = encoding.decode("utf-8")
            shape = tuple(int(value) for value in node.attrs.get("shape", ()))
            if encoding != "csr_matrix" or shape != (N_DEV, N_SELECTED):
                raise RuntimeError(f"Expected CSR E161 X; observed {encoding} {shape}")
            indptr = np.asarray(node["indptr"][: N_TRAIN + 1], dtype=np.int64)
            if (
                indptr.shape != (N_TRAIN + 1,)
                or indptr[0] != 0
                or np.any(np.diff(indptr) < 0)
            ):
                raise RuntimeError("Malformed train-prefix CSR indptr")
            stop = int(indptr[-1])
            data = np.asarray(node["data"][:stop], dtype=np.float64)
            indices = np.asarray(node["indices"][:stop], dtype=np.int64)
            if len(data) != stop or len(indices) != stop:
                raise RuntimeError("Truncated train-prefix CSR payload")
            train_x = sp.csr_matrix(
                (data, indices, indptr),
                shape=(N_TRAIN, N_SELECTED),
                dtype=np.float64,
            )
            storage = "csr_train_prefix_only"
        else:
            raise RuntimeError("Unsupported E161 X storage")
    after = E161_H5AD.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeError("E161 development H5AD changed during train read")
    if train_x.shape != (N_TRAIN, N_SELECTED) or not np.isfinite(train_x.data).all():
        raise RuntimeError("E161 train expression is malformed or non-finite")
    return train_x, conditions[:N_TRAIN], {
        "storage": storage,
        "train_X_rows_indexed_materialized_transformed": N_TRAIN,
        "validation_X_rows_indexed_materialized_transformed": 0,
        "test_X_rows_indexed_materialized_transformed": 0,
        "excluded_X_rows_indexed_materialized_transformed": 0,
    }


def build_systema_reference(
    train_x: sp.csr_matrix,
    train_conditions: np.ndarray,
    train_order: list[str],
    control: np.ndarray,
    pca_mean: np.ndarray,
    components: np.ndarray,
    genes: list[str],
) -> dict[str, Any]:
    if len(train_order) != N_TRAIN_CONDITIONS or train_order[0] != "ctrl":
        raise RuntimeError("Frozen E160 train condition order changed")
    if set(train_conditions.tolist()) != set(train_order):
        raise RuntimeError("E161 train condition membership differs from E160")
    noncontrol = [condition for condition in train_order if condition != "ctrl"]
    if len(noncontrol) != N_NONCONTROL_TRAIN_CONDITIONS:
        raise RuntimeError("Expected exactly 71 noncontrol train conditions")
    counts = pd.Series(train_conditions).value_counts().to_dict()
    condition_means: list[np.ndarray] = []
    for condition in noncontrol:
        mask = np.flatnonzero(train_conditions == condition)
        if len(mask) < 2:
            raise RuntimeError(f"Train condition has fewer than two cells: {condition}")
        condition_means.append(
            np.asarray(train_x[mask].mean(axis=0), dtype=np.float64).reshape(-1)
        )
    centroid = np.mean(np.stack(condition_means), axis=0, dtype=np.float64)
    noncontrol_rows = np.flatnonzero(train_conditions != "ctrl")
    cell_weighted = np.asarray(
        train_x[noncontrol_rows].mean(axis=0), dtype=np.float64
    ).reshape(-1)
    if (
        centroid.shape != (N_SELECTED,)
        or cell_weighted.shape != (N_SELECTED,)
        or not np.isfinite(centroid).all()
        or not np.isfinite(cell_weighted).all()
    ):
        raise RuntimeError("Systema train-only centroid is malformed")
    coordinate = (centroid - pca_mean.astype(np.float64)) @ components.astype(
        np.float64
    ).T
    effect = centroid - control.astype(np.float64)
    if coordinate.shape != (N_PCA,) or not np.isfinite(coordinate).all():
        raise RuntimeError("Systema PCA10 projection failed")
    return {
        "profile": centroid,
        "pca10": coordinate,
        "cell_weighted_recomputed": cell_weighted,
        "condition_counts": {condition: int(counts[condition]) for condition in noncontrol},
        "n_noncontrol_cells": int(len(noncontrol_rows)),
        "effect_rms": float(np.sqrt(np.mean(effect * effect))),
        "condition_balanced_minus_cell_weighted_rms": float(
            np.sqrt(np.mean((centroid - cell_weighted) ** 2))
        ),
        "genes": genes,
    }


def build_baseline_assets(
    common: dict[str, Any],
    systema: dict[str, Any],
    control: np.ndarray,
    pca_mean: np.ndarray,
    components: np.ndarray,
    genes: list[str],
) -> dict[str, Any]:
    interface = common["e162b"]["interface"]
    post_path = E162B_RELEASE / interface["paths"]["post_profiles"]
    source = pd.read_csv(post_path)
    if source.columns[:2].tolist() != ["baseline", "condition"]:
        raise RuntimeError("E162b post-profile index columns changed")
    if source.columns[2:].astype(str).tolist() != genes:
        raise RuntimeError("E162b post-profile gene axis changed")
    expected_rows = [
        (baseline, condition)
        for baseline in E162B_BASELINE_ORDER
        for condition in common["test_labels"]
    ]
    observed_rows = list(
        zip(source["baseline"].astype(str), source["condition"].astype(str))
    )
    if observed_rows != expected_rows:
        raise RuntimeError("E162b post-profile baseline/task order changed")

    cell_block = source[
        source["baseline"].astype(str).eq("cell_weighted_perturbed_mean")
    ]
    cell_values = cell_block[genes].to_numpy(dtype=np.float64)
    if cell_values.shape != (N_TEST, N_SELECTED):
        raise RuntimeError("E162b cell-weighted profile shape changed")
    within_task_delta = float(np.max(np.abs(cell_values - cell_values[0])))
    recomputed_delta = float(
        np.max(np.abs(cell_values[0] - systema["cell_weighted_recomputed"]))
    )
    if within_task_delta > 1e-12 or recomputed_delta > 5e-12:
        raise RuntimeError("E162b cell-weighted mean no longer matches train-only recomputation")

    blocks: list[pd.DataFrame] = []
    for baseline in BASELINE_ORDER:
        if baseline == "condition_balanced_perturbed_mean":
            block = pd.DataFrame(
                np.repeat(systema["profile"][None, :], N_TEST, axis=0),
                columns=genes,
            )
            block.insert(0, "condition", common["test_labels"])
            block.insert(0, "baseline", baseline)
        else:
            source_name = "control" if baseline == "control_no_change" else baseline
            block = source[source["baseline"].astype(str).eq(source_name)].copy()
            block.loc[:, "baseline"] = baseline
        blocks.append(block)
    combined = pd.concat(blocks, ignore_index=True)
    if list(zip(combined["baseline"], combined["condition"])) != [
        (baseline, condition)
        for baseline in BASELINE_ORDER
        for condition in common["test_labels"]
    ]:
        raise RuntimeError("E164 combined baseline order failed")

    values = combined[genes].to_numpy(dtype=np.float64).reshape(
        len(BASELINE_ORDER), N_TEST, N_SELECTED
    )
    control_pca = (control.astype(np.float64) - pca_mean.astype(np.float64)) @ components.astype(
        np.float64
    ).T
    rows: list[dict[str, Any]] = []
    for baseline_index, baseline in enumerate(BASELINE_ORDER):
        post_pca = (values[baseline_index] - pca_mean.astype(np.float64)) @ components.astype(
            np.float64
        ).T
        for task_index, condition in enumerate(common["test_labels"]):
            row: dict[str, Any] = {
                "baseline": baseline,
                "condition": condition,
                "test_index": task_index,
            }
            row.update(
                {
                    f"post_PC{index + 1}": float(post_pca[task_index, index])
                    for index in range(N_PCA)
                }
            )
            row.update(
                {
                    f"effect_PC{index + 1}": float(
                        post_pca[task_index, index] - control_pca[index]
                    )
                    for index in range(N_PCA)
                }
            )
            rows.append(row)

    risk_path = E162B_RELEASE / interface["paths"]["risk_wide"]
    risk = pd.read_csv(risk_path)
    if risk["condition"].astype(str).tolist() != common["test_labels"]:
        raise RuntimeError("E162b risk/task order changed")
    risk["condition_balanced_effect_rms"] = systema["effect_rms"]
    risk["condition_balanced_magnitude_confidence"] = -systema["effect_rms"]
    risk["condition_balanced_magnitude_estimability"] = (
        "constant_or_nonfinite_baseline"
    )

    audit = pd.DataFrame(
        [
            {
                "check": "noncontrol_train_conditions",
                "observed": N_NONCONTROL_TRAIN_CONDITIONS,
                "required": N_NONCONTROL_TRAIN_CONDITIONS,
                "gate_passed": True,
            },
            {
                "check": "noncontrol_train_cells",
                "observed": systema["n_noncontrol_cells"],
                "required": ">0",
                "gate_passed": systema["n_noncontrol_cells"] > 0,
            },
            {
                "check": "E162b_cellweighted_within_task_max_abs_delta",
                "observed": within_task_delta,
                "required": "<=1e-12",
                "gate_passed": within_task_delta <= 1e-12,
            },
            {
                "check": "E162b_cellweighted_vs_recomputed_max_abs_delta",
                "observed": recomputed_delta,
                "required": "<=5e-12",
                "gate_passed": recomputed_delta <= 5e-12,
            },
            {
                "check": "conditionbalanced_vs_cellweighted_RMS",
                "observed": systema["condition_balanced_minus_cell_weighted_rms"],
                "required": "reported_not_gated",
                "gate_passed": True,
            },
            {
                "check": "Systema_uses_validation_or_test_X",
                "observed": False,
                "required": False,
                "gate_passed": True,
            },
        ]
    )
    if not audit["gate_passed"].astype(bool).all():
        raise RuntimeError("E164 Systema/baseline audit failed")
    return {
        "post_profiles": combined,
        "pca10": pd.DataFrame(rows),
        "risk_wide": risk,
        "audit": audit,
    }


def exact_unique_scalar(values: np.ndarray) -> int:
    return int(len(np.unique(np.asarray(values, dtype=np.float64))))


def exact_unique_vectors(values: np.ndarray) -> int:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if array.ndim != 2:
        raise RuntimeError("Expected a 2D vector matrix")
    return int(len({tuple(row.tolist()) for row in array}))


def score_estimability(values: np.ndarray, *, seed: int, score: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = bool(np.isfinite(array).all())
    unique = exact_unique_scalar(array) if finite else 0
    std = float(np.std(array, ddof=1)) if finite and len(array) > 1 else float("nan")
    estimable = bool(finite and unique >= 2 and std > ESTIMABILITY_STD_MINIMUM)
    return {
        "seed": seed,
        "split": "test_label_only",
        "score": score,
        "n_rows": int(len(array)),
        "all_finite": finite,
        "exact_unique": unique,
        "sample_std_ddof1": std,
        "required_unique": 2,
        "required_std_strictly_greater_than": ESTIMABILITY_STD_MINIMUM,
        "estimable": estimable,
        "failure_code": "" if estimable else "constant_or_nonfinite_baseline",
        "downstream_statistic": "eligible" if estimable else "NA",
    }


def raw_score_gate(table: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    raw = table["raw_log_prob"].to_numpy(dtype=np.float64)
    prediction = table[
        [f"predicted_pca_{index}" for index in range(N_PCA)]
    ].to_numpy(dtype=np.float64)
    finite = bool(np.isfinite(raw).all())
    unique = exact_unique_scalar(raw) if finite else 0
    std = float(np.std(raw, ddof=1)) if finite else float("nan")
    coordinate_std = np.std(prediction, axis=0, ddof=1)
    passed = bool(
        len(table) == N_TEST
        and finite
        and unique >= RAW_UNIQUE_MINIMUM
        and std > RAW_STD_MINIMUM
    )
    return {
        "schema": "safeconf_e164_raw_score_gate_v1",
        "seed": seed,
        "split": "test_label_only",
        "condition_order_sha256": sha256_bytes(
            ("\n".join(table["condition"].astype(str)) + "\n").encode("utf-8")
        ),
        "n_rows": int(len(table)),
        "required_rows": N_TEST,
        "raw_log_prob_all_finite": finite,
        "raw_log_prob_exact_unique": unique,
        "required_minimum_exact_unique": RAW_UNIQUE_MINIMUM,
        "raw_log_prob_sample_std_ddof1": std,
        "required_std_strictly_greater_than": RAW_STD_MINIMUM,
        "passed": passed,
        "prediction_all_finite": bool(np.isfinite(prediction).all()),
        "prediction_exact_unique_vectors": exact_unique_vectors(prediction),
        "prediction_coordinate_sample_std_ddof1": coordinate_std.tolist(),
        "prediction_constant": bool(exact_unique_vectors(prediction) < 2),
        "prediction_is_diagnostic_not_part_of_this_gate": True,
        "E162_validation_failure_overwritten": False,
    }


def query_prescribe(
    common: dict[str, Any],
    modules: dict[int, Any],
    control: np.ndarray,
    pca_mean: np.ndarray,
    components: np.ndarray,
    genes: list[str],
    persist_root: Path,
) -> dict[str, Any]:
    e162 = common["e162_module"]
    tables: dict[int, pd.DataFrame] = {}
    profiles: dict[int, pd.DataFrame] = {}
    gates: dict[int, dict[str, Any]] = {}
    estimability: list[dict[str, Any]] = []
    degeneracy: list[dict[str, Any]] = []
    for seed in SEEDS:
        table, _old_gate, _old_estimability = e162.query_label_only_scores(
            modules[seed],
            common["test_labels"],
            control,
            pca_mean,
            components,
            genes,
            seed=seed,
            split="test",
        )
        if table["condition"].astype(str).tolist() != common["test_labels"]:
            raise RuntimeError(f"Seed {seed} query order changed")
        gate = raw_score_gate(table, seed=seed)
        predicted = table[
            [f"predicted_pca_{index}" for index in range(N_PCA)]
        ].to_numpy(dtype=np.float64)
        reconstructed = predicted @ components.astype(np.float64) + pca_mean.astype(
            np.float64
        )
        profile = pd.DataFrame(reconstructed, columns=genes)
        profile.insert(0, "condition", common["test_labels"])
        profile.insert(0, "seed", seed)
        tables[seed] = table
        profiles[seed] = profile
        gates[seed] = gate
        # Persist every completed seed immediately.  If a later forward fails,
        # the irreversible event and already observed outputs remain auditable.
        atomic_csv(
            table,
            persist_root
            / f"tables/E164_PRESCRIBE_TEST_LABEL_ONLY_SCORES_SEED{seed}.csv",
        )
        atomic_json(
            persist_root / f"tables/E164_RAW_SCORE_GATE_SEED{seed}.json", gate
        )
        atomic_gzip_csv(
            profile,
            persist_root
            / f"profiles/E164_PRESCRIBE_TEST_POST_PROFILES_SEED{seed}.csv.gz",
        )
        for column, score in (
            ("raw_log_prob", "raw_log_prob"),
            ("official_combined_confidence", "official_combined_confidence"),
            ("predicted_magnitude_rms", "predicted_magnitude_rms"),
        ):
            estimability.append(
                score_estimability(table[column].to_numpy(float), seed=seed, score=score)
            )
        coordinate_std = np.std(predicted, axis=0, ddof=1)
        for coordinate in range(N_PCA):
            degeneracy.append(
                {
                    "seed": seed,
                    "object": "predicted_pca_coordinate",
                    "coordinate_zero_based": coordinate,
                    "all_finite": bool(np.isfinite(predicted[:, coordinate]).all()),
                    "exact_unique": exact_unique_scalar(predicted[:, coordinate]),
                    "sample_std_ddof1": float(coordinate_std[coordinate]),
                    "constant_at_1e_minus_12": bool(
                        coordinate_std[coordinate] <= ESTIMABILITY_STD_MINIMUM
                    ),
                    "E162_failure_overwritten": False,
                }
            )
        vector_estimable = bool(
            np.isfinite(predicted).all()
            and exact_unique_vectors(predicted) >= 2
            and np.any(coordinate_std > ESTIMABILITY_STD_MINIMUM)
        )
        estimability.append(
            {
                "seed": seed,
                "split": "test_label_only",
                "score": "predicted_pca10_vector",
                "n_rows": N_TEST,
                "all_finite": bool(np.isfinite(predicted).all()),
                "exact_unique": exact_unique_vectors(predicted),
                "sample_std_ddof1": float(np.max(coordinate_std)),
                "required_unique": 2,
                "required_std_strictly_greater_than": ESTIMABILITY_STD_MINIMUM,
                "estimable": vector_estimable,
                "failure_code": "" if vector_estimable else "constant_or_nonfinite_prediction",
                "downstream_statistic": "retain_prediction_and_disclose_collapse",
            }
        )
    return {
        "tables": tables,
        "profiles": profiles,
        "gates": gates,
        "estimability": pd.DataFrame(estimability),
        "degeneracy": pd.DataFrame(degeneracy),
    }


def add_prescribe_risk_columns(risk: pd.DataFrame, queried: dict[str, Any]) -> pd.DataFrame:
    result = risk.copy()
    for seed in SEEDS:
        table = queried["tables"][seed]
        if table["condition"].astype(str).tolist() != result["condition"].astype(str).tolist():
            raise RuntimeError(f"Seed {seed} score/risk task order differs")
        result[f"prescribe_raw_log_prob_seed{seed}_confidence"] = table[
            "raw_log_prob"
        ].to_numpy(float)
    main = queried["tables"][MAIN_SEED]
    result["prescribe_official_seed3407_confidence"] = main[
        "official_combined_confidence"
    ].to_numpy(float)
    result["prescribe_negative_magnitude_seed3407_confidence"] = -main[
        "predicted_magnitude_rms"
    ].to_numpy(float)
    result["prescribe_magnitude_raw_seed3407"] = main[
        "predicted_magnitude_rms"
    ].to_numpy(float)
    return result


def build_e165_spec(*, prescribe_arm_authorized: bool) -> dict[str, Any]:
    coverage = [round(value, 2) for value in np.arange(0.50, 1.001, 0.05)]
    retained = [int(np.ceil(value * N_TEST)) for value in coverage]
    return {
        "schema": "safeconf_e165_wessels_evaluation_spec_v1",
        "experiment": "E165_wessels_frozen_truth_evaluation",
        "frozen_by": "E164_wessels_pretruth_lock",
        "frozen_at": now_text(),
        "analysis_timing": "before_first_raw_Wessels_or_test_truth_open",
        "baseline_arm_authorized": True,
        "prescribe_arm_authorized": prescribe_arm_authorized,
        "no_adaptive_changes_after_truth_unseal": True,
        "truth_unseal": {
            "precondition_schema": "safeconf_e164_to_e165_v1",
            "required_baseline_arm_authorized": True,
            "required_for_prescribe_endpoints": {
                "prescribe_arm_authorized": True,
                "main_raw_gate_passed": True,
            },
            "first_action": (
                "atomically write and fsync a permanent E165 truth-unseal event "
                "binding the E164 interface, manifest, raw identity and runner Git blob"
            ),
            "raw_identity_and_axes_to_reverify": [
                "E160 raw file identity and full byte hashes",
                "E160 canonical row-to-condition audit",
                "20631 endogenous feature axis",
                "2023 E161 selected-gene axis and raw indices",
                "8 engineered construct columns excluded",
                "413 guide/barcode columns excluded",
            ],
            "permitted_X": {
                "test_rows": 9902,
                "columns": "first 20631 endogenous features only",
                "engineered_construct_columns": 0,
                "guide_barcode_columns": 0,
                "validation_rows": 0,
                "excluded_rows": 0,
            },
            "normalization": (
                "per cell over all 20631 endogenous raw-count genes: "
                "log1p(10000*x/library); select the frozen 2023 genes only afterward"
            ),
            "task_truth": (
                "arithmetic mean of normalized selected-gene vectors over all test "
                "cells in each canonical condition"
            ),
            "control_truth": "E161 train-only control_gene_mean; never a test control",
            "pca10_truth": (
                "(raw_test_task_mean - E161_PCA_mean) @ E161_components.T; inverse "
                "transform with the same mean/components for the primary truth profile"
            ),
            "mandatory_truth_outputs": [
                "raw normalized 2023-gene task means and effects",
                "train-PCA10 projected/inverse-transformed task means and effects",
                "task cell counts and normalization audit",
            ],
        },
        "predictors": {
            "fixed_order": list(BASELINE_ORDER)
            + (["prescribe_seed3407"] if prescribe_arm_authorized else []),
            "seed_sensitivity_only": (
                ["prescribe_seed3408", "prescribe_seed3409"]
                if prescribe_arm_authorized
                else []
            ),
            "profile_kind": "one fixed 2023-gene post-state centroid per test task",
            "single_cell_prediction_distribution_available": False,
        },
        "primary_truth_and_accuracy": {
            "PRESCRIBE_primary_truth": "train-only PCA10 inverse-transform effect",
            "PRESCRIBE_primary_task_accuracy": (
                "Pearson(predicted_effect_2023, PCA10_projected_truth_effect_2023)"
            ),
            "mandatory_raw_sensitivity": (
                "Pearson(predicted_effect_2023, raw_normalized_truth_effect_2023)"
            ),
            "raw_sensitivity_never_replaces_primary": True,
            "constant_or_weak_signal_rule": (
                "Pearson/cosine is NA when either compared vector is nonfinite or has "
                "sample std <=1e-12; never fill with zero"
            ),
            "direction_rule": (
                "per gene hit iff prediction_effect * truth_effect > 0; prediction or "
                "truth equal to zero is a miss"
            ),
        },
        "confirmatory_families": {
            "familywise_alpha": 0.05,
            "PRESCRIBE_own_family": {
                "alpha": 0.025,
                "enabled_only_if_prescribe_arm_authorized": True,
                "score": "prescribe_seed3407.raw_log_prob; higher is more confident",
                "statistic": (
                    "Spearman(raw_log_prob, PCA10_inverse_transform_own_model_Pearson)"
                ),
                "expected_direction": "greater_than_zero",
                "confirmatory_gate": (
                    "point estimate >0 and both task-bootstrap and component-gene "
                    "cluster-bootstrap 95% CI lower bounds >0"
                ),
                "mandatory_sensitivity": [
                    "same score versus raw 2023-gene Pearson effect accuracy",
                    "same score versus PCA10 and raw RMSE with negative expected sign",
                    "same score versus strict direction fraction",
                    "seeds 3408 and 3409 reported without replacing seed 3407",
                ],
                "disclosure": (
                    "E162 failed its original prediction nondegeneracy gate; E163 is a "
                    "validation-informed futility diagnostic, not external confirmation"
                ),
            },
            "baseline_family_fixed_sequence": {
                "alpha": 0.025,
                "H1": {
                    "name": "matching_vs_cellweighted_PCA10_RMSE",
                    "task_contrast": (
                        "RMSE10_cell_weighted_perturbed_mean - RMSE10_matching_single_mean"
                    ),
                    "aggregate": "arithmetic mean over all 48 tasks",
                    "expected_direction": "greater_than_zero_favors_matching",
                    "confirmatory_gate": (
                        "point estimate >0 and both task-bootstrap and component-gene "
                        "cluster-bootstrap 95% CI lower bounds >0"
                    ),
                },
                "H2": {
                    "name": "matching_SE_risk",
                    "tested_only_if_H1_passes": True,
                    "statistic": (
                        "Spearman(matching_se_pca10_confidence, -RMSE10_matching)"
                    ),
                    "expected_direction": "greater_than_zero",
                    "confirmatory_gate": (
                        "point estimate >0 and both task-bootstrap and component-gene "
                        "cluster-bootstrap 95% CI lower bounds >0"
                    ),
                    "if_H1_fails": "descriptive_only",
                },
            },
        },
        "resampling": {
            "rng": "numpy.random.default_rng(3407)",
            "replicates": 10000,
            "CI": "percentile [0.025,0.975], numpy.quantile method=linear",
            "minimum_valid_replicates": 9500,
            "paired_indices_shared_by_all_predictors_and_scores": True,
            "task_bootstrap": "sample 48 task indices with replacement",
            "component_gene_cluster_bootstrap": (
                "let K be distinct component genes; sample K genes with replacement; "
                "for every sampled gene append all containing test tasks in canonical "
                "condition order; a pair can enter twice through its two components"
            ),
            "leave_one_component_gene_out": {
                "required": True,
                "per_gene": [
                    "removed gene",
                    "removed task count",
                    "remaining task count",
                    "effect or rho",
                    "estimability status",
                ],
                "summary": ["minimum", "median", "maximum", "fraction_positive"],
            },
        },
        "split_half_experimental_reproducibility_reference": {
            "name_must_not_contain": ["upper_bound", "upper bound", "theoretical ceiling"],
            "interpretation": (
                "experimental reproducibility benchmark/reference; half-sample noise can "
                "make it lower than full-sample reproducibility"
            ),
            "cell_id_source": "frozen raw obs index; exact string, no row-number fallback",
            "digest": (
                'SHA256("E165|Wessels|split-half|3407\\t" + canonical_condition '
                '+ "\\t" + cell_id)'
            ),
            "partition": (
                "sort by (digest_hex, cell_id), assign alternating rows A/B; for odd n, "
                "A contains one more cell"
            ),
            "small_task_rule": "n<4 => every split-half endpoint is NA; do not borrow cells",
            "normalization": (
                "normalize/log1p every cell first; form A and B means independently"
            ),
            "endpoints": [
                "PCA10-projected Pearson and RMSE",
                "raw 2023-gene Pearson and RMSE",
                "PCA10-truth-top20 and raw-truth-top20 RMSE/Pearson/direction",
            ],
            "role": "contextual_descriptive_not_confirmatory",
            "citations": {
                "TxPert": "10.1038/s41587-026-03113-4",
                "2026_SBB_evaluation": "10.64898/2026.04.20.719650",
            },
            "TxPert_context": (
                "seen-single/unseen-double evaluation requires strong additive/simple "
                "baselines; split-half is not a model-performance upper bound"
            ),
        },
        "all_required_task_metrics": {
            "absolute": [
                "PCA10_projected_RMSE_2023",
                "raw_RMSE_2023",
                "PCA10_projected_MSE_2023",
                "raw_MSE_2023",
            ],
            "control_reference": [
                "Pearson_delta_PCA10",
                "cosine_delta_PCA10",
                "Pearson_delta_raw",
                "cosine_delta_raw",
            ],
            "train_Systema_reference": [
                "Pearson(predicted_post-Systema_train_centroid, truth_post-Systema_train_centroid)",
                "cosine(predicted_post-Systema_train_centroid, truth_post-Systema_train_centroid)",
            ],
            "direction": ["strict_direction_fraction_all", "strict_direction_fraction_top20"],
            "RMSE_reference_invariance_note": (
                "RMSE is unchanged by subtracting a shared reference; retain it to "
                "distinguish matching from the 2x matching single-additive amplitude"
            ),
        },
        "Systema_compatible_evaluation": {
            "train_reference": (
                "E164 condition-balanced mean of 71 train perturbation-condition centroids"
            ),
            "simple_predictor": "condition_balanced_perturbed_mean",
            "Euclidean_centroid_accuracy": (
                "for each predicted post state, find the nearest among all 48 true test "
                "post centroids by Euclidean distance; canonical condition breaks exact "
                "ties; hit iff nearest condition is the task itself; report mean hit rate"
            ),
            "panel_metric_role": "secondary_descriptive_perturbation_specificity",
            "citation": "10.1038/s41587-025-02777-8",
        },
        "top20": {
            "PCA10_truth_top20": (
                "20 genes with largest absolute PCA10-projected truth effect; exact ties "
                "by zero-based E161 selected-gene index"
            ),
            "raw_truth_top20": (
                "20 genes with largest absolute raw normalized truth effect; exact ties "
                "by zero-based E161 selected-gene index"
            ),
            "metrics": ["RMSE", "Pearson_delta", "strict_direction_fraction"],
            "prediction_union_set_forbidden": True,
            "role": "secondary_test_dependent_truth_selected_signal_sensitive",
        },
        "scPerturBench_compatibility": {
            "citation": "10.1038/s41592-025-02980-0",
            "centroid_metrics_to_report": ["MSE", "RMSE", "PCC_delta"],
            "distribution_metrics_fixed_NA": [
                "E-distance",
                "Wasserstein",
                "KL_divergence",
                "Common-DEGs",
            ],
            "NA_reason": "all frozen predictors provide centroids, not predicted cell distributions",
        },
        "risk_and_selective_prediction": {
            "orientation": "every analysis score is higher_expected_accuracy",
            "scores": [
                "matching_se_pca10_confidence",
                "matching_se_gene_confidence",
                "min_single_cell_count_confidence",
                "min_train_pair_degree_confidence",
                "matching_magnitude_confidence",
                "hash_random_confidence",
                "constant_confidence",
                "exact_pair_support_confidence",
            ]
            + (
                [
                    "prescribe_raw_log_prob_seed3407_confidence",
                    "prescribe_raw_log_prob_seed3408_confidence",
                    "prescribe_raw_log_prob_seed3409_confidence",
                    "prescribe_official_seed3407_confidence",
                    "prescribe_negative_magnitude_seed3407_confidence",
                ]
                if prescribe_arm_authorized
                else []
            ),
            "constant_rule": (
                "nonfinite, exact unique<2 or sample std<=1e-12 => rho, CI, AURC and "
                "coverage contrasts NA with constant_or_nonfinite_baseline"
            ),
            "coverage_grid": coverage,
            "retained_task_counts": retained,
            "ordering": "score descending, canonical condition ascending tie-break",
            "outputs": [
                "Spearman versus each error/accuracy endpoint",
                "selective risk curve and AURC",
                "lowest-confidence quartile error enrichment",
                "paired delta-rho versus predicted magnitude",
            ],
        },
        "multiplicity": {
            "confirmatory": (
                "PRESCRIBE family alpha=0.025 plus fixed-sequence baseline family "
                "alpha=0.025; total FWER<=0.05"
            ),
            "secondary_prediction_contrasts": "Holm within one family",
            "secondary_risk_contrasts": "Holm within one separate family",
            "everything_else": "descriptive; report raw and adjusted p-values where defined",
        },
        "forbidden_adaptations": [
            "change task set or label order",
            "change score sign after seeing truth",
            "jitter or perturb constant scores/predictions",
            "replace an NA correlation with zero",
            "add a prediction-union top20 set",
            "call split-half an upper bound",
            "claim E162 validation failure was passed",
        ],
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: Path, payload: bytes, *, replace: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        # link(2) is an atomic no-replace publication on this filesystem: the
        # target is created only if it did not exist at that exact instant.
        # This closes the exists-check/rename race for the irreversible event.
        try:
            os.link(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        temporary.unlink()
    fsync_directory(path.parent)
    return sha256_bytes(payload)


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> str:
    return atomic_write(path, json_bytes(value), replace=replace)


def atomic_csv(frame: pd.DataFrame, path: Path, *, replace: bool = False) -> str:
    payload = frame.to_csv(
        index=False, float_format="%.17g", lineterminator="\n"
    ).encode("utf-8")
    return atomic_write(path, payload, replace=replace)


def atomic_gzip_csv(frame: pd.DataFrame, path: Path, *, replace: bool = False) -> str:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as gz_handle:
        with io.TextIOWrapper(gz_handle, encoding="utf-8", newline="") as text_handle:
            frame.to_csv(
                text_handle,
                index=False,
                float_format="%.17g",
                lineterminator="\n",
            )
    return atomic_write(path, buffer.getvalue(), replace=replace)


def fsync_tree(root: Path) -> None:
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.is_symlink():
            raise RuntimeError(f"Symlink rejected in E164 staging: {path}")
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    directories = sorted(
        [root, *(value for value in root.rglob("*") if value.is_dir())],
        key=lambda value: len(value.parts),
        reverse=True,
    )
    for directory in directories:
        fsync_directory(directory)


def source_hash_table(common: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(role: str, path: Path, *, expected: str | None = None, git_match: Any = False) -> None:
        regular_file(path, role)
        observed = sha256_file(path)
        if expected is not None and observed != expected:
            raise RuntimeError(f"Source changed while building table: {role}")
        rows.append(
            {
                "source_role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": observed,
                "expected_sha256": expected or observed,
                "matches_git_head_blob": git_match,
                "git_head": common["head"],
            }
        )

    add(
        "E164_runner",
        RUNNER,
        expected=common["own_git"]["runner"]["working_sha256"],
        git_match=common["own_git"]["runner"]["matches_head_blob"],
    )
    add(
        "E164_contract",
        CONTRACT,
        expected=common["own_git"]["contract"]["working_sha256"],
        git_match=common["own_git"]["contract"]["matches_head_blob"],
    )
    fixed_paths = {
        "E163_status": E163_STATUS,
        "E163_interface": E163_INTERFACE,
        "E163_manifest": E163_MANIFEST,
        "E163_authorization": E163_AUTHORIZATION,
        "E162_status": E162_STATUS,
        "E162b_status": E162B_STATUS,
        "E162b_interface": E162B_INTERFACE,
        "E162b_manifest": E162B_MANIFEST,
        "E160_split": E160_SPLIT,
        "E160_condition_audit": E160_CONDITION_AUDIT,
        "E161_interface": E161_INTERFACE,
        "E161_genes": E161_GENES,
        "E161_pca": E161_PCA,
        "E161_control": E161_CONTROL,
        "E161_h5ad": E161_H5AD,
    }
    for role, path in fixed_paths.items():
        add(role, path, expected=EXPECTED_SHA256[role])
    for seed in SEEDS:
        checkpoint = common["e162"]["checkpoints"][str(seed)]["locked_checkpoint"]
        add(
            f"E162_locked_checkpoint_seed{seed}",
            Path(checkpoint["path"]),
            expected=checkpoint["sha256"],
        )
    for relative, record in sorted(common["e163"]["manifest_records"].items()):
        add(
            f"E163_payload::{relative}",
            E163_RELEASE / relative,
            expected=record["sha256"],
        )
    for relative, record in sorted(common["e162b"]["manifest_records"].items()):
        add(
            f"E162b_payload::{relative}",
            E162B_RELEASE / relative,
            expected=record["sha256"],
        )
    if QUERY_EVENT.exists():
        add("E164_test_label_query_event", QUERY_EVENT)
    return pd.DataFrame(rows).sort_values("source_role").reset_index(drop=True)


def access_ledger(*, prescribe_query_performed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asset": "raw_Wessels_H5AD",
                "access": "file_open",
                "count": 0,
                "detail": "raw_file_opened=false",
            },
            {
                "asset": "E161_development_H5AD_X",
                "access": "train_rows_indexed_materialized_transformed",
                "count": N_TRAIN,
                "detail": "Systema condition-balanced reference only",
            },
            {
                "asset": "E161_development_H5AD_X",
                "access": "validation_rows_indexed_materialized_transformed",
                "count": 0,
                "detail": "metadata inspected; X suffix not sliced",
            },
            {
                "asset": "test_or_excluded_X",
                "access": "rows_indexed_materialized_transformed",
                "count": 0,
                "detail": "sealed",
            },
            {
                "asset": "test_condition_strings",
                "access": "label_only_forward_inputs",
                "count": N_TEST if prescribe_query_performed else 0,
                "detail": "48 canonical strings; no expression or cell count",
            },
            {
                "asset": "test_graphs",
                "access": "constructed_or_loaded",
                "count": 0,
                "detail": "label-only Batch uses x/pert/batch/ptr only",
            },
            {
                "asset": "test_truth_effect_error_DE_cell_count",
                "access": "queried_or_computed",
                "count": 0,
                "detail": "all false",
            },
        ]
    )


def build_query_event(common: dict[str, Any], *, gpu_index: int) -> dict[str, Any]:
    return {
        "schema": "safeconf_e164_test_label_query_event_v1",
        "event_id": uuid.uuid4().hex,
        "experiment": "E164_wessels_pretruth_lock",
        "written_and_fsynced_at": now_text(),
        "irreversible": True,
        "must_exist_before_first_forward": True,
        "E163": {
            "status_sha256": EXPECTED_SHA256["E163_status"],
            "interface_sha256": EXPECTED_SHA256["E163_interface"],
            "manifest_sha256": EXPECTED_SHA256["E163_manifest"],
            "authorization_sha256": EXPECTED_SHA256["E163_authorization"],
            "validation_gate_passed": common["e163"]["validation_gate_passed"],
            "authorize_future_test_label_lock": common["e163"]["interface"][
                "authorize_future_test_label_lock"
            ],
            "manifest_payload_sha256": {
                relative: record["sha256"]
                for relative, record in sorted(
                    common["e163"]["manifest_records"].items()
                )
            },
        },
        "E162_failure": {
            "status_sha256": EXPECTED_SHA256["E162_status"],
            "phase": common["e162"]["status"]["phase"],
            "E162_failure_overwritten": False,
            "checkpoints": common["e162"]["checkpoints"],
        },
        "E162b": {
            "status_sha256": EXPECTED_SHA256["E162b_status"],
            "interface_sha256": EXPECTED_SHA256["E162b_interface"],
            "manifest_sha256": EXPECTED_SHA256["E162b_manifest"],
            "manifest_payload_sha256": {
                relative: record["sha256"]
                for relative, record in sorted(
                    common["e162b"]["manifest_records"].items()
                )
            },
        },
        "E160_E161": {
            "E160_split_sha256": EXPECTED_SHA256["E160_split"],
            "E160_condition_audit_sha256": EXPECTED_SHA256[
                "E160_condition_audit"
            ],
            "test_label_order": common["test_labels"],
            "test_label_order_sha256": common["test_label_order_sha256"],
            "test_conditions_are_dual_pairs": True,
            "all_test_components_seen_as_train_singles": True,
            "test_cell_count_from_E160_metadata_only": common[
                "frozen_test_cell_count_metadata_only"
            ],
            "E161_interface_sha256": EXPECTED_SHA256["E161_interface"],
            "selected_gene_order_sha256": EXPECTED_SHA256["E161_genes"],
            "PCA_model_sha256": EXPECTED_SHA256["E161_pca"],
            "control_prior_sha256": EXPECTED_SHA256["E161_control"],
            "development_H5AD_sha256": EXPECTED_SHA256["E161_h5ad"],
        },
        "source_and_runtime": {
            "git_head": common["head"],
            "runner": common["own_git"]["runner"],
            "contract": common["own_git"]["contract"],
            "native_gate_fingerprint_sha256": common["native_gate"][
                "gate_fingerprint_sha256"
            ],
            "PRESCRIBE_commit": common["native_gate"]["prescribe_commit"],
            "PRESCRIBE_source_sha256": {
                row["relative_path"]: row["actual_sha256"]
                for row in common["native_gate"]["prescribe_sources"]
            },
            "python": sys.version.split()[0],
            "gpu_physical_index": gpu_index,
            "gpu_internal_index": 0,
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
        },
        "query": {
            "seeds": list(SEEDS),
            "conditions": "48 canonical strings in frozen order",
            "x": "E161 train-only control_gene_mean repeated per task",
            "allowed_batch_fields": ["x", "pert", "batch", "ptr"],
            "forbidden_fields": [
                "y",
                "y_pca",
                "y_n",
                "y_d",
                "y_s",
                "de_idx",
                "cell_count",
                "test_truth",
                "test_expression",
                "error",
            ],
        },
        "access_before_first_forward": {
            "raw_Wessels_opened": False,
            "test_X_accessed": False,
            "test_truth_accessed": False,
            "test_endpoint_computed": False,
            "test_graphs": 0,
            "validation_X_rows_materialized_by_E164": 0,
        },
    }


def write_release(
    common: dict[str, Any],
    systema: dict[str, Any],
    baseline: dict[str, Any],
    queried: dict[str, Any] | None,
    transaction_id: str,
    started_at: str,
) -> dict[str, Any]:
    query_performed = queried is not None
    main_raw_gate_passed: bool | None = (
        bool(queried["gates"][MAIN_SEED]["passed"]) if queried is not None else None
    )
    prescribe_arm_authorized = bool(
        common["e163"]["validation_gate_passed"] and main_raw_gate_passed is True
    )
    if not common["e163"]["validation_gate_passed"]:
        phase = "complete_baseline_arm_pretruth_lock_prescribe_not_authorized"
    elif prescribe_arm_authorized:
        phase = "complete_dual_arm_pretruth_lock_no_test_X_or_truth"
    else:
        phase = "complete_baseline_arm_pretruth_lock_prescribe_raw_gate_failed"

    atomic_gzip_csv(
        baseline["post_profiles"],
        STAGING / "profiles/E164_BASELINE_TEST_POST_PROFILES.csv.gz",
    )
    atomic_csv(
        baseline["pca10"],
        STAGING / "profiles/E164_BASELINE_TEST_PCA10_COORDINATES.csv",
    )
    systema_profile = pd.DataFrame([systema["profile"]], columns=systema["genes"])
    systema_profile.insert(0, "n_noncontrol_train_cells", systema["n_noncontrol_cells"])
    systema_profile.insert(
        0, "n_noncontrol_train_conditions", N_NONCONTROL_TRAIN_CONDITIONS
    )
    systema_profile.insert(0, "reference", "condition_balanced_perturbed_mean")
    atomic_gzip_csv(
        systema_profile,
        STAGING
        / "profiles/E164_SYSTEMA_CONDITION_BALANCED_PERTURBED_MEAN.csv.gz",
    )
    systema_pca = pd.DataFrame(
        [
            {
                "reference": "condition_balanced_perturbed_mean",
                "n_noncontrol_train_conditions": N_NONCONTROL_TRAIN_CONDITIONS,
                "n_noncontrol_train_cells": systema["n_noncontrol_cells"],
                **{
                    f"post_PC{index + 1}": float(systema["pca10"][index])
                    for index in range(N_PCA)
                },
            }
        ]
    )
    atomic_csv(
        systema_pca,
        STAGING
        / "profiles/E164_SYSTEMA_CONDITION_BALANCED_PERTURBED_MEAN_PCA10.csv",
    )

    risk_wide = baseline["risk_wide"]
    if prescribe_arm_authorized:
        if queried is None:
            raise RuntimeError("Authorized PRESCRIBE arm has no query outputs")
        risk_wide = add_prescribe_risk_columns(risk_wide, queried)
    atomic_csv(risk_wide, STAGING / "tables/E164_RISK_WIDE.csv")
    atomic_csv(
        baseline["audit"], STAGING / "tables/E164_SYSTEMA_REFERENCE_AUDIT.csv"
    )
    atomic_csv(
        access_ledger(prescribe_query_performed=query_performed),
        STAGING / "tables/E164_X_ACCESS_LEDGER.csv",
    )
    if queried is not None:
        atomic_csv(
            queried["estimability"], STAGING / "tables/E164_ESTIMABILITY.csv"
        )
        atomic_csv(
            queried["degeneracy"], STAGING / "tables/E164_DEGENERACY_AUDIT.csv"
        )

    evaluation_spec = build_e165_spec(
        prescribe_arm_authorized=prescribe_arm_authorized
    )
    atomic_json(STAGING / "E165_EVALUATION_SPEC.json", evaluation_spec)
    atomic_csv(
        source_hash_table(common), STAGING / "tables/E164_SOURCE_HASHES.csv"
    )
    atomic_csv(
        pd.DataFrame(common["runtime"]),
        STAGING / "tables/E164_RUNTIME_ENVIRONMENT.csv",
    )

    e163_rhos = common["e163"]["interface"].get("seed_primary_spearman_rho", {})
    report = f"""# E164 Wessels test truth解封前锁

## 授权结果

- baseline arm：已冻结并授权；
- E163 validation gate：{'通过' if common['e163']['validation_gate_passed'] else '未通过'}；
- PRESCRIBE label-only forward：{'已执行' if query_performed else '未执行'}；
- seed3407 raw score gate：{main_raw_gate_passed if main_raw_gate_passed is not None else 'NA'}；
- PRESCRIBE arm最终授权：{prescribe_arm_authorized}。

E163三seed validation rho为3407={e163_rhos.get('3407', 'NA')}、3408={e163_rhos.get('3408', 'NA')}、3409={e163_rhos.get('3409', 'NA')}。E163只是validation-informed futility diagnostic，不是外部确认。

## 不改写的失败

E162正式phase仍是`failed_main_validation_nondegeneracy_gate_no_test_label_query`。三个validation prediction都只有一个exact vector。E164只在新的、test truth仍封存的合同下锁定raw score；没有把E162失败改写为通过。

## baseline与Systema

E162b四个预锁预测器原样进入五预测器层级；新增`condition_balanced_perturbed_mean`由71个train非control condition先分别求细胞均值，再对condition等权。它与E162b cell-weighted mean的2023基因RMS差为{systema['condition_balanced_minus_cell_weighted_rms']:.17g}。该值只来自11,779个train rows。

## 访问边界

raw Wessels文件未打开；test、validation和excluded X rows均为0；test truth/effect/error/DE和test graph均为0。若执行PRESCRIBE，输入只有48个condition字符串及E161 train-control mean。E165评价规则已在`E165_EVALUATION_SPEC.json`冻结，本阶段没有任何test endpoint。
"""
    atomic_write(STAGING / "reports/E164_REPORT.md", report.encode("utf-8"))
    atomic_write(
        STAGING / "README_先看这个.md",
        (
            "# E164\n\n先读 `reports/E164_REPORT.md`，再核对 "
            "`E164_E165_INTERFACE.json`、`E165_EVALUATION_SPEC.json`、访问账本和哈希表。\n"
        ).encode("utf-8"),
    )

    pre_interface_paths = sorted(
        path
        for path in STAGING.rglob("*")
        if path.is_file()
        and path.name
        not in {"E164_E165_INTERFACE.json", "RUN_STATUS.json", "RESULTS_SHA256.csv"}
    )
    artifact_sha256 = {
        path.relative_to(STAGING).as_posix(): sha256_file(path)
        for path in pre_interface_paths
    }
    paths: dict[str, Any] = {
        "baseline_post_profiles": "profiles/E164_BASELINE_TEST_POST_PROFILES.csv.gz",
        "baseline_pca10_coordinates": "profiles/E164_BASELINE_TEST_PCA10_COORDINATES.csv",
        "risk_wide": "tables/E164_RISK_WIDE.csv",
        "systema_post_profile": (
            "profiles/E164_SYSTEMA_CONDITION_BALANCED_PERTURBED_MEAN.csv.gz"
        ),
        "systema_pca10": (
            "profiles/E164_SYSTEMA_CONDITION_BALANCED_PERTURBED_MEAN_PCA10.csv"
        ),
        "evaluation_spec": "E165_EVALUATION_SPEC.json",
        "access_ledger": "tables/E164_X_ACCESS_LEDGER.csv",
    }
    if prescribe_arm_authorized:
        paths["prescribe_scores"] = {
            str(seed): f"tables/E164_PRESCRIBE_TEST_LABEL_ONLY_SCORES_SEED{seed}.csv"
            for seed in SEEDS
        }
        paths["prescribe_post_profiles"] = {
            str(seed): f"profiles/E164_PRESCRIBE_TEST_POST_PROFILES_SEED{seed}.csv.gz"
            for seed in SEEDS
        }
        paths["raw_score_gates"] = {
            str(seed): f"tables/E164_RAW_SCORE_GATE_SEED{seed}.json"
            for seed in SEEDS
        }
    for exposed in [
        value
        for value in paths.values()
        if isinstance(value, str)
    ] + [
        item
        for value in paths.values()
        if isinstance(value, dict)
        for item in value.values()
    ]:
        if exposed not in artifact_sha256:
            raise RuntimeError(f"Interface path is not artifact-bound: {exposed}")

    event_sha256 = sha256_file(QUERY_EVENT) if QUERY_EVENT.exists() else None
    raw_gates = (
        {str(seed): queried["gates"][seed] for seed in SEEDS}
        if queried is not None
        else {}
    )
    interface = {
        "schema": "safeconf_e164_to_e165_v1",
        "experiment": "E164_wessels_pretruth_lock",
        "phase": phase,
        "git_head": common["head"],
        "transaction_id": transaction_id,
        "baseline_arm_authorized": True,
        "prescribe_arm_authorized": prescribe_arm_authorized,
        "e163_validation_gate_passed": common["e163"]["validation_gate_passed"],
        "e163_authorize_future_test_label_lock": common["e163"]["interface"][
            "authorize_future_test_label_lock"
        ],
        "main_raw_gate_passed": main_raw_gate_passed,
        "raw_score_gates": raw_gates,
        "baseline_order": list(BASELINE_ORDER),
        "test_label_order": common["test_labels"],
        "test_label_order_sha256": common["test_label_order_sha256"],
        "n_test_labels": N_TEST,
        "n_selected_genes": N_SELECTED,
        "selected_gene_order_sha256": EXPECTED_SHA256["E161_genes"],
        "n_pca_coordinates": N_PCA,
        "E162_failure_phase": common["e162"]["status"]["phase"],
        "E162_failure_overwritten": False,
        "E163_binding": {
            "status_sha256": EXPECTED_SHA256["E163_status"],
            "interface_sha256": EXPECTED_SHA256["E163_interface"],
            "manifest_sha256": EXPECTED_SHA256["E163_manifest"],
            "authorization_gate_sha256": EXPECTED_SHA256["E163_authorization"],
        },
        "E162b_binding": {
            "status_sha256": EXPECTED_SHA256["E162b_status"],
            "interface_sha256": EXPECTED_SHA256["E162b_interface"],
            "manifest_sha256": EXPECTED_SHA256["E162b_manifest"],
        },
        "query_event": {
            "created": QUERY_EVENT.exists(),
            "path": (
                "../TEST_LABEL_QUERY_EVENT.json" if QUERY_EVENT.exists() else None
            ),
            "sha256": event_sha256,
        },
        "paths": paths,
        "artifact_sha256": artifact_sha256,
        "access_boundary": {
            "raw_Wessels_opened": False,
            "train_X_rows_indexed_materialized_transformed": N_TRAIN,
            "validation_X_rows_indexed_materialized_transformed": 0,
            "test_X_rows_indexed_materialized_transformed": 0,
            "excluded_X_rows_indexed_materialized_transformed": 0,
            "test_label_strings_forwarded": N_TEST if query_performed else 0,
            "test_graphs": 0,
            "test_truth_effect_error_DE_or_endpoint_used": False,
        },
    }
    atomic_json(STAGING / "E164_E165_INTERFACE.json", interface)

    manifest_paths = sorted(
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
        for path in manifest_paths
    ]
    manifest = pd.DataFrame(manifest_rows)
    atomic_csv(manifest, STAGING / "RESULTS_SHA256.csv")

    completed_at = now_text()
    status = {
        "schema": "safeconf_e164_pretruth_lock_status_v1",
        "experiment": "E164_wessels_pretruth_lock",
        "phase": phase,
        "transaction_id": transaction_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "git_head": common["head"],
        "baseline_arm_authorized": True,
        "prescribe_arm_authorized": prescribe_arm_authorized,
        "e163_validation_gate_passed": common["e163"]["validation_gate_passed"],
        "e163_authorize_future_test_label_lock": common["e163"]["interface"][
            "authorize_future_test_label_lock"
        ],
        "test_label_queries_started": query_performed,
        "test_label_strings_forwarded": N_TEST if query_performed else 0,
        "main_raw_gate_passed": main_raw_gate_passed,
        "raw_score_gates": raw_gates,
        "raw_Wessels_opened": False,
        "train_X_rows_indexed_materialized_transformed": N_TRAIN,
        "validation_X_rows_indexed_materialized_transformed": 0,
        "test_X_rows_indexed_materialized_transformed": 0,
        "excluded_X_rows_indexed_materialized_transformed": 0,
        "test_X_accessed": False,
        "test_truth_accessed": False,
        "test_endpoint_computed": False,
        "test_graphs": 0,
        "E162_failure_phase": common["e162"]["status"]["phase"],
        "E162_failure_overwritten": False,
        "n_baseline_predictors": len(BASELINE_ORDER),
        "baseline_order": list(BASELINE_ORDER),
        "n_test_tasks": N_TEST,
        "n_selected_genes": N_SELECTED,
        "n_pca_coordinates": N_PCA,
        "systema_condition_balanced_minus_cellweighted_RMS": systema[
            "condition_balanced_minus_cell_weighted_rms"
        ],
        "test_label_query_event_sha256": event_sha256,
        "results_manifest_sha256": sha256_file(STAGING / "RESULTS_SHA256.csv"),
        "artifact_sha256": {
            row["relative_path"]: row["sha256"] for row in manifest_rows
        },
    }
    atomic_json(STAGING / "RUN_STATUS.json", status)

    expected = set(BASE_RELEASE_FILES)
    if query_performed:
        expected.update(PRESCRIBE_RELEASE_FILES)
    observed = {
        path.relative_to(STAGING).as_posix()
        for path in STAGING.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise RuntimeError(
            f"E164 release allowlist mismatch: {sorted(observed ^ expected)}"
        )
    if any(path.is_symlink() for path in STAGING.rglob("*")):
        raise RuntimeError("Symlink rejected in E164 release staging")
    for row in manifest_rows:
        path = STAGING / row["relative_path"]
        if (
            path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise RuntimeError(f"Post-write hash mismatch: {row['relative_path']}")
    fsync_tree(STAGING)
    STAGING.replace(RELEASE)
    fsync_directory(OUT)
    return status


def record_failure(error: BaseException) -> None:
    FAILURES.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    payload = {
        "schema": "safeconf_e164_failure_v1",
        "experiment": "E164_wessels_pretruth_lock",
        "failed_at": now_text(),
        "error_type": type(error).__name__,
        "error": repr(error),
        "traceback": traceback.format_exc(),
        "query_event_exists": QUERY_EVENT.exists(),
        "query_event_sha256": sha256_file(QUERY_EVENT) if QUERY_EVENT.exists() else None,
        "staging_preserved": STAGING.exists(),
        "release_exists": RELEASE.exists(),
        "replay_forbidden": True,
    }
    atomic_json(FAILURES / f"E164_FAILURE_{stamp}.json", payload)


def preflight(common: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "safeconf_e164_preflight_v1",
        "experiment": "E164_wessels_pretruth_lock",
        "mode": "preflight",
        "phase": "metadata_hash_contracts_passed_no_development_X_graph_or_checkpoint_opened",
        "git_head": common["head"],
        "runner_committed_and_matching": common["own_git"]["runner"][
            "matches_head_blob"
        ],
        "contract_committed_and_matching": common["own_git"]["contract"][
            "matches_head_blob"
        ],
        "baseline_arm_will_be_authorized": True,
        "e163_validation_gate_passed": common["e163"]["validation_gate_passed"],
        "prescribe_label_query_planned": common["e163"]["validation_gate_passed"],
        "test_labels": N_TEST,
        "test_label_order_sha256": common["test_label_order_sha256"],
        "development_H5AD_opened": False,
        "development_graph_cache_opened": False,
        "checkpoint_opened": False,
        "raw_Wessels_opened": False,
        "test_X_accessed": False,
        "test_truth_accessed": False,
    }


def formal(common: dict[str, Any], *, gpu_index: int) -> dict[str, Any]:
    if gpu_index < 0:
        raise RuntimeError("GPU physical index must be nonnegative")
    if RELEASE.exists() or STAGING.exists() or QUERY_EVENT.exists() or FAILURES.exists():
        raise RuntimeError(
            "E164 is append-only: existing release/staging/query-event/failure forbids replay"
        )
    e162 = common["e162_module"]
    control, pca_mean, components, genes = e162.load_control_and_pca(
        common["native_gate"]["interface"]
    )
    if (
        len(genes) != N_SELECTED
        or sha256_bytes(("\n".join(genes) + "\n").encode("utf-8"))
        != EXPECTED_SHA256["E161_genes"]
    ):
        raise RuntimeError("E161 selected-gene axis changed during formal load")
    train_x, train_conditions, _train_access = load_train_expression_only(genes)
    systema = build_systema_reference(
        train_x,
        train_conditions,
        [str(value) for value in common["native_gate"]["split"]["train"]],
        control,
        pca_mean,
        components,
        genes,
    )
    baseline = build_baseline_assets(
        common, systema, control, pca_mean, components, genes
    )
    del train_x

    started_at = now_text()
    transaction_id = uuid.uuid4().hex
    native = None
    queried: dict[str, Any] | None = None
    try:
        modules: dict[int, Any] = {}
        if common["e163"]["validation_gate_passed"]:
            if "torch" in sys.modules:
                raise RuntimeError(
                    "torch was imported before E164 fixed CUDA_VISIBLE_DEVICES"
                )
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            native = e162.import_native_stack()
            # Only two scalar architecture fields are required to reconstruct
            # the already-trained modules.  This deliberately avoids opening
            # E161 development graphs or validation expression.
            architecture = SimpleNamespace(
                nodes_num=N_SELECTED,
                num_pert=len(common["perturbation_genes"]),
            )
            for seed in SEEDS:
                module, _checkpoint_audit = e162.load_locked_seed_module(
                    native, architecture, E162_ATTEMPT, seed
                )
                modules[seed] = module

        STAGING.mkdir(parents=True, exist_ok=False)
        transaction = {
            "schema": "e164_atomic_staging_v1",
            "experiment": "E164_wessels_pretruth_lock",
            "transaction_id": transaction_id,
            "target": str(RELEASE),
            "created_at": started_at,
            "baseline_arm_authorized": True,
            "e163_validation_gate_passed": common["e163"][
                "validation_gate_passed"
            ],
            "planned_label_query": common["e163"]["validation_gate_passed"],
        }
        atomic_json(STAGING / ".E164_TRANSACTION.json", transaction)

        if common["e163"]["validation_gate_passed"]:
            event = build_query_event(common, gpu_index=gpu_index)
            atomic_json(QUERY_EVENT, event)
            if not QUERY_EVENT.is_file() or sha256_file(QUERY_EVENT) != sha256_bytes(
                json_bytes(event)
            ):
                raise RuntimeError("E164 query event did not persist exactly")
            queried = query_prescribe(
                common,
                modules,
                control,
                pca_mean,
                components,
                genes,
                STAGING,
            )
    finally:
        if native is not None:
            e162.restore_import_context(native)

    return write_release(
        common,
        systema,
        baseline,
        queried,
        transaction_id,
        started_at,
    )


def main() -> None:
    args = parse_args()
    try:
        common = common_preflight(formal=args.mode == "formal")
        result = (
            preflight(common)
            if args.mode == "preflight"
            else formal(common, gpu_index=args.gpu_index)
        )
    except BaseException as error:
        if args.mode == "formal":
            record_failure(error)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
