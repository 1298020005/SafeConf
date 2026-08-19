#!/usr/bin/env python3
"""Evaluate the sealed E168 test donor after the committed pretruth gate.

This process accepts only the F2 prediction release and the F3 isolated truth
bundle.  It never opens the 44.6 GB source H5AD.  A PASS snapshot must be
byte-identical to a commit present on both configured remotes before any F3
array is loaded.  All registered tests, ties, and negative outcomes are
written without changing the frozen endpoint or decision lines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping
import uuid

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E168_primary_human_cd4_fresh_confirmation_20260716"
PRETRUTH = OUT / "pretruth_release"
SNAPSHOT = PRETRUTH / "PRETRUTH_GATE_SNAPSHOT.json"
F2 = Path(
    "/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/"
    "isolated/F2_pretruth"
)
F3 = Path(
    "/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/"
    "isolated/F3_postgate"
)
RELEASE = OUT / "postgate_release"
STAGING = OUT / ".postgate_release.staging"
STAT_LOCK = OUT / "STATISTICAL_ANALYSIS_LOCK.json"
MODEL_LOCK = OUT / "MODEL_INPUT_LOCK.json"
TASK_LOCK = OUT / "manifests/E168_TASK_MANIFEST.csv"
TARGET_LOCK = OUT / "manifests/E168_SELECTED_TARGETS.csv"
PRETRUTH_RUNNER = ROOT / "tools/scripts/run_e168_primary_cd4_pretruth.py"
ASSET_BUILDER = ROOT / "tools/scripts/build_e168_primary_cd4_isolated_assets.py"
CODE_ROOT = ROOT / "code/20260426_154505_perturb_transport_final_push"

STATES = ("Rest", "Stim8hr", "Stim48hr")
N_GENES = 512
N_TARGETS = 200
N_SEEN = 160
N_UNSEEN = 40
TOLERANCE = 1e-6
COVERAGES = np.linspace(0.20, 1.00, 17)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2026071681
PERMUTATION_DRAWS = 100_000
PERMUTATION_SEED = 2026071682
PREDICTOR_NAMES = (
    "scGPT_seed3407", "scGPT_seed3408", "scGPT_seed3409",
    "GEARS_seed3407", "GEARS_seed3408", "GEARS_seed3409",
    "scGPT_seed_mean", "GEARS_seed_mean", "ensemble_seed_family_mean",
)
F2_ALLOWLIST = {
    "GENE_PANEL.csv", "CONTROL_PROFILES.npz", "SEEN_TARGET_EFFECTS.npz",
    "PRETRUTH_TASKS.csv", "PRETRUTH_GUIDE_EFFECT_INDEX.csv",
    "TRAIN_NTC_COEXPRESSION_EDGES.csv",
    "TRAIN_NTC_COEXPRESSION_PROFILE_INDEX.csv", "ROW_ACCESS_AUDIT.csv",
    "ACCESS_ATTESTATION.json", "MANIFEST.sha256",
}
F3_ALLOWLIST = {
    "TEST_TARGET_EFFECTS.npz", "TEST_GUIDE_EFFECTS.npz",
    "TEST_GUIDE_EFFECT_INDEX.csv", "TEST_TASKS.csv", "ROW_ACCESS_AUDIT.csv",
    "ACCESS_ATTESTATION.json", "MANIFEST.sha256",
}


class IntegrityFailure(RuntimeError):
    """The sealed evaluation contract was not satisfied."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def git_text(*args: str) -> str:
    return git(*args).stdout.decode("utf-8").strip()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def strict_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise IntegrityFailure(f"{name} contains a non-boolean value")
    return normalized.eq("true")


def require_committed(path: Path, head: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise IntegrityFailure(f"missing/symlinked code-freeze input: {path}")
    relative = path.relative_to(ROOT).as_posix()
    try:
        committed = git("show", f"{head}:{relative}").stdout
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"file is not committed at HEAD: {relative}") from exc
    if committed != path.read_bytes():
        raise IntegrityFailure(f"working file differs from HEAD: {relative}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_gate_commit(gate_commit: str, branch: str) -> tuple[dict[str, Any], dict[str, str]]:
    head = git_text("rev-parse", "HEAD")
    if git("cat-file", "-e", f"{gate_commit}^{{commit}}", check=False).returncode:
        raise IntegrityFailure("supplied gate commit is unavailable")
    if git("merge-base", "--is-ancestor", gate_commit, head, check=False).returncode:
        raise IntegrityFailure("current HEAD does not contain the supplied gate commit")
    relative = SNAPSHOT.relative_to(ROOT).as_posix()
    try:
        committed_snapshot = git("show", f"{gate_commit}:{relative}").stdout
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure("gate snapshot is absent from the supplied commit") from exc
    if committed_snapshot != SNAPSHOT.read_bytes():
        raise IntegrityFailure("local gate snapshot differs from committed bytes")
    script_relative = SCRIPT.relative_to(ROOT).as_posix()
    try:
        committed_evaluator = git("show", f"{gate_commit}:{script_relative}").stdout
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure("postgate evaluator was not frozen before the gate") from exc
    if committed_evaluator != SCRIPT.read_bytes():
        raise IntegrityFailure("postgate evaluator changed after the pretruth gate")

    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched = f"refs/remotes/{remote}/{branch}"
        result = git(
            "fetch", "--quiet", remote, f"refs/heads/{branch}:{fetched}", check=False,
        )
        if result.returncode:
            raise IntegrityFailure(
                f"cannot verify gate commit on {remote}: "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        remote_head = git_text("rev-parse", fetched)
        if git("merge-base", "--is-ancestor", gate_commit, remote_head, check=False).returncode:
            raise IntegrityFailure(f"gate commit is absent from {remote}/{branch}")
        remote_heads[remote] = remote_head

    snapshot = json.loads(committed_snapshot)
    required = {
        "schema": "safeconf_e168_pretruth_gate_snapshot_v1",
        "experiment": "E168_primary_human_cd4_fresh_confirmation",
        "stage": "F2_PRETRUTH_GATE",
        "status": "PASS",
        "all_registered_gates_passed": True,
        "test_targeting_x_values_read": 0,
        "forbidden_column_unseen_x_values_read": 0,
        "test_query_graphs_containing_y": 0,
        "train_reference_task_count": 960,
        "validation_query_count": 600,
        "test_query_count": 600,
        "deployment_authorized": False,
    }
    mismatches = {
        key: {"expected": value, "observed": snapshot.get(key)}
        for key, value in required.items() if snapshot.get(key) != value
    }
    if mismatches:
        raise IntegrityFailure(f"gate snapshot is not an exact PASS: {mismatches}")
    if snapshot.get("runner_sha256") != sha256_file(PRETRUTH_RUNNER):
        raise IntegrityFailure("gate snapshot is not bound to the current pretruth runner")
    if snapshot.get("asset_builder_sha256") != sha256_file(ASSET_BUILDER):
        raise IntegrityFailure("gate snapshot is not bound to the current asset builder")
    return snapshot, remote_heads


def parse_flat_manifest(directory: Path, allowlist: set[str]) -> tuple[dict[str, str], str]:
    if directory.is_symlink() or not directory.is_dir():
        raise IntegrityFailure(f"isolated directory is missing or symlinked: {directory}")
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if observed != allowlist or any(path.is_dir() for path in directory.iterdir()):
        raise IntegrityFailure(f"isolated directory allowlist failed: {directory}")
    manifest_path = directory / "MANIFEST.sha256"
    rows: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if len(digest) != 64 or "/" in name or name in rows:
            raise IntegrityFailure(f"invalid isolated manifest entry: {line}")
        rows[name] = digest
    if set(rows) != allowlist - {"MANIFEST.sha256"}:
        raise IntegrityFailure("isolated manifest entries differ from allowlist")
    for name, digest in rows.items():
        if sha256_file(directory / name) != digest:
            raise IntegrityFailure(f"isolated asset hash mismatch: {directory / name}")
    return rows, sha256_file(manifest_path)


def verify_pretruth_files(snapshot: dict[str, Any], gate_commit: str) -> None:
    registered = snapshot.get("pretruth_files_sha256")
    if not isinstance(registered, dict) or not registered:
        raise IntegrityFailure("snapshot contains no pretruth file hashes")
    permitted_extra = {
        "PRETRUTH_GATE_SNAPSHOT.json", "reports/E168_PRETRUTH_REPORT.md"
    }
    observed = {
        path.relative_to(PRETRUTH).as_posix()
        for path in PRETRUTH.rglob("*") if path.is_file()
    }
    if observed != set(registered) | permitted_extra:
        raise IntegrityFailure(
            f"pretruth release file set changed: extra={sorted(observed-set(registered)-permitted_extra)}"
        )
    for relative, expected in registered.items():
        path = PRETRUTH / relative
        if path.is_symlink() or sha256_file(path) != expected:
            raise IntegrityFailure(f"pretruth release hash changed: {relative}")
        git_path = path.relative_to(ROOT).as_posix()
        try:
            committed = git("show", f"{gate_commit}:{git_path}").stdout
        except subprocess.CalledProcessError as exc:
            raise IntegrityFailure(f"pretruth file absent from gate commit: {relative}") from exc
        if hashlib.sha256(committed).hexdigest() != expected or committed != path.read_bytes():
            raise IntegrityFailure(f"pretruth file is not byte-identical to gate commit: {relative}")


def verify_gate_certificates(snapshot: dict[str, Any]) -> dict[str, Any]:
    specifications = {
        "G2_SCORE_CERTIFICATES.csv": 6,
        "G3_PREDICTOR_CERTIFICATES.csv": 24,
        "G4_SEED_STABILITY.csv": 6,
        "SYNTHETIC_REGRESSION_TESTS.csv": 10,
    }
    result: dict[str, Any] = {}
    for name, expected_rows in specifications.items():
        frame = pd.read_csv(PRETRUTH / "tables" / name, keep_default_na=False)
        if len(frame) != expected_rows or "passed" not in frame.columns:
            raise IntegrityFailure(f"pretruth gate certificate count/schema failed: {name}")
        passed = strict_bool(frame.passed, f"{name}:passed")
        if not passed.all():
            raise IntegrityFailure(f"pretruth gate certificate contains a failed unit: {name}")
        result[name] = {"rows": len(frame), "passed": int(passed.sum())}
    if snapshot.get("registered_g2_units") != 6 or snapshot.get("registered_g4_units") != 6:
        raise IntegrityFailure("snapshot registered gate-unit counts changed")
    if snapshot.get("synthetic_tests_passed") != 10:
        raise IntegrityFailure("snapshot synthetic-test count changed")
    g5 = pd.read_csv(
        PRETRUTH / "tables/G5_MAGNITUDE_EQUIVALENCE.csv", keep_default_na=False
    )
    if len(g5) != 6 or "risk_magnitude_operational_weak_order_identical" not in g5:
        raise IntegrityFailure("G5 magnitude-equivalence certificate failed")
    equivalent = strict_bool(
        g5.risk_magnitude_operational_weak_order_identical,
        "G5:risk_magnitude_operational_weak_order_identical",
    )
    result["G5_MAGNITUDE_EQUIVALENCE.csv"] = {
        "rows": len(g5), "equivalent_units": int(equivalent.sum())
    }
    return result


def load_npz_vectors(path: Path, expected_size: int | None = None) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            values = np.asarray(archive[key], dtype=np.float32)
            if values.ndim != 1 or values.shape != (N_GENES,) or not np.isfinite(values).all():
                raise IntegrityFailure(f"invalid vector: {path.name}/{key}/{values.shape}")
            result[str(key)] = values
    if expected_size is not None and len(result) != expected_size:
        raise IntegrityFailure(f"unexpected vector count in {path.name}: {len(result)}")
    return result


def load_and_validate_inputs(
    gate_commit: str,
    branch: str,
) -> tuple[
    pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray],
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[str, str],
    list[dict[str, Any]],
]:
    head = git_text("rev-parse", "HEAD")
    code_files = [
        SCRIPT, PRETRUTH_RUNNER, ASSET_BUILDER, STAT_LOCK, MODEL_LOCK,
        TASK_LOCK, TARGET_LOCK,
    ]
    code_hashes = [require_committed(path, head) for path in code_files]
    snapshot, remote_heads = verify_gate_commit(gate_commit, branch)
    verify_pretruth_files(snapshot, gate_commit)
    gate_certificates = verify_gate_certificates(snapshot)
    _f2_entries, f2_manifest_sha = parse_flat_manifest(F2, F2_ALLOWLIST)
    _f3_entries, f3_manifest_sha = parse_flat_manifest(F3, F3_ALLOWLIST)
    if snapshot.get("f2_manifest_sha256") != f2_manifest_sha:
        raise IntegrityFailure("snapshot does not bind the current F2 manifest")

    f2_attestation = json.loads((F2 / "ACCESS_ATTESTATION.json").read_text())
    f3_attestation = json.loads((F3 / "ACCESS_ATTESTATION.json").read_text())
    if f2_attestation.get("source_full_sha256") != snapshot.get("source_full_sha256"):
        raise IntegrityFailure("F2 source SHA differs from the gate snapshot")
    required_f3 = {
        "stage": "F3_POSTGATE_ISOLATED_TRUTH_BUILD",
        "status": "PASS",
        "all_registered_pretruth_gates_passed": True,
        "f2_manifest_sha256": f2_manifest_sha,
        "source_full_sha256": snapshot.get("source_full_sha256"),
        "gate_commit": gate_commit,
        "postgate_test_targeting_x_values_read": 1200,
        "forbidden_column_unseen_x_values_read": 0,
        "train_or_validation_targeting_x_values_read_in_postgate": 0,
        "n_test_target_effects": 600,
        "n_test_guide_effects": 1200,
        "test_performance_metrics_computed": 0,
    }
    mismatches = {
        key: {"expected": value, "observed": f3_attestation.get(key)}
        for key, value in required_f3.items() if f3_attestation.get(key) != value
    }
    if mismatches:
        raise IntegrityFailure(f"F3 access attestation failed: {mismatches}")
    if f3_attestation.get("builder_sha256") != sha256_file(ASSET_BUILDER):
        raise IntegrityFailure("F3 truth is not bound to the current asset builder")
    if f3_attestation.get("gate_snapshot_sha256") != sha256_file(SNAPSHOT):
        raise IntegrityFailure("F3 truth is not bound to the committed gate snapshot")

    f3_access = pd.read_csv(F3 / "ROW_ACCESS_AUDIT.csv", keep_default_na=False)
    if len(f3_access) != 1200 or f3_access.metadata_row_index.astype(int).duplicated().any():
        raise IntegrityFailure("F3 row-access audit is not exactly 1,200 unique rows")
    if f3_access.x_access_phase.value_counts().to_dict() != {
        "POSTGATE_TEST_TRUTH_X": 1200
    } or not f3_access.logical_x_row_read_count.astype(int).eq(1).all():
        raise IntegrityFailure("F3 row-access phase/exactly-once audit failed")

    panel = pd.read_csv(F2 / "GENE_PANEL.csv", keep_default_na=False)
    if len(panel) != N_GENES or panel.panel_index.astype(int).tolist() != list(range(N_GENES)):
        raise IntegrityFailure("F2 panel order/count changed")
    if panel.ensembl_id.astype(str).nunique() != N_GENES:
        raise IntegrityFailure("F2 panel Ensembl IDs are not unique")
    gene_order_hash = "sha256:" + hashlib.sha256(
        "\n".join(panel.scgpt_token.astype(str)).encode()
    ).hexdigest()
    panel["gene_order_hash"] = gene_order_hash

    scoring = pd.read_csv(
        PRETRUTH / "tables/PRETRUTH_SCORING_INTERFACE.csv", keep_default_na=False
    )
    if len(scoring) != 2160 or scoring.task_id.astype(str).nunique() != 2160:
        raise IntegrityFailure("pretruth scoring interface count/key failed")
    test_scores = scoring.loc[scoring.donor_role.eq("test")].copy().reset_index(drop=True)
    if len(test_scores) != 600:
        raise IntegrityFailure("pretruth interface does not contain 600 test queries")
    required_score_columns = {
        "task_id", "donor_id", "culture_condition", "perturbed_gene_id",
        "perturbed_gene_name", "target_stratum", "safeconf_risk",
        "predicted_magnitude", "model_disagreement_rmse", "z_context_train960",
        "z_log_support_train960", "z_disagreement_train960",
    }
    if not required_score_columns.issubset(test_scores.columns):
        raise IntegrityFailure("pretruth scoring interface schema failed")
    numeric_score_columns = [
        "safeconf_risk", "predicted_magnitude", "model_disagreement_rmse",
        "z_context_train960", "z_log_support_train960", "z_disagreement_train960",
    ]
    if not np.isfinite(test_scores[numeric_score_columns].to_numpy(float)).all():
        raise IntegrityFailure("pretruth test scores contain a non-finite value")

    with np.load(
        PRETRUTH / "arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False
    ) as archive:
        if set(archive.files) != set(PREDICTOR_NAMES):
            raise IntegrityFailure("pretruth predictor array names changed")
        prediction_matrices = {
            name: np.asarray(archive[name], dtype=np.float32) for name in archive.files
        }
    for name, matrix in prediction_matrices.items():
        if matrix.shape != (2160, N_GENES) or not np.isfinite(matrix).all():
            raise IntegrityFailure(f"pretruth prediction matrix failed: {name}/{matrix.shape}")

    task_to_pretruth_row = {
        task: index for index, task in enumerate(scoring.task_id.astype(str))
    }
    test_prediction_matrices = {
        name: np.stack([
            matrix[task_to_pretruth_row[task]] for task in test_scores.task_id.astype(str)
        ]).astype(np.float32)
        for name, matrix in prediction_matrices.items()
    }
    truth = load_npz_vectors(F3 / "TEST_TARGET_EFFECTS.npz", expected_size=600)
    guide_truth = load_npz_vectors(F3 / "TEST_GUIDE_EFFECTS.npz", expected_size=1200)
    if set(truth) != set(test_scores.task_id.astype(str)):
        raise IntegrityFailure("F3 target truth keys differ from pretruth test queries")

    f3_tasks = pd.read_csv(F3 / "TEST_TASKS.csv", keep_default_na=False)
    if len(f3_tasks) != 600 or set(f3_tasks.task_id.astype(str)) != set(truth):
        raise IntegrityFailure("F3 test-task table differs from truth arrays")
    locked_tasks = pd.read_csv(TASK_LOCK, keep_default_na=False)
    locked_test = locked_tasks.loc[strict_bool(locked_tasks.primary_test_task, "primary_test_task")]
    shared = list(locked_tasks.columns)
    if not f3_tasks[shared].sort_values("task_id").reset_index(drop=True).astype(str).equals(
        locked_test[shared].sort_values("task_id").reset_index(drop=True).astype(str)
    ):
        raise IntegrityFailure("F3 test-task metadata differs from frozen task manifest")

    seen_effects = load_npz_vectors(F2 / "SEEN_TARGET_EFFECTS.npz", expected_size=1440)
    metadata = {
        "snapshot": snapshot,
        "remote_heads": remote_heads,
        "f2_manifest_sha256": f2_manifest_sha,
        "f3_manifest_sha256": f3_manifest_sha,
        "f2_attestation": f2_attestation,
        "f3_attestation": f3_attestation,
        "gene_order_hash": gene_order_hash,
        "gate_commit": gate_commit,
        "git_head": head,
        "gate_certificates": gate_certificates,
    }
    return (
        panel, test_scores, test_prediction_matrices, truth, guide_truth,
        seen_effects, metadata, remote_heads, code_hashes,
    )


def quantize(values: np.ndarray, tolerance: float = TOLERANCE) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise IntegrityFailure("cannot quantize non-finite values")
    return np.rint(values / tolerance).astype(np.int64)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def cosine_error(a: np.ndarray, b: np.ndarray) -> float:
    left, right = np.asarray(a, float), np.asarray(b, float)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0 if np.linalg.norm(left - right) <= 1e-12 else 1.0
    return float(1.0 - np.dot(left, right) / denominator)


def spearman(a: Iterable[float], b: Iterable[float]) -> float:
    left, right = np.asarray(list(a), float), np.asarray(list(b), float)
    if len(left) < 4 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return float("nan")
    return float(np.corrcoef(
        rankdata(left, method="average"), rankdata(right, method="average")
    )[0, 1])


def risk_at_k(score: np.ndarray, loss: np.ndarray, k: int) -> dict[str, float]:
    labels = quantize(score)
    order = np.argsort(labels, kind="mergesort")
    threshold = labels[order[k - 1]]
    strict_loss = loss[labels < threshold]
    tied_loss = np.sort(loss[labels == threshold])
    slots = int(k - len(strict_loss))
    if slots < 1 or slots > len(tied_loss):
        raise IntegrityFailure("invalid tie boundary while constructing RC curve")
    strict_sum = float(np.sum(strict_loss))
    return {
        "risk_tie_average": (
            strict_sum + slots / len(tied_loss) * float(np.sum(tied_loss))
        ) / k,
        "risk_best_legal_tie_order": (
            strict_sum + float(np.sum(tied_loss[:slots]))
        ) / k,
        "risk_worst_legal_tie_order": (
            strict_sum + float(np.sum(tied_loss[-slots:]))
        ) / k,
    }


def tie_aware_curve(score: np.ndarray, loss: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    values, errors = np.asarray(score, float), np.asarray(loss, float)
    if len(values) != len(errors) or len(values) < 2 or not np.isfinite(errors).all():
        raise IntegrityFailure("invalid score/loss block for RC curve")
    rows = []
    for coverage in COVERAGES:
        k = max(1, int(math.ceil(float(coverage) * len(values))))
        rows.append({
            "coverage": float(coverage), "selected_k": k,
            **risk_at_k(values, errors, k),
        })
    curve = pd.DataFrame(rows)
    span = float(COVERAGES[-1] - COVERAGES[0])
    summaries = {
        "aurc_tie_average": float(np.trapezoid(
            curve.risk_tie_average, curve.coverage
        ) / span),
        "aurc_best_legal_tie_order": float(np.trapezoid(
            curve.risk_best_legal_tie_order, curve.coverage
        ) / span),
        "aurc_worst_legal_tie_order": float(np.trapezoid(
            curve.risk_worst_legal_tie_order, curve.coverage
        ) / span),
    }
    return curve, summaries


def tie_aware_aurc_value(score: np.ndarray, loss: np.ndarray) -> float:
    """Tie-average AURC with one quantization/sort pass per ranking batch."""
    labels = quantize(score)
    errors = np.asarray(loss, dtype=np.float64)
    if len(labels) != len(errors) or not np.isfinite(errors).all():
        raise IntegrityFailure("invalid score/loss block for fast AURC")
    order = np.argsort(labels, kind="mergesort")
    ordered_labels = labels[order]
    ordered_loss = errors[order]
    prefix = np.concatenate(([0.0], np.cumsum(ordered_loss, dtype=np.float64)))
    risks = []
    for coverage in COVERAGES:
        k = max(1, int(math.ceil(float(coverage) * len(labels))))
        threshold = ordered_labels[k - 1]
        left = int(np.searchsorted(ordered_labels, threshold, side="left"))
        right = int(np.searchsorted(ordered_labels, threshold, side="right"))
        slots = k - left
        tied_sum = prefix[right] - prefix[left]
        expected_sum = prefix[left] + slots / (right - left) * tied_sum
        risks.append(expected_sum / k)
    return float(np.trapezoid(risks, COVERAGES) / (COVERAGES[-1] - COVERAGES[0]))


def build_task_metrics(
    scores: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    seen_effects: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    task_ids = scores.task_id.astype(str).tolist()
    truth_matrix = np.stack([truth[task] for task in task_ids]).astype(np.float32)
    ensemble = predictions["ensemble_seed_family_mean"]
    if ensemble.shape != truth_matrix.shape:
        raise IntegrityFailure("ensemble and truth matrix shapes differ")

    f2_tasks = pd.read_csv(F2 / "PRETRUTH_TASKS.csv", keep_default_na=False)
    train_donors = sorted(set(f2_tasks.loc[f2_tasks.donor_role.eq("train"), "donor_id"].astype(str)))
    if len(train_donors) != 2:
        raise IntegrityFailure("TrainDonorEffectMean requires exactly two train donors")
    donor_mean: dict[str, np.ndarray] = {}
    for row in scores.itertuples(index=False):
        if str(row.target_stratum) != "DONOR_UNSEEN_ONLY":
            continue
        keys = [
            f"E168::{donor}::{row.culture_condition}::{row.perturbed_gene_id}"
            for donor in train_donors
        ]
        if any(key not in seen_effects for key in keys):
            raise IntegrityFailure(f"missing train-donor effect for {row.task_id}")
        donor_mean[str(row.task_id)] = np.mean(
            np.stack([seen_effects[key] for key in keys]), axis=0
        ).astype(np.float32)
    if len(donor_mean) != 3 * N_SEEN:
        raise IntegrityFailure("TrainDonorEffectMean coverage is not 480 seen test tasks")

    metrics = scores.copy()
    metrics["true_error_rmse"] = np.sqrt(
        np.mean((ensemble.astype(float) - truth_matrix.astype(float)) ** 2, axis=1)
    )
    metrics["nochange_error_rmse"] = np.sqrt(
        np.mean(truth_matrix.astype(float) ** 2, axis=1)
    )
    metrics["ensemble_beats_nochange"] = (
        metrics.true_error_rmse < metrics.nochange_error_rmse
    )
    metrics["train_donor_effect_mean_error_rmse"] = np.nan
    task_index = {task: index for index, task in enumerate(task_ids)}
    for task, vector in donor_mean.items():
        index = task_index[task]
        metrics.loc[index, "train_donor_effect_mean_error_rmse"] = rmse(
            vector, truth_matrix[index]
        )
    metrics["disagreement_only_risk"] = metrics.model_disagreement_rmse.astype(float)
    metrics["support_only_risk"] = -metrics.z_log_support_train960.astype(float)
    metrics["context_only_risk"] = -metrics.z_context_train960.astype(float)
    metrics["magnitude_risk"] = metrics.predicted_magnitude.astype(float)
    if not np.isfinite(metrics.true_error_rmse.to_numpy(float)).all():
        raise IntegrityFailure("formal ensemble errors contain non-finite values")
    return metrics, {task: truth_matrix[i] for i, task in enumerate(task_ids)}, donor_mean


def registered_blocks(frame: pd.DataFrame) -> Iterable[tuple[str, str, pd.DataFrame]]:
    for state in STATES:
        state_frame = frame.loc[frame.culture_condition.eq(state)]
        masks = {
            "all_200": np.ones(len(state_frame), dtype=bool),
            "seen_160": state_frame.target_stratum.eq("DONOR_UNSEEN_ONLY").to_numpy(),
            "column_unseen_40_descriptive": state_frame.target_stratum.eq("COLUMN_UNSEEN").to_numpy(),
        }
        for stratum, mask in masks.items():
            block = state_frame.loc[mask].copy()
            expected = {
                "all_200": N_TARGETS,
                "seen_160": N_SEEN,
                "column_unseen_40_descriptive": N_UNSEEN,
            }[stratum]
            if len(block) != expected or block.perturbed_gene_id.astype(str).nunique() != expected:
                raise IntegrityFailure(f"registered state/stratum block failed: {state}/{stratum}")
            yield state, stratum, block


def compute_curves_and_aurc(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_columns = {
        "SafeConf": "safeconf_risk",
        "magnitude": "magnitude_risk",
        "disagreement_only": "disagreement_only_risk",
        "support_only": "support_only_risk",
        "context_only": "context_only_risk",
    }
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for state, stratum, block in registered_blocks(frame):
        loss = block.true_error_rmse.to_numpy(float)
        for score_name, column in score_columns.items():
            curve, summary = tie_aware_curve(block[column].to_numpy(float), loss)
            curve.insert(0, "score_name", score_name)
            curve.insert(0, "stratum", stratum)
            curve.insert(0, "culture_condition", state)
            curve_rows.extend(curve.to_dict("records"))
            labels = quantize(block[column].to_numpy(float))
            summary_rows.append({
                "culture_condition": state,
                "stratum": stratum,
                "score_name": score_name,
                "n_tasks": len(block),
                "score_quantized_levels": int(np.unique(labels).size),
                "score_max_tie_fraction": float(
                    np.unique(labels, return_counts=True)[1].max() / len(labels)
                ),
                **summary,
            })
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def state_delta_table(aurc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stratum in ("all_200", "seen_160", "column_unseen_40_descriptive"):
        for state in STATES:
            block = aurc.loc[
                aurc.culture_condition.eq(state) & aurc.stratum.eq(stratum)
            ].set_index("score_name")
            if not {"SafeConf", "magnitude"}.issubset(block.index):
                raise IntegrityFailure("AURC table lacks SafeConf or magnitude")
            rows.append({
                "stratum": stratum,
                "culture_condition": state,
                "safeconf_aurc": float(block.loc["SafeConf", "aurc_tie_average"]),
                "magnitude_aurc": float(block.loc["magnitude", "aurc_tie_average"]),
                "delta_magnitude_minus_safeconf": float(
                    block.loc["magnitude", "aurc_tie_average"]
                    - block.loc["SafeConf", "aurc_tie_average"]
                ),
            })
    return pd.DataFrame(rows)


def mean_state_delta(
    frame: pd.DataFrame,
    candidate: np.ndarray | None = None,
    comparator: np.ndarray | None = None,
) -> float:
    working = frame.copy()
    if candidate is not None:
        working["_candidate"] = candidate
        candidate_column = "_candidate"
    else:
        candidate_column = "safeconf_risk"
    if comparator is not None:
        working["_comparator"] = comparator
        comparator_column = "_comparator"
    else:
        comparator_column = "magnitude_risk"
    deltas = []
    for state in STATES:
        block = working.loc[working.culture_condition.eq(state)]
        loss = block.true_error_rmse.to_numpy(float)
        _, candidate_summary = tie_aware_curve(
            block[candidate_column].to_numpy(float), loss
        )
        _, comparator_summary = tie_aware_curve(
            block[comparator_column].to_numpy(float), loss
        )
        deltas.append(
            comparator_summary["aurc_tie_average"]
            - candidate_summary["aurc_tie_average"]
        )
    return float(np.mean(deltas))


def cluster_matrices(frame: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    genes = sorted(frame.perturbed_gene_id.astype(str).unique())
    candidate = np.empty((len(genes), len(STATES)), dtype=np.float64)
    comparator = np.empty_like(candidate)
    loss = np.empty_like(candidate)
    for gene_index, gene in enumerate(genes):
        block = frame.loc[frame.perturbed_gene_id.astype(str).eq(gene)].set_index(
            "culture_condition"
        )
        if set(block.index.astype(str)) != set(STATES) or len(block) != 3:
            raise IntegrityFailure("cluster matrix requires exactly three states per target")
        for state_index, state in enumerate(STATES):
            row = block.loc[state]
            candidate[gene_index, state_index] = float(row.safeconf_risk)
            comparator[gene_index, state_index] = float(row.magnitude_risk)
            loss[gene_index, state_index] = float(row.true_error_rmse)
    return genes, candidate, comparator, loss


def matrix_mean_state_delta(
    candidate: np.ndarray,
    comparator: np.ndarray,
    loss: np.ndarray,
) -> float:
    deltas = [
        tie_aware_aurc_value(comparator[:, state], loss[:, state])
        - tie_aware_aurc_value(candidate[:, state], loss[:, state])
        for state in range(len(STATES))
    ]
    return float(np.mean(deltas))


def cluster_bootstrap(
    frame: pd.DataFrame,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    genes, candidate, comparator, loss = cluster_matrices(frame)
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        take = rng.integers(0, len(genes), size=len(genes))
        values[draw] = matrix_mean_state_delta(
            candidate[take], comparator[take], loss[take]
        )
    return values


def paired_cluster_permutation(
    frame: pd.DataFrame,
    observed: float,
    draws: int = PERMUTATION_DRAWS,
    seed: int = PERMUTATION_SEED,
) -> tuple[float, np.ndarray]:
    genes, candidate, comparator, loss = cluster_matrices(frame)
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=np.float64)
    exceed = 0
    for draw in range(draws):
        swap = rng.integers(0, 2, size=(len(genes), 1), dtype=np.int8).astype(bool)
        permuted_candidate = np.where(swap, comparator, candidate)
        permuted_comparator = np.where(swap, candidate, comparator)
        value = matrix_mean_state_delta(permuted_candidate, permuted_comparator, loss)
        null[draw] = value
        exceed += int(value >= observed)
    return (1 + exceed) / (draws + 1), null


def infer_stratum(frame: pd.DataFrame, stratum: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if stratum == "all_200":
        block = frame.copy()
    elif stratum == "seen_160":
        block = frame.loc[frame.target_stratum.eq("DONOR_UNSEEN_ONLY")].copy()
    else:
        raise ValueError(stratum)
    _genes, candidate, comparator, loss = cluster_matrices(block)
    observed = matrix_mean_state_delta(candidate, comparator, loss)
    bootstrap = cluster_bootstrap(block)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    p_value, null = paired_cluster_permutation(block, observed)
    result = {
        "stratum": stratum,
        "n_genes": int(block.perturbed_gene_id.astype(str).nunique()),
        "n_tasks": len(block),
        "observed_mean_state_delta": observed,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_ci95_lower": float(ci_low),
        "bootstrap_ci95_upper": float(ci_high),
        "permutation_draws": PERMUTATION_DRAWS,
        "permutation_seed": PERMUTATION_SEED,
        "permutation_p_one_sided": float(p_value),
        "permutation_null_mean": float(np.mean(null)),
        "permutation_null_std": float(np.std(null, ddof=0)),
        "permutation_null_sha256_float64": hashlib.sha256(
            np.ascontiguousarray(null, dtype=np.float64).tobytes()
        ).hexdigest(),
    }
    return result, bootstrap, null


def secondary_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for state, stratum, block in registered_blocks(frame):
        risk = block.safeconf_risk.to_numpy(float)
        loss = block.true_error_rmse.to_numpy(float)
        n = len(block)
        k = max(1, int(math.ceil(0.20 * n)))
        high_risk = np.argsort(quantize(risk), kind="mergesort")[::-1][:k]
        high_error = np.argsort(loss, kind="mergesort")[::-1][:k]
        overlap = len(set(high_risk) & set(high_error))
        accepted = np.setdiff1d(np.arange(n), high_risk)
        full_mean = float(np.mean(loss))
        accepted_mean = float(np.mean(loss[accepted]))
        rows.append({
            "culture_condition": state,
            "stratum": stratum,
            "n_tasks": n,
            "spearman_risk_loss": spearman(risk, loss),
            "high_risk_k": k,
            "high_error_k": k,
            "top20_overlap": overlap,
            "top20_high_error_capture": overlap / k,
            "top20_high_error_enrichment": (overlap / k) / (k / n),
            "full_mean_rmse": full_mean,
            "accepted_after_review20_mean_rmse": accepted_mean,
            "review20_relative_rmse_reduction": (
                (full_mean - accepted_mean) / full_mean if full_mean > 0 else 0.0
            ),
        })
        baseline_rows.append({
            "culture_condition": state,
            "stratum": stratum,
            "n_tasks": n,
            "ensemble_beats_nochange_fraction": float(
                block.ensemble_beats_nochange.astype(bool).mean()
            ),
            "ensemble_mean_rmse": float(block.true_error_rmse.mean()),
            "nochange_mean_rmse": float(block.nochange_error_rmse.mean()),
            "train_donor_mean_available_tasks": int(
                block.train_donor_effect_mean_error_rmse.notna().sum()
            ),
            "train_donor_mean_mean_rmse": float(
                block.train_donor_effect_mean_error_rmse.mean()
            ) if block.train_donor_effect_mean_error_rmse.notna().any() else np.nan,
        })
    return pd.DataFrame(rows), pd.DataFrame(baseline_rows)


def guide_consistency(guide_truth: dict[str, np.ndarray]) -> pd.DataFrame:
    index = pd.read_csv(F3 / "TEST_GUIDE_EFFECT_INDEX.csv", keep_default_na=False)
    if len(index) != 1200 or index.task_id.astype(str).nunique() != 600:
        raise IntegrityFailure("test guide index count failed")
    rows = []
    used_keys: set[str] = set()
    for task, block in index.groupby("task_id", sort=True):
        keys = block.guide_effect_asset_key.astype(str).tolist()
        if len(keys) != 2 or len(set(keys)) != 2 or any(key not in guide_truth for key in keys):
            raise IntegrityFailure(f"test task does not have exactly two frozen guides: {task}")
        used_keys.update(keys)
        left, right = guide_truth[keys[0]], guide_truth[keys[1]]
        rows.append({
            "task_id": str(task),
            "donor_id": str(block.donor_id.iloc[0]),
            "culture_condition": str(block.culture_condition.iloc[0]),
            "ensembl_id": str(block.ensembl_id.iloc[0]),
            "target_stratum": str(block.target_stratum.iloc[0]),
            "guide_1": str(block.guide_id.iloc[0]),
            "guide_2": str(block.guide_id.iloc[1]),
            "guide_effect_rmse": rmse(left, right),
            "guide_effect_cosine_similarity": 1.0 - cosine_error(left, right),
            "guide_effect_spearman": spearman(left, right),
        })
    if used_keys != set(guide_truth):
        raise IntegrityFailure("unregistered guide truth array detected")
    return pd.DataFrame(rows)


def build_prediction_records(
    metrics: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    donor_mean: dict[str, np.ndarray],
    gene_order_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    task_ids = metrics.task_id.astype(str).tolist()
    task_index = {task: index for index, task in enumerate(task_ids)}
    predicted_arrays: dict[str, np.ndarray] = {}
    true_arrays = {
        f"{task}::truth": np.asarray(vector, dtype=np.float32)
        for task, vector in sorted(truth.items())
    }
    records: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    all_predictors = [*PREDICTOR_NAMES, "NoChange", "TrainDonorEffectMean"]
    for row in metrics.itertuples(index=False):
        task = str(row.task_id)
        index = task_index[task]
        predictor_vectors = {
            name: predictions[name][index] for name in PREDICTOR_NAMES
        }
        predictor_vectors["NoChange"] = np.zeros(N_GENES, dtype=np.float32)
        if task in donor_mean:
            predictor_vectors["TrainDonorEffectMean"] = donor_mean[task]
        for predictor in all_predictors:
            if predictor not in predictor_vectors:
                continue
            vector = np.asarray(predictor_vectors[predictor], dtype=np.float32)
            true_vector = truth[task]
            record_id = f"{task}::{predictor}"
            predicted_key = f"{record_id}::prediction"
            true_key = f"{task}::truth"
            predicted_arrays[predicted_key] = vector
            shared = {
                "record_id": record_id,
                "task_id": task,
                "task_key": task,
                "dataset_name": "PrimaryHumanCD4_GWCD4i_E168_formal512",
                "fold_id": 16801,
                "split": "test",
                "context": f"{row.donor_id}::{row.culture_condition}",
                "perturbation": str(row.perturbed_gene_name),
                "predictor_name": predictor,
            }
            records.append({
                "schema_version": "safeconf_prediction_record_v1",
                **shared,
                "dataset_group": "fresh_primary_human_cd4_crispri",
                "run_type": "formal",
                "gene_panel_id": "E168_trainNTC_target200_scgpt512_v1",
                "gene_order_hash": gene_order_hash,
                "effect_definition": "mean_diff",
                "normalization_id": (
                    "pseudobulk_UMI_CP10K_log1p_equal_guide_mean_matched_NTC_v1"
                ),
                "error_normalization": "raw_rmse",
                "predicted_effect_key": predicted_key,
                "true_effect_key": true_key,
                "true_error_rmse": rmse(vector, true_vector),
                "true_error_cosine": cosine_error(vector, true_vector),
            })
            features.append({
                **shared,
                "context_similarity_max": float(row.context_similarity_max),
                "perturbation_support_count": int(row.perturbation_support_count),
                "model_disagreement_rmse": float(row.model_disagreement_rmse),
                "safeconf_risk": float(row.safeconf_risk),
                "safeconf_confidence": float(row.safeconf_confidence),
                "predicted_magnitude": float(row.predicted_magnitude),
                "target_stratum": str(row.target_stratum),
            })
    record_frame, feature_frame = pd.DataFrame(records), pd.DataFrame(features)
    if len(record_frame) != 600 * 11 - 120 or record_frame.record_id.duplicated().any():
        raise IntegrityFailure(f"PredictionRecord coverage failed: {len(record_frame)}")
    if set(predicted_arrays) != set(record_frame.predicted_effect_key.astype(str)):
        raise IntegrityFailure("predicted array keys differ from PredictionRecords")
    return record_frame, feature_frame, predicted_arrays, true_arrays


def write_manifest(directory: Path) -> str:
    hashes = {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    payload = "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items()))
    atomic_bytes(directory / "MANIFEST.sha256", payload.encode())
    return sha256_file(directory / "MANIFEST.sha256")


def formal_decision(
    inference: pd.DataFrame,
    state_deltas: pd.DataFrame,
    baselines: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    all_test = inference.loc[inference.stratum.eq("all_200")].iloc[0]
    seen_test = inference.loc[inference.stratum.eq("seen_160")].iloc[0]
    all_state = state_deltas.loc[state_deltas.stratum.eq("all_200")]
    nochange = baselines.loc[baselines.stratum.eq("all_200")]
    nochange_pass = bool(
        len(nochange) == 3
        and (nochange.ensemble_beats_nochange_fraction.to_numpy(float) > 0.5).all()
    )
    positive_states = int(
        (all_state.delta_magnitude_minus_safeconf.to_numpy(float) > 0).sum()
    )
    all_inference_pass = bool(
        all_test.observed_mean_state_delta > 0
        and all_test.bootstrap_ci95_lower > 0
        and all_test.permutation_p_one_sided < 0.05
        and positive_states >= 2
        and nochange_pass
    )
    seen_pass = bool(
        seen_test.observed_mean_state_delta > 0
        and seen_test.bootstrap_ci95_lower > 0
        and seen_test.permutation_p_one_sided < 0.05
    )
    if all_inference_pass and seen_pass:
        decision = "CONFIRMATION_PASS_NONTRIVIAL"
    elif all_inference_pass:
        decision = "PARTIAL_SUPPORT_STRATIFICATION_ONLY"
    else:
        decision = "NO_CONFIRMATION"
    return decision, {
        "all_200_inference_passed": all_inference_pass,
        "seen_160_hierarchical_passed": seen_pass,
        "ensemble_beats_nochange_each_state": nochange_pass,
        "positive_state_delta_count": positive_states,
    }


def run_formal(gate_commit: str, branch: str) -> dict[str, Any]:
    started = time.time()
    if RELEASE.exists() or STAGING.exists():
        raise IntegrityFailure("E168 postgate release is append-only and already exists")
    (
        panel, scores, predictions, truth, guide_truth, seen_effects, metadata,
        remote_heads, code_hashes,
    ) = load_and_validate_inputs(gate_commit, branch)
    metrics, truth_by_task, donor_mean = build_task_metrics(
        scores, predictions, truth, seen_effects
    )
    curves, aurc = compute_curves_and_aurc(metrics)
    state_deltas = state_delta_table(aurc)

    inference_rows = []
    inference_draws: dict[str, np.ndarray] = {}
    for stratum in ("all_200", "seen_160"):
        result, bootstrap, null = infer_stratum(metrics, stratum)
        inference_rows.append(result)
        inference_draws[f"{stratum}::cluster_bootstrap_delta"] = bootstrap
        inference_draws[f"{stratum}::paired_permutation_null_delta"] = null
    inference = pd.DataFrame(inference_rows)
    secondary, baselines = secondary_metrics(metrics)
    guides = guide_consistency(guide_truth)
    decision, checks = formal_decision(inference, state_deltas, baselines)

    records, features, predicted_arrays, true_arrays = build_prediction_records(
        metrics, predictions, truth_by_task, donor_mean, metadata["gene_order_hash"]
    )
    for subdir in ("tables", "arrays", "reports"):
        (STAGING / subdir).mkdir(parents=True, exist_ok=False)
    table_outputs = {
        "TASK_METRICS.csv": metrics,
        "RC_CURVES.csv": curves,
        "AURC_SUMMARY.csv": aurc,
        "STATE_DELTAS.csv": state_deltas,
        "PRIMARY_INFERENCE.csv": inference,
        "SECONDARY_METRICS.csv": secondary,
        "BASELINE_COMPARISONS.csv": baselines,
        "GUIDE_CONSISTENCY.csv": guides,
        "PREDICTION_RECORDS.csv": records,
        "CONFIDENCE_FEATURES.csv": features,
        "INPUT_HASHES.csv": pd.DataFrame(code_hashes),
    }
    for name, frame in table_outputs.items():
        atomic_csv(STAGING / "tables" / name, frame)
    atomic_npz(STAGING / "arrays/predicted_effects.npz", predicted_arrays)
    atomic_npz(STAGING / "arrays/true_effects.npz", true_arrays)
    atomic_npz(STAGING / "arrays/INFERENCE_DRAWS.npz", inference_draws)

    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    from safetrans_confidence.data.records import (
        assert_no_feature_label_leakage,
        validate_prediction_record_artifacts,
    )

    feature_columns = [
        column for column in features.columns
        if column not in {
            "record_id", "task_id", "task_key", "dataset_name", "fold_id", "split",
            "context", "perturbation", "predictor_name",
        }
    ]
    assert_no_feature_label_leakage(feature_columns)
    contract_issues = validate_prediction_record_artifacts(
        STAGING, records=records, strict=True, require_effect_arrays=True
    )
    if contract_issues:
        raise IntegrityFailure(f"strict PredictionRecord validation failed: {contract_issues}")

    all_result = inference.loc[inference.stratum.eq("all_200")].iloc[0]
    seen_result = inference.loc[inference.stratum.eq("seen_160")].iloc[0]
    status = {
        "schema": "safeconf_e168_postgate_result_v1",
        "experiment": "E168_primary_human_cd4_fresh_confirmation",
        "stage": "F3_POSTGATE_FORMAL_EVALUATION",
        "status": "COMPLETE",
        "decision": decision,
        "deployment_authorized": False,
        "gate_commit": gate_commit,
        "gate_snapshot_sha256": sha256_file(SNAPSHOT),
        "gate_remote_heads": remote_heads,
        "source_full_sha256": metadata["snapshot"]["source_full_sha256"],
        "f2_manifest_sha256": metadata["f2_manifest_sha256"],
        "f3_manifest_sha256": metadata["f3_manifest_sha256"],
        "n_test_tasks": len(metrics),
        "n_prediction_records": len(records),
        "gene_panel_size": N_GENES,
        "gene_order_hash": metadata["gene_order_hash"],
        "primary_delta": float(all_result.observed_mean_state_delta),
        "primary_ci95": [
            float(all_result.bootstrap_ci95_lower),
            float(all_result.bootstrap_ci95_upper),
        ],
        "primary_permutation_p_one_sided": float(
            all_result.permutation_p_one_sided
        ),
        "seen_delta": float(seen_result.observed_mean_state_delta),
        "seen_ci95": [
            float(seen_result.bootstrap_ci95_lower),
            float(seen_result.bootstrap_ci95_upper),
        ],
        "seen_permutation_p_one_sided": float(
            seen_result.permutation_p_one_sided
        ),
        "decision_checks": checks,
        "pretruth_gate_certificates": metadata["gate_certificates"],
        "risk_magnitude_equivalent_registered_units": metadata[
            "gate_certificates"
        ]["G5_MAGNITUDE_EQUIVALENCE.csv"]["equivalent_units"],
        "prediction_record_contract_issues": contract_issues,
        "test_truth_used_for_training_scoring_or_threshold": False,
        "test_truth_first_used_after_committed_dual_remote_gate": True,
        "single_test_donor_population_generalization_claim": False,
        "wall_seconds": time.time() - started,
    }
    atomic_json(STAGING / "RUN_STATUS.json", status)
    report = (
        "# E168｜Primary Human CD4+ T-cell sealed test result\n\n"
        f"正式判定：**{decision}**。\n\n"
        f"全 200 targets 的三状态平均 Δ(AURC_magnitude−AURC_SafeConf)="
        f"{all_result.observed_mean_state_delta:.6g}，95% CI "
        f"[{all_result.bootstrap_ci95_lower:.6g}, {all_result.bootstrap_ci95_upper:.6g}]，"
        f"单侧置换 p={all_result.permutation_p_one_sided:.6g}。\n\n"
        f"seen 160 targets 的 Δ={seen_result.observed_mean_state_delta:.6g}，95% CI "
        f"[{seen_result.bootstrap_ci95_lower:.6g}, {seen_result.bootstrap_ci95_upper:.6g}]，"
        f"p={seen_result.permutation_p_one_sided:.6g}。\n\n"
        "该结果来自一位完整留出的供体、三个状态和 512 基因 reduced-panel "
        "pseudobulk benchmark；不代表四个独立供体，也不形成部署或临床授权。\n"
    )
    atomic_bytes(STAGING / "reports/E168_POSTGATE_REPORT.md", report.encode())
    write_manifest(STAGING)
    os.replace(STAGING, RELEASE)
    return status


def synthetic_tests() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any) -> None:
        rows.append({"test_id": name, "passed": bool(passed), "observed": str(observed)})

    loss = np.linspace(0.05, 1.0, 40)
    constant = np.ones(40)
    _curve, constant_summary = tie_aware_curve(constant, loss)
    add(
        "S1_full_tie_average_equals_population_mean",
        abs(constant_summary["aurc_tie_average"] - float(np.mean(loss))) < 1e-12,
        constant_summary,
    )
    _curve, good = tie_aware_curve(loss, loss)
    _curve, bad = tie_aware_curve(-loss, loss)
    add(
        "S2_informative_order_beats_reverse_order",
        good["aurc_tie_average"] < bad["aurc_tie_average"],
        {"good": good, "bad": bad},
    )
    rng = np.random.default_rng(168)
    synthetic_rows = []
    for gene in range(20):
        latent = gene / 19
        for state_index, state in enumerate(STATES):
            error = latent + 0.01 * state_index
            synthetic_rows.append({
                "perturbed_gene_id": f"G{gene:02d}",
                "culture_condition": state,
                "safeconf_risk": error + rng.normal(0, 0.005),
                "magnitude_risk": -error + rng.normal(0, 0.005),
                "true_error_rmse": error,
            })
    synthetic = pd.DataFrame(synthetic_rows)
    observed = mean_state_delta(synthetic)
    boot = cluster_bootstrap(synthetic, draws=200, seed=17)
    p_value, null = paired_cluster_permutation(
        synthetic, observed, draws=500, seed=18
    )
    add("S3_positive_signal_delta", observed > 0, observed)
    add("S4_cluster_bootstrap_finite", len(boot) == 200 and np.isfinite(boot).all(), np.quantile(boot, [0.025, 0.975]))
    add("S5_cluster_permutation_finite", 0 < p_value <= 1 and len(null) == 500 and np.isfinite(null).all(), p_value)
    tied_score = np.round(rng.normal(size=80), 1)
    tied_loss = rng.random(80)
    slow_aurc = tie_aware_curve(tied_score, tied_loss)[1]["aurc_tie_average"]
    fast_aurc = tie_aware_aurc_value(tied_score, tied_loss)
    add("S6_fast_tie_aurc_matches_reference", abs(slow_aurc - fast_aurc) < 1e-12, {"slow": slow_aurc, "fast": fast_aurc})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-commit")
    parser.add_argument("--branch")
    parser.add_argument("--synthetic-test-only", action="store_true")
    args = parser.parse_args()
    if args.synthetic_test_only:
        tests = synthetic_tests()
        print(tests.to_string(index=False))
        if len(tests) != 6 or not tests.passed.astype(bool).all():
            raise SystemExit(2)
        return
    if not args.gate_commit:
        parser.error("formal postgate evaluation requires --gate-commit")
    branch = args.branch or git_text("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        parser.error("formal postgate evaluation requires a named branch")
    result = run_formal(args.gate_commit, branch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
