#!/usr/bin/env python3
"""Joint sealed evaluation of all four E170 fresh-target panels.

The evaluator never opens the 44.6 GB source object.  It accepts only four
committed PASS gates, four isolated F2 prediction bundles, and four isolated
F3 truth bundles.  E168's 200 unsealed targets are absent from every formal
inference table produced here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping
import uuid

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
JOINT_HELPER = Path(__file__).resolve()
SCRIPT = JOINT_HELPER
EXPERIMENT_CODE = "E170"
EXPERIMENT_ID = "E170_primary_cd4_multipanel_precision"
ISOLATED_NAMESPACE = "E170"
TASK_PREFIX = "E170"
PRETRUTH_SCHEMA_PREFIX = "safeconf_e170"
PRETRUTH_REPORT_PREFIX = "E170"
EXPECTED_G4_RISK_ESTIMATOR: str | None = None
STATUS_SCHEMA = "safeconf_e170_joint_postgate_result_v1"
JOINT_STAGE = "JOINT_F3_POSTGATE_FORMAL_EVALUATION"
JOINT_REPORT_NAME = "E170_JOINT_POSTGATE_REPORT.md"
JOINT_REPORT_TITLE = "E170｜Primary CD4 未读目标多面板结果"
EXPERIMENT = ROOT / "docs/实验结果/E170_primary_cd4_multipanel_precision_20260718"
DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
RELEASE = EXPERIMENT / "postgate_release"
STAGING = EXPERIMENT / ".postgate_release.staging"
PANELS = ("P01", "P02", "P03", "P04")
STATES = ("Rest", "Stim8hr", "Stim48hr")
N_GENES = 512
SEEDS = (3407, 3408, 3409)
PREDICTOR_NAMES = (
    "scGPT_seed3407", "scGPT_seed3408", "scGPT_seed3409",
    "GEARS_seed3407", "GEARS_seed3408", "GEARS_seed3409",
    "scGPT_seed_mean", "GEARS_seed_mean", "ensemble_seed_family_mean",
)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2026071801
PERMUTATION_DRAWS = 100_000
PERMUTATION_SEED = 2026071802
TOLERANCE = 1e-6

PRETRUTH_WRAPPER = ROOT / "tools/scripts/run_e170_primary_cd4_panel_pretruth.py"
PRETRUTH_HELPER = ROOT / "tools/scripts/run_e168_primary_cd4_pretruth.py"
ASSET_WRAPPER = ROOT / "tools/scripts/build_e170_primary_cd4_panel_assets.py"
ASSET_HELPER = ROOT / "tools/scripts/build_e168_primary_cd4_isolated_assets.py"
POSTGATE_HELPER = ROOT / "tools/scripts/run_e168_primary_cd4_postgate.py"
FREEZE_SCRIPT = ROOT / "tools/scripts/freeze_e170_primary_cd4_multipanel.py"
STAT_LOCK = EXPERIMENT / "STATISTICAL_ANALYSIS_LOCK.json"
MODEL_LOCK = EXPERIMENT / "MODEL_INPUT_LOCK.json"
ALL_TARGETS = EXPERIMENT / "manifests/E170_ALL_SELECTED_TARGETS.csv"
ALL_TASKS = EXPERIMENT / "manifests/E170_ALL_TASKS.csv"
EXECUTION_PLAN = EXPERIMENT / "PRETRUTH_CODE_FREEZE_PLAN.md"
CORRECTION_LOG = EXPERIMENT / "PRETRUTH_RUNTIME_CORRECTION_LOG.md"
PRIOR_TARGET_MANIFESTS = (
    ROOT
    / "docs/实验结果/E168_primary_human_cd4_fresh_confirmation_20260716"
    / "manifests/E168_SELECTED_TARGETS.csv",
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
    """A sealed input, code freeze, or output contract was violated."""


def import_postgate_helper() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e170_postgate_helper", POSTGATE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import postgate helper: {POSTGATE_HELPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def strict_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values = series.astype(str).str.strip().str.lower()
    if not values.isin({"true", "false"}).all():
        raise IntegrityFailure(f"{name} contains a non-boolean value")
    return values.eq("true")


def require_committed(path: Path, head: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise IntegrityFailure(f"missing or symlinked code-freeze input: {path}")
    relative = path.relative_to(ROOT).as_posix()
    try:
        committed = git("show", f"{head}:{relative}").stdout
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"uncommitted code-freeze input: {relative}") from exc
    if committed != path.read_bytes():
        raise IntegrityFailure(f"working file differs from HEAD: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def parse_flat_manifest(directory: Path, allowlist: set[str]) -> tuple[dict[str, str], str]:
    if directory.is_symlink() or not directory.is_dir():
        raise IntegrityFailure(f"isolated directory missing or symlinked: {directory}")
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if observed != allowlist or any(path.is_dir() for path in directory.iterdir()):
        raise IntegrityFailure(f"isolated allowlist failed: {directory}")
    rows: dict[str, str] = {}
    manifest = directory / "MANIFEST.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if len(digest) != 64 or "/" in name or name in rows:
            raise IntegrityFailure(f"unsafe manifest entry: {line}")
        rows[name] = digest
    if set(rows) != allowlist - {"MANIFEST.sha256"}:
        raise IntegrityFailure(f"manifest entries changed: {directory}")
    for name, digest in rows.items():
        if sha256_file(directory / name) != digest:
            raise IntegrityFailure(f"isolated asset hash mismatch: {directory/name}")
    return rows, sha256_file(manifest)


def load_npz_vectors(path: Path, expected: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            value = np.asarray(archive[key], dtype=np.float32)
            if value.shape != (N_GENES,) or not np.isfinite(value).all():
                raise IntegrityFailure(f"invalid vector {path.name}/{key}/{value.shape}")
            result[str(key)] = value
    if len(result) != expected:
        raise IntegrityFailure(f"unexpected vector count {path}: {len(result)} != {expected}")
    return result


def verify_gate_commit(gate_commit: str, branch: str) -> dict[str, str]:
    head = git_text("rev-parse", "HEAD")
    if git("cat-file", "-e", f"{gate_commit}^{{commit}}", check=False).returncode:
        raise IntegrityFailure("gate commit is unavailable")
    if git("merge-base", "--is-ancestor", gate_commit, head, check=False).returncode:
        raise IntegrityFailure("current HEAD does not descend from the gate commit")
    code_files = [
        SCRIPT, JOINT_HELPER, PRETRUTH_WRAPPER, PRETRUTH_HELPER, ASSET_WRAPPER, ASSET_HELPER,
        POSTGATE_HELPER, FREEZE_SCRIPT, STAT_LOCK, MODEL_LOCK, ALL_TARGETS, ALL_TASKS,
        EXECUTION_PLAN, CORRECTION_LOG,
    ]
    for path in code_files:
        relative = path.relative_to(ROOT).as_posix()
        try:
            frozen = git("show", f"{gate_commit}:{relative}").stdout
        except subprocess.CalledProcessError as exc:
            raise IntegrityFailure(f"{relative} was not frozen before truth access") from exc
        if frozen != path.read_bytes():
            raise IntegrityFailure(f"{relative} changed after the gate")
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched = f"refs/remotes/{remote}/{branch}"
        result = git("fetch", "--quiet", remote, f"refs/heads/{branch}:{fetched}", check=False)
        if result.returncode:
            raise IntegrityFailure(f"cannot verify gate on {remote}: {result.stderr.decode(errors='replace')}")
        remote_head = git_text("rev-parse", fetched)
        if git("merge-base", "--is-ancestor", gate_commit, remote_head, check=False).returncode:
            raise IntegrityFailure(f"gate commit absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    return remote_heads


def verify_pretruth_release(panel: str, gate_commit: str) -> tuple[dict[str, Any], Path]:
    release = EXPERIMENT / "pretruth_release" / panel
    snapshot_path = release / "PRETRUTH_GATE_SNAPSHOT.json"
    relative = snapshot_path.relative_to(ROOT).as_posix()
    try:
        committed_snapshot = git("show", f"{gate_commit}:{relative}").stdout
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"{panel} snapshot absent from gate commit") from exc
    if committed_snapshot != snapshot_path.read_bytes():
        raise IntegrityFailure(f"{panel} local snapshot differs from gate commit")
    snapshot = json.loads(committed_snapshot)
    required = {
        "schema": f"{PRETRUTH_SCHEMA_PREFIX}_{panel.lower()}_pretruth_gate_snapshot_v1",
        "experiment": f"{EXPERIMENT_ID}::{panel}",
        "stage": f"{EXPERIMENT_CODE}_{panel}_F2_PRETRUTH_GATE",
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
    if EXPECTED_G4_RISK_ESTIMATOR is not None:
        required["g4_risk_estimator"] = EXPECTED_G4_RISK_ESTIMATOR
    mismatches = {key: {"expected": value, "observed": snapshot.get(key)} for key, value in required.items() if snapshot.get(key) != value}
    if mismatches:
        raise IntegrityFailure(f"{panel} gate snapshot failed: {mismatches}")
    if snapshot.get("runner_sha256") != sha256_file(PRETRUTH_WRAPPER):
        raise IntegrityFailure(f"{panel} gate runner hash changed")
    if snapshot.get("asset_builder_sha256") != sha256_file(ASSET_WRAPPER):
        raise IntegrityFailure(f"{panel} gate asset-builder hash changed")
    registered = snapshot.get("pretruth_files_sha256", {})
    report_name = f"reports/{PRETRUTH_REPORT_PREFIX}_{panel}_PRETRUTH_REPORT.md"
    observed = {path.relative_to(release).as_posix() for path in release.rglob("*") if path.is_file()}
    if observed != set(registered) | {"PRETRUTH_GATE_SNAPSHOT.json", report_name}:
        raise IntegrityFailure(f"{panel} pretruth release file set changed")
    for rel, digest in registered.items():
        path = release / rel
        if sha256_file(path) != digest:
            raise IntegrityFailure(f"{panel} pretruth hash changed: {rel}")
        committed = git("show", f"{gate_commit}:{path.relative_to(ROOT).as_posix()}").stdout
        if committed != path.read_bytes() or hashlib.sha256(committed).hexdigest() != digest:
            raise IntegrityFailure(f"{panel} pretruth file not identical to gate commit: {rel}")
    report_path = release / report_name
    try:
        committed_report = git(
            "show", f"{gate_commit}:{report_path.relative_to(ROOT).as_posix()}"
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"{panel} pretruth report absent from gate commit") from exc
    if committed_report != report_path.read_bytes():
        raise IntegrityFailure(f"{panel} pretruth report differs from gate commit")
    for name, rows in {
        "G2_SCORE_CERTIFICATES.csv": 6,
        "G3_PREDICTOR_CERTIFICATES.csv": 24,
        "G4_SEED_STABILITY.csv": 6,
        "SYNTHETIC_REGRESSION_TESTS.csv": 10,
    }.items():
        table = pd.read_csv(release / "tables" / name, keep_default_na=False)
        if len(table) != rows or not strict_bool(table.passed, f"{panel}:{name}").all():
            raise IntegrityFailure(f"{panel} gate certificate failed: {name}")
    return snapshot, release


def load_panel(panel: str, gate_commit: str) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    snapshot, pretruth = verify_pretruth_release(panel, gate_commit)
    isolated = DATA_ROOT / "isolated" / ISOLATED_NAMESPACE / panel
    f2, f3 = isolated / "F2_pretruth", isolated / "F3_postgate"
    _f2_files, f2_manifest = parse_flat_manifest(f2, F2_ALLOWLIST)
    _f3_files, f3_manifest = parse_flat_manifest(f3, F3_ALLOWLIST)
    if snapshot.get("f2_manifest_sha256") != f2_manifest:
        raise IntegrityFailure(f"{panel} snapshot does not bind F2")
    f2_att = json.loads((f2 / "ACCESS_ATTESTATION.json").read_text())
    f3_att = json.loads((f3 / "ACCESS_ATTESTATION.json").read_text())
    required_f2 = {
        "experiment": f"{EXPERIMENT_ID}::{panel}",
        "stage": f"{EXPERIMENT_CODE}_{panel}_F2_PRETRUTH_ISOLATED_ASSET_BUILD",
        "status": "PASS", "test_targeting_x_values_read": 0,
        "forbidden_column_unseen_x_values_read": 0,
    }
    required_f3 = {
        "experiment": f"{EXPERIMENT_ID}::{panel}",
        "stage": f"{EXPERIMENT_CODE}_{panel}_F3_POSTGATE_ISOLATED_TRUTH_BUILD",
        "status": "PASS", "all_registered_pretruth_gates_passed": True,
        "f2_manifest_sha256": f2_manifest,
        "source_full_sha256": snapshot["source_full_sha256"],
        "gate_commit": gate_commit,
        "postgate_test_targeting_x_values_read": 1200,
        "forbidden_column_unseen_x_values_read": 0,
        "train_or_validation_targeting_x_values_read_in_postgate": 0,
        "n_test_target_effects": 600, "n_test_guide_effects": 1200,
        "test_performance_metrics_computed": 0,
    }
    for label, att, required in (("F2", f2_att, required_f2), ("F3", f3_att, required_f3)):
        mismatches = {key: {"expected": value, "observed": att.get(key)} for key, value in required.items() if att.get(key) != value}
        if mismatches:
            raise IntegrityFailure(f"{panel} {label} attestation failed: {mismatches}")
        if att.get("builder_sha256") != sha256_file(ASSET_WRAPPER):
            raise IntegrityFailure(f"{panel} {label} builder hash changed")
    if f3_att.get("gate_snapshot_sha256") != sha256_file(pretruth / "PRETRUTH_GATE_SNAPSHOT.json"):
        raise IntegrityFailure(f"{panel} F3 does not bind gate snapshot")
    if f3_att.get("source_full_sha256") != f2_att.get("source_full_sha256"):
        raise IntegrityFailure(f"{panel} F2/F3 source differs")
    if f3_manifest != sha256_file(f3 / "MANIFEST.sha256"):
        raise AssertionError("unreachable manifest mismatch")

    panel_table = pd.read_csv(f2 / "GENE_PANEL.csv", keep_default_na=False)
    if len(panel_table) != N_GENES or panel_table.panel_index.astype(int).tolist() != list(range(N_GENES)):
        raise IntegrityFailure(f"{panel} gene panel changed")
    gene_order_hash = "sha256:" + hashlib.sha256("\n".join(panel_table.scgpt_token.astype(str)).encode()).hexdigest()
    scoring = pd.read_csv(pretruth / "tables/PRETRUTH_SCORING_INTERFACE.csv", keep_default_na=False)
    if len(scoring) != 2160 or scoring.task_id.nunique() != 2160 or set(scoring.panel_id.astype(str)) != {panel}:
        raise IntegrityFailure(f"{panel} scoring interface changed")
    test_scores = scoring.loc[scoring.donor_role.eq("test")].copy().reset_index(drop=True)
    if len(test_scores) != 600:
        raise IntegrityFailure(f"{panel} test score count changed")
    with np.load(pretruth / "arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(PREDICTOR_NAMES):
            raise IntegrityFailure(f"{panel} predictor set changed")
        matrices = {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files}
    for name, matrix in matrices.items():
        if matrix.shape != (2160, N_GENES) or not np.isfinite(matrix).all():
            raise IntegrityFailure(f"{panel} invalid prediction matrix: {name}/{matrix.shape}")
    row_by_task = {task: index for index, task in enumerate(scoring.task_id.astype(str))}
    predictions = {
        name: np.stack([matrix[row_by_task[task]] for task in test_scores.task_id.astype(str)]).astype(np.float32)
        for name, matrix in matrices.items()
    }
    truth = load_npz_vectors(f3 / "TEST_TARGET_EFFECTS.npz", 600)
    guide_truth = load_npz_vectors(f3 / "TEST_GUIDE_EFFECTS.npz", 1200)
    if set(truth) != set(test_scores.task_id.astype(str)):
        raise IntegrityFailure(f"{panel} test truth keys differ from pretruth queries")
    seen = load_npz_vectors(f2 / "SEEN_TARGET_EFFECTS.npz", 1440)

    locked_tasks = pd.read_csv(
        EXPERIMENT / "manifests" / panel / f"{EXPERIMENT_CODE}_{panel}_TASK_MANIFEST.csv",
        keep_default_na=False,
    )
    locked_test = locked_tasks.loc[strict_bool(locked_tasks.primary_test_task, f"{panel}:primary_test_task")]
    f3_tasks = pd.read_csv(f3 / "TEST_TASKS.csv", keep_default_na=False)
    shared = list(locked_tasks.columns)
    if len(f3_tasks) != 600 or not f3_tasks[shared].sort_values("task_id").reset_index(drop=True).astype(str).equals(
        locked_test[shared].sort_values("task_id").reset_index(drop=True).astype(str)
    ):
        raise IntegrityFailure(f"{panel} F3 tasks differ from frozen task manifest")

    ensemble = predictions["ensemble_seed_family_mean"]
    task_ids = test_scores.task_id.astype(str).tolist()
    truth_matrix = np.stack([truth[task] for task in task_ids]).astype(np.float32)
    metrics = test_scores.copy()
    metrics["panel_id"] = panel
    metrics["true_error_rmse"] = np.sqrt(np.mean((ensemble.astype(float) - truth_matrix.astype(float)) ** 2, axis=1))
    metrics["nochange_error_rmse"] = np.sqrt(np.mean(truth_matrix.astype(float) ** 2, axis=1))
    metrics["ensemble_beats_nochange"] = metrics.true_error_rmse < metrics.nochange_error_rmse
    metrics["train_donor_effect_mean_error_rmse"] = np.nan
    train_donors = sorted(set(locked_tasks.loc[locked_tasks.donor_role.eq("train"), "donor_id"].astype(str)))
    donor_mean: dict[str, np.ndarray] = {}
    for index, row in enumerate(metrics.itertuples(index=False)):
        if str(row.target_stratum) != "DONOR_UNSEEN_ONLY":
            continue
        keys = [
            f"{TASK_PREFIX}::{panel}::{donor}::{row.culture_condition}::{row.perturbed_gene_id}"
            for donor in train_donors
        ]
        if len(keys) != 2 or any(key not in seen for key in keys):
            raise IntegrityFailure(f"{panel} missing train donor mean input for {row.task_id}")
        vector = np.mean(np.stack([seen[key] for key in keys]), axis=0).astype(np.float32)
        donor_mean[str(row.task_id)] = vector
        metrics.loc[index, "train_donor_effect_mean_error_rmse"] = float(np.sqrt(np.mean((vector.astype(float) - truth_matrix[index].astype(float)) ** 2)))
    if len(donor_mean) != 480:
        raise IntegrityFailure(f"{panel} train donor mean coverage changed")
    metrics["disagreement_only_risk"] = metrics.model_disagreement_rmse.astype(float)
    metrics["support_only_risk"] = -metrics.z_log_support_train960.astype(float)
    metrics["context_only_risk"] = -metrics.z_context_train960.astype(float)
    metrics["magnitude_risk"] = metrics.predicted_magnitude.astype(float)
    if not np.isfinite(metrics.true_error_rmse.to_numpy(float)).all():
        raise IntegrityFailure(f"{panel} true errors are non-finite")

    guide_index = pd.read_csv(f3 / "TEST_GUIDE_EFFECT_INDEX.csv", keep_default_na=False)
    guide_rows = []
    used: set[str] = set()
    for task, block in guide_index.groupby("task_id", sort=True):
        keys = [f"{task}::{guide}" for guide in block.sort_values("guide_id").guide_id.astype(str)]
        if len(keys) != 2 or any(key not in guide_truth for key in keys):
            raise IntegrityFailure(f"{panel} guide truth mismatch for {task}")
        used.update(keys)
        left, right = guide_truth[keys[0]], guide_truth[keys[1]]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        cosine = float(np.dot(left, right) / denominator) if denominator > 1e-12 else np.nan
        guide_rows.append({
            "panel_id": panel, "task_id": task,
            "culture_condition": str(block.culture_condition.iloc[0]),
            "perturbed_gene_id": str(block.ensembl_id.iloc[0]),
            "target_stratum": str(block.target_stratum.iloc[0]),
            "guide_1": keys[0].rsplit("::", 1)[-1], "guide_2": keys[1].rsplit("::", 1)[-1],
            "guide_effect_rmse": float(np.sqrt(np.mean((left.astype(float) - right.astype(float)) ** 2))),
            "guide_effect_cosine_similarity": cosine,
        })
    if used != set(guide_truth):
        raise IntegrityFailure(f"{panel} unused guide truth array")
    metadata = {
        "panel": panel, "panel_table": panel_table, "gene_order_hash": gene_order_hash,
        "predictions": predictions, "truth": truth, "truth_matrix": truth_matrix,
        "donor_mean": donor_mean, "guide_rows": pd.DataFrame(guide_rows),
        "snapshot": snapshot, "f2_manifest_sha256": f2_manifest,
        "f3_manifest_sha256": f3_manifest,
    }
    hashes = [
        {"panel_id": panel, "path": str(path), "sha256": sha256_file(path)}
        for path in (f2 / "MANIFEST.sha256", f3 / "MANIFEST.sha256", pretruth / "PRETRUTH_GATE_SNAPSHOT.json")
    ]
    return metrics, metadata, hashes


def registered_blocks(frame: pd.DataFrame):
    for panel in PANELS:
        for state in STATES:
            unit = frame.loc[frame.panel_id.eq(panel) & frame.culture_condition.eq(state)]
            masks = {
                "all_200": np.ones(len(unit), dtype=bool),
                "seen_160": unit.target_stratum.eq("DONOR_UNSEEN_ONLY").to_numpy(),
                "column_unseen_40_descriptive": unit.target_stratum.eq("COLUMN_UNSEEN").to_numpy(),
            }
            for stratum, mask in masks.items():
                block = unit.loc[mask].copy()
                expected = {"all_200": 200, "seen_160": 160, "column_unseen_40_descriptive": 40}[stratum]
                if len(block) != expected:
                    raise IntegrityFailure(f"{panel}/{state}/{stratum} count {len(block)} != {expected}")
                yield panel, state, stratum, block


def compute_aurc(frame: pd.DataFrame, helper: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_columns = {
        "SafeConf": "safeconf_risk", "magnitude": "magnitude_risk",
        "disagreement_only": "disagreement_only_risk", "support_only": "support_only_risk",
        "context_only": "context_only_risk",
    }
    curve_rows, summary_rows, resolution_rows = [], [], []
    for panel, state, stratum, block in registered_blocks(frame):
        loss = block.true_error_rmse.to_numpy(float)
        for score_name, column in score_columns.items():
            score = block[column].to_numpy(float)
            curve, summary = helper.tie_aware_curve(score, loss)
            for row in curve.to_dict("records"):
                curve_rows.append({"panel_id": panel, "culture_condition": state, "stratum": stratum, "score_name": score_name, **row})
            labels = np.rint(score / TOLERANCE).astype(np.int64)
            summary_rows.append({
                "panel_id": panel, "culture_condition": state, "stratum": stratum,
                "score_name": score_name, "n_tasks": len(block),
                "n_operational_levels": int(np.unique(labels).size),
                "max_tie_fraction": float(np.unique(labels, return_counts=True)[1].max() / len(labels)),
                **summary,
            })
        resolution_rows.append({
            "panel_id": panel, "culture_condition": state, "stratum": stratum,
            "n_tasks": len(block),
            "context_unique": int(np.unique(np.rint(block.context_similarity_max.to_numpy(float) / TOLERANCE)).size),
            "support_unique": int(block.perturbation_support_count.astype(int).nunique()),
            "disagreement_unique": int(np.unique(np.rint(block.model_disagreement_rmse.to_numpy(float) / TOLERANCE)).size),
            "safeconf_unique": int(np.unique(np.rint(block.safeconf_risk.to_numpy(float) / TOLERANCE)).size),
        })
    aurc = pd.DataFrame(summary_rows)
    deltas = []
    for panel in PANELS:
        for state in STATES:
            for stratum in ("all_200", "seen_160", "column_unseen_40_descriptive"):
                block = aurc.loc[aurc.panel_id.eq(panel) & aurc.culture_condition.eq(state) & aurc.stratum.eq(stratum)].set_index("score_name")
                deltas.append({
                    "panel_id": panel, "culture_condition": state, "stratum": stratum,
                    "safeconf_aurc": float(block.loc["SafeConf", "aurc_tie_average"]),
                    "magnitude_aurc": float(block.loc["magnitude", "aurc_tie_average"]),
                    "delta_magnitude_minus_safeconf": float(block.loc["magnitude", "aurc_tie_average"] - block.loc["SafeConf", "aurc_tie_average"]),
                })
    return pd.DataFrame(curve_rows), aurc, pd.DataFrame(deltas).merge(pd.DataFrame(resolution_rows), on=["panel_id", "culture_condition", "stratum"], how="left")


def panel_cluster_matrices(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    matrices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for panel in PANELS:
        panel_frame = frame.loc[frame.panel_id.eq(panel)]
        genes = sorted(panel_frame.perturbed_gene_id.astype(str).unique())
        candidate = np.empty((len(genes), 3), dtype=np.float64)
        comparator = np.empty_like(candidate); loss = np.empty_like(candidate)
        for gene_index, gene in enumerate(genes):
            block = panel_frame.loc[panel_frame.perturbed_gene_id.astype(str).eq(gene)].set_index("culture_condition")
            if len(block) != 3 or set(block.index.astype(str)) != set(STATES):
                raise IntegrityFailure(f"{panel}/{gene} lacks three-state cluster")
            for state_index, state in enumerate(STATES):
                row = block.loc[state]
                candidate[gene_index, state_index] = float(row.safeconf_risk)
                comparator[gene_index, state_index] = float(row.magnitude_risk)
                loss[gene_index, state_index] = float(row.true_error_rmse)
        matrices[panel] = candidate, comparator, loss
    return matrices


def joint_delta(matrices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], helper: Any) -> float:
    values = []
    for candidate, comparator, loss in matrices.values():
        for state in range(3):
            values.append(helper.tie_aware_aurc_value(comparator[:, state], loss[:, state]) - helper.tie_aware_aurc_value(candidate[:, state], loss[:, state]))
    return float(np.mean(values))


def stratified_bootstrap(matrices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], helper: Any, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = {}
        for panel, (candidate, comparator, loss) in matrices.items():
            take = rng.integers(0, len(candidate), size=len(candidate))
            sampled[panel] = candidate[take], comparator[take], loss[take]
        values[draw] = joint_delta(sampled, helper)
    return values


def paired_permutation(matrices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], helper: Any, observed: float, draws: int, seed: int) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed); null = np.empty(draws, dtype=np.float64); exceed = 0
    for draw in range(draws):
        permuted = {}
        for panel, (candidate, comparator, loss) in matrices.items():
            swap = rng.integers(0, 2, size=(len(candidate), 1), dtype=np.int8).astype(bool)
            permuted[panel] = np.where(swap, comparator, candidate), np.where(swap, candidate, comparator), loss
        value = joint_delta(permuted, helper); null[draw] = value; exceed += int(value >= observed)
    return (1 + exceed) / (draws + 1), null


def infer(frame: pd.DataFrame, helper: Any, stratum: str, draws_scale: float = 1.0) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if stratum == "all_800":
        block = frame.copy()
    elif stratum == "seen_640":
        block = frame.loc[frame.target_stratum.eq("DONOR_UNSEEN_ONLY")].copy()
    else:
        raise ValueError(stratum)
    matrices = panel_cluster_matrices(block)
    observed = joint_delta(matrices, helper)
    bdraws = max(100, int(round(BOOTSTRAP_DRAWS * draws_scale)))
    pdraws = max(200, int(round(PERMUTATION_DRAWS * draws_scale)))
    bootstrap = stratified_bootstrap(matrices, helper, bdraws, BOOTSTRAP_SEED)
    p_value, null = paired_permutation(matrices, helper, observed, pdraws, PERMUTATION_SEED)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "stratum": stratum, "n_targets": int(block.groupby(["panel_id", "perturbed_gene_id"]).ngroups),
        "n_tasks": len(block), "n_panel_state_units": 12,
        "observed_equal_panel_state_delta": observed,
        "bootstrap_draws": bdraws, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_ci95_lower": float(low), "bootstrap_ci95_upper": float(high),
        "permutation_draws": pdraws, "permutation_seed": PERMUTATION_SEED,
        "permutation_p_one_sided": float(p_value),
        "permutation_null_mean": float(np.mean(null)), "permutation_null_std": float(np.std(null)),
        "permutation_null_sha256_float64": hashlib.sha256(np.ascontiguousarray(null).tobytes()).hexdigest(),
    }, bootstrap, null


def secondary_tables(frame: pd.DataFrame, helper: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    secondary, baselines = [], []
    for panel, state, stratum, block in registered_blocks(frame):
        loss = block.true_error_rmse.to_numpy(float); risk = block.safeconf_risk.to_numpy(float)
        k = max(1, int(math.ceil(0.20 * len(block)))); labels = np.rint(risk / TOLERANCE).astype(np.int64)
        order = np.argsort(labels, kind="mergesort"); top = loss[order[-k:]]
        secondary.append({
            "panel_id": panel, "culture_condition": state, "stratum": stratum,
            "spearman_safeconf_vs_error": helper.spearman(risk, loss),
            "top20_high_risk_error_mean": float(np.mean(top)),
            "population_error_mean": float(np.mean(loss)),
            "top20_error_enrichment": float(np.mean(top) / np.mean(loss)) if np.mean(loss) > 0 else np.nan,
        })
        baseline = block.train_donor_effect_mean_error_rmse.dropna().to_numpy(float)
        baselines.append({
            "panel_id": panel, "culture_condition": state, "stratum": stratum,
            "ensemble_beats_nochange_fraction": float(block.ensemble_beats_nochange.mean()),
            "ensemble_error_mean": float(block.true_error_rmse.mean()),
            "nochange_error_mean": float(block.nochange_error_rmse.mean()),
            "train_donor_effect_mean_error": float(np.mean(baseline)) if len(baseline) else np.nan,
            "n_train_donor_mean_available": len(baseline),
        })
    return pd.DataFrame(secondary), pd.DataFrame(baselines)


def build_records(metrics: pd.DataFrame, panel_meta: dict[str, dict[str, Any]], helper: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    records, features = [], []; predicted_arrays: dict[str, np.ndarray] = {}; true_arrays: dict[str, np.ndarray] = {}
    for panel in PANELS:
        block = metrics.loc[metrics.panel_id.eq(panel)].reset_index(drop=True)
        meta = panel_meta[panel]; predictions = meta["predictions"]; truth = meta["truth"]
        task_index = {task: index for index, task in enumerate(block.task_id.astype(str))}
        for task, vector in truth.items():
            true_arrays[f"{task}::truth"] = np.asarray(vector, dtype=np.float32)
        for row in block.itertuples(index=False):
            task = str(row.task_id); index = task_index[task]
            vectors = {name: predictions[name][index] for name in PREDICTOR_NAMES}
            vectors["NoChange"] = np.zeros(N_GENES, dtype=np.float32)
            if task in meta["donor_mean"]:
                vectors["TrainDonorEffectMean"] = meta["donor_mean"][task]
            for predictor, vector in vectors.items():
                record_id = f"{task}::{predictor}"; predicted_key = f"{record_id}::prediction"; true_key = f"{task}::truth"
                predicted_arrays[predicted_key] = np.asarray(vector, dtype=np.float32)
                shared = {
                    "record_id": record_id, "task_id": task, "task_key": task,
                    "dataset_name": f"PrimaryHumanCD4_GWCD4i_{EXPERIMENT_CODE}_{panel}_formal512",
                    "fold_id": str(row.fold_id), "split": "test", "panel_id": panel,
                    "context": f"{row.donor_id}::{row.culture_condition}",
                    "perturbation": str(row.perturbed_gene_name), "predictor_name": predictor,
                }
                records.append({
                    "schema_version": "safeconf_prediction_record_v1", **shared,
                    "dataset_group": "fresh_target_replication_same_primary_human_cd4_study",
                    "run_type": "formal", "gene_panel_id": f"{EXPERIMENT_CODE}_{panel}_trainNTC_target200_scgpt512_v1",
                    "gene_order_hash": meta["gene_order_hash"], "effect_definition": "mean_diff",
                    "normalization_id": "pseudobulk_UMI_CP10K_log1p_equal_guide_mean_matched_NTC_v1",
                    "error_normalization": "raw_rmse", "predicted_effect_key": predicted_key,
                    "true_effect_key": true_key,
                    "true_error_rmse": helper.rmse(vector, truth[task]),
                    "true_error_cosine": helper.cosine_error(vector, truth[task]),
                })
                features.append({
                    **shared, "context_similarity_max": float(row.context_similarity_max),
                    "perturbation_support_count": int(row.perturbation_support_count),
                    "model_disagreement_rmse": float(row.model_disagreement_rmse),
                    "safeconf_risk": float(row.safeconf_risk),
                    "safeconf_confidence": float(row.safeconf_confidence),
                    "predicted_magnitude": float(row.predicted_magnitude),
                    "target_stratum": str(row.target_stratum),
                })
    record_frame, feature_frame = pd.DataFrame(records), pd.DataFrame(features)
    if len(record_frame) != 4 * 6480 or record_frame.record_id.duplicated().any():
        raise IntegrityFailure(f"joint PredictionRecord coverage failed: {len(record_frame)}")
    if set(predicted_arrays) != set(record_frame.predicted_effect_key.astype(str)):
        raise IntegrityFailure("PredictionRecord keys differ from predicted arrays")
    return record_frame, feature_frame, predicted_arrays, true_arrays


def write_manifest(directory: Path) -> str:
    hashes = {path.relative_to(directory).as_posix(): sha256_file(path) for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "MANIFEST.sha256"}
    atomic_bytes(directory / "MANIFEST.sha256", "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())).encode())
    return sha256_file(directory / "MANIFEST.sha256")


def synthetic_tests() -> pd.DataFrame:
    helper = import_postgate_helper(); rng = np.random.default_rng(170); rows = []
    def add(name: str, passed: bool, observed: Any) -> None:
        rows.append({"test_id": name, "passed": bool(passed), "observed": str(observed)})
    loss = np.linspace(0.05, 1.0, 40)
    constant = helper.tie_aware_curve(np.ones(40), loss)[1]["aurc_tie_average"]
    add("S1_full_tie_equals_mean_loss", abs(constant - np.mean(loss)) < 1e-12, constant)
    matrices = {}
    for panel in PANELS:
        latent = np.linspace(-1, 1, 24)[:, None] + np.zeros((24, 3))
        matrices[panel] = latent + rng.normal(0, .01, latent.shape), -latent, latent + 1.2
    observed = joint_delta(matrices, helper)
    add("S2_joint_delta_finite", math.isfinite(observed), observed)
    boot = stratified_bootstrap(matrices, helper, 100, 1)
    add("S3_stratified_bootstrap_count", len(boot) == 100 and np.isfinite(boot).all(), np.quantile(boot, [0.025, .975]))
    p, null = paired_permutation(matrices, helper, observed, 200, 2)
    add("S4_paired_permutation_count", len(null) == 200 and 0 < p <= 1 and np.isfinite(null).all(), p)
    task_rows = []
    for panel_index, panel in enumerate(PANELS):
        for gene in range(200):
            latent = (gene + 1) / 200 + panel_index * 0.01
            for state_index, state in enumerate(STATES):
                error = latent + state_index * 0.02
                task_rows.append({
                    "panel_id": panel, "culture_condition": state,
                    "perturbed_gene_id": f"{panel}_G{gene:03d}",
                    "target_stratum": "DONOR_UNSEEN_ONLY" if gene < 160 else "COLUMN_UNSEEN",
                    "safeconf_risk": error, "magnitude_risk": -error,
                    "disagreement_only_risk": error, "support_only_risk": float(gene >= 160),
                    "context_only_risk": float(state_index), "true_error_rmse": error,
                    "context_similarity_max": 0.9 + state_index * .01,
                    "perturbation_support_count": 6 if gene < 160 else 0,
                    "model_disagreement_rmse": error,
                    "ensemble_beats_nochange": True,
                    "nochange_error_rmse": error + .1,
                    "train_donor_effect_mean_error_rmse": error if gene < 160 else np.nan,
                })
    synthetic_frame = pd.DataFrame(task_rows)
    curves, aurc, deltas = compute_aurc(synthetic_frame, helper)
    add(
        "S5_full_registered_unit_counts",
        len(curves) == 3060 and len(aurc) == 180 and len(deltas) == 36,
        {"curves": len(curves), "aurc": len(aurc), "deltas": len(deltas)},
    )
    synthetic_inference, synthetic_boot, synthetic_null = infer(
        synthetic_frame, helper, "all_800", draws_scale=.001
    )
    add(
        "S6_full_joint_inference_contract",
        synthetic_inference["n_targets"] == 800
        and synthetic_inference["n_tasks"] == 2400
        and len(synthetic_boot) == 100
        and len(synthetic_null) == 200,
        synthetic_inference,
    )
    secondary, baselines = secondary_tables(synthetic_frame, helper)
    add(
        "S7_secondary_unit_counts",
        len(secondary) == 36 and len(baselines) == 36,
        {"secondary": len(secondary), "baselines": len(baselines)},
    )
    return pd.DataFrame(rows)


def run_formal(gate_commit: str, branch: str) -> dict[str, Any]:
    started = time.time()
    if RELEASE.exists() or STAGING.exists():
        raise IntegrityFailure(f"{EXPERIMENT_CODE} joint postgate release is append-only and already exists")
    helper = import_postgate_helper()
    remote_heads = verify_gate_commit(gate_commit, branch)
    head = git_text("rev-parse", "HEAD")
    code_files = [SCRIPT, JOINT_HELPER, PRETRUTH_WRAPPER, PRETRUTH_HELPER, ASSET_WRAPPER, ASSET_HELPER, POSTGATE_HELPER, FREEZE_SCRIPT, STAT_LOCK, MODEL_LOCK, ALL_TARGETS, ALL_TASKS, EXECUTION_PLAN, CORRECTION_LOG]
    code_hashes = [require_committed(path, head) for path in code_files]
    panel_meta: dict[str, dict[str, Any]] = {}; metrics_list, guide_list, input_hashes = [], [], []
    for panel in PANELS:
        metrics, metadata, hashes = load_panel(panel, gate_commit)
        metrics_list.append(metrics); guide_list.append(metadata["guide_rows"]); panel_meta[panel] = metadata; input_hashes.extend(hashes)
    metrics = pd.concat(metrics_list, ignore_index=True)
    if len(metrics) != 2400 or metrics.task_id.nunique() != 2400:
        raise IntegrityFailure("joint metrics must contain 2,400 unique fresh test tasks")
    observed_targets = set(metrics.perturbed_gene_id.astype(str))
    locked_targets = set(pd.read_csv(ALL_TARGETS).ensembl_core.astype(str))
    if observed_targets != locked_targets:
        raise IntegrityFailure(
            f"{EXPERIMENT_CODE} inference targets differ from its frozen target manifest"
        )
    for prior_manifest in PRIOR_TARGET_MANIFESTS:
        prior_targets = set(pd.read_csv(prior_manifest).ensembl_core.astype(str))
        if observed_targets & prior_targets:
            raise IntegrityFailure(
                f"a prior selected target leaked into {EXPERIMENT_CODE} inference"
            )

    curves, aurc, deltas = compute_aurc(metrics, helper)
    inference_rows, inference_arrays = [], {}
    for stratum in ("all_800", "seen_640"):
        result, bootstrap, null = infer(metrics, helper, stratum)
        inference_rows.append(result)
        inference_arrays[f"{stratum}::cluster_bootstrap_delta"] = bootstrap
        inference_arrays[f"{stratum}::paired_permutation_null_delta"] = null
    inference = pd.DataFrame(inference_rows)
    secondary, baselines = secondary_tables(metrics, helper)
    guides = pd.concat(guide_list, ignore_index=True)
    records, features, predicted_arrays, true_arrays = build_records(metrics, panel_meta, helper)

    all_result = inference.loc[inference.stratum.eq("all_800")].iloc[0]
    seen_result = inference.loc[inference.stratum.eq("seen_640")].iloc[0]
    all_units = deltas.loc[deltas.stratum.eq("all_200")]
    nochange = baselines.loc[baselines.stratum.eq("all_200")]
    checks = {
        "all_four_pretruth_gates_passed": True,
        "ensemble_beats_nochange_all_12_units": bool(len(nochange) == 12 and (nochange.ensemble_beats_nochange_fraction.to_numpy(float) > .5).all()),
        "all800_delta_positive": bool(all_result.observed_equal_panel_state_delta > 0),
        "all800_ci_lower_positive": bool(all_result.bootstrap_ci95_lower > 0),
        "all800_permutation_p_lt_0_05": bool(all_result.permutation_p_one_sided < .05),
        "positive_panel_state_units": int((all_units.delta_magnitude_minus_safeconf > 0).sum()),
        "positive_panel_state_units_at_least_8": bool((all_units.delta_magnitude_minus_safeconf > 0).sum() >= 8),
        "seen640_ci_lower_positive": bool(seen_result.bootstrap_ci95_lower > 0),
        "seen640_permutation_p_lt_0_05": bool(seen_result.permutation_p_one_sided < .05),
    }
    all_pass = all(checks.values())
    all_population_pass = all([
        checks["all_four_pretruth_gates_passed"], checks["ensemble_beats_nochange_all_12_units"],
        checks["all800_delta_positive"], checks["all800_ci_lower_positive"],
        checks["all800_permutation_p_lt_0_05"], checks["positive_panel_state_units_at_least_8"],
    ])
    if all_pass:
        decision = "TARGET_REPLICATION_PASS_NONTRIVIAL"
    elif all_population_pass:
        decision = "PARTIAL_SUPPORT_STRATIFICATION_ONLY"
    else:
        decision = "NO_TARGET_REPLICATION"

    for sub in ("tables", "arrays", "reports"):
        (STAGING / sub).mkdir(parents=True, exist_ok=False)
    for name, table in {
        "TASK_METRICS.csv": metrics, "RC_CURVES.csv": curves, "AURC_SUMMARY.csv": aurc,
        "PANEL_STATE_DELTAS.csv": deltas, "JOINT_PRIMARY_INFERENCE.csv": inference,
        "SECONDARY_METRICS.csv": secondary, "BASELINE_COMPARISONS.csv": baselines,
        "GUIDE_CONSISTENCY.csv": guides, "PREDICTION_RECORDS.csv": records,
        "CONFIDENCE_FEATURES.csv": features,
        "INPUT_HASHES.csv": pd.DataFrame(code_hashes + input_hashes),
    }.items():
        atomic_csv(STAGING / "tables" / name, table)
    atomic_npz(STAGING / "arrays/predicted_effects.npz", predicted_arrays)
    atomic_npz(STAGING / "arrays/true_effects.npz", true_arrays)
    atomic_npz(STAGING / "arrays/INFERENCE_DRAWS.npz", inference_arrays)

    code_root = ROOT / "code/20260426_154505_perturb_transport_final_push"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from safetrans_confidence.data.records import assert_no_feature_label_leakage, validate_prediction_record_artifacts
    identifier_columns = {"record_id", "task_id", "task_key", "dataset_name", "fold_id", "split", "panel_id", "context", "perturbation", "predictor_name"}
    assert_no_feature_label_leakage([column for column in features.columns if column not in identifier_columns])
    issues = validate_prediction_record_artifacts(STAGING, records=records, strict=True, require_effect_arrays=True)
    if issues:
        raise IntegrityFailure(f"strict PredictionRecord validation failed: {issues}")

    status = {
        "schema": STATUS_SCHEMA,
        "experiment": EXPERIMENT_ID, "stage": JOINT_STAGE,
        "status": "COMPLETE", "decision": decision, "deployment_authorized": False,
        "gate_commit": gate_commit, "gate_remote_heads": remote_heads,
        "n_fresh_test_targets": 800, "n_test_tasks": len(metrics), "n_panels": 4,
        "n_prediction_records": len(records), "prior_selected_targets_in_inference": 0,
        "primary_delta": float(all_result.observed_equal_panel_state_delta),
        "primary_ci95": [float(all_result.bootstrap_ci95_lower), float(all_result.bootstrap_ci95_upper)],
        "primary_permutation_p_one_sided": float(all_result.permutation_p_one_sided),
        "seen_delta": float(seen_result.observed_equal_panel_state_delta),
        "seen_ci95": [float(seen_result.bootstrap_ci95_lower), float(seen_result.bootstrap_ci95_upper)],
        "seen_permutation_p_one_sided": float(seen_result.permutation_p_one_sided),
        "decision_checks": checks, "prediction_record_contract_issues": issues,
        "test_truth_used_for_training_scoring_threshold_or_panel_selection": False,
        "same_test_donor_study_target_replication_only": True,
        "independent_donor_or_study_replication_claim": False,
        "wall_seconds": time.time() - started,
    }
    atomic_json(STAGING / "RUN_STATUS.json", status)
    report = f"""# {JOINT_REPORT_TITLE}

正式判定：**{decision}**。

{EXPERIMENT_CODE} 新 800 targets、12 个 panel×state 等权的 Δ(AURC_magnitude−AURC_SafeConf)={all_result.observed_equal_panel_state_delta:.6g}，95% CI [{all_result.bootstrap_ci95_lower:.6g}, {all_result.bootstrap_ci95_upper:.6g}]，单侧置换 p={all_result.permutation_p_one_sided:.6g}。

seen 640 targets 的 Δ={seen_result.observed_equal_panel_state_delta:.6g}，95% CI [{seen_result.bootstrap_ci95_lower:.6g}, {seen_result.bootstrap_ci95_upper:.6g}]，p={seen_result.permutation_p_one_sided:.6g}。12 个 panel×state 中 {checks['positive_panel_state_units']} 个点估计为正。

所有先前选择的目标均未进入本次推断。四个面板提高的是同一 test donor/study 内的目标层面精度，不能当成四个新供体或独立研究；结果不形成部署或临床授权。
"""
    atomic_bytes(STAGING / f"reports/{JOINT_REPORT_NAME}", report.encode())
    write_manifest(STAGING); os.replace(STAGING, RELEASE)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-commit")
    parser.add_argument("--branch")
    parser.add_argument("--synthetic-test-only", action="store_true")
    args = parser.parse_args()
    if args.synthetic_test_only:
        tests = synthetic_tests(); print(tests.to_string(index=False))
        if len(tests) != 7 or not tests.passed.astype(bool).all():
            raise SystemExit(2)
        return
    if not args.gate_commit or not args.branch:
        parser.error("formal evaluation requires --gate-commit and --branch")
    result = run_formal(args.gate_commit, args.branch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
