#!/usr/bin/env python3
"""Open E177 calibration truth and freeze split-conformal upper bounds."""

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
from typing import Any, Mapping, Sequence
import uuid

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
EXPERIMENT = "E177_sunshine_external_certificate"
OUT = ROOT / "docs/实验结果/E177_sunshine_external_certificate_20260719"
RELEASE = OUT / "calibration_release"
STAGING = OUT / f".calibration_release.staging.{os.getpid()}"
PRETRUTH = OUT / "pretruth_release"
PRETRUTH_SNAPSHOT = PRETRUTH / "PRETRUTH_GATE_SNAPSHOT.json"
PRETRUTH_PREDICTIONS = PRETRUTH / "arrays/PRETRUTH_PREDICTIONS.npz"
F2_ASSETS = Path("/home/yyf/data/safeconf_e177_external/isolated/F2_pretruth")
ASSET_BUILDER = ROOT / "tools/scripts/build_e177_sunshine_pretruth_assets.py"
PRETRUTH_RUNNER = ROOT / "tools/scripts/run_e177_sunshine_pretruth.py"
SOURCE_LOCK = OUT / "SOURCE_LOCK.json"
TASKS = OUT / "manifests/E177_TASK_MANIFEST.csv"
ROW_ACCESS = OUT / "manifests/E177_ROW_ACCESS_MANIFEST.csv"

TECH_GROUPS = tuple(str(value) for value in range(1, 9))
N_GENES = 512
N_CALIBRATION_TARGETS = 30
N_CALIBRATION_TASKS = 240
TARGET_COVERAGE = 0.90
CALIBRATION_PHASE = "POSTGATE_CALIBRATION_TRUTH_X"
EVALUATION_PHASE = "POSTCALIBRATION_EVALUATION_TRUTH_X"


class IntegrityFailure(RuntimeError):
    """A calibration-stage boundary or frozen-input check failed."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("xb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value, np.float32) for key, value in sorted(arrays.items())})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_dual_remote_contains_head(head: str) -> tuple[str, dict[str, str]]:
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        raise IntegrityFailure("E177 calibration requires a named Git branch")
    remotes: dict[str, str] = {}
    for remote in ("origin", "github"):
        result = subprocess.run(
            ["git", "fetch", "--quiet", remote, f"refs/heads/{branch}:refs/remotes/{remote}/{branch}"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            raise IntegrityFailure(f"cannot verify {remote}: {result.stderr.decode(errors='replace').strip()}")
        remote_head = git_text("rev-parse", f"refs/remotes/{remote}/{branch}")
        if subprocess.run(["git", "merge-base", "--is-ancestor", head, remote_head], cwd=ROOT, check=False).returncode:
            raise IntegrityFailure(f"HEAD {head} is absent from {remote}/{branch}")
        remotes[remote] = remote_head
    return branch, remotes


def require_committed(path: Path, head: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    payload = path.read_bytes()
    try:
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"required file is not committed: {relative}") from exc
    if hashlib.sha256(payload).digest() != hashlib.sha256(committed).digest():
        raise IntegrityFailure(f"working file differs from HEAD: {relative}")
    return {"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def verify_inputs() -> tuple[str, str, dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    head = git_text("rev-parse", "HEAD")
    branch, remotes = verify_dual_remote_contains_head(head)
    files = [
        RUNNER,
        PRETRUTH_RUNNER,
        ASSET_BUILDER,
        PRETRUTH_SNAPSHOT,
        PRETRUTH / "tables/PRETRUTH_SCORING_INTERFACE.csv",
        PRETRUTH / "tables/G4_SEED_STABILITY.csv",
        PRETRUTH_PREDICTIONS,
        SOURCE_LOCK,
        TASKS,
        ROW_ACCESS,
    ]
    hashes = [require_committed(path, head) for path in files]
    snapshot = json.loads(PRETRUTH_SNAPSHOT.read_text())
    if snapshot.get("status") != "PASS" or not snapshot.get("all_registered_gates_passed"):
        raise IntegrityFailure("E177 pretruth gate is not PASS")
    for relative, expected in snapshot.get("pretruth_files_sha256", {}).items():
        path = PRETRUTH / relative
        if path.is_file() and sha256_file(path) != expected:
            raise IntegrityFailure(f"pretruth release artifact changed: {relative}")
    if snapshot.get("calibration_target_x_rows_read") != 0 or snapshot.get("evaluation_target_x_rows_read") != 0:
        raise IntegrityFailure("pretruth snapshot already read calibration/evaluation rows")
    return head, branch, remotes, hashes, snapshot


def import_builder() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e177_asset_builder_for_calibration", ASSET_BUILDER)
    if spec is None or spec.loader is None:
        raise IntegrityFailure("cannot import E177 asset builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_log1p(x: Any) -> sp.csr_matrix:
    matrix = x.tocsr().astype(np.float32) if sp.issparse(x) else sp.csr_matrix(np.asarray(x, dtype=np.float32))
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise IntegrityFailure("invalid library size in selected calibration rows")
    scale = np.divide(1.0e4, totals, out=np.zeros_like(totals), where=totals > 0)
    matrix = (sp.diags(scale, dtype=np.float32) @ matrix).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix


def read_rows_full(adata: ad.AnnData, row_indices: Sequence[int]) -> sp.csr_matrix:
    rows = np.asarray(row_indices, dtype=np.int64)
    if len(rows) == 0 or len(np.unique(rows)) != len(rows) or (rows[1:] <= rows[:-1]).any():
        raise IntegrityFailure("calibration row indices must be non-empty, unique, and sorted")
    values = adata.X[rows, :]
    matrix = values.tocsr() if sp.issparse(values) else sp.csr_matrix(values)
    if matrix.shape != (len(rows), adata.n_vars):
        raise IntegrityFailure(f"unexpected calibration X slice shape: {matrix.shape}")
    return normalize_log1p(matrix)


def load_npz_vectors(path: Path, expected: int | None = None) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            value = np.asarray(archive[key], dtype=np.float32)
            if value.shape != (N_GENES,) or not np.isfinite(value).all():
                raise IntegrityFailure(f"invalid vector {path}:{key}/{value.shape}")
            result[str(key)] = value
    if expected is not None and len(result) != expected:
        raise IntegrityFailure(f"unexpected NPZ vector count: {len(result)} != {expected}")
    return result


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def conformal_rank(n_clusters: int, coverage: float) -> int:
    return min(n_clusters, math.ceil((n_clusters + 1) * coverage))


def build_calibration_truth(batch_size: int) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, int]]:
    source_lock = json.loads(SOURCE_LOCK.read_text())
    source = Path(source_lock["source_path"])
    if not source.is_file() or sha256_file(source) != source_lock["source_sha256"]:
        raise IntegrityFailure("E177 source bytes changed")
    panel = pd.read_csv(F2_ASSETS / "GENE_PANEL.csv", keep_default_na=False)
    panel_columns = panel.source_column_index.to_numpy(dtype=np.int64)
    controls = load_npz_vectors(F2_ASSETS / "CONTROL_PROFILES.npz", expected=len(TECH_GROUPS))
    rows = pd.read_csv(ROW_ACCESS, keep_default_na=False)
    rows["source_row_index"] = pd.to_numeric(rows.source_row_index, errors="raise").astype(int)
    rows["technical_group"] = rows.technical_group.astype(str)
    calibration = rows.loc[rows.truth_access_phase.eq(CALIBRATION_PHASE)].copy()
    if calibration.empty or calibration.source_row_index.duplicated().any():
        raise IntegrityFailure("invalid calibration row manifest")
    if int(rows.truth_access_phase.eq(EVALUATION_PHASE).sum()) <= 0:
        raise IntegrityFailure("evaluation rows disappeared from row manifest")
    task_sum: dict[str, np.ndarray] = {}
    task_count: dict[str, int] = {}
    access_rows: list[dict[str, Any]] = []
    adata = ad.read_h5ad(source, backed="r")
    try:
        for start in range(0, len(calibration), batch_size):
            batch = calibration.sort_values("source_row_index", kind="stable").iloc[start : start + batch_size].copy()
            norm = read_rows_full(adata, batch.source_row_index.astype(int).tolist())
            values = norm[:, panel_columns].toarray().astype(np.float32)
            for meta, vector in zip(batch.itertuples(index=False), values):
                group = str(meta.technical_group)
                task_id = f"E177::G{group}::{meta.perturbation}"
                task_sum.setdefault(task_id, np.zeros(N_GENES, dtype=np.float64))
                task_sum[task_id] += vector
                task_count[task_id] = task_count.get(task_id, 0) + 1
                access_rows.append({
                    "source_row_index": int(meta.source_row_index),
                    "truth_access_phase": CALIBRATION_PHASE,
                    "logical_x_row_read_count": 1,
                    "asset_stage": "F4_CALIBRATION_TRUTH",
                    "purpose": "split_conformal_calibration_target_truth",
                })
    finally:
        adata.file.close()
    true_effects: dict[str, np.ndarray] = {}
    table_rows: list[dict[str, Any]] = []
    for task_id in sorted(task_sum):
        group = task_id.split("::", 3)[1]
        count = int(task_count[task_id])
        true_effects[task_id] = (task_sum[task_id] / count - controls[group]).astype(np.float32)
        table_rows.append({"task_id": task_id, "technical_group": group.removeprefix("G"), "perturbation": task_id.rsplit("::", 1)[-1], "n_exact_cells_merged": count, "true_effect_key": task_id})
    if len(true_effects) != N_CALIBRATION_TASKS:
        raise IntegrityFailure(f"calibration task count changed: {len(true_effects)}")
    access = pd.DataFrame(access_rows)
    if access.truth_access_phase.value_counts().to_dict() != {CALIBRATION_PHASE: int(len(calibration))}:
        raise IntegrityFailure("calibration X access count changed")
    return true_effects, pd.DataFrame(table_rows), {CALIBRATION_PHASE: int(len(calibration))}


def calibrate(true_effects: Mapping[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scores = pd.read_csv(PRETRUTH / "tables/PRETRUTH_SCORING_INTERFACE.csv", keep_default_na=False)
    with np.load(PRETRUTH_PREDICTIONS, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key], dtype=np.float32) for key in archive.files}
    task_order = scores.task_id.astype(str).tolist()
    index = {task: i for i, task in enumerate(task_order)}
    calibration_tasks = scores.loc[scores.target_split.eq("calibration"), "task_id"].astype(str).tolist()
    if set(calibration_tasks) != set(true_effects):
        raise IntegrityFailure("calibration true-effect keys do not match pretruth scoring rows")
    rows: list[dict[str, Any]] = []
    for task_id in calibration_tasks:
        i = index[task_id]
        truth = true_effects[task_id]
        sc = arrays["scGPT_seed_mean"][i]
        ge = arrays["GEARS_seed_mean"][i]
        ensemble = arrays["ensemble_seed_family_mean"][i]
        pair_mean = 0.5 * (rmse(sc, truth) + rmse(ge, truth))
        pair_max = max(rmse(sc, truth), rmse(ge, truth))
        disagreement = float(scores.loc[scores.task_id.eq(task_id), "model_disagreement_rmse"].iloc[0])
        magnitude = float(scores.loc[scores.task_id.eq(task_id), "predicted_magnitude"].iloc[0])
        pair_lower = disagreement / 2.0
        rows.append({
            "task_id": task_id,
            "technical_group": str(scores.loc[scores.task_id.eq(task_id), "technical_group"].iloc[0]),
            "perturbation": str(scores.loc[scores.task_id.eq(task_id), "perturbation"].iloc[0]),
            "ensemble_rmse": rmse(ensemble, truth),
            "pair_mean_rmse": pair_mean,
            "pair_max_rmse": pair_max,
            "model_disagreement_rmse": disagreement,
            "pair_lower_bound_rmse": pair_lower,
            "predicted_magnitude": magnitude,
            "ensemble_base_prediction": max(0.0, magnitude),
            "pair_mean_base_prediction": max(pair_lower, magnitude),
            "calibration_or_evaluation_truth_used_for_score_selection": False,
        })
    task_errors = pd.DataFrame(rows)
    residual_rows: list[dict[str, Any]] = []
    calibration_model: dict[str, Any] = {
        "schema": "safeconf_e177_split_conformal_calibration_v1",
        "experiment": EXPERIMENT,
        "coverage": TARGET_COVERAGE,
        "cluster": "perturbation_target_across_eight_technical_groups",
        "n_calibration_targets": N_CALIBRATION_TARGETS,
        "n_calibration_tasks": N_CALIBRATION_TASKS,
        "finite_sample_order_rank_one_based": conformal_rank(N_CALIBRATION_TARGETS, TARGET_COVERAGE),
        "base_model_selection_used_calibration_truth": False,
        "base_model_spec": {
            "ensemble_rmse": "predicted_magnitude_from_pretruth_predictions",
            "pair_mean_rmse": "max(predicted_magnitude_from_pretruth_predictions, model_disagreement_rmse/2)",
        },
        "outcomes": {},
    }
    rank = conformal_rank(N_CALIBRATION_TARGETS, TARGET_COVERAGE)
    for outcome, base_col in (
        ("ensemble_rmse", "ensemble_base_prediction"),
        ("pair_mean_rmse", "pair_mean_base_prediction"),
    ):
        work = task_errors[["perturbation", outcome, base_col]].copy()
        work["residual"] = work[outcome].to_numpy(float) - work[base_col].to_numpy(float)
        clustered = work.groupby("perturbation", sort=True).residual.max().sort_values().to_numpy(float)
        quantile = float(clustered[rank - 1])
        calibration_model["outcomes"][outcome] = {
            "base_prediction_column": base_col,
            "cluster_score": "max_technical_group(observed_error_minus_base_prediction)",
            "quantile": quantile,
            "cluster_residual_sha256_float64": hashlib.sha256(np.ascontiguousarray(clustered, dtype=np.float64).tobytes()).hexdigest(),
        }
        for perturbation, group in work.groupby("perturbation", sort=True):
            residual_rows.append({
                "perturbation": perturbation,
                "outcome": outcome,
                "n_tasks": len(group),
                "cluster_residual_max": float(group.residual.max()),
                "quantile_used": quantile,
            })
    return task_errors, pd.DataFrame(residual_rows), calibration_model


def write_release(true_effects: Mapping[str, np.ndarray], true_index: pd.DataFrame, task_errors: pd.DataFrame, residuals: pd.DataFrame, calibration_model: dict[str, Any], head: str, branch: str, remotes: dict[str, str], input_hashes: list[dict[str, Any]], snapshot: dict[str, Any], access_counts: dict[str, int]) -> Path:
    if RELEASE.exists() or STAGING.exists():
        raise IntegrityFailure("E177 calibration release is append-only and already exists")
    try:
        for sub in ("arrays", "tables", "reports"):
            (STAGING / sub).mkdir(parents=True, exist_ok=False)
        atomic_npz(STAGING / "arrays/CALIBRATION_TRUE_EFFECTS.npz", dict(true_effects))
        atomic_csv(STAGING / "tables/CALIBRATION_TRUE_EFFECT_INDEX.csv", true_index)
        atomic_csv(STAGING / "tables/CALIBRATION_TASK_ERRORS.csv", task_errors)
        atomic_csv(STAGING / "tables/CALIBRATION_RESIDUALS_BY_TARGET.csv", residuals)
        atomic_csv(STAGING / "tables/INPUT_HASHES.csv", pd.DataFrame(input_hashes))
        atomic_json(STAGING / "CALIBRATION_MODEL.json", calibration_model)
        files = sorted(path for path in STAGING.rglob("*") if path.is_file())
        file_hashes = {path.relative_to(STAGING).as_posix(): sha256_file(path) for path in files}
        attestation = {
            "schema": "safeconf_e177_calibration_access_attestation_v1",
            "experiment": EXPERIMENT,
            "stage": "F4_CALIBRATION_TRUTH_AND_BOUND_FREEZE",
            "status": "PASS",
            "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "git_head": head,
            "git_branch": branch,
            "code_freeze_remote_heads": remotes,
            "runner_sha256": sha256_file(RUNNER),
            "pretruth_snapshot_sha256": sha256_file(PRETRUTH_SNAPSHOT),
            "pretruth_snapshot_status": snapshot["status"],
            "calibration_x_rows_read_by_phase": access_counts,
            "evaluation_target_x_rows_read": 0,
            "n_calibration_tasks": int(len(task_errors)),
            "n_calibration_targets": int(task_errors.perturbation.nunique()),
            "calibration_truth_used_to_select_base_model": False,
            "calibration_truth_used_to_compute_quantile": True,
            "release_files_sha256": file_hashes,
            "public_processed_data_only": True,
            "operational_wetlab_protocol_in_scope": False,
        }
        atomic_json(STAGING / "ACCESS_ATTESTATION.json", attestation)
        report = (
            "# E177 calibration report\n\n"
            "Calibration truth opened for 30 targets and 240 tasks. Evaluation target truth remains sealed.\n\n"
            f"Ensemble quantile: `{calibration_model['outcomes']['ensemble_rmse']['quantile']:.6f}`.\n\n"
            f"Pair-mean quantile: `{calibration_model['outcomes']['pair_mean_rmse']['quantile']:.6f}`.\n"
        )
        atomic_bytes(STAGING / "reports/E177_CALIBRATION_REPORT.md", report.encode())
        os.replace(STAGING, RELEASE)
        return RELEASE / "ACCESS_ATTESTATION.json"
    except Exception:
        shutil.rmtree(STAGING, ignore_errors=True)
        raise


def run(batch_size: int) -> dict[str, Any]:
    head, branch, remotes, input_hashes, snapshot = verify_inputs()
    true_effects, true_index, access_counts = build_calibration_truth(batch_size)
    errors, residuals, model = calibrate(true_effects)
    attestation = write_release(true_effects, true_index, errors, residuals, model, head, branch, remotes, input_hashes, snapshot, access_counts)
    return {
        "status": "PASS",
        "attestation": str(attestation.relative_to(ROOT)),
        "attestation_sha256": sha256_file(attestation),
        "ensemble_quantile": model["outcomes"]["ensemble_rmse"]["quantile"],
        "pair_mean_quantile": model["outcomes"]["pair_mean_rmse"]["quantile"],
        "evaluation_target_x_rows_read": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    result = run(args.batch_size)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
