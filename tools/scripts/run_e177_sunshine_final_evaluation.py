#!/usr/bin/env python3
"""Open E177 evaluation truth and apply the frozen calibration bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import beta, rankdata


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
EXPERIMENT = "E177_sunshine_external_certificate"
OUT = ROOT / "docs/实验结果/E177_sunshine_external_certificate_20260719"
RELEASE = OUT / "final_evaluation"
STAGING = OUT / f".final_evaluation.staging.{os.getpid()}"
PRETRUTH = OUT / "pretruth_release"
CALIBRATION = OUT / "calibration_release"
PRETRUTH_PREDICTIONS = PRETRUTH / "arrays/PRETRUTH_PREDICTIONS.npz"
CALIBRATION_MODEL = CALIBRATION / "CALIBRATION_MODEL.json"
CALIBRATION_ATTESTATION = CALIBRATION / "ACCESS_ATTESTATION.json"
F2_ASSETS = Path("/home/yyf/data/safeconf_e177_external/isolated/F2_pretruth")
SOURCE_LOCK = OUT / "SOURCE_LOCK.json"
TASKS = OUT / "manifests/E177_TASK_MANIFEST.csv"
ROW_ACCESS = OUT / "manifests/E177_ROW_ACCESS_MANIFEST.csv"
CALIBRATION_RUNNER = ROOT / "tools/scripts/run_e177_sunshine_calibration.py"
PRETRUTH_RUNNER = ROOT / "tools/scripts/run_e177_sunshine_pretruth.py"

TECH_GROUPS = tuple(str(value) for value in range(1, 9))
N_GENES = 512
N_EVALUATION_TARGETS = 50
N_EVALUATION_TASKS = 400
EVALUATION_PHASE = "POSTCALIBRATION_EVALUATION_TRUTH_X"
EPS = 1e-7


class IntegrityFailure(RuntimeError):
    """A final-evaluation boundary or frozen-input check failed."""


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
        raise IntegrityFailure("E177 final evaluation requires a named Git branch")
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


def verify_inputs() -> tuple[str, str, dict[str, str], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    head = git_text("rev-parse", "HEAD")
    branch, remotes = verify_dual_remote_contains_head(head)
    files = [
        RUNNER,
        CALIBRATION_RUNNER,
        PRETRUTH_RUNNER,
        PRETRUTH / "PRETRUTH_GATE_SNAPSHOT.json",
        PRETRUTH / "tables/PRETRUTH_SCORING_INTERFACE.csv",
        PRETRUTH_PREDICTIONS,
        CALIBRATION_ATTESTATION,
        CALIBRATION_MODEL,
        CALIBRATION / "tables/CALIBRATION_TASK_ERRORS.csv",
        CALIBRATION / "arrays/CALIBRATION_TRUE_EFFECTS.npz",
        SOURCE_LOCK,
        TASKS,
        ROW_ACCESS,
    ]
    hashes = [require_committed(path, head) for path in files]
    calibration_attestation = json.loads(CALIBRATION_ATTESTATION.read_text())
    if calibration_attestation.get("status") != "PASS":
        raise IntegrityFailure("E177 calibration release is not PASS")
    if calibration_attestation.get("evaluation_target_x_rows_read") != 0:
        raise IntegrityFailure("evaluation truth was already read before final evaluator")
    calibration_model = json.loads(CALIBRATION_MODEL.read_text())
    if calibration_model.get("n_calibration_targets") != 30 or calibration_model.get("finite_sample_order_rank_one_based") != 28:
        raise IntegrityFailure("E177 calibration model changed")
    return head, branch, remotes, hashes, calibration_attestation, calibration_model


def normalize_log1p(x: Any) -> sp.csr_matrix:
    matrix = x.tocsr().astype(np.float32) if sp.issparse(x) else sp.csr_matrix(np.asarray(x, dtype=np.float32))
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise IntegrityFailure("invalid library size in selected evaluation rows")
    scale = np.divide(1.0e4, totals, out=np.zeros_like(totals), where=totals > 0)
    matrix = (sp.diags(scale, dtype=np.float32) @ matrix).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix


def read_rows_full(adata: ad.AnnData, row_indices: Sequence[int]) -> sp.csr_matrix:
    rows = np.asarray(row_indices, dtype=np.int64)
    if len(rows) == 0 or len(np.unique(rows)) != len(rows) or (rows[1:] <= rows[:-1]).any():
        raise IntegrityFailure("evaluation row indices must be non-empty, unique, and sorted")
    values = adata.X[rows, :]
    matrix = values.tocsr() if sp.issparse(values) else sp.csr_matrix(values)
    if matrix.shape != (len(rows), adata.n_vars):
        raise IntegrityFailure(f"unexpected evaluation X slice shape: {matrix.shape}")
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


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 4 or np.unique(a[keep]).size < 2 or np.unique(b[keep]).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a[keep], method="average"), rankdata(b[keep], method="average"))[0, 1])


def binomial_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def build_evaluation_truth(batch_size: int) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, int]]:
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
    evaluation = rows.loc[rows.truth_access_phase.eq(EVALUATION_PHASE)].copy()
    if evaluation.empty or evaluation.source_row_index.duplicated().any():
        raise IntegrityFailure("invalid evaluation row manifest")
    task_sum: dict[str, np.ndarray] = {}
    task_count: dict[str, int] = {}
    access_rows: list[dict[str, Any]] = []
    adata = ad.read_h5ad(source, backed="r")
    try:
        ordered = evaluation.sort_values("source_row_index", kind="stable").reset_index(drop=True)
        for start in range(0, len(ordered), batch_size):
            batch = ordered.iloc[start : start + batch_size].copy()
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
                    "truth_access_phase": EVALUATION_PHASE,
                    "logical_x_row_read_count": 1,
                    "asset_stage": "F5_FINAL_EVALUATION_TRUTH",
                    "purpose": "hidden_evaluation_target_truth",
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
    if len(true_effects) != N_EVALUATION_TASKS:
        raise IntegrityFailure(f"evaluation task count changed: {len(true_effects)}")
    return true_effects, pd.DataFrame(table_rows), {EVALUATION_PHASE: int(len(evaluation))}


def evaluate(true_effects: Mapping[str, np.ndarray], calibration_model: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scores = pd.read_csv(PRETRUTH / "tables/PRETRUTH_SCORING_INTERFACE.csv", keep_default_na=False)
    with np.load(PRETRUTH_PREDICTIONS, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key], dtype=np.float32) for key in archive.files}
    order = scores.task_id.astype(str).tolist()
    index = {task: i for i, task in enumerate(order)}
    evaluation_tasks = scores.loc[scores.target_split.eq("evaluation"), "task_id"].astype(str).tolist()
    if set(evaluation_tasks) != set(true_effects):
        raise IntegrityFailure("evaluation true-effect keys do not match frozen scoring rows")
    ensemble_q = float(calibration_model["outcomes"]["ensemble_rmse"]["quantile"])
    pair_q = float(calibration_model["outcomes"]["pair_mean_rmse"]["quantile"])
    rows: list[dict[str, Any]] = []
    for task_id in evaluation_tasks:
        i = index[task_id]
        truth = true_effects[task_id]
        sc = arrays["scGPT_seed_mean"][i]
        ge = arrays["GEARS_seed_mean"][i]
        ensemble = arrays["ensemble_seed_family_mean"][i]
        sc_error = rmse(sc, truth)
        ge_error = rmse(ge, truth)
        pair_mean = 0.5 * (sc_error + ge_error)
        pair_max = max(sc_error, ge_error)
        row = scores.loc[scores.task_id.eq(task_id)].iloc[0]
        disagreement = float(row.model_disagreement_rmse)
        magnitude = float(row.predicted_magnitude)
        pair_lower = disagreement / 2.0
        ensemble_base = max(0.0, magnitude)
        pair_base = max(pair_lower, magnitude)
        ensemble_upper = max(0.0, ensemble_base + ensemble_q)
        pair_upper = max(pair_lower, pair_base + pair_q)
        rows.append({
            "task_id": task_id,
            "technical_group": str(row.technical_group),
            "perturbation": str(row.perturbation),
            "scgpt_rmse": sc_error,
            "gears_rmse": ge_error,
            "ensemble_rmse": rmse(ensemble, truth),
            "pair_mean_rmse": pair_mean,
            "pair_max_rmse": pair_max,
            "model_disagreement_rmse": disagreement,
            "pair_lower_bound_rmse": pair_lower,
            "predicted_magnitude": magnitude,
            "safeconf_risk": float(row.safeconf_risk),
            "ensemble_base_prediction": ensemble_base,
            "ensemble_upper_bound": ensemble_upper,
            "pair_mean_base_prediction": pair_base,
            "pair_mean_upper_bound": pair_upper,
            "ensemble_covered": rmse(ensemble, truth) <= ensemble_upper + EPS,
            "pair_mean_covered": pair_mean <= pair_upper + EPS,
            "pair_lower_mean_violation": pair_lower > pair_mean + EPS,
            "pair_lower_max_violation": pair_lower > pair_max + EPS,
        })
    task = pd.DataFrame(rows)
    cluster_rows: list[dict[str, Any]] = []
    for perturbation, group in task.groupby("perturbation", sort=True):
        cluster_rows.append({
            "perturbation": perturbation,
            "n_tasks": len(group),
            "ensemble_cluster_covered": bool(group.ensemble_covered.all()),
            "pair_mean_cluster_covered": bool(group.pair_mean_covered.all()),
            "max_ensemble_error": float(group.ensemble_rmse.max()),
            "max_ensemble_upper": float(group.ensemble_upper_bound.min()),
            "max_pair_mean_error": float(group.pair_mean_rmse.max()),
            "min_pair_mean_upper": float(group.pair_mean_upper_bound.min()),
        })
    clusters = pd.DataFrame(cluster_rows)
    diag_rows = []
    for outcome in ("ensemble_rmse", "pair_mean_rmse", "pair_max_rmse"):
        for score in ("safeconf_risk", "predicted_magnitude", "model_disagreement_rmse", "pair_lower_bound_rmse"):
            diag_rows.append({"outcome": outcome, "score": score, "n_tasks": len(task), "spearman": spearman(task[score], task[outcome])})
    diagnostics = pd.DataFrame(diag_rows)

    ens_k = int(clusters.ensemble_cluster_covered.sum())
    pair_k = int(clusters.pair_mean_cluster_covered.sum())
    ens_ci = binomial_ci(ens_k, len(clusters))
    pair_ci = binomial_ci(pair_k, len(clusters))
    summary = {
        "schema": "safeconf_e177_final_evaluation_summary_v1",
        "experiment": EXPERIMENT,
        "status": "COMPLETE",
        "n_evaluation_targets": int(len(clusters)),
        "n_evaluation_tasks": int(len(task)),
        "ensemble_target_cluster_covered": ens_k,
        "ensemble_target_cluster_coverage": ens_k / len(clusters),
        "ensemble_target_cluster_ci95": list(ens_ci),
        "pair_mean_target_cluster_covered": pair_k,
        "pair_mean_target_cluster_coverage": pair_k / len(clusters),
        "pair_mean_target_cluster_ci95": list(pair_ci),
        "ensemble_task_coverage": float(task.ensemble_covered.mean()),
        "pair_mean_task_coverage": float(task.pair_mean_covered.mean()),
        "pair_lower_mean_violations": int(task.pair_lower_mean_violation.sum()),
        "pair_lower_max_violations": int(task.pair_lower_max_violation.sum()),
        "ensemble_quantile": float(ensemble_q),
        "pair_mean_quantile": float(pair_q),
        "median_ensemble_rmse": float(task.ensemble_rmse.median()),
        "median_pair_mean_rmse": float(task.pair_mean_rmse.median()),
        "median_pair_lower_tightness": float((task.pair_lower_bound_rmse / task.pair_mean_rmse).median()),
        "public_processed_data_only": True,
        "operational_wetlab_protocol_in_scope": False,
        "deployment_authorized": False,
    }
    return task, clusters, diagnostics, summary


def write_release(true_effects: Mapping[str, np.ndarray], truth_index: pd.DataFrame, task: pd.DataFrame, clusters: pd.DataFrame, diagnostics: pd.DataFrame, summary: dict[str, Any], head: str, branch: str, remotes: dict[str, str], input_hashes: list[dict[str, Any]], calibration_attestation: dict[str, Any], access_counts: dict[str, int]) -> Path:
    if RELEASE.exists() or STAGING.exists():
        raise IntegrityFailure("E177 final evaluation release is append-only and already exists")
    try:
        for sub in ("arrays", "tables", "reports"):
            (STAGING / sub).mkdir(parents=True, exist_ok=False)
        atomic_npz(STAGING / "arrays/EVALUATION_TRUE_EFFECTS.npz", dict(true_effects))
        atomic_csv(STAGING / "tables/EVALUATION_TRUE_EFFECT_INDEX.csv", truth_index)
        atomic_csv(STAGING / "tables/EVALUATION_TASK_RESULTS.csv", task)
        atomic_csv(STAGING / "tables/EVALUATION_TARGET_CLUSTER_COVERAGE.csv", clusters)
        atomic_csv(STAGING / "tables/EVALUATION_RANKING_DIAGNOSTICS.csv", diagnostics)
        atomic_csv(STAGING / "tables/INPUT_HASHES.csv", pd.DataFrame(input_hashes))
        atomic_json(STAGING / "E177_FINAL_SUMMARY.json", summary)
        files = sorted(path for path in STAGING.rglob("*") if path.is_file())
        file_hashes = {path.relative_to(STAGING).as_posix(): sha256_file(path) for path in files}
        attestation = {
            "schema": "safeconf_e177_final_evaluation_attestation_v1",
            "experiment": EXPERIMENT,
            "stage": "F5_FINAL_EVALUATION",
            "status": "COMPLETE",
            "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "git_head": head,
            "git_branch": branch,
            "code_freeze_remote_heads": remotes,
            "runner_sha256": sha256_file(RUNNER),
            "calibration_attestation_sha256": sha256_file(CALIBRATION_ATTESTATION),
            "calibration_status": calibration_attestation["status"],
            "evaluation_x_rows_read_by_phase": access_counts,
            "release_files_sha256": file_hashes,
            "public_processed_data_only": True,
            "operational_wetlab_protocol_in_scope": False,
        }
        atomic_json(STAGING / "ACCESS_ATTESTATION.json", attestation)
        report = (
            "# E177 final evaluation report\n\n"
            f"Evaluation targets/tasks: {summary['n_evaluation_targets']}/{summary['n_evaluation_tasks']}.\n\n"
            f"Ensemble target-cluster coverage: {summary['ensemble_target_cluster_covered']}/{summary['n_evaluation_targets']} "
            f"({summary['ensemble_target_cluster_coverage']:.3f}).\n\n"
            f"Pair-mean target-cluster coverage: {summary['pair_mean_target_cluster_covered']}/{summary['n_evaluation_targets']} "
            f"({summary['pair_mean_target_cluster_coverage']:.3f}).\n\n"
            f"Pair lower-bound violations: mean={summary['pair_lower_mean_violations']}, max={summary['pair_lower_max_violations']}.\n"
        )
        atomic_bytes(STAGING / "reports/E177_FINAL_EVALUATION_REPORT.md", report.encode())
        os.replace(STAGING, RELEASE)
        return RELEASE / "E177_FINAL_SUMMARY.json"
    except Exception:
        shutil.rmtree(STAGING, ignore_errors=True)
        raise


def run(batch_size: int) -> dict[str, Any]:
    head, branch, remotes, hashes, calibration_attestation, calibration_model = verify_inputs()
    true_effects, truth_index, access_counts = build_evaluation_truth(batch_size)
    task, clusters, diagnostics, summary = evaluate(true_effects, calibration_model)
    summary_path = write_release(true_effects, truth_index, task, clusters, diagnostics, summary, head, branch, remotes, hashes, calibration_attestation, access_counts)
    return {
        "status": "COMPLETE",
        "summary": str(summary_path.relative_to(ROOT)),
        "summary_sha256": sha256_file(summary_path),
        "ensemble_target_cluster_coverage": summary["ensemble_target_cluster_coverage"],
        "pair_mean_target_cluster_coverage": summary["pair_mean_target_cluster_coverage"],
        "pair_lower_mean_violations": summary["pair_lower_mean_violations"],
        "pair_lower_max_violations": summary["pair_lower_max_violations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    result = run(args.batch_size)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
