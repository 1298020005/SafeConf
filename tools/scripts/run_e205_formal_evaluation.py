#!/usr/bin/env python3
"""Evaluate E205 only after a separately committed truth-release authorization.

This runner cannot create an authorization and cannot discover target truth on
its own.  It accepts the previously released E201 target centroids only after
the E205 pretruth risk table has existed in an earlier Git commit and a later,
tracked authorization names that exact commit and both sealed-file hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
E201_FINAL = (
    ROOT
    / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
    / "formal_core_evaluation/E201_CORE_FINAL_STATUS.json"
)
GAT_RISK_STATUS = (
    ROOT
    / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
    / "E201_PRETRUTH_RISK_STATUS.json"
)
TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
N_TASKS = 2_008
N_PRIMARY = 1_808
N_GENES = 3_352
BUDGETS = (0.05, 0.10, 0.20, 0.30)
PREDICTORS = (
    "safeconf_m_4to1",
    "predicted_magnitude",
    "safeconf_e205_risk",
    "family_disagreement",
    "gat_family_disagreement",
    "registered_family_disagreement",
    "cross_family_disagreement",
)
REGISTERED_ROUTING_PREDICTORS = (
    "certificate_priority_q80",
    "registered_predicted_magnitude",
    "registered_family_disagreement",
)


class EvaluationFailure(RuntimeError):
    """Fail-closed E205 evaluation error."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--risk-table", type=Path, required=True)
    parser.add_argument("--risk-status", type=Path, required=True)
    parser.add_argument("--release-authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5_000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_data_path(value: str, data_root: Path) -> Path:
    if not value.startswith("DATA/"):
        raise EvaluationFailure(f"not a DATA-relative path: {value}")
    path = (data_root / value[len("DATA/") :]).resolve()
    try:
        path.relative_to(data_root.resolve())
    except ValueError as exc:
        raise EvaluationFailure(f"DATA path escapes root: {value}") from exc
    return path


def verify_record(path: Path, record: dict, label: str) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise EvaluationFailure(f"sealed {label} changed: {path}")


def git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True
    ).strip()


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise EvaluationFailure(f"tracked input is outside repository: {path}") from exc


def tracked_clean(path: Path) -> bool:
    try:
        relative = repo_relative(path)
    except EvaluationFailure:
        return False
    commands = (
        ["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{relative}"],
        ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", relative],
        ["git", "-C", str(ROOT), "diff", "--cached", "--quiet", "HEAD", "--", relative],
    )
    return all(
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
        for command in commands
    )


def verify_git_release(*paths: Path) -> tuple[str, str]:
    required = (SCRIPT, E201_FINAL, GAT_RISK_STATUS, *paths)
    if not all(path.is_file() and tracked_clean(path) for path in required):
        raise EvaluationFailure("evaluation code and sealed inputs must be tracked and clean")
    branch, head = git_text("branch", "--show-current"), git_text("rev-parse", "HEAD")
    if not branch:
        raise EvaluationFailure("detached HEAD is not allowed")
    for remote in ("origin", "github"):
        if git_text("rev-parse", f"{remote}/{branch}") != head:
            raise EvaluationFailure(f"{remote}/{branch} differs from local HEAD")
    return branch, head


def file_sha_at_commit(commit: str, path: Path) -> str:
    relative = repo_relative(path)
    process = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if process.returncode != 0:
        raise EvaluationFailure(f"risk seal missing at commit {commit}: {relative}")
    return hashlib.sha256(process.stdout).hexdigest()


def validate_authorization(
    authorization: dict,
    risk_status_sha: str,
    risk_table_sha: str,
    e201_final_sha: str,
) -> str:
    required = {
        "experiment": "E205_cross_family_exphormer",
        "stage": "TARGET_TRUTH_REUSE_AUTHORIZATION",
        "status": "AUTHORIZED",
        "allow_released_e201_target_centroids": True,
        "risk_sealed_and_pushed_before_authorization": True,
    }
    if any(authorization.get(key) != value for key, value in required.items()):
        raise EvaluationFailure("E205 target-truth authorization fields failed")
    if authorization.get("pretruth_risk_status_sha256") != risk_status_sha:
        raise EvaluationFailure("authorization names another risk status")
    if authorization.get("pretruth_risk_table_sha256") != risk_table_sha:
        raise EvaluationFailure("authorization names another risk table")
    if authorization.get("e201_core_final_status_sha256") != e201_final_sha:
        raise EvaluationFailure("authorization names another E201 truth source")
    commit = str(authorization.get("risk_seal_commit", ""))
    if len(commit) < 7:
        raise EvaluationFailure("authorization lacks the earlier risk-seal commit")
    return commit


def verify_authorization_history(
    authorization: dict,
    risk_status: Path,
    risk_table: Path,
    current_head: str,
) -> None:
    commit = str(authorization["risk_seal_commit"])
    if commit == current_head:
        raise EvaluationFailure("authorization must be committed after the risk seal")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, current_head],
        check=False,
    ).returncode == 0
    if not ancestor:
        raise EvaluationFailure("risk-seal commit is not an ancestor of evaluation HEAD")
    if file_sha_at_commit(commit, risk_status) != sha256_file(risk_status):
        raise EvaluationFailure("risk status changed after its named seal commit")
    if file_sha_at_commit(commit, risk_table) != sha256_file(risk_table):
        raise EvaluationFailure("risk table changed after its named seal commit")


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise EvaluationFailure("rank requires a finite one-dimensional vector")
    return rankdata(values, method="average") / len(values)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = percentile_rank(left), percentile_rank(right)
    if np.std(left_rank) <= 0 or np.std(right_rank) <= 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def partial_spearman(
    predictor: np.ndarray, outcome: np.ndarray, covariate: np.ndarray
) -> float:
    x, y, z = percentile_rank(predictor), percentile_rank(outcome), percentile_rank(covariate)
    design = np.column_stack([np.ones(len(z)), z])
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(x_residual) <= 0 or np.std(y_residual) <= 0:
        return float("nan")
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def stable_ties(task_ids: np.ndarray, occurrences: np.ndarray) -> np.ndarray:
    values = []
    for task_id, occurrence in zip(map(str, task_ids), occurrences):
        payload = f"E205\0{task_id}\0{int(occurrence)}".encode("utf-8")
        values.append(int(hashlib.sha256(payload).hexdigest()[:16], 16))
    return np.asarray(values, dtype=np.uint64)


def review_metrics(
    score: np.ndarray,
    outcome: np.ndarray,
    task_ids: np.ndarray,
    budget: float,
    occurrences: np.ndarray | None = None,
) -> dict[str, float]:
    score, outcome = np.asarray(score, float), np.asarray(outcome, float)
    task_ids = np.asarray(list(map(str, task_ids)))
    if occurrences is None:
        occurrences = np.zeros(len(score), dtype=int)
    occurrences = np.asarray(occurrences, dtype=int)
    if (
        not 0 < budget < 1
        or not (len(score) == len(outcome) == len(task_ids) == len(occurrences))
        or len(score) < 5
        or not np.isfinite(score).all()
        or not np.isfinite(outcome).all()
    ):
        raise EvaluationFailure("invalid review-metric inputs")
    n_select = int(math.ceil(budget * len(score)))
    ties = stable_ties(task_ids, occurrences)
    selected = np.lexsort((ties, -score))[:n_select]
    oracle = np.lexsort((ties, -outcome))[:n_select]
    selected_mean, oracle_mean = float(outcome[selected].mean()), float(outcome[oracle].mean())
    overall_mean = float(outcome.mean())
    denominator = oracle_mean - overall_mean
    utility = (selected_mean - overall_mean) / denominator if denominator > 1e-15 else float("nan")
    return {
        "budget": budget,
        "n_tasks": len(score),
        "n_selected": n_select,
        "high_error_capture": len(set(selected) & set(oracle)) / n_select,
        "selected_mean_error": selected_mean,
        "overall_mean_error": overall_mean,
        "error_lift": selected_mean / overall_mean,
        "oracle_mean_error": oracle_mean,
        "oracle_normalized_utility": float(utility),
    }


def load_risk_inputs(
    data_root: Path, risk_table: Path, risk_status_path: Path
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    status = json.loads(risk_status_path.read_text(encoding="utf-8"))
    if (
        status.get("experiment") != "E205_cross_family_exphormer"
        or status.get("stage") != "PRETRUTH_RISK_FEATURES"
        or status.get("status") != "PASS"
        or int(status.get("n_tasks", -1)) != N_TASKS
        or int(status.get("n_primary_tasks", -1)) != N_PRIMARY
        or int(status.get("target_expression_nonzero_values_seen", -1)) != 0
        or status.get("target_truth_materialized") is not False
        or status.get("target_outcomes_evaluated") is not False
    ):
        raise EvaluationFailure("E205 pretruth risk status failed")
    record = status.get("risk_table", {})
    if record.get("path") != repo_relative(risk_table):
        raise EvaluationFailure("E205 pretruth risk-table path changed")
    verify_record(risk_table, record, "E205 risk table")
    features = pd.read_csv(risk_table, keep_default_na=True)
    required = {
        "task_id", "target", "condition", "analysis_stratum", "source_mean_delta_row",
        "predicted_magnitude", "safeconf_e205_risk", "safeconf_m_4to1",
        "family_disagreement", "gat_family_disagreement",
        "registered_family_disagreement", "cross_family_disagreement",
        "registered_predicted_magnitude", "certificate_priority_q80",
    }
    if not required.issubset(features.columns):
        raise EvaluationFailure(f"E205 risk schema missing: {sorted(required-set(features.columns))}")
    if (
        len(features) != N_TASKS
        or features.task_id.nunique() != N_TASKS
        or int(features.analysis_stratum.eq("primary_ge30").sum()) != N_PRIMARY
        or not np.array_equal(features.source_mean_delta_row.to_numpy(int), np.arange(N_TASKS))
        or tuple(pd.unique(features.target.astype(str))) != TARGETS
        or not np.isfinite(features[list(PREDICTORS)].to_numpy(float)).all()
        or not np.isfinite(
            features[list(REGISTERED_ROUTING_PREDICTORS)].to_numpy(float)
        ).all()
    ):
        raise EvaluationFailure("E205 risk task contract changed")
    records = {Path(item["path"]).name: item for item in status.get("vector_files", [])}
    expected = {
        "E205_SEED_CENTROIDS.npy": (4, N_TASKS, N_GENES),
        "E205_FAMILY_CENTROIDS.npy": (N_TASKS, N_GENES),
        "E205_REGISTERED_FAMILY_CENTROIDS.npy": (N_TASKS, N_GENES),
        "E205_CONTROL_CENTROIDS.npy": (N_TASKS, N_GENES),
        "E205_SOURCE_TRANSFER_CENTROIDS.npy": (N_TASKS, N_GENES),
    }
    if set(records) != set(expected):
        raise EvaluationFailure("E205 vector family is incomplete")
    vectors = {}
    for name, shape in expected.items():
        path = resolve_data_path(records[name]["path"], data_root)
        verify_record(path, records[name], name)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != shape or str(values.dtype) != "float32":
            raise EvaluationFailure(f"E205 vector contract changed: {name}")
        vectors[name] = values
    gat_status = json.loads(GAT_RISK_STATUS.read_text(encoding="utf-8"))
    if (
        sha256_file(GAT_RISK_STATUS)
        != status.get("gat_pretruth_risk_status_sha256")
        or gat_status.get("status") != "PASS"
        or int(gat_status.get("n_tasks", -1)) != N_TASKS
        or gat_status.get("target_truth_materialized") is not False
        or gat_status.get("target_outcomes_evaluated") is not False
    ):
        raise EvaluationFailure("sealed E201 GAT prediction family changed")
    gat_records = {
        Path(item["path"]).name: item for item in gat_status.get("vector_files", [])
    }
    gat_record = gat_records.get("E201_SEED_CENTROIDS.npy", {})
    gat_path = resolve_data_path(gat_record.get("path", ""), data_root)
    verify_record(gat_path, gat_record, "E201 GAT seed centroids")
    gat_seed = np.load(gat_path, mmap_mode="r", allow_pickle=False)
    if gat_seed.shape != (4, N_TASKS, N_GENES) or str(gat_seed.dtype) != "float32":
        raise EvaluationFailure("E201 GAT seed-centroid contract changed")
    vectors["E201_GAT_SEED_CENTROIDS.npy"] = gat_seed
    return features, vectors, status


def load_truth_centroids(data_root: Path) -> tuple[np.ndarray, dict]:
    status = json.loads(E201_FINAL.read_text(encoding="utf-8"))
    if (
        status.get("experiment") != "E201_txpert_multitarget_retraining"
        or status.get("stage") != "FORMAL_CORE_EVALUATION"
        or status.get("status") != "PASS"
        or status.get("target_truth_released_after_risk_seal") is not True
    ):
        raise EvaluationFailure("released E201 truth-centroid source failed")
    record = status.get("truth_centroid_file", {})
    path = resolve_data_path(record.get("path", ""), data_root)
    verify_record(path, record, "E201 truth centroids")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != (N_TASKS, N_GENES) or str(values.dtype) != "float32":
        raise EvaluationFailure("E201 truth-centroid contract changed")
    return values, status


def task_errors(
    features: pd.DataFrame,
    vectors: dict[str, np.ndarray],
    truth: np.ndarray,
) -> pd.DataFrame:
    seed = np.asarray(vectors["E205_SEED_CENTROIDS.npy"], dtype=np.float64)
    gat_seed = np.asarray(
        vectors["E201_GAT_SEED_CENTROIDS.npy"], dtype=np.float64
    )
    family = np.asarray(vectors["E205_FAMILY_CENTROIDS.npy"], dtype=np.float64)
    registered_family = np.asarray(
        vectors["E205_REGISTERED_FAMILY_CENTROIDS.npy"], dtype=np.float64
    )
    truth64 = np.asarray(truth, dtype=np.float64)
    seed_rmse = np.sqrt(np.mean(np.square(seed - truth64[None, :]), axis=2))
    gat_seed_rmse = np.sqrt(
        np.mean(np.square(gat_seed - truth64[None, :]), axis=2)
    )
    family_rmse = np.sqrt(np.mean(np.square(family - truth64), axis=1))
    family_rms = np.sqrt(np.mean(np.square(seed - truth64[None, :]), axis=(0, 2)))
    registered_members = np.concatenate([gat_seed, seed], axis=0)
    recomputed_registered_family = registered_members.mean(axis=0)
    if not np.allclose(
        recomputed_registered_family,
        registered_family,
        atol=2e-6,
        rtol=0,
    ):
        raise EvaluationFailure("registered-family centroid changed after sealing")
    registered_seed_rmse = np.sqrt(
        np.mean(np.square(registered_members - truth64[None, :]), axis=2)
    )
    registered_centroid_rmse = np.sqrt(
        np.mean(np.square(registered_family - truth64), axis=1)
    )
    registered_rms = np.sqrt(np.mean(np.square(registered_seed_rmse), axis=0))
    result = features.copy()
    for index, number in enumerate((1, 2, 3, 4)):
        result[f"exphormer_seed_{number}_rmse"] = seed_rmse[index]
        result[f"gat_seed_{number}_rmse"] = gat_seed_rmse[index]
    result["family_centroid_rmse"] = family_rmse
    result["family_rms_error"] = family_rms
    result["worst_seed_error"] = seed_rmse.max(axis=0)
    result["family_identity_residual"] = np.square(family_rms) - (
        np.square(family_rmse) + np.square(result.family_disagreement.to_numpy(float))
    )
    result["registered_family_centroid_rmse"] = registered_centroid_rmse
    result["registered_family_rms_error"] = registered_rms
    result["registered_family_worst_member_error"] = registered_seed_rmse.max(axis=0)
    result["registered_family_identity_residual"] = np.square(registered_rms) - (
        np.square(registered_centroid_rmse)
        + np.square(result.registered_family_disagreement.to_numpy(float))
    )
    scale = np.maximum(
        np.square(registered_rms),
        np.square(registered_centroid_rmse)
        + np.square(result.registered_family_disagreement.to_numpy(float)),
    )
    result["registered_family_identity_tolerance"] = np.maximum(
        1e-12, 1e-10 * scale
    )
    result["registered_family_lower_tightness"] = np.divide(
        result.registered_family_disagreement.to_numpy(float),
        registered_rms,
        out=np.full_like(registered_rms, np.nan),
        where=registered_rms > 0,
    )
    return result


def certificate_curves(frame: pd.DataFrame, tau_grid: list[dict]) -> pd.DataFrame:
    """Evaluate frozen high-error certificates for tasks and perturbation clusters."""
    rows: list[dict] = []
    scopes = [("pooled", frame)] + [
        (target, frame.loc[frame.target.eq(target)]) for target in TARGETS
    ]
    for scope, block in scopes:
        for record in tau_grid:
            tau = float(record["tau"])
            true_high = block.registered_family_rms_error.to_numpy(float) > tau
            certified = (
                block.registered_family_disagreement.to_numpy(float) > tau
            )
            n_true = int(true_high.sum())
            n_certified = int(certified.sum())
            n_true_positive = int((true_high & certified).sum())
            rows.append(
                {
                    "unit": "task",
                    "scope": scope,
                    "quantile": float(record["quantile"]),
                    "tau": tau,
                    "n_units": len(block),
                    "n_true_high": n_true,
                    "n_certified_high": n_certified,
                    "n_false_certificates": int((certified & ~true_high).sum()),
                    "certified_high_coverage": n_certified / len(block),
                    "certified_high_recall": (
                        n_true_positive / n_true if n_true else float("nan")
                    ),
                    "certified_high_precision": (
                        n_true_positive / n_certified
                        if n_certified
                        else float("nan")
                    ),
                    "unknown_fraction": 1.0 - n_certified / len(block),
                }
            )

    # A perturbation is truly/certifiably high when any of its context tasks
    # exceeds tau.  Maxima preserve the deterministic lower-bound implication.
    target = (
        frame.groupby("condition", observed=True)
        .agg(
            registered_family_rms_error=("registered_family_rms_error", "max"),
            registered_family_disagreement=(
                "registered_family_disagreement",
                "max",
            ),
        )
        .reset_index()
    )
    for record in tau_grid:
        tau = float(record["tau"])
        true_high = target.registered_family_rms_error.to_numpy(float) > tau
        certified = target.registered_family_disagreement.to_numpy(float) > tau
        n_true = int(true_high.sum())
        n_certified = int(certified.sum())
        n_true_positive = int((true_high & certified).sum())
        rows.append(
            {
                "unit": "perturbation_cluster",
                "scope": "pooled",
                "quantile": float(record["quantile"]),
                "tau": tau,
                "n_units": len(target),
                "n_true_high": n_true,
                "n_certified_high": n_certified,
                "n_false_certificates": int((certified & ~true_high).sum()),
                "certified_high_coverage": n_certified / len(target),
                "certified_high_recall": (
                    n_true_positive / n_true if n_true else float("nan")
                ),
                "certified_high_precision": (
                    n_true_positive / n_certified
                    if n_certified
                    else float("nan")
                ),
                "unknown_fraction": 1.0 - n_certified / len(target),
            }
        )
    return pd.DataFrame(rows)


def scope_rows(frame: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    associations, utilities = [], []
    scopes = [("pooled", frame)] + [(target, frame.loc[frame.target.eq(target)]) for target in TARGETS]
    for scope, block in scopes:
        outcome = block.family_rms_error.to_numpy(float)
        for predictor in PREDICTORS:
            associations.append(
                {
                    "scope": scope,
                    "predictor": predictor,
                    "outcome": "family_rms_error",
                    "n_tasks": len(block),
                    "spearman": spearman(block[predictor].to_numpy(float), outcome),
                    "partial_spearman_control_magnitude": (
                        partial_spearman(
                            block[predictor].to_numpy(float),
                            outcome,
                            block.predicted_magnitude.to_numpy(float),
                        )
                        if predictor != "predicted_magnitude"
                        else float("nan")
                    ),
                }
            )
            for budget in BUDGETS:
                utilities.append(
                    {
                        "scope": scope,
                        "predictor": predictor,
                        **review_metrics(
                            block[predictor].to_numpy(float),
                            outcome,
                            block.task_id.to_numpy(),
                            budget,
                        ),
                    }
                )
    return associations, utilities


def registered_routing_rows(frame: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Evaluate the frozen certificate-priority router on its declared family."""
    associations, utilities = [], []
    scopes = [("pooled", frame)] + [
        (target, frame.loc[frame.target.eq(target)]) for target in TARGETS
    ]
    for scope, block in scopes:
        outcome = block.registered_family_rms_error.to_numpy(float)
        for predictor in REGISTERED_ROUTING_PREDICTORS:
            associations.append(
                {
                    "scope": scope,
                    "predictor": predictor,
                    "outcome": "registered_family_rms_error",
                    "n_tasks": len(block),
                    "spearman": spearman(
                        block[predictor].to_numpy(float), outcome
                    ),
                }
            )
            for budget in BUDGETS:
                utilities.append(
                    {
                        "scope": scope,
                        "predictor": predictor,
                        **review_metrics(
                            block[predictor].to_numpy(float),
                            outcome,
                            block.task_id.to_numpy(),
                            budget,
                        ),
                    }
                )
    return associations, utilities


def bootstrap_increment(frame: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    if n_bootstrap < 100:
        raise EvaluationFailure("formal E205 bootstrap requires at least 100 draws")
    clusters = sorted(frame.condition.astype(str).unique())
    members = [np.flatnonzero(frame.condition.astype(str).to_numpy() == item) for item in clusters]
    rng = np.random.default_rng(205_202_609)
    rows = []
    for draw in range(n_bootstrap):
        chosen = rng.integers(0, len(members), len(members))
        indices = np.concatenate([members[index] for index in chosen])
        block = frame.iloc[indices]
        outcome = block.family_rms_error.to_numpy(float)
        combined = block.safeconf_m_4to1.to_numpy(float)
        magnitude = block.predicted_magnitude.to_numpy(float)
        occurrences = block.groupby("task_id", sort=False).cumcount().to_numpy(int)
        combined_utility = review_metrics(
            combined, outcome, block.task_id.to_numpy(), 0.20, occurrences
        )["oracle_normalized_utility"]
        magnitude_utility = review_metrics(
            magnitude, outcome, block.task_id.to_numpy(), 0.20, occurrences
        )["oracle_normalized_utility"]
        rows.append(
            {
                "draw": draw,
                "delta_spearman": spearman(combined, outcome) - spearman(magnitude, outcome),
                "delta_utility_20": combined_utility - magnitude_utility,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_registered_router(
    frame: pd.DataFrame, n_bootstrap: int
) -> pd.DataFrame:
    """Cluster-bootstrap certificate-priority routing versus family magnitude."""
    if n_bootstrap < 100:
        raise EvaluationFailure("formal E205 bootstrap requires at least 100 draws")
    clusters = sorted(frame.condition.astype(str).unique())
    members = [
        np.flatnonzero(frame.condition.astype(str).to_numpy() == item)
        for item in clusters
    ]
    rng = np.random.default_rng(216_202_609)
    rows = []
    for draw in range(n_bootstrap):
        chosen = rng.integers(0, len(members), len(members))
        indices = np.concatenate([members[index] for index in chosen])
        block = frame.iloc[indices]
        outcome = block.registered_family_rms_error.to_numpy(float)
        routed = block.certificate_priority_q80.to_numpy(float)
        magnitude = block.registered_predicted_magnitude.to_numpy(float)
        occurrences = block.groupby("task_id", sort=False).cumcount().to_numpy(int)
        routed_utility = review_metrics(
            routed,
            outcome,
            block.task_id.to_numpy(),
            0.20,
            occurrences,
        )["oracle_normalized_utility"]
        magnitude_utility = review_metrics(
            magnitude,
            outcome,
            block.task_id.to_numpy(),
            0.20,
            occurrences,
        )["oracle_normalized_utility"]
        rows.append(
            {
                "draw": draw,
                "delta_spearman": spearman(routed, outcome)
                - spearman(magnitude, outcome),
                "delta_utility_20": routed_utility - magnitude_utility,
            }
        )
    return pd.DataFrame(rows)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    risk_table, risk_status = args.risk_table.resolve(), args.risk_status.resolve()
    authorization_path, output_dir = args.release_authorization.resolve(), args.output_dir.resolve()
    if output_dir.exists():
        raise EvaluationFailure(f"refusing to overwrite: {output_dir}")
    _, head = verify_git_release(risk_table, risk_status, authorization_path)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    risk_status_sha, risk_table_sha = sha256_file(risk_status), sha256_file(risk_table)
    e201_final_sha = sha256_file(E201_FINAL)
    validate_authorization(authorization, risk_status_sha, risk_table_sha, e201_final_sha)
    verify_authorization_history(authorization, risk_status, risk_table, head)

    features, vectors, pretruth_status = load_risk_inputs(data_root, risk_table, risk_status)
    truth, truth_status = load_truth_centroids(data_root)
    evaluated = task_errors(features, vectors, truth)
    primary = evaluated.loc[evaluated.analysis_stratum.eq("primary_ge30")].copy()
    if len(primary) != N_PRIMARY or not np.isfinite(primary.family_rms_error).all():
        raise EvaluationFailure("E205 primary outcome construction failed")
    max_identity = float(primary.family_identity_residual.abs().max())
    lower_violations = int(
        (primary.family_disagreement > primary.family_rms_error + 1e-12).sum()
    )
    registered_residual = primary.registered_family_identity_residual.abs().to_numpy(
        float
    )
    registered_tolerance = primary.registered_family_identity_tolerance.to_numpy(
        float
    )
    registered_identity_failures = int(
        (registered_residual > registered_tolerance).sum()
    )
    registered_lower_violations = int(
        (
            primary.registered_family_disagreement.to_numpy(float)
            > primary.registered_family_rms_error.to_numpy(float) + 1e-12
        ).sum()
    )
    registered_tightness_median = float(
        np.nanmedian(primary.registered_family_lower_tightness.to_numpy(float))
    )
    tau_grid = pretruth_status.get("registered_family", {}).get("tau_grid", [])
    if (
        len(tau_grid) != 9
        or [float(row.get("quantile", -1)) for row in tau_grid]
        != [number / 10 for number in range(1, 10)]
        or any(not math.isfinite(float(row.get("tau", float("nan")))) for row in tau_grid)
    ):
        raise EvaluationFailure("registered-family tau grid changed after sealing")
    certificate = certificate_curves(primary, tau_grid)
    useful_rows = certificate.loc[
        certificate.unit.eq("task")
        & certificate.scope.eq("pooled")
        & certificate.n_true_high.ge(20)
    ]
    useful_certificate = bool(
        len(useful_rows)
        and useful_rows.certified_high_recall.fillna(0).max() >= 0.05
    )
    registered_certificate_supported = bool(
        registered_identity_failures == 0
        and registered_lower_violations == 0
        and registered_tightness_median >= 0.15
        and useful_certificate
    )
    associations, utilities = scope_rows(primary)
    draws = bootstrap_increment(primary, args.n_bootstrap)
    intervals = []
    for column in ("delta_spearman", "delta_utility_20"):
        values = draws[column].to_numpy(float)
        intervals.append(
            {
                "measure": column,
                "estimate": float(
                    (
                        next(
                            row["spearman"]
                            for row in associations
                            if row["scope"] == "pooled" and row["predictor"] == "safeconf_m_4to1"
                        )
                        - next(
                            row["spearman"]
                            for row in associations
                            if row["scope"] == "pooled" and row["predictor"] == "predicted_magnitude"
                        )
                    )
                    if column == "delta_spearman"
                    else (
                        next(
                            row["oracle_normalized_utility"]
                            for row in utilities
                            if row["scope"] == "pooled"
                            and row["predictor"] == "safeconf_m_4to1"
                            and math.isclose(row["budget"], 0.20)
                        )
                        - next(
                            row["oracle_normalized_utility"]
                            for row in utilities
                            if row["scope"] == "pooled"
                            and row["predictor"] == "predicted_magnitude"
                            and math.isclose(row["budget"], 0.20)
                        )
                    )
                ),
                "ci95_lower": float(np.quantile(values, 0.025)),
                "ci95_upper": float(np.quantile(values, 0.975)),
                "positive_fraction": float(np.mean(values > 0)),
            }
        )

    registered_associations, registered_utilities = registered_routing_rows(primary)
    registered_draws = bootstrap_registered_router(primary, args.n_bootstrap)
    registered_intervals = []
    for column in ("delta_spearman", "delta_utility_20"):
        values = registered_draws[column].to_numpy(float)
        routed_association = next(
            row["spearman"]
            for row in registered_associations
            if row["scope"] == "pooled"
            and row["predictor"] == "certificate_priority_q80"
        )
        magnitude_association = next(
            row["spearman"]
            for row in registered_associations
            if row["scope"] == "pooled"
            and row["predictor"] == "registered_predicted_magnitude"
        )
        routed_utility = next(
            row["oracle_normalized_utility"]
            for row in registered_utilities
            if row["scope"] == "pooled"
            and row["predictor"] == "certificate_priority_q80"
            and math.isclose(row["budget"], 0.20)
        )
        magnitude_utility = next(
            row["oracle_normalized_utility"]
            for row in registered_utilities
            if row["scope"] == "pooled"
            and row["predictor"] == "registered_predicted_magnitude"
            and math.isclose(row["budget"], 0.20)
        )
        registered_intervals.append(
            {
                "measure": column,
                "estimate": float(
                    routed_association - magnitude_association
                    if column == "delta_spearman"
                    else routed_utility - magnitude_utility
                ),
                "ci95_lower": float(np.quantile(values, 0.025)),
                "ci95_upper": float(np.quantile(values, 0.975)),
                "positive_fraction": float(np.mean(values > 0)),
            }
        )

    output_dir.mkdir(parents=True)
    atomic_csv(output_dir / "E205_TASK_METRICS.csv", evaluated)
    atomic_csv(output_dir / "E205_RISK_ASSOCIATIONS.csv", pd.DataFrame(associations))
    atomic_csv(output_dir / "E205_REVIEW_UTILITY.csv", pd.DataFrame(utilities))
    atomic_csv(output_dir / "E205_INCREMENTAL_BOOTSTRAP_DRAWS.csv", draws)
    atomic_csv(output_dir / "E205_INCREMENTAL_INTERVALS.csv", pd.DataFrame(intervals))
    atomic_csv(output_dir / "E205_REGISTERED_CERTIFICATE_CURVES.csv", certificate)
    atomic_csv(
        output_dir / "E205_REGISTERED_ROUTING_ASSOCIATIONS.csv",
        pd.DataFrame(registered_associations),
    )
    atomic_csv(
        output_dir / "E205_REGISTERED_ROUTING_UTILITY.csv",
        pd.DataFrame(registered_utilities),
    )
    atomic_csv(
        output_dir / "E205_REGISTERED_ROUTING_BOOTSTRAP_DRAWS.csv",
        registered_draws,
    )
    atomic_csv(
        output_dir / "E205_REGISTERED_ROUTING_INTERVALS.csv",
        pd.DataFrame(registered_intervals),
    )
    utility_interval = next(row for row in intervals if row["measure"] == "delta_utility_20")
    registered_utility_interval = next(
        row
        for row in registered_intervals
        if row["measure"] == "delta_utility_20"
    )
    status = {
        "experiment": "E205_cross_family_exphormer",
        "stage": "FORMAL_EVALUATION",
        "execution_status": "PASS",
        "primary_increment_status": (
            "SUPPORTED" if utility_interval["ci95_lower"] > 0 else "NOT_SUPPORTED"
        ),
        "generated_at": now(),
        "git_head": head,
        "n_tasks": len(evaluated),
        "n_primary_tasks": len(primary),
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_unit": "perturbation condition cluster across targets",
        "pretruth_risk_status_sha256": risk_status_sha,
        "pretruth_risk_table_sha256": risk_table_sha,
        "release_authorization_sha256": sha256_file(authorization_path),
        "e201_core_final_status_sha256": e201_final_sha,
        "e201_truth_centroid_sha256": truth_status["truth_centroid_file"]["sha256"],
        "pretruth_target_outcomes_evaluated": pretruth_status["target_outcomes_evaluated"],
        "family_identity_max_abs_residual": max_identity,
        "family_lower_bound_violations": lower_violations,
        "registered_family_certificate_status": (
            "SUPPORTED" if registered_certificate_supported else "NOT_SUPPORTED"
        ),
        "registered_family_identity_max_abs_residual": float(
            registered_residual.max()
        ),
        "registered_family_identity_failures": registered_identity_failures,
        "registered_family_lower_bound_violations": registered_lower_violations,
        "registered_family_lower_tightness_median": registered_tightness_median,
        "registered_family_has_useful_certificate_threshold": useful_certificate,
        "registered_family_gate": (
            "scale-aware identity failures=0; lower violations=0; median "
            "tightness>=0.15; task recall>=0.05 at a frozen tau with >=20 "
            "observed high-error tasks"
        ),
        "primary_gate": "pooled 20% SafeConf-M minus magnitude utility cluster-bootstrap CI lower > 0",
        "primary_observed": utility_interval,
        "certificate_priority_router_status": (
            "SUPPORTED"
            if registered_utility_interval["ci95_lower"] > 0
            else "NOT_SUPPORTED"
        ),
        "certificate_priority_router_role": (
            "preregistered secondary; does not replace the frozen SafeConf-M "
            "primary gate or the registered-family certificate gate"
        ),
        "certificate_priority_router_observed": registered_utility_interval,
    }
    atomic_json(output_dir / "E205_FORMAL_EVALUATION_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
