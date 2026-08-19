#!/usr/bin/env python3
"""Evaluate the frozen E200 K562 cross-context release."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E200_txpert_cross_context_k562_20260802"
FREEZE = OUT / "FORMAL_EVALUATION_FREEZE.md"
FINAL = OUT / "formal_evaluation"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"
FIGURES = FINAL / "figures"

PRETRUTH = OUT / "pretruth_release"
PRETRUTH_STATUS = PRETRUTH / "E200_PRETRUTH_STATUS.json"
FEATURES = PRETRUTH / "tables/E200_PRETRUTH_FEATURES.csv"
PRETRUTH_HASHES = OUT / "E200_PRETRUTH_OUTPUT_HASHES.csv"
PREDICTION_SEAL = OUT / "E200_PREDICTION_SEAL_STATUS.json"

DATA = Path("/home/yyf/data/txpert_official_20260802")
PRED = DATA / "e200/predictions"
REFERENCE = DATA / "cache/K562_cross_cell_lines/de_adata_test.h5ad"
GAT = PRED / "gat/test_predictions.h5ad"
TRUTH = PRED / "gat/test_ground_truth.h5ad"
CONTROL = PRED / "gat/test_controls.h5ad"
GENERAL = PRED / "general_baseline/test_predictions.h5ad"

SCPERTEVAL = Path("/home/yyf/archive/external/scPertEval")
SCPERTEVAL_COMMIT = "8709eb07a0e7d4ecf1c60c977f2018690a749975"
EXPECTED = {
    GAT: (
        2_235_398_016,
        "7647d4c2665ee4c546ea32e429c49d40700b90ca27104515dccb4084f41ec09f",
    ),
    TRUTH: (
        2_235_398_016,
        "c7a0ddd2aa902f8778ae1657e79841179dad7fe52ece2991f79e6ca432c0f9b7",
    ),
    CONTROL: (
        2_235_398_016,
        "6737ebaf794776a59ef6110cd1ae131c086119152901d62bdb4b462bae8bbed8",
    ),
    GENERAL: (
        2_025_572_536,
        "0d2200b0762b5aa4f7f29314bbda99032a78a4f959f937c9d14cbd444b437d30",
    ),
    REFERENCE: (
        7_767_053_064,
        "1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8",
    ),
    FEATURES: (
        167_962,
        "897fc081d38fffca7eef8fd9763be9bb5dc15cbfd8be76d7b4e5b73247378a42",
    ),
    PRETRUTH_STATUS: (
        483,
        "04f353e97a574498721b979acd499702a239c2ac079f82ab428626450208a90d",
    ),
    PRETRUTH_HASHES: (
        865,
        "c44e9c72d56efb1b6d0492789653defbecf8b6f7206bcc8df868367dd02ae2c4",
    ),
    PREDICTION_SEAL: (
        642,
        "ccb01511b5a5685dc4c44ea46ff9df6efb573038bd56e75cc8bbb71b4743d486",
    ),
    SCPERTEVAL / "src/scperteval/protocols/metrics.py": (
        13_482,
        "422eb55036aae1a6fe838143b102af93c3348a62711c8e5dc367300edd3dd293",
    ),
    SCPERTEVAL / "src/scperteval/protocols/table.py": (
        5_745,
        "33c789c027018d5fc1b35a8097f1dd50b943e6d2160bb71c9877b59d17ee7d7d",
    ),
}

N_FULL_CELLS = 150_472
N_STRICT_CELLS = 80_153
N_GENES = 3_352
N_TASKS = 580
N_PRIMARY = 566
N_SENSITIVITY = 14
MIN_PRIMARY = 30
MIN_SENSITIVITY = 10
N_BOOTSTRAP = 5_000
BUDGET = 0.20
SUBSAMPLE = 2_048
WORKERS = 8
SEED = 20_260_802
FEATURE_TOL = 1e-12

OBS_HASH = "33cc9fbfc6ea04da16e1e6d82368e5913242f3a88dc8de34e6f781d0d968521c"
VAR_HASH = "d67c176fda6515159421fea6fbaca860240cb6980ccc51745bff619dfec489ca"
PERT_HASH = "77770556a104100464288fa0fdb9caa95fb6cb3be90a4b5ef04348c601346f62"
BATCH_HASH = "5691048375390ffc72cefc3334b73529c28d0230a570fc09d3663bff52931519"

PREDICTORS = ("gat", "general_baseline", "batch_matched_control")
ENDPOINTS = (
    "mse",
    "pearson_pert",
    "rank",
    "energy_distance_pca_k=50",
    "de_auprc",
)
LOWER_IS_BETTER = {
    "mse": True,
    "pearson_pert": False,
    "rank": True,
    "energy_distance_pca_k=50": True,
    "de_auprc": False,
}
RISK_FEATURES = (
    "transfer_risk",
    "predicted_magnitude",
    "model_baseline_gap",
    "training_delta_dispersion",
    "negative_log_train_cells",
    "support_context_deficit",
)


class EvaluationFailure(RuntimeError):
    """Fail-closed E200 formal-evaluation error."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def order_hash(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def tracked_clean(path: Path) -> bool:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    commands = (
        ["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{relative}"],
        ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", relative],
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--cached",
            "--quiet",
            "HEAD",
            "--",
            relative,
        ],
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


def remote_tip(remote: str, branch: str) -> str:
    line = git_output(ROOT, "ls-remote", remote, f"refs/heads/{branch}")
    if not line:
        raise EvaluationFailure(f"missing remote branch: {remote}/{branch}")
    return line.split()[0]


def verify_git_release() -> str:
    required = (
        SCRIPT,
        FREEZE,
        FEATURES,
        PRETRUTH_STATUS,
        PRETRUTH_HASHES,
    )
    if not all(tracked_clean(path) for path in required):
        raise EvaluationFailure("formal code/freeze/pretruth release is not clean")
    branch = git_output(ROOT, "branch", "--show-current")
    head = git_output(ROOT, "rev-parse", "HEAD")
    if not branch:
        raise EvaluationFailure("detached HEAD is not allowed")
    for remote in ("origin", "github"):
        if remote_tip(remote, branch) != head:
            raise EvaluationFailure(f"{remote}/{branch} differs from local HEAD")
    if git_output(SCPERTEVAL, "rev-parse", "HEAD") != SCPERTEVAL_COMMIT:
        raise EvaluationFailure("scPertEval commit changed")
    if git_output(SCPERTEVAL, "status", "--porcelain"):
        raise EvaluationFailure("scPertEval source tree is dirty")
    return head


def logical_path(path: Path) -> str:
    resolved = path.resolve()
    for root, prefix in (
        (ROOT.resolve(), ""),
        (Path("/home/yyf/data").resolve(), "DATA/"),
        (Path("/home/yyf/archive/external").resolve(), "EXTERNAL/"),
    ):
        try:
            return prefix + resolved.relative_to(root).as_posix()
        except ValueError:
            pass
    return path.name


def verify_inputs() -> pd.DataFrame:
    rows = []
    for path, (expected_bytes, expected_sha) in EXPECTED.items():
        if not path.is_file():
            raise EvaluationFailure(f"missing formal input: {path}")
        observed_bytes = path.stat().st_size
        observed_sha = sha256_file(path)
        if observed_bytes != expected_bytes or observed_sha != expected_sha:
            raise EvaluationFailure(f"formal input mismatch: {path}")
        rows.append(
            {
                "path": logical_path(path),
                "bytes": observed_bytes,
                "sha256": observed_sha,
                "status": "PASS",
            }
        )
    status = json.loads(PRETRUTH_STATUS.read_text(encoding="utf-8"))
    if (
        status.get("status") != "PASS"
        or status.get("n_tasks") != N_TASKS
        or status.get("target_K562_perturbation_expression_rows_opened") != 0
        or status.get("target_outcomes_evaluated") is not False
    ):
        raise EvaluationFailure("pretruth release contract failed")
    return pd.DataFrame(rows)


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def condition_from_label(label: str) -> str:
    prefix, suffix = "K562_", "_1+1"
    if not label.startswith(prefix) or not label.endswith(suffix):
        raise EvaluationFailure(f"unexpected TxPert label: {label}")
    condition = label[len(prefix) : -len(suffix)]
    if len(condition.split("+")) != 2 or not condition.endswith("+ctrl"):
        raise EvaluationFailure(f"not a single-gene task: {label}")
    return condition


def verify_full_metadata(handle: ad.AnnData, label: str) -> None:
    if handle.shape != (N_FULL_CELLS, N_GENES):
        raise EvaluationFailure(f"{label} full shape changed: {handle.shape}")
    checks = {
        "obs": (order_hash(handle.obs_names), OBS_HASH),
        "var": (order_hash(handle.var_names), VAR_HASH),
        "pert": (order_hash(handle.obs.pert_cond_names.astype(str)), PERT_HASH),
        "batch": (
            order_hash(handle.obs.experimental_batches.astype(str)),
            BATCH_HASH,
        ),
    }
    failed = [name for name, pair in checks.items() if pair[0] != pair[1]]
    if failed:
        raise EvaluationFailure(f"{label} metadata changed: {failed}")


def finite_matrix(values: np.ndarray, label: str, block: int = 512) -> None:
    for start in range(0, values.shape[0], block):
        if not np.isfinite(values[start : start + block]).all():
            raise EvaluationFailure(f"non-finite values in {label}")


def load_strict_matrix(
    path: Path,
    label: str,
    strict_indices: np.ndarray,
) -> ad.AnnData:
    backed = ad.read_h5ad(path, backed="r")
    try:
        verify_full_metadata(backed, label)
        subset = backed[strict_indices].to_memory()
    finally:
        backed.file.close()
    if subset.shape != (N_STRICT_CELLS, N_GENES):
        raise EvaluationFailure(f"{label} strict shape changed: {subset.shape}")
    finite_matrix(np.asarray(subset.X), label)
    return subset


def load_outcomes(
    features: pd.DataFrame,
) -> tuple[ad.AnnData, dict[str, ad.AnnData]]:
    strict = set(features.task_id.astype(str))
    reference = ad.read_h5ad(GAT, backed="r")
    try:
        verify_full_metadata(reference, "gat")
        labels = reference.obs.pert_cond_names.astype(str).to_numpy()
        conditions = np.asarray([condition_from_label(label) for label in labels])
        strict_indices = np.flatnonzero(np.isin(conditions, list(strict)))
    finally:
        reference.file.close()
    if len(strict_indices) != N_STRICT_CELLS:
        raise EvaluationFailure(f"strict cell count changed: {len(strict_indices)}")
    handles = {
        "gat": load_strict_matrix(GAT, "gat", strict_indices),
        "truth": load_strict_matrix(TRUTH, "truth", strict_indices),
        "batch_matched_control": load_strict_matrix(
            CONTROL, "batch_matched_control", strict_indices
        ),
        "general_baseline": load_strict_matrix(
            GENERAL, "general_baseline", strict_indices
        ),
    }
    truth = handles.pop("truth")
    ref_obs = truth.obs_names.astype(str).tolist()
    ref_var = truth.var_names.astype(str).tolist()
    ref_pert = truth.obs.pert_cond_names.astype(str).tolist()
    ref_batch = truth.obs.experimental_batches.astype(str).tolist()
    for label, handle in handles.items():
        if (
            handle.obs_names.astype(str).tolist() != ref_obs
            or handle.var_names.astype(str).tolist() != ref_var
            or handle.obs.pert_cond_names.astype(str).tolist() != ref_pert
            or handle.obs.experimental_batches.astype(str).tolist() != ref_batch
        ):
            raise EvaluationFailure(f"strict alignment failed: {label}")
    observed_tasks = {
        condition_from_label(label) for label in pd.unique(ref_pert)
    }
    if observed_tasks != strict:
        raise EvaluationFailure("strict outcome tasks differ from pretruth tasks")
    return truth, handles


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(delta))))


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    if (
        keep.sum() < 4
        or np.unique(left[keep]).size < 2
        or np.unique(right[keep]).size < 2
    ):
        return float("nan")
    return float(
        np.corrcoef(
            rankdata(left[keep], method="average"),
            rankdata(right[keep], method="average"),
        )[0, 1]
    )


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"E200::{label}".encode()).hexdigest()[:8], 16)


def tie_values(task_ids) -> np.ndarray:
    return np.asarray(
        [
            int(hashlib.sha256(f"E200\0{task}".encode()).hexdigest()[:16], 16)
            for task in map(str, task_ids)
        ],
        dtype=np.uint64,
    )


def utility_arrays(
    risk: np.ndarray,
    outcome: np.ndarray,
    ties: np.ndarray,
) -> dict[str, float]:
    risk = np.asarray(risk, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    keep = np.isfinite(risk) & np.isfinite(outcome)
    risk, outcome, ties = risk[keep], outcome[keep], ties[keep]
    n_select = int(math.ceil(len(risk) * BUDGET))
    risk_order = np.lexsort((ties, -risk))[:n_select]
    oracle_order = np.lexsort((ties, -outcome))[:n_select]
    selected = float(outcome[risk_order].mean())
    oracle = float(outcome[oracle_order].mean())
    overall = float(outcome.mean())
    denominator = oracle - overall
    return {
        "budget": BUDGET,
        "n_tasks": len(risk),
        "n_selected": n_select,
        "high_error_capture": float(
            len(set(risk_order.tolist()) & set(oracle_order.tolist())) / n_select
        ),
        "random_expected_capture": float(n_select / len(risk)),
        "selected_mean_error": selected,
        "overall_mean_error": overall,
        "error_lift": float(selected / overall),
        "oracle_mean_error": oracle,
        "oracle_normalized_utility": (
            float((selected - overall) / denominator)
            if denominator > 1e-15
            else float("nan")
        ),
    }


def bootstrap_spearman(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
) -> dict[str, float]:
    x = frame[predictor].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    rng = np.random.default_rng(stable_seed(f"rho::{predictor}::{outcome}"))
    values = []
    for _ in range(N_BOOTSTRAP):
        take = rng.integers(0, len(frame), len(frame))
        value = spearman(x[take], y[take])
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise EvaluationFailure(f"no valid bootstrap rho: {predictor}/{outcome}")
    return {
        "spearman": spearman(x, y),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "bootstrap_valid": len(values),
    }


def bootstrap_utility(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
) -> dict[str, float]:
    x = frame[predictor].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    base_ties = tie_values(frame.task_id)
    point = utility_arrays(x, y, base_ties)
    rng = np.random.default_rng(stable_seed(f"utility::{predictor}::{outcome}"))
    salt = np.uint64(0x9E3779B97F4A7C15)
    occurrence = np.arange(len(frame), dtype=np.uint64) * salt
    values = []
    for _ in range(N_BOOTSTRAP):
        take = rng.integers(0, len(frame), len(frame))
        ties = base_ties[take] ^ occurrence
        value = utility_arrays(x[take], y[take], ties)[
            "oracle_normalized_utility"
        ]
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise EvaluationFailure(f"no valid utility bootstrap: {predictor}")
    return {
        **point,
        "utility_ci95_lower": float(np.quantile(values, 0.025)),
        "utility_ci95_upper": float(np.quantile(values, 0.975)),
        "utility_bootstrap_valid": len(values),
    }


def paired_increment(
    frame: pd.DataFrame,
    estimator: str,
    baseline: str,
    outcome: str,
) -> pd.DataFrame:
    est = frame[estimator].to_numpy(float)
    base = frame[baseline].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    base_ties = tie_values(frame.task_id)
    point_rho = spearman(est, y) - spearman(base, y)
    point_utility = (
        utility_arrays(est, y, base_ties)["oracle_normalized_utility"]
        - utility_arrays(base, y, base_ties)["oracle_normalized_utility"]
    )
    rng = np.random.default_rng(
        stable_seed(f"paired::{estimator}::{baseline}::{outcome}")
    )
    salt = np.uint64(0x9E3779B97F4A7C15)
    occurrence = np.arange(len(frame), dtype=np.uint64) * salt
    rho_values, utility_values = [], []
    for _ in range(N_BOOTSTRAP):
        take = rng.integers(0, len(frame), len(frame))
        delta_rho = spearman(est[take], y[take]) - spearman(base[take], y[take])
        if math.isfinite(delta_rho):
            rho_values.append(delta_rho)
        ties = base_ties[take] ^ occurrence
        delta_utility = (
            utility_arrays(est[take], y[take], ties)[
                "oracle_normalized_utility"
            ]
            - utility_arrays(base[take], y[take], ties)[
                "oracle_normalized_utility"
            ]
        )
        if math.isfinite(delta_utility):
            utility_values.append(delta_utility)
    rows = []
    for measure, point, values in (
        ("delta_spearman", point_rho, rho_values),
        ("delta_oracle_normalized_utility", point_utility, utility_values),
    ):
        if not values:
            raise EvaluationFailure(f"no valid paired bootstrap: {measure}")
        rows.append(
            {
                "estimator": estimator,
                "baseline": baseline,
                "outcome": outcome,
                "measure": measure,
                "estimate": point,
                "ci95_lower": float(np.quantile(values, 0.025)),
                "ci95_upper": float(np.quantile(values, 0.975)),
                "bootstrap_valid": len(values),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_delta(values: np.ndarray, label: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise EvaluationFailure(f"no finite paired deltas: {label}")
    rng = np.random.default_rng(stable_seed(f"mean-delta::{label}"))
    take = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    boot = values[take].mean(axis=1)
    return {
        "mean_delta": float(values.mean()),
        "ci95_lower": float(np.quantile(boot, 0.025)),
        "ci95_upper": float(np.quantile(boot, 0.975)),
        "bootstrap_valid": N_BOOTSTRAP,
    }


def task_certificate(
    truth: ad.AnnData,
    predictions: dict[str, ad.AnnData],
    features: pd.DataFrame,
) -> pd.DataFrame:
    labels = truth.obs.pert_cond_names.astype(str).to_numpy()
    condition_to_label = {
        condition_from_label(str(label)): str(label) for label in pd.unique(labels)
    }
    rows = []
    z_columns = [
        "z_model_baseline_gap",
        "z_training_delta_dispersion",
        "z_negative_log_train_cells",
        "z_support_context_deficit",
    ]
    for feature in features.itertuples(index=False):
        condition = str(feature.task_id)
        label = condition_to_label[condition]
        indices = np.flatnonzero(labels == label)
        truth_center = np.asarray(truth.X[indices], dtype=np.float64).mean(0)
        gat_center = np.asarray(
            predictions["gat"].X[indices], dtype=np.float64
        ).mean(0)
        general_center = np.asarray(
            predictions["general_baseline"].X[indices], dtype=np.float64
        ).mean(0)
        control_center = np.asarray(
            predictions["batch_matched_control"].X[indices], dtype=np.float64
        ).mean(0)
        gap = rmse(gat_center, general_center)
        magnitude = rmse(gat_center, control_center)
        identity = float(
            feature.transfer_risk
            - np.mean([getattr(feature, column) for column in z_columns])
        )
        rows.append(
            {
                "task_id": condition,
                "condition_label": label,
                "gene": condition.split("+")[0],
                "n_cells": len(indices),
                "gat_centroid_rmse": rmse(gat_center, truth_center),
                "general_baseline_centroid_rmse": rmse(
                    general_center, truth_center
                ),
                "control_centroid_rmse": rmse(control_center, truth_center),
                "recomputed_model_baseline_gap": gap,
                "recomputed_predicted_magnitude": magnitude,
                "pretruth_feature_max_abs_residual": max(
                    abs(gap - float(feature.model_baseline_gap)),
                    abs(magnitude - float(feature.predicted_magnitude)),
                    abs(identity),
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != N_TASKS or frame.task_id.nunique() != N_TASKS:
        raise EvaluationFailure("task certificate cardinality changed")
    if frame.pretruth_feature_max_abs_residual.max() >= FEATURE_TOL:
        raise EvaluationFailure("pretruth feature recomputation failed")
    return frame


def set_prediction_key(handle: ad.AnnData) -> None:
    handle.obs["perturbation"] = handle.obs.pert_cond_names.astype(str).to_numpy()


def build_scperteval_dataset(truth: ad.AnnData) -> ad.AnnData:
    source = ad.read_h5ad(REFERENCE, backed="r")
    try:
        obs = source.obs
        mask = (
            obs.cell_line.astype(str).eq("K562") & obs.control.astype(bool)
        ).to_numpy()
        controls = source[np.flatnonzero(mask)].to_memory()
    finally:
        source.file.close()
    if controls.n_obs != 10_691 or controls.var_names.tolist() != truth.var_names.tolist():
        raise EvaluationFailure("K562 control population changed")
    controls.obs["perturbation"] = "control"
    dataset = ad.concat(
        [truth, controls],
        axis=0,
        join="inner",
        merge="same",
        index_unique="::",
    )
    if dataset.shape != (N_STRICT_CELLS + 10_691, N_GENES):
        raise EvaluationFailure(f"scPertEval dataset shape changed: {dataset.shape}")
    return dataset


def run_scperteval(
    dataset: ad.AnnData,
    predictions: dict[str, ad.AnnData],
    task_counts: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sys.path.insert(0, str(SCPERTEVAL / "src"))
    import scperteval

    rows, runtimes = [], []
    for stratum, minimum in (
        ("primary_ge30", MIN_PRIMARY),
        ("sensitivity_10_29", MIN_SENSITIVITY),
    ):
        start = perf_counter()
        prepared = scperteval.prepare(
            dataset,
            list(ENDPOINTS),
            subsample=SUBSAMPLE,
            seed=SEED,
            min_cells=minimum,
            perturbation_key="perturbation",
            control_label="control",
            workers=WORKERS,
            name=f"E200_K562_cross_context_{stratum}",
        )
        runtimes.append(
            {
                "stage": stratum,
                "predictor": "ALL",
                "endpoint": "prepare",
                "seconds": perf_counter() - start,
            }
        )
        for predictor, prediction in predictions.items():
            for endpoint in ENDPOINTS:
                start = perf_counter()
                result = scperteval.score(
                    prepared,
                    endpoint,
                    prediction,
                    de_method="t-test",
                )
                runtimes.append(
                    {
                        "stage": stratum,
                        "predictor": predictor,
                        "endpoint": endpoint,
                        "seconds": perf_counter() - start,
                    }
                )
                for record in result.per_perturbation.to_dict("records"):
                    label = str(record["perturbation"])
                    condition = condition_from_label(label)
                    n_cells = task_counts[condition]
                    if stratum == "primary_ge30" and n_cells < MIN_PRIMARY:
                        continue
                    if stratum == "sensitivity_10_29" and not (
                        MIN_SENSITIVITY <= n_cells < MIN_PRIMARY
                    ):
                        continue
                    score = float(record["score"])
                    oriented = score if LOWER_IS_BETTER[endpoint] else 1.0 - score
                    rows.append(
                        {
                            "stratum": stratum,
                            "task_id": condition,
                            "condition_label": label,
                            "n_cells": n_cells,
                            "predictor": predictor,
                            "endpoint": endpoint,
                            "score": score,
                            "oriented_error": oriented,
                        }
                    )
        del prepared
        gc.collect()
    frame = pd.DataFrame(rows)
    expected = len(PREDICTORS) * len(ENDPOINTS) * (N_PRIMARY + N_SENSITIVITY)
    if len(frame) != expected:
        raise EvaluationFailure(
            f"scPertEval row count changed: {len(frame)} != {expected}"
        )
    return frame, pd.DataFrame(runtimes)


def performance_comparisons(
    task_frame: pd.DataFrame,
    scpert: pd.DataFrame,
) -> pd.DataFrame:
    primary = task_frame.loc[task_frame.n_cells.ge(MIN_PRIMARY)]
    rows = []
    for baseline, column in (
        ("general_baseline", "general_baseline_centroid_rmse"),
        ("batch_matched_control", "control_centroid_rmse"),
    ):
        delta = primary.gat_centroid_rmse.to_numpy() - primary[column].to_numpy()
        boot = bootstrap_mean_delta(delta, f"centroid_rmse::{baseline}")
        rows.append(
            {
                "endpoint": "centroid_rmse",
                "predictor": "gat",
                "baseline": baseline,
                "n_tasks": len(primary),
                "predictor_mean_error": float(primary.gat_centroid_rmse.mean()),
                "baseline_mean_error": float(primary[column].mean()),
                "task_win_rate": float((delta < 0).mean()),
                **boot,
            }
        )
    primary_sc = scpert.loc[scpert.stratum.eq("primary_ge30")]
    pivot = primary_sc.pivot(
        index=["task_id", "endpoint"],
        columns="predictor",
        values="oriented_error",
    )
    for endpoint in ENDPOINTS:
        block = pivot.xs(endpoint, level="endpoint")
        for baseline in ("general_baseline", "batch_matched_control"):
            delta = block.gat.to_numpy() - block[baseline].to_numpy()
            boot = bootstrap_mean_delta(delta, f"{endpoint}::{baseline}")
            rows.append(
                {
                    "endpoint": endpoint,
                    "predictor": "gat",
                    "baseline": baseline,
                    "n_tasks": len(block),
                    "predictor_mean_error": float(block.gat.mean()),
                    "baseline_mean_error": float(block[baseline].mean()),
                    "task_win_rate": float((delta < 0).mean()),
                    **boot,
                }
            )
    return pd.DataFrame(rows)


def risk_analysis(
    task_frame: pd.DataFrame,
    scpert: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = task_frame.loc[task_frame.n_cells.ge(MIN_PRIMARY)].copy()
    associations = []
    for outcome in ("gat_centroid_rmse", "general_baseline_centroid_rmse"):
        for predictor in RISK_FEATURES:
            associations.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "n_tasks": len(primary),
                    **bootstrap_spearman(primary, predictor, outcome),
                }
            )
    utilities = []
    for predictor in RISK_FEATURES:
        utilities.append(
            {
                "predictor": predictor,
                "outcome": "gat_centroid_rmse",
                **bootstrap_utility(primary, predictor, "gat_centroid_rmse"),
            }
        )
    increments = paired_increment(
        primary,
        "transfer_risk",
        "predicted_magnitude",
        "gat_centroid_rmse",
    )
    primary_gat = scpert.loc[
        scpert.stratum.eq("primary_ge30") & scpert.predictor.eq("gat")
    ]
    endpoint_rows = []
    for endpoint in ENDPOINTS:
        endpoint_frame = primary_gat.loc[
            primary_gat.endpoint.eq(endpoint), ["task_id", "oriented_error"]
        ].merge(
            primary[["task_id", "transfer_risk", "predicted_magnitude"]],
            on="task_id",
            validate="one_to_one",
        )
        for predictor in ("transfer_risk", "predicted_magnitude"):
            endpoint_rows.append(
                {
                    "endpoint": endpoint,
                    "predictor": predictor,
                    "n_tasks": len(endpoint_frame),
                    **bootstrap_spearman(
                        endpoint_frame, predictor, "oriented_error"
                    ),
                }
            )
    sensitivity = task_frame.loc[
        task_frame.n_cells.between(MIN_SENSITIVITY, MIN_PRIMARY - 1)
    ]
    sensitivity_rows = [
        {
            "predictor": predictor,
            "outcome": "gat_centroid_rmse",
            "n_tasks": len(sensitivity),
            "spearman": spearman(
                sensitivity[predictor].to_numpy(),
                sensitivity.gat_centroid_rmse.to_numpy(),
            ),
        }
        for predictor in ("transfer_risk", "predicted_magnitude")
    ]
    return (
        pd.DataFrame(associations),
        pd.DataFrame(utilities),
        increments,
        pd.DataFrame(endpoint_rows),
        pd.DataFrame(sensitivity_rows),
    )


def summary_table(scpert: pd.DataFrame) -> pd.DataFrame:
    return (
        scpert.groupby(
            ["stratum", "predictor", "endpoint"], observed=True
        ).oriented_error.agg(n="count", mean_error="mean", median_error="median")
        .reset_index()
    )


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_figure(
    task_frame: pd.DataFrame,
    utilities: pd.DataFrame,
    performance: pd.DataFrame,
    path: Path,
) -> None:
    primary = task_frame.loc[task_frame.n_cells.ge(MIN_PRIMARY)].copy()
    colors = {1: "#4C78A8", 2: "#F28E2B", 3: "#59A14F"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2))
    for count in (1, 2, 3):
        block = primary.loc[primary.n_train_contexts.eq(count)]
        axes[0, 0].scatter(
            block.transfer_risk,
            block.gat_centroid_rmse,
            s=16,
            alpha=0.62,
            color=colors[count],
            linewidths=0,
            label=f"{count} training context{'s' if count > 1 else ''}",
        )
    axes[0, 0].set_xlabel("Frozen transfer risk")
    axes[0, 0].set_ylabel("GAT centroid RMSE")
    axes[0, 0].set_title("A  Cross-context task risk")
    axes[0, 0].legend(frameon=False, fontsize=8)

    primary["risk_decile"] = pd.qcut(
        primary.transfer_risk.rank(method="first"), 10, labels=False
    ) + 1
    decile = primary.groupby("risk_decile").gat_centroid_rmse.agg(
        mean="mean", sem="sem"
    )
    axes[0, 1].errorbar(
        decile.index,
        decile["mean"],
        yerr=decile["sem"],
        marker="o",
        color="#4C78A8",
        ecolor="#777777",
        capsize=2,
        linewidth=1.4,
    )
    axes[0, 1].set_xlabel("Frozen risk decile")
    axes[0, 1].set_ylabel("Mean GAT centroid RMSE")
    axes[0, 1].set_title("B  Error across risk deciles")

    utility = utilities.set_index("predictor").loc[
        ["transfer_risk", "predicted_magnitude"]
    ]
    values = utility.oracle_normalized_utility.to_numpy(float)
    lower = np.maximum(values - utility.utility_ci95_lower.to_numpy(float), 0)
    upper = np.maximum(utility.utility_ci95_upper.to_numpy(float) - values, 0)
    axes[1, 0].bar(
        [0, 1], values, color=["#4C78A8", "#B7B7B7"], width=0.62
    )
    axes[1, 0].errorbar(
        [0, 1],
        values,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#333333",
        capsize=3,
        linewidth=0.9,
    )
    axes[1, 0].axhline(0, color="#555555", linewidth=0.8)
    axes[1, 0].set_xticks([0, 1], ["Transfer risk", "Magnitude"])
    axes[1, 0].set_ylabel("Oracle-normalized utility")
    axes[1, 0].set_title("C  Fixed 20% review budget")

    forest = performance.loc[
        performance.baseline.eq("general_baseline")
        & performance.endpoint.isin(ENDPOINTS)
    ].set_index("endpoint").loc[list(ENDPOINTS)]
    y = np.arange(len(forest))
    axes[1, 1].errorbar(
        forest.mean_delta,
        y,
        xerr=np.vstack(
            [
                np.maximum(forest.mean_delta - forest.ci95_lower, 0),
                np.maximum(forest.ci95_upper - forest.mean_delta, 0),
            ]
        ),
        fmt="o",
        color="#E15759",
        ecolor="#555555",
        capsize=2,
        markersize=4,
    )
    axes[1, 1].axvline(0, color="#555555", linestyle="--", linewidth=0.8)
    axes[1, 1].set_yticks(
        y, [endpoint.replace("_pca_k=50", " PCA50") for endpoint in ENDPOINTS]
    )
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("GAT minus general-baseline error")
    axes[1, 1].set_title("D  Five frozen endpoints")

    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E7E7E7", linewidth=0.55, zorder=0)
        axis.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if FINAL.exists():
        raise EvaluationFailure("formal evaluation is append-only and already exists")
    head = verify_git_release()
    input_hashes = verify_inputs()
    features = pd.read_csv(FEATURES)
    if len(features) != N_TASKS or features.task_id.nunique() != N_TASKS:
        raise EvaluationFailure("pretruth feature table changed")
    required_features = set(RISK_FEATURES) | {
        "task_id",
        "condition_label",
        "gene",
        "n_prediction_cells",
        "n_train_contexts",
        "z_model_baseline_gap",
        "z_training_delta_dispersion",
        "z_negative_log_train_cells",
        "z_support_context_deficit",
    }
    if not required_features.issubset(features.columns):
        raise EvaluationFailure("pretruth feature schema changed")

    truth, predictions = load_outcomes(features)
    for handle in (truth, *predictions.values()):
        set_prediction_key(handle)
    certificate = task_certificate(truth, predictions, features)
    task_frame = features.merge(
        certificate,
        on=["task_id", "condition_label", "gene"],
        validate="one_to_one",
    )
    if not np.array_equal(
        task_frame.n_prediction_cells.to_numpy(int),
        task_frame.n_cells.to_numpy(int),
    ):
        raise EvaluationFailure("task cell counts changed after outcome opening")

    dataset = build_scperteval_dataset(truth)
    task_counts = task_frame.set_index("task_id").n_cells.astype(int).to_dict()
    scpert, runtimes = run_scperteval(dataset, predictions, task_counts)
    del dataset
    gc.collect()
    scpert_summary = summary_table(scpert)
    performance = performance_comparisons(task_frame, scpert)
    (
        associations,
        utilities,
        increments,
        endpoint_risk,
        sensitivity,
    ) = risk_analysis(task_frame, scpert)

    primary = task_frame.loc[task_frame.n_cells.ge(MIN_PRIMARY)].copy()
    agreement_frame = primary[[
        "task_id",
        "gat_centroid_rmse",
        "general_baseline_centroid_rmse",
    ]]
    agreement = pd.DataFrame(
        [
            {
                "predictor_a": "gat",
                "predictor_b": "general_baseline",
                "n_tasks": len(agreement_frame),
                **bootstrap_spearman(
                    agreement_frame,
                    "gat_centroid_rmse",
                    "general_baseline_centroid_rmse",
                ),
            }
        ]
    )
    main_assoc = associations.loc[
        associations.predictor.eq("transfer_risk")
        & associations.outcome.eq("gat_centroid_rmse")
    ].iloc[0]
    main_utility = utilities.loc[
        utilities.predictor.eq("transfer_risk")
    ].iloc[0]
    delta_rho = increments.loc[
        increments.measure.eq("delta_spearman")
    ].iloc[0]
    delta_utility = increments.loc[
        increments.measure.eq("delta_oracle_normalized_utility")
    ].iloc[0]
    general_perf = performance.loc[
        performance.endpoint.eq("centroid_rmse")
        & performance.baseline.eq("general_baseline")
    ].iloc[0]
    feature_residual = float(task_frame.pretruth_feature_max_abs_residual.max())
    integrity = bool(
        feature_residual < FEATURE_TOL
        and len(scpert) == len(PREDICTORS) * len(ENDPOINTS) * N_TASKS
    )
    routing = bool(
        main_assoc.ci95_lower > 0 and main_utility.utility_ci95_lower > 0
    )
    incremental = bool(
        delta_rho.ci95_lower > 0 or delta_utility.ci95_lower > 0
    )
    gat_beats_general = bool(general_perf.ci95_upper < 0)
    gates = pd.DataFrame(
        [
            {
                "gate": "integrity",
                "passed": integrity,
                "observed": feature_residual,
                "criterion": "pretruth residual <1e-12 and complete endpoint rows",
            },
            {
                "gate": "empirical_routing",
                "passed": routing,
                "observed": f"rho_lower={main_assoc.ci95_lower:.6g};utility_lower={main_utility.utility_ci95_lower:.6g}",
                "criterion": "both 95% CI lower bounds >0",
            },
            {
                "gate": "incremental_vs_magnitude",
                "passed": incremental,
                "observed": f"delta_rho_lower={delta_rho.ci95_lower:.6g};delta_utility_lower={delta_utility.ci95_lower:.6g}",
                "criterion": "either paired 95% CI lower bound >0",
            },
            {
                "gate": "gat_vs_general_centroid_rmse",
                "passed": gat_beats_general,
                "observed": f"mean_delta={general_perf.mean_delta:.6g};upper={general_perf.ci95_upper:.6g}",
                "criterion": "paired mean-delta 95% CI upper <0",
            },
        ]
    )

    for directory in (TABLES, REPORTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_csv(TABLES / "E200_FORMAL_INPUT_HASHES.csv", input_hashes)
    atomic_csv(TABLES / "E200_TASK_METRICS.csv", task_frame)
    atomic_csv(TABLES / "E200_SCPERTEVAL_TASK_METRICS.csv", scpert)
    atomic_csv(TABLES / "E200_SCPERTEVAL_SUMMARY.csv", scpert_summary)
    atomic_csv(TABLES / "E200_PERFORMANCE_COMPARISONS.csv", performance)
    atomic_csv(TABLES / "E200_RISK_ASSOCIATIONS.csv", associations)
    atomic_csv(TABLES / "E200_REVIEW_UTILITY.csv", utilities)
    atomic_csv(TABLES / "E200_INCREMENTAL_TESTS.csv", increments)
    atomic_csv(TABLES / "E200_ENDPOINT_RISK_ASSOCIATIONS.csv", endpoint_risk)
    atomic_csv(TABLES / "E200_MODEL_ERROR_AGREEMENT.csv", agreement)
    atomic_csv(TABLES / "E200_SENSITIVITY_ASSOCIATIONS.csv", sensitivity)
    atomic_csv(TABLES / "E200_RUNTIME.csv", runtimes)
    atomic_csv(TABLES / "E200_FORMAL_GATES.csv", gates)
    make_figure(
        task_frame,
        utilities,
        performance,
        FIGURES / "E200_cross_context_audit.png",
    )

    primary_summary = scpert_summary.loc[
        scpert_summary.stratum.eq("primary_ge30")
    ]
    report_lines = [
        "# E200 K562 整体背景留出审计",
        "",
        f"- 完整性：**{'PASS' if integrity else 'FAIL'}**。",
        f"- 20% 复核路由：**{'ENABLED' if routing else 'ABSTAIN'}**。",
        f"- 相对 predicted magnitude 的新增价值：**{'SUPPORTED' if incremental else 'NOT SUPPORTED'}**。",
        f"- GAT 主 RMSE 优于 general baseline：**{'YES' if gat_beats_general else 'NO'}**。",
        "",
        "## 主误差与基线",
        "",
        markdown_table(
            performance.loc[
                performance.endpoint.eq("centroid_rmse"),
                [
                    "baseline",
                    "n_tasks",
                    "predictor_mean_error",
                    "baseline_mean_error",
                    "task_win_rate",
                    "mean_delta",
                    "ci95_lower",
                    "ci95_upper",
                ],
            ]
        ),
        "",
        "## 冻结风险量",
        "",
        markdown_table(
            associations.loc[
                associations.outcome.eq("gat_centroid_rmse"),
                ["predictor", "spearman", "ci95_lower", "ci95_upper"],
            ]
        ),
        "",
        markdown_table(
            utilities.loc[
                utilities.predictor.isin(["transfer_risk", "predicted_magnitude"]),
                [
                    "predictor",
                    "n_selected",
                    "high_error_capture",
                    "error_lift",
                    "oracle_normalized_utility",
                    "utility_ci95_lower",
                    "utility_ci95_upper",
                ],
            ]
        ),
        "",
        "## 五个独立端点",
        "",
        markdown_table(
            primary_summary[
                ["predictor", "endpoint", "n", "mean_error", "median_error"]
            ]
        ),
        "",
        "## 边界",
        "",
        "E200 只是 K562 目标背景上的公开 GAT checkpoint。"
        "它能回答整行留出的一个真实实例，不代表其他目标细胞系、"
        "多模型家族或跨独立数据集已经回答。",
        "",
    ]
    atomic_text(REPORTS / "E200_REPORT.md", "\n".join(report_lines))
    status = {
        "experiment": "E200_txpert_cross_context_k562",
        "stage": "FORMAL_EVALUATION",
        "generated_at": now(),
        "status": "PASS" if integrity else "FAIL",
        "git_head": head,
        "n_tasks": N_TASKS,
        "n_primary_tasks": N_PRIMARY,
        "n_sensitivity_tasks": N_SENSITIVITY,
        "pretruth_feature_max_abs_residual": feature_residual,
        "integrity_gate_pass": integrity,
        "routing_activation_gate_pass": routing,
        "routing_status": "ENABLED" if routing else "ABSTAIN",
        "incremental_vs_magnitude_gate_pass": incremental,
        "incremental_claim_status": (
            "SUPPORTED" if incremental else "NOT_SUPPORTED"
        ),
        "gat_beats_general_centroid_rmse": gat_beats_general,
        "cross_context_answered": True,
        "cross_dataset_transfer_answered": False,
        "multiple_model_families_answered": False,
        "other_target_cell_lines_answered": False,
        "performance_is_not_integrity_gate": True,
        "retrospective_public_data": True,
    }
    atomic_json(FINAL / "E200_FINAL_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
