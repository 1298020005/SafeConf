#!/usr/bin/env python3
"""Evaluate the frozen E199 TxPert public-model release.

This runner is deliberately separate from prediction and pretruth-feature
generation.  It verifies their sealed hashes, opens target expression for the
first time in an outcome calculation, evaluates the preregistered scPertEval
panel, reproduces TxPert's batch-matched secondary endpoints, and applies the
three frozen SafeConf decisions independently.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E199_txpert_public_k562_20260802"
FREEZE = OUT / "FORMAL_EVALUATION_FREEZE.md"
FINAL = OUT / "formal_evaluation"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"
FIGURES = FINAL / "figures"

PRETRUTH = OUT / "pretruth_release"
PRETRUTH_STATUS = PRETRUTH / "E199_PRETRUTH_STATUS.json"
FEATURES = PRETRUTH / "tables/E199_PRETRUTH_FEATURES.csv"

DATA = Path("/home/yyf/data/txpert_official_20260802")
PRED = DATA / "e199/predictions"
ORIGINAL = DATA / "cache/K562_single_cell_line/de_adata_test.h5ad"
TXPERT = Path("/home/yyf/archive/external/TxPert")
SCPERTEVAL = Path("/home/yyf/archive/external/scPertEval")

MODEL_PATHS = {
    "gat": PRED / "gat/test_predictions.h5ad",
    "exphormer": PRED / "exphormer/test_predictions.h5ad",
    "exphormer_mg": PRED / "exphormer_mg/test_predictions.h5ad",
}
TRUTH = PRED / "gat/test_ground_truth.h5ad"
CONTROL = PRED / "gat/test_controls.h5ad"
GENERAL = PRED / "general_baseline/test_predictions.h5ad"

EXPECTED = {
    MODEL_PATHS["gat"]: (
        1_084_480_564,
        "5e2d2bdfc67a368c3d2fd0f987c30f4c179e32158aea3f8eca9286557bd379a5",
    ),
    MODEL_PATHS["exphormer"]: (
        1_084_480_564,
        "ff475ed22837d0bb5fda92744e49019614e30e78a2465d8978b2601602f493db",
    ),
    MODEL_PATHS["exphormer_mg"]: (
        1_084_480_564,
        "764d6a864a99127904bd29f8d9e185ba2797681d19a45d8d334980feb5c62df0",
    ),
    TRUTH: (
        1_084_480_564,
        "f4b605bbea37d016c24dec96a4c1a8c6cf01d36b123980a46ab617b62b4700a2",
    ),
    CONTROL: (
        1_084_480_564,
        "3ae24c04804b33eb495c27ae75f4a3fa3930afdcdecf61c3f9cd3b272e9837a3",
    ),
    GENERAL: (
        771_973_828,
        "49e81f1743d29b332859fcb36c4b60dccb11d00ff8b141fffdb5e30682cc16a8",
    ),
    ORIGINAL: (
        2_839_201_052,
        "d9a6e4ff52b97c97dfcaaf48869b0aa1476eb251fa427ee0240bd2dcee26da40",
    ),
    FEATURES: (
        58_618,
        "5f05fae1b08f02f9a7963894945b5d4c049075842544892fab01d3ebd8694f6e",
    ),
    PRETRUTH_STATUS: (
        396,
        "1078e51814e4753ab23235998d12d4c0687eac1a87700e468b97f6834e2ee08f",
    ),
    TXPERT / "gspp/metrics.py": (
        19_257,
        "a8e7c34833edc3f9c9bfd575432ab77963ac2e0a22167f87581a56c3ad90b15c",
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

TXPERT_COMMIT = "08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
SCPERTEVAL_COMMIT = "8709eb07a0e7d4ecf1c60c977f2018690a749975"
MODEL_KEYS = tuple(MODEL_PATHS)
PREDICTOR_KEYS = (
    "gat",
    "exphormer",
    "exphormer_mg",
    "family_centroid",
    "general_baseline",
    "batch_matched_control",
)
PRIMARY_PROTOCOLS = (
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
    "diversity_lower_bound",
    "predicted_magnitude",
    "model_baseline_gap",
    "negative_string_train_neighbor_count",
    "negative_go_train_neighbor_count",
    "graph_isolated_risk",
)

N_CELLS = 38_475
N_GENES = 5_000
N_TASKS = 272
N_PRIMARY = 263
N_SENSITIVITY = 9
MIN_PRIMARY = 30
MIN_SENSITIVITY = 10
SUBSAMPLE = 2_048
WORKERS = 8
SEED = 20_260_802
N_BOOTSTRAP = 5_000
BUDGET = 0.20
IDENTITY_TOL = 1e-8
FEATURE_TOL = 1e-12


class EvaluationFailure(RuntimeError):
    """Fail-closed E199 evaluation error."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def tracked_clean(path: Path) -> bool:
    rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
    commands = (
        ["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{rel}"],
        ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel],
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--cached",
            "--quiet",
            "HEAD",
            "--",
            rel,
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
    required = (SCRIPT, FREEZE, FEATURES, PRETRUTH_STATUS)
    if not all(tracked_clean(path) for path in required):
        raise EvaluationFailure("formal runner/freeze/pretruth release is not tracked and clean")
    branch = git_output(ROOT, "branch", "--show-current")
    head = git_output(ROOT, "rev-parse", "HEAD")
    if not branch:
        raise EvaluationFailure("detached HEAD is not allowed")
    for remote in ("origin", "github"):
        if remote_tip(remote, branch) != head:
            raise EvaluationFailure(f"{remote}/{branch} does not equal local HEAD")
    for repo, expected in (
        (TXPERT, TXPERT_COMMIT),
        (SCPERTEVAL, SCPERTEVAL_COMMIT),
    ):
        if git_output(repo, "rev-parse", "HEAD") != expected:
            raise EvaluationFailure(f"external source commit changed: {repo}")
        if git_output(repo, "status", "--porcelain"):
            raise EvaluationFailure(f"external source tree is dirty: {repo}")
    return head


def logical_path(path: Path) -> str:
    resolved = path.resolve()
    for base, prefix in (
        (ROOT.resolve(), ""),
        (Path("/home/yyf/data").resolve(), "DATA/"),
        (Path("/home/yyf/archive/external").resolve(), "EXTERNAL/"),
    ):
        try:
            return prefix + resolved.relative_to(base).as_posix()
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
        or status.get("target_expression_files_opened") != 0
    ):
        raise EvaluationFailure("pretruth release contract failed")
    return pd.DataFrame(rows)


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def condition_from_label(label: str) -> str:
    prefix = "K562_"
    suffix = "_1+1"
    if not label.startswith(prefix) or not label.endswith(suffix):
        raise EvaluationFailure(f"unexpected TxPert label: {label}")
    condition = label[len(prefix) : -len(suffix)]
    if len(condition.split("+")) != 2 or not condition.endswith("+ctrl"):
        raise EvaluationFailure(f"not a single-gene task: {label}")
    return condition


def finite_matrix(X: np.ndarray, label: str, block: int = 512) -> None:
    for start in range(0, X.shape[0], block):
        if not np.isfinite(X[start : start + block]).all():
            raise EvaluationFailure(f"non-finite values in {label}")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if np.std(a) <= 0 or np.std(b) <= 0:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else 0.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 4 or np.unique(a[keep]).size < 2 or np.unique(b[keep]).size < 2:
        return float("nan")
    return float(
        np.corrcoef(
            rankdata(a[keep], method="average"),
            rankdata(b[keep], method="average"),
        )[0, 1]
    )


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"E199::{label}".encode()).hexdigest()[:8], 16)


def tie_values(task_ids) -> np.ndarray:
    return np.asarray(
        [
            int(hashlib.sha256(f"E199\0{task}".encode()).hexdigest()[:16], 16)
            for task in map(str, task_ids)
        ],
        dtype=np.uint64,
    )


def utility_arrays(
    risk: np.ndarray,
    outcome: np.ndarray,
    ties: np.ndarray,
    budget: float = BUDGET,
) -> dict[str, float]:
    risk = np.asarray(risk, float)
    outcome = np.asarray(outcome, float)
    keep = np.isfinite(risk) & np.isfinite(outcome)
    risk, outcome, ties = risk[keep], outcome[keep], ties[keep]
    n_select = int(math.ceil(len(risk) * budget))
    risk_order = np.lexsort((ties, -risk))[:n_select]
    oracle_order = np.lexsort((ties, -outcome))[:n_select]
    selected_mean = float(outcome[risk_order].mean())
    oracle_mean = float(outcome[oracle_order].mean())
    overall_mean = float(outcome.mean())
    denominator = oracle_mean - overall_mean
    return {
        "budget": budget,
        "n_tasks": len(risk),
        "n_selected": n_select,
        "high_error_capture": float(
            len(set(risk_order.tolist()) & set(oracle_order.tolist())) / n_select
        ),
        "random_expected_capture": float(n_select / len(risk)),
        "selected_mean_error": selected_mean,
        "overall_mean_error": overall_mean,
        "error_lift": float(selected_mean / overall_mean),
        "oracle_mean_error": oracle_mean,
        "oracle_normalized_utility": (
            float((selected_mean - overall_mean) / denominator)
            if denominator > 1e-15
            else float("nan")
        ),
    }


def bootstrap_spearman(
    frame: pd.DataFrame, predictor: str, outcome: str
) -> dict[str, float]:
    x = frame[predictor].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    rng = np.random.default_rng(stable_seed(f"rho::{predictor}::{outcome}"))
    boot = []
    for _ in range(N_BOOTSTRAP):
        take = rng.integers(0, len(frame), size=len(frame))
        value = spearman(x[take], y[take])
        if math.isfinite(value):
            boot.append(value)
    if not boot:
        raise EvaluationFailure(f"no valid bootstrap correlations: {predictor}")
    return {
        "spearman": spearman(x, y),
        "ci95_lower": float(np.quantile(boot, 0.025)),
        "ci95_upper": float(np.quantile(boot, 0.975)),
        "bootstrap_valid": len(boot),
    }


def bootstrap_utility(
    frame: pd.DataFrame, predictor: str, outcome: str
) -> dict[str, float]:
    x = frame[predictor].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    base_ties = tie_values(frame.task_id)
    point = utility_arrays(x, y, base_ties)
    rng = np.random.default_rng(stable_seed(f"utility::{predictor}::{outcome}"))
    boot = []
    salt = np.uint64(0x9E3779B97F4A7C15)
    occurrence = np.arange(len(frame), dtype=np.uint64) * salt
    for _ in range(N_BOOTSTRAP):
        take = rng.integers(0, len(frame), size=len(frame))
        ties = base_ties[take] ^ occurrence
        value = utility_arrays(x[take], y[take], ties)[
            "oracle_normalized_utility"
        ]
        if math.isfinite(value):
            boot.append(value)
    if not boot:
        raise EvaluationFailure(f"no valid bootstrap utilities: {predictor}")
    return {
        **point,
        "utility_ci95_lower": float(np.quantile(boot, 0.025)),
        "utility_ci95_upper": float(np.quantile(boot, 0.975)),
        "utility_bootstrap_valid": len(boot),
    }


def paired_increment(
    frame: pd.DataFrame,
    estimator: str,
    baseline: str,
    outcome: str,
) -> list[dict[str, Any]]:
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
    rho_boot = []
    utility_boot = []
    salt = np.uint64(0x9E3779B97F4A7C15)
    occurrence = np.arange(len(frame), dtype=np.uint64) * salt
    for _ in range(N_BOOTSTRAP):
        take = rng.integers(0, len(frame), size=len(frame))
        delta_rho = spearman(est[take], y[take]) - spearman(base[take], y[take])
        if math.isfinite(delta_rho):
            rho_boot.append(delta_rho)
        ties = base_ties[take] ^ occurrence
        delta_utility = (
            utility_arrays(est[take], y[take], ties)["oracle_normalized_utility"]
            - utility_arrays(base[take], y[take], ties)[
                "oracle_normalized_utility"
            ]
        )
        if math.isfinite(delta_utility):
            utility_boot.append(delta_utility)
    if not rho_boot or not utility_boot:
        raise EvaluationFailure(
            f"paired bootstrap failed: {estimator} vs {baseline} for {outcome}"
        )
    return [
        {
            "estimator": estimator,
            "baseline": baseline,
            "outcome": outcome,
            "measure": "delta_spearman",
            "estimate": point_rho,
            "ci95_lower": float(np.quantile(rho_boot, 0.025)),
            "ci95_upper": float(np.quantile(rho_boot, 0.975)),
            "bootstrap_valid": len(rho_boot),
        },
        {
            "estimator": estimator,
            "baseline": baseline,
            "outcome": outcome,
            "measure": "delta_oracle_normalized_utility",
            "estimate": point_utility,
            "ci95_lower": float(np.quantile(utility_boot, 0.025)),
            "ci95_upper": float(np.quantile(utility_boot, 0.975)),
            "bootstrap_valid": len(utility_boot),
        },
    ]


def bootstrap_mean_delta(values: np.ndarray, label: str) -> dict[str, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
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


def set_prediction_key(adata: ad.AnnData) -> None:
    adata.obs["perturbation"] = adata.obs["pert_cond_names"].astype(str).to_numpy()


def validate_aligned(adatas: dict[str, ad.AnnData]) -> None:
    reference = adatas["gat"]
    if reference.shape != (N_CELLS, N_GENES):
        raise EvaluationFailure(f"reference shape changed: {reference.shape}")
    reference_obs = reference.obs_names.astype(str).tolist()
    reference_var = reference.var_names.astype(str).tolist()
    reference_pert = reference.obs["pert_cond_names"].astype(str).tolist()
    reference_batch = reference.obs["experimental_batches"].astype(str).tolist()
    for label, handle in adatas.items():
        if (
            handle.shape != reference.shape
            or handle.obs_names.astype(str).tolist() != reference_obs
            or handle.var_names.astype(str).tolist() != reference_var
            or handle.obs["pert_cond_names"].astype(str).tolist() != reference_pert
            or handle.obs["experimental_batches"].astype(str).tolist()
            != reference_batch
        ):
            raise EvaluationFailure(f"aligned matrix contract failed: {label}")
        finite_matrix(np.asarray(handle.X), label)


def load_predictions() -> tuple[
    ad.AnnData,
    ad.AnnData,
    dict[str, ad.AnnData],
    dict[str, ad.AnnData],
]:
    truth = ad.read_h5ad(TRUTH)
    control = ad.read_h5ad(CONTROL)
    members = {key: ad.read_h5ad(path) for key, path in MODEL_PATHS.items()}
    general = ad.read_h5ad(GENERAL)
    aligned = {**members, "truth": truth, "control": control, "general": general}
    validate_aligned(aligned)
    family_x = np.asarray(members["gat"].X, dtype=np.float64).copy()
    for key in ("exphormer", "exphormer_mg"):
        family_x += np.asarray(members[key].X, dtype=np.float64)
    family_x /= len(MODEL_KEYS)
    family = ad.AnnData(
        X=family_x,
        obs=members["gat"].obs.copy(),
        var=members["gat"].var.copy(),
    )
    finite_matrix(family.X, "family_centroid")
    predictions = {
        **members,
        "family_centroid": family,
        "general_baseline": general,
        "batch_matched_control": control,
    }
    for handle in (truth, control, *predictions.values()):
        set_prediction_key(handle)
    return truth, control, members, predictions


def build_scperteval_dataset(truth: ad.AnnData) -> ad.AnnData:
    source = ad.read_h5ad(ORIGINAL, backed="r")
    try:
        mask = source.obs["control"].astype(bool).to_numpy()
        raw_control = source[mask].to_memory()
    finally:
        source.file.close()
    if raw_control.n_obs != 10_691 or raw_control.var_names.tolist() != truth.var_names.tolist():
        raise EvaluationFailure("original control population changed")
    raw_control.obs["perturbation"] = "control"
    dataset = ad.concat(
        [truth, raw_control],
        axis=0,
        # AnnData 0.11 accepts inner/outer only.  Exact gene names and order
        # were checked immediately above, so inner cannot silently drop a gene.
        join="inner",
        merge="same",
        index_unique="::",
    )
    if dataset.n_obs != N_CELLS + 10_691 or dataset.n_vars != N_GENES:
        raise EvaluationFailure("scPertEval dataset assembly failed")
    return dataset


def run_scperteval(
    dataset: ad.AnnData,
    predictions: dict[str, ad.AnnData],
    task_counts: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sys.path.insert(0, str(SCPERTEVAL / "src"))
    import scperteval

    rows = []
    runtimes = []
    for stratum, min_cells in (
        ("primary_ge30", MIN_PRIMARY),
        ("sensitivity_10_29", MIN_SENSITIVITY),
    ):
        start = perf_counter()
        prepared = scperteval.prepare(
            dataset,
            list(PRIMARY_PROTOCOLS),
            subsample=SUBSAMPLE,
            seed=SEED,
            min_cells=min_cells,
            perturbation_key="perturbation",
            control_label="control",
            workers=WORKERS,
            name=f"E199_K562_{stratum}",
        )
        runtimes.append(
            {
                "stage": stratum,
                "predictor": "ALL",
                "endpoint": "prepare",
                "seconds": perf_counter() - start,
            }
        )
        for predictor, pred_adata in predictions.items():
            for endpoint in PRIMARY_PROTOCOLS:
                start = perf_counter()
                result = scperteval.score(
                    prepared,
                    endpoint,
                    pred_adata,
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
                    raw = float(record["score"])
                    oriented = raw if LOWER_IS_BETTER[endpoint] else 1.0 - raw
                    rows.append(
                        {
                            "stratum": stratum,
                            "task_id": condition,
                            "condition_label": label,
                            "gene": condition.split("+")[0],
                            "n_cells": n_cells,
                            "predictor": predictor,
                            "endpoint": endpoint,
                            "direction": (
                                "lower_better"
                                if LOWER_IS_BETTER[endpoint]
                                else "higher_better"
                            ),
                            "score": raw,
                            "oriented_error": oriented,
                        }
                    )
        del prepared
        gc.collect()
    frame = pd.DataFrame(rows)
    expected = (
        len(PREDICTOR_KEYS)
        * len(PRIMARY_PROTOCOLS)
        * (N_PRIMARY + N_SENSITIVITY)
    )
    if len(frame) != expected:
        raise EvaluationFailure(f"scPertEval row count changed: {len(frame)} != {expected}")
    return frame, pd.DataFrame(runtimes)


def task_certificate(
    truth: ad.AnnData,
    control: ad.AnnData,
    members: dict[str, ad.AnnData],
    predictions: dict[str, ad.AnnData],
    features: pd.DataFrame,
) -> pd.DataFrame:
    labels = truth.obs["pert_cond_names"].astype(str).to_numpy()
    condition_to_label = {
        condition_from_label(str(label)): str(label) for label in pd.unique(labels)
    }
    gene_pos = {str(gene): i for i, gene in enumerate(truth.var_names)}
    rows = []
    for feature in features.itertuples(index=False):
        condition = str(feature.task_id)
        label = condition_to_label[condition]
        indices = np.flatnonzero(labels == label)
        member_centroids = np.stack(
            [
                np.asarray(members[key].X[indices], dtype=np.float64).mean(0)
                for key in MODEL_KEYS
            ],
            axis=0,
        )
        true_centroid = np.asarray(truth.X[indices], dtype=np.float64).mean(0)
        control_centroid = np.asarray(control.X[indices], dtype=np.float64).mean(0)
        general_centroid = np.asarray(
            predictions["general_baseline"].X[indices], dtype=np.float64
        ).mean(0)
        family_centroid = member_centroids.mean(0)
        member_mse = np.mean(
            (member_centroids - true_centroid[None, :]) ** 2, axis=1
        )
        member_rmse = np.sqrt(member_mse)
        centroid_error = rmse(family_centroid, true_centroid)
        family_rms = float(np.sqrt(member_mse.mean()))
        family_worst = float(member_rmse.max())
        diversity_mse = float(
            np.mean((member_centroids - family_centroid[None, :]) ** 2)
        )
        diversity_lower = math.sqrt(max(diversity_mse, 0.0))
        recomputed_magnitude = rmse(family_centroid, control_centroid)
        recomputed_gap = rmse(family_centroid, general_centroid)
        feature_residual = max(
            abs(diversity_mse - float(feature.family_diversity)),
            abs(diversity_lower - float(feature.diversity_lower_bound)),
            abs(recomputed_magnitude - float(feature.predicted_magnitude)),
            abs(recomputed_gap - float(feature.model_baseline_gap)),
        )
        gene = str(feature.gene)
        gene_index = gene_pos.get(gene)
        if gene_index is None:
            observed_target_effect = float("nan")
            predicted_target_effect = float("nan")
            target_effect_error = float("nan")
            target_direction_correct = float("nan")
        else:
            observed_target_effect = float(
                true_centroid[gene_index] - control_centroid[gene_index]
            )
            predicted_target_effect = float(
                family_centroid[gene_index] - control_centroid[gene_index]
            )
            target_effect_error = predicted_target_effect - observed_target_effect
            target_direction_correct = float(
                np.sign(predicted_target_effect) == np.sign(observed_target_effect)
            )
        row = {
            "task_id": condition,
            "condition_label": label,
            "gene": gene,
            "n_cells": len(indices),
            "family_rms_error": family_rms,
            "family_worst_error": family_worst,
            "centroid_error": centroid_error,
            "diversity_lower_bound": diversity_lower,
            "rms_identity_residual": abs(
                family_rms**2 - (centroid_error**2 + diversity_lower**2)
            ),
            "family_rms_lower_violation": diversity_lower > family_rms + 1e-12,
            "family_worst_lower_violation": diversity_lower
            > family_worst + 1e-12,
            "pretruth_feature_max_abs_residual": feature_residual,
            "batch_matched_control_rmse": rmse(control_centroid, true_centroid),
            "general_baseline_rmse": rmse(general_centroid, true_centroid),
            "observed_target_effect": observed_target_effect,
            "predicted_target_effect": predicted_target_effect,
            "target_effect_error": target_effect_error,
            "target_direction_correct": target_direction_correct,
        }
        for index, key in enumerate(MODEL_KEYS):
            row[f"{key}_rmse"] = float(member_rmse[index])
        rows.append(row)
    frame = pd.DataFrame(rows)
    if (
        len(frame) != N_TASKS
        or frame.pretruth_feature_max_abs_residual.max() > FEATURE_TOL
    ):
        raise EvaluationFailure("sealed pretruth features failed recomputation")
    return frame


def run_txpert_endpoints(
    truth: ad.AnnData,
    control: ad.AnnData,
    predictions: dict[str, ad.AnnData],
) -> pd.DataFrame:
    sys.path.insert(0, str(TXPERT))
    import gspp.metrics as txmetrics

    original = ad.read_h5ad(ORIGINAL)
    txmetrics.CELL_TYPE_MEAN_CONTROL.clear()
    txmetrics.PERT_MEANS.clear()
    txmetrics.cache_perturbation_means(original, "K562")
    retrieval = txmetrics.RetrievalMetric(subset=True)
    labels = truth.obs["pert_cond_names"].astype(str).to_numpy()
    rows = []
    for label in pd.unique(labels):
        condition = condition_from_label(str(label))
        indices = np.flatnonzero(labels == label)
        truth_block = np.asarray(truth.X[indices])
        control_block = np.asarray(control.X[indices])
        for predictor, handle in predictions.items():
            prediction_block = np.asarray(handle.X[indices])
            predicted_delta = (prediction_block - control_block).mean(0)
            delta_is_constant = bool(np.ptp(predicted_delta) == 0)
            if delta_is_constant:
                # TxPert pearson_delta explicitly maps an undefined constant-input
                # Pearson value to zero.  RetrievalMetric has no analogous guard:
                # sorting NaN correlations yields a finite but meaningless rank.
                pearson_delta = 0.0
                fast_retrieval = float("nan")
            else:
                pearson_delta = txmetrics.pearson_delta(
                    prediction_block,
                    truth_block,
                    "K562",
                    control_block,
                    original,
                    condition,
                )
                # RetrievalMetric repeatedly recomputes the same predicted mean
                # inside its 100-reference loop.  Supplying the already-computed
                # delta as one row is algebraically identical and avoids that
                # redundant O(n_cells * n_references * n_genes) work.
                collapsed_prediction = predicted_delta[None, :]
                collapsed_control = np.zeros_like(collapsed_prediction)
                fast_retrieval = retrieval(
                    collapsed_prediction,
                    truth_block[:1],  # unused by TxPert RetrievalMetric
                    "K562",
                    collapsed_control,
                    original,
                    condition,
                )
            rows.extend(
                [
                    {
                        "task_id": condition,
                        "gene": condition.split("+")[0],
                        "n_cells": len(indices),
                        "predictor": predictor,
                        "endpoint": "pearson_delta_batch_matched",
                        "direction": "higher_better",
                        "score": float(pearson_delta),
                        "mathematically_defined": not delta_is_constant,
                        "score_status": (
                            "official_nan_to_zero_convention"
                            if delta_is_constant
                            else "defined"
                        ),
                    },
                    {
                        "task_id": condition,
                        "gene": condition.split("+")[0],
                        "n_cells": len(indices),
                        "predictor": predictor,
                        "endpoint": "fast_retrieval_official",
                        "direction": "higher_better",
                        "score": float(fast_retrieval),
                        "mathematically_defined": not delta_is_constant,
                        "score_status": (
                            "undefined_constant_predicted_delta"
                            if delta_is_constant
                            else "defined"
                        ),
                    },
                ]
            )
    del original
    txmetrics.CELL_TYPE_MEAN_CONTROL.clear()
    txmetrics.PERT_MEANS.clear()
    gc.collect()
    frame = pd.DataFrame(rows)
    expected = N_TASKS * len(PREDICTOR_KEYS) * 2
    expected_constant = frame.predictor.eq("batch_matched_control")
    observed_constant = ~frame.mathematically_defined
    retrieval = frame.endpoint.eq("fast_retrieval_official")
    valid_scores = np.isfinite(frame.score) | (retrieval & observed_constant)
    if (
        len(frame) != expected
        or not np.array_equal(observed_constant.to_numpy(), expected_constant.to_numpy())
        or not valid_scores.all()
        or frame.loc[retrieval & observed_constant, "score"].notna().any()
    ):
        raise EvaluationFailure("TxPert secondary endpoint contract failed")
    return frame


def scperteval_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["stratum", "predictor", "endpoint", "direction"],
            observed=True,
        )
        .score.agg(n="count", mean="mean", median="median", std="std")
        .reset_index()
    )


def baseline_comparisons(task_frame: pd.DataFrame) -> pd.DataFrame:
    primary = task_frame.loc[task_frame.n_cells.ge(MIN_PRIMARY)].copy()
    columns = {
        "gat": "gat_rmse",
        "exphormer": "exphormer_rmse",
        "exphormer_mg": "exphormer_mg_rmse",
        "family_centroid": "centroid_error",
        "family_member_rms": "family_rms_error",
        "general_baseline": "general_baseline_rmse",
        "batch_matched_control": "batch_matched_control_rmse",
    }
    rows = []
    for predictor, column in columns.items():
        for baseline, baseline_column in (
            ("general_baseline", "general_baseline_rmse"),
            ("batch_matched_control", "batch_matched_control_rmse"),
        ):
            delta = primary[column].to_numpy(float) - primary[baseline_column].to_numpy(float)
            rows.append(
                {
                    "predictor": predictor,
                    "baseline": baseline,
                    "n_tasks": len(primary),
                    "predictor_mean_rmse": float(primary[column].mean()),
                    "baseline_mean_rmse": float(primary[baseline_column].mean()),
                    "task_win_rate": float(np.mean(delta < 0)),
                    **bootstrap_mean_delta(delta, f"{predictor}::{baseline}"),
                }
            )
    return pd.DataFrame(rows)


def endpoint_baseline_comparisons(scpert: pd.DataFrame) -> pd.DataFrame:
    primary = scpert.loc[scpert.stratum.eq("primary_ge30")]
    rows = []
    for endpoint in PRIMARY_PROTOCOLS:
        block = primary.loc[primary.endpoint.eq(endpoint)]
        pivot = block.pivot(index="task_id", columns="predictor", values="oriented_error")
        for baseline in ("general_baseline", "batch_matched_control"):
            delta = (
                pivot["family_centroid"].to_numpy(float)
                - pivot[baseline].to_numpy(float)
            )
            finite = np.isfinite(delta)
            rows.append(
                {
                    "endpoint": endpoint,
                    "baseline": baseline,
                    "n_tasks": int(finite.sum()),
                    "task_win_rate": float(np.mean(delta[finite] < 0)),
                    **bootstrap_mean_delta(
                        delta[finite], f"endpoint::{endpoint}::{baseline}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def risk_analysis(
    features: pd.DataFrame,
    certificate: pd.DataFrame,
    scpert: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_frame = features.rename(columns={"n_prediction_cells": "n_cells"})
    frame = feature_frame.merge(
        certificate,
        on=["task_id", "condition_label", "gene", "n_cells"],
        suffixes=("_pretruth", ""),
        validate="one_to_one",
    )
    if (
        np.max(
            np.abs(
                frame.diversity_lower_bound_pretruth.to_numpy(float)
                - frame.diversity_lower_bound.to_numpy(float)
            )
        )
        > FEATURE_TOL
    ):
        raise EvaluationFailure("risk table changed the sealed diversity feature")
    frame = frame.drop(columns=["diversity_lower_bound"]).rename(
        columns={"diversity_lower_bound_pretruth": "diversity_lower_bound"}
    )
    frame["negative_string_train_neighbor_count"] = -frame[
        "string_train_neighbor_count"
    ].astype(float)
    frame["negative_go_train_neighbor_count"] = -frame[
        "go_train_neighbor_count"
    ].astype(float)
    frame["graph_isolated_risk"] = frame.graph_isolated.astype(float)
    primary = frame.loc[frame.n_cells.ge(MIN_PRIMARY)].copy()
    associations = []
    utilities = []
    for predictor in RISK_FEATURES:
        associations.append(
            {
                "predictor": predictor,
                "outcome": "family_rms_error",
                "n_tasks": len(primary),
                **bootstrap_spearman(primary, predictor, "family_rms_error"),
            }
        )
        utilities.append(
            {
                "predictor": predictor,
                "outcome": "family_rms_error",
                **bootstrap_utility(primary, predictor, "family_rms_error"),
            }
        )
    increments = pd.DataFrame(
        paired_increment(
            primary,
            "diversity_lower_bound",
            "predicted_magnitude",
            "family_rms_error",
        )
    )
    family_endpoints = scpert.loc[
        scpert.stratum.eq("primary_ge30")
        & scpert.predictor.eq("family_centroid")
    ][["task_id", "endpoint", "oriented_error"]]
    endpoint_rows = []
    for endpoint, endpoint_frame in family_endpoints.groupby("endpoint", observed=True):
        merged = primary.merge(endpoint_frame, on="task_id", validate="one_to_one")
        for predictor in RISK_FEATURES:
            endpoint_rows.append(
                {
                    "endpoint": endpoint,
                    "predictor": predictor,
                    "n_tasks": int(
                        (
                            np.isfinite(merged[predictor])
                            & np.isfinite(merged.oriented_error)
                        ).sum()
                    ),
                    "spearman": spearman(
                        merged[predictor].to_numpy(float),
                        merged.oriented_error.to_numpy(float),
                    ),
                }
            )
    return (
        pd.DataFrame(associations),
        pd.DataFrame(utilities),
        increments,
        pd.DataFrame(endpoint_rows),
    )


def model_difficulty(certificate: pd.DataFrame) -> pd.DataFrame:
    primary = certificate.loc[certificate.n_cells.ge(MIN_PRIMARY)].copy()
    rows = []
    for left, right in combinations(MODEL_KEYS, 2):
        left_col = f"{left}_rmse"
        right_col = f"{right}_rmse"
        rows.append(
            {
                "comparison": f"{left}_vs_{right}",
                "quantity": "task_error_rank_agreement",
                "n_tasks": len(primary),
                **bootstrap_spearman(primary, left_col, right_col),
            }
        )
    for model in MODEL_KEYS:
        rows.append(
            {
                "comparison": f"diversity_vs_{model}",
                "quantity": "family_risk_to_member_error",
                "n_tasks": len(primary),
                **bootstrap_spearman(
                    primary, "diversity_lower_bound", f"{model}_rmse"
                ),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(map(str, frame.columns))

    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "NA" if not math.isfinite(float(value)) else f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(render, row)) + " |")
    return "\n".join(lines)


def make_figure(
    task_frame: pd.DataFrame,
    utilities: pd.DataFrame,
    endpoint_comparisons: pd.DataFrame,
    path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    primary = task_frame.loc[task_frame.n_cells.ge(MIN_PRIMARY)]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0))
    predictors = [
        ("GAT", "gat_rmse", "#4C78A8"),
        ("Exphormer", "exphormer_rmse", "#72A0C1"),
        ("Exphormer-MG", "exphormer_mg_rmse", "#54A24B"),
        ("Family centroid", "centroid_error", "#E45756"),
        ("General baseline", "general_baseline_rmse", "#B279A2"),
        ("Matched control", "batch_matched_control_rmse", "#9D9D9D"),
    ]
    axes[0, 0].bar(
        range(len(predictors)),
        [primary[column].mean() for _, column, _ in predictors],
        color=[color for _, _, color in predictors],
        width=0.72,
    )
    axes[0, 0].set_xticks(
        range(len(predictors)),
        [label for label, _, _ in predictors],
        rotation=28,
        ha="right",
    )
    axes[0, 0].set_ylabel("Mean centroid RMSE")
    axes[0, 0].set_title("A  Public-model performance")

    axes[0, 1].scatter(
        primary.diversity_lower_bound,
        primary.family_rms_error,
        s=16,
        alpha=0.58,
        color="#4C78A8",
        edgecolors="none",
    )
    axes[0, 1].plot(
        [0, primary.diversity_lower_bound.max()],
        [0, primary.diversity_lower_bound.max()],
        color="#777777",
        linestyle="--",
        linewidth=0.9,
    )
    axes[0, 1].set_xlabel("Diversity lower bound")
    axes[0, 1].set_ylabel("Family RMS error")
    axes[0, 1].set_title("B  Certificate and empirical routing")

    order = [
        "diversity_lower_bound",
        "predicted_magnitude",
        "model_baseline_gap",
        "negative_string_train_neighbor_count",
        "negative_go_train_neighbor_count",
        "graph_isolated_risk",
    ]
    utility = utilities.set_index("predictor").loc[order]
    values = utility.oracle_normalized_utility.to_numpy(float)
    lower = np.maximum(
        values - utility.utility_ci95_lower.to_numpy(float), 0.0
    )
    upper = np.maximum(
        utility.utility_ci95_upper.to_numpy(float) - values, 0.0
    )
    axes[1, 0].bar(
        range(len(order)),
        values,
        color=["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#9D9D9D"],
        width=0.7,
    )
    axes[1, 0].errorbar(
        range(len(order)),
        values,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#333333",
        capsize=2,
        linewidth=0.8,
    )
    axes[1, 0].axhline(0, color="#555555", linewidth=0.8)
    axes[1, 0].set_xticks(
        range(len(order)),
        ["Diversity", "Magnitude", "Baseline gap", "STRING support", "GO support", "Graph isolated"],
        rotation=28,
        ha="right",
    )
    axes[1, 0].set_ylabel("Oracle-normalized utility")
    axes[1, 0].set_title("C  Fixed 20% review budget")

    forest = endpoint_comparisons.loc[
        endpoint_comparisons.baseline.eq("general_baseline")
    ].set_index("endpoint").loc[list(PRIMARY_PROTOCOLS)]
    y = np.arange(len(forest))
    axes[1, 1].errorbar(
        forest.mean_delta,
        y,
        xerr=np.vstack(
            [
                np.maximum(forest.mean_delta - forest.ci95_lower, 0.0),
                np.maximum(forest.ci95_upper - forest.mean_delta, 0.0),
            ]
        ),
        fmt="o",
        color="#E45756",
        ecolor="#555555",
        capsize=2,
        markersize=4,
    )
    axes[1, 1].axvline(0, color="#555555", linestyle="--", linewidth=0.8)
    axes[1, 1].set_yticks(y, [p.replace("_pca_k=50", " PCA50") for p in PRIMARY_PROTOCOLS])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Family minus general-baseline error")
    axes[1, 1].set_title("D  Five preregistered endpoints")

    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.55, zorder=0)
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
    features = pd.read_csv(FEATURES, keep_default_na=False)
    if len(features) != N_TASKS or features.task_id.nunique() != N_TASKS:
        raise EvaluationFailure("pretruth feature table changed")

    truth, control, members, predictions = load_predictions()
    task_counts = (
        features.set_index("task_id").n_prediction_cells.astype(int).to_dict()
    )
    certificate = task_certificate(
        truth, control, members, predictions, features
    )
    dataset = build_scperteval_dataset(truth)
    scpert, runtimes = run_scperteval(dataset, predictions, task_counts)
    scpert_summary = scperteval_summary(scpert)
    del dataset
    gc.collect()

    txpert_endpoints = run_txpert_endpoints(truth, control, predictions)
    task_frame = features.merge(
        certificate,
        on=["task_id", "condition_label", "gene"],
        suffixes=("_pretruth", ""),
        validate="one_to_one",
    )
    if not np.array_equal(
        task_frame.n_prediction_cells.to_numpy(int),
        task_frame.n_cells.to_numpy(int),
    ):
        raise EvaluationFailure("task cell counts changed after outcome opening")
    task_frame = task_frame.drop(columns=["n_prediction_cells"])

    baseline = baseline_comparisons(task_frame)
    endpoint_baseline = endpoint_baseline_comparisons(scpert)
    associations, utilities, increments, endpoint_risk = risk_analysis(
        features, certificate, scpert
    )
    difficulty = model_difficulty(certificate)

    primary = task_frame.loc[task_frame.n_cells.ge(MIN_PRIMARY)]
    diversity_assoc = associations.loc[
        associations.predictor.eq("diversity_lower_bound")
    ].iloc[0]
    diversity_utility = utilities.loc[
        utilities.predictor.eq("diversity_lower_bound")
    ].iloc[0]
    delta_rho = increments.loc[increments.measure.eq("delta_spearman")].iloc[0]
    delta_utility = increments.loc[
        increments.measure.eq("delta_oracle_normalized_utility")
    ].iloc[0]
    certificate_gate = bool(
        task_frame.rms_identity_residual.max() <= IDENTITY_TOL
        and not task_frame.family_rms_lower_violation.any()
        and not task_frame.family_worst_lower_violation.any()
        and task_frame.pretruth_feature_max_abs_residual.max() <= FEATURE_TOL
    )
    routing_gate = bool(
        diversity_assoc.ci95_lower > 0
        and diversity_utility.utility_ci95_lower > 0
    )
    incremental_gate = bool(
        delta_rho.ci95_lower > 0 or delta_utility.ci95_lower > 0
    )
    family_general = baseline.loc[
        baseline.predictor.eq("family_centroid")
        & baseline.baseline.eq("general_baseline")
    ].iloc[0]
    family_beats_general = bool(family_general.ci95_upper < 0)

    gates = pd.DataFrame(
        [
            {
                "gate": "certificate_integrity",
                "passed": certificate_gate,
                "observed": task_frame.rms_identity_residual.max(),
                "criterion": f"identity residual <= {IDENTITY_TOL}; zero lower-bound violations",
            },
            {
                "gate": "empirical_routing",
                "passed": routing_gate,
                "observed": f"rho_lower={diversity_assoc.ci95_lower:.6g};utility_lower={diversity_utility.utility_ci95_lower:.6g}",
                "criterion": "both 95% CI lower bounds > 0",
            },
            {
                "gate": "incremental_vs_magnitude",
                "passed": incremental_gate,
                "observed": f"delta_rho_lower={delta_rho.ci95_lower:.6g};delta_utility_lower={delta_utility.ci95_lower:.6g}",
                "criterion": "either paired 95% CI lower bound > 0",
            },
            {
                "gate": "family_centroid_vs_general_mse",
                "passed": family_beats_general,
                "observed": f"mean_delta={family_general.mean_delta:.6g};upper={family_general.ci95_upper:.6g}",
                "criterion": "paired RMSE delta 95% CI upper < 0",
            },
        ]
    )

    for directory in (TABLES, REPORTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    write_csv(TABLES / "E199_INPUT_HASHES.csv", input_hashes)
    write_csv(TABLES / "E199_TASK_CERTIFICATE.csv", task_frame)
    write_csv(TABLES / "E199_SCPERTEVAL_TASK_METRICS.csv", scpert)
    write_csv(TABLES / "E199_SCPERTEVAL_SUMMARY.csv", scpert_summary)
    write_csv(TABLES / "E199_TXPERT_SECONDARY_ENDPOINTS.csv", txpert_endpoints)
    write_csv(TABLES / "E199_BASELINE_COMPARISONS.csv", baseline)
    write_csv(TABLES / "E199_ENDPOINT_BASELINE_COMPARISONS.csv", endpoint_baseline)
    write_csv(TABLES / "E199_RISK_ASSOCIATIONS.csv", associations)
    write_csv(TABLES / "E199_REVIEW_UTILITY.csv", utilities)
    write_csv(TABLES / "E199_INCREMENTAL_TESTS.csv", increments)
    write_csv(TABLES / "E199_ENDPOINT_RISK_ASSOCIATIONS.csv", endpoint_risk)
    write_csv(TABLES / "E199_MODEL_DIFFICULTY.csv", difficulty)
    write_csv(TABLES / "E199_RUNTIME.csv", runtimes)
    write_csv(TABLES / "E199_FORMAL_GATES.csv", gates)
    make_figure(
        task_frame,
        utilities,
        endpoint_baseline,
        FIGURES / "E199_txpert_public_k562.png",
    )

    tx_summary = (
        txpert_endpoints.groupby(["predictor", "endpoint"], observed=True)
        .score.agg(n="count", mean="mean", median="median")
        .reset_index()
    )
    write_csv(TABLES / "E199_TXPERT_SECONDARY_SUMMARY.csv", tx_summary)
    primary_scpert = scpert_summary.loc[
        scpert_summary.stratum.eq("primary_ge30")
    ][["predictor", "endpoint", "n", "mean", "median"]]
    report = [
        "# E199 TxPert 公开 K562 未见扰动审计",
        "",
        f"确定性证书：**{'PASS' if certificate_gate else 'FAIL'}**。",
        f"经验路由：**{'ENABLED' if routing_gate else 'ABSTAIN'}**。",
        f"相对 predicted magnitude 的新增价值：**{'SUPPORTED' if incremental_gate else 'NOT SUPPORTED'}**。",
        "",
        "## 五个冻结端点（主分析）",
        "",
        markdown_table(primary_scpert),
        "",
        "## 简单基线对照（centroid RMSE）",
        "",
        markdown_table(
            baseline.loc[
                baseline.baseline.eq("general_baseline"),
                [
                    "predictor",
                    "predictor_mean_rmse",
                    "baseline_mean_rmse",
                    "task_win_rate",
                    "mean_delta",
                    "ci95_lower",
                    "ci95_upper",
                ],
            ]
        ),
        "",
        "## SafeConf 风险量",
        "",
        markdown_table(
            associations[
                ["predictor", "spearman", "ci95_lower", "ci95_upper"]
            ]
        ),
        "",
        markdown_table(
            utilities[
                [
                    "predictor",
                    "high_error_capture",
                    "error_lift",
                    "oracle_normalized_utility",
                    "utility_ci95_lower",
                    "utility_ci95_upper",
                ]
            ]
        ),
        "",
        "## 边界",
        "",
        "E199 只是 K562 内未见单基因扰动。它能回答模型家族是否显示任务级难度，"
        "不能代替整个细胞背景留出或跨数据集迁移。公开 Exphormer-MG 只使用 STRING+GO，"
        "不包含 TxPert 论文最强配置中未公开的 PxMap/TxMap。",
        "",
        f"主分析 {len(primary)} 个任务；目标基因效应方向命中率为 "
        f"{primary.target_direction_correct.mean():.3f}，该项是预先写入 formal runner 的失败模式诊断。",
        "",
    ]
    write_text(REPORTS / "E199_REPORT.md", "\n".join(report))

    status = {
        "experiment": "E199_txpert_public_k562",
        "stage": "FORMAL_EVALUATION",
        "generated_at": now(),
        "status": "PASS" if certificate_gate else "FAIL",
        "git_head": head,
        "n_tasks": N_TASKS,
        "n_primary_tasks": N_PRIMARY,
        "n_sensitivity_tasks": N_SENSITIVITY,
        "max_rms_identity_residual": float(
            task_frame.rms_identity_residual.max()
        ),
        "family_rms_lower_violations": int(
            task_frame.family_rms_lower_violation.sum()
        ),
        "family_worst_lower_violations": int(
            task_frame.family_worst_lower_violation.sum()
        ),
        "pretruth_feature_max_abs_residual": float(
            task_frame.pretruth_feature_max_abs_residual.max()
        ),
        "certificate_gate_pass": certificate_gate,
        "routing_activation_gate_pass": routing_gate,
        "routing_status": "ENABLED" if routing_gate else "ABSTAIN",
        "incremental_vs_magnitude_gate_pass": incremental_gate,
        "incremental_claim_status": (
            "SUPPORTED" if incremental_gate else "NOT_SUPPORTED"
        ),
        "family_centroid_beats_general_mse_gate": family_beats_general,
        "cross_context_answered": False,
        "cross_dataset_transfer_answered": False,
        "public_graphs_only": True,
        "retrospective_public_data": True,
        "performance_is_not_a_certificate_pass_gate": True,
    }
    write_json(FINAL / "E199_FINAL_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
