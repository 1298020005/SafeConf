#!/usr/bin/env python3
"""One-shot authorized E216 evaluation after the pretruth seal commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy.stats import rankdata


TASK_KEYS = ("condition", "cell_type", "treatment")
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 216_202_609


class EvaluationFailure(RuntimeError):
    """The authorization, truth, or registered evaluation contract failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise EvaluationFailure(f"invalid JSON: {path}") from exc


def verify_authorization(repo: Path, authorization_path: Path, pretruth_dir: Path) -> dict:
    authorization = load_json(authorization_path)
    required = {
        "experiment": "E216_jiang24_resource_bounded_confirmation",
        "stage": "TEST_TRUTH_RELEASE_AUTHORIZATION",
        "status": "AUTHORIZED",
        "allow_test_perturbed_expression": True,
        "risk_sealed_and_pushed_before_authorization": True,
    }
    if any(authorization.get(key) != value for key, value in required.items()):
        raise EvaluationFailure("test-truth authorization fields failed")
    seal_commit = str(authorization.get("risk_seal_commit", ""))
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    if len(seal_commit) < 7 or seal_commit == current_head:
        raise EvaluationFailure("authorization must follow a distinct seal commit")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", seal_commit, current_head], cwd=repo
    ).returncode == 0
    if not ancestor:
        raise EvaluationFailure("risk seal is not an ancestor of evaluation HEAD")

    expected_files = {
        "E216_PRETRUTH_STATUS.json",
        "E216_PRETRUTH_TASK_SCORES.csv",
        "E216_PRETRUTH_STATE_STANDARDIZATION.csv",
        "E216_CERTIFICATE_THRESHOLDS.csv",
        "E216_ARCHITECTURE_BALANCED_FAMILY.npz",
        "E216_PRETRUTH_MANIFEST.csv",
    }
    records = authorization.get("pretruth_files", {})
    if set(records) != expected_files:
        raise EvaluationFailure("authorization does not identify the complete pretruth release")
    for name in expected_files:
        path = pretruth_dir / name
        if not path.is_file() or sha256(path) != records[name]:
            raise EvaluationFailure(f"pretruth hash changed: {name}")

    sealed_repo_files = authorization.get("sealed_repo_files", {})
    if len(sealed_repo_files) < 4:
        raise EvaluationFailure("authorization lacks committed risk-table evidence")
    for relative_path, expected_hash in sealed_repo_files.items():
        result = subprocess.run(
            ["git", "show", f"{seal_commit}:{relative_path}"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        if bytes_sha256(result.stdout) != expected_hash:
            raise EvaluationFailure(f"sealed Git artifact changed: {relative_path}")
    return authorization


def read_selected_csr_blocks(
    h5ad_path: Path,
    positions: list[int],
    gene_positions: np.ndarray,
    *,
    n_total_genes: int,
    max_rows_per_block: int = 4096,
) -> csr_matrix:
    """Read selected CSR rows in consecutive blocks on a fixed gene panel."""

    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise EvaluationFailure("truth positions must be sorted and unique")
    blocks: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1 or position - start >= max_rows_per_block:
            blocks.append((start, previous + 1))
            start = position
        previous = position
    blocks.append((start, previous + 1))

    result = []
    with h5py.File(h5ad_path, "r") as handle:
        matrix = handle["X"]
        encoding = matrix.attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        if encoding != "csr_matrix":
            raise EvaluationFailure("Jiang24 X is no longer CSR")
        indptr_dataset = matrix["indptr"]
        for first, stop_row in blocks:
            local_indptr = np.asarray(indptr_dataset[first:stop_row + 1], dtype=np.int64)
            data_start, data_stop = int(local_indptr[0]), int(local_indptr[-1])
            local = csr_matrix(
                (
                    np.asarray(matrix["data"][data_start:data_stop]),
                    np.asarray(matrix["indices"][data_start:data_stop], dtype=np.int64),
                    local_indptr - data_start,
                ),
                shape=(stop_row - first, n_total_genes),
            )
            result.append(local[:, gene_positions])
    return vstack(result, format="csr")


def truth_centroids(
    h5ad_path: Path,
    resource_split_path: Path,
    tasks: pd.DataFrame,
    hvg: list[str],
    perturbench_repo: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    sys.path.insert(0, str(perturbench_repo / "src"))
    from perturbench.data.utils import load_dataframe_from_h5

    obs = load_dataframe_from_h5(
        str(h5ad_path), "obs", ["condition", "cell_type", "treatment"]
    )
    var = load_dataframe_from_h5(str(h5ad_path), "var", [])
    split = pd.read_csv(
        resource_split_path,
        header=None,
        names=["cell_barcode", "resource_split"],
        dtype=str,
    )
    if not split.cell_barcode.astype(str).equals(pd.Series(obs.index.astype(str))):
        raise EvaluationFailure("resource split order differs from Jiang24")
    task_index = pd.MultiIndex.from_frame(tasks.loc[:, TASK_KEYS].astype(str))
    obs_index = pd.MultiIndex.from_frame(obs.loc[:, TASK_KEYS].astype(str))
    selected = split.resource_split.eq("test").to_numpy() & obs_index.isin(task_index)
    positions = np.flatnonzero(selected).astype(int).tolist()
    selected_obs = obs.iloc[positions].loc[:, TASK_KEYS].astype(str).reset_index(drop=True)
    observed_counts = selected_obs.value_counts(list(TASK_KEYS)).rename("observed_n_cells")
    expected_counts = tasks.set_index(list(TASK_KEYS)).n_cells.astype(int)
    if not observed_counts.sort_index().equals(expected_counts.sort_index()):
        raise EvaluationFailure("test truth membership differs from the frozen 224 tasks")
    gene_names = var.index.astype(str)
    gene_positions = gene_names.get_indexer(hvg)
    if np.any(gene_positions < 0):
        raise EvaluationFailure("registered HVG axis is absent from Jiang24")
    expression = read_selected_csr_blocks(
        h5ad_path,
        positions,
        gene_positions,
        n_total_genes=len(gene_names),
    )
    selected_obs["_row"] = np.arange(len(selected_obs), dtype=int)
    centroids = {}
    for key, block in selected_obs.groupby(list(TASK_KEYS), observed=True, sort=True):
        take = block._row.to_numpy(dtype=int)
        centroids[tuple(str(value) for value in key)] = np.asarray(
            expression[take].mean(axis=0)
        ).ravel()
    ordered_keys = list(tasks.loc[:, TASK_KEYS].itertuples(index=False, name=None))
    truth = np.stack([centroids[key] for key in ordered_keys]).astype(np.float32)
    return truth, selected_obs.drop(columns="_row")


def rmse(left: np.ndarray, right: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.mean(np.square(np.asarray(left, float) - np.asarray(right, float)), axis=axis))


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rankdata(np.asarray(left, float), method="average")
    right_rank = rankdata(np.asarray(right, float), method="average")
    if np.std(left_rank) <= 0 or np.std(right_rank) <= 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def macro_state_spearman(frame: pd.DataFrame, score: str, outcome: str) -> float:
    values = [
        spearman(block[score].to_numpy(float), block[outcome].to_numpy(float))
        for _, block in frame.groupby(["cell_type", "treatment"], sort=True)
    ]
    if len(values) != 12 or not np.isfinite(values).all():
        raise EvaluationFailure("macro-state Spearman is not evaluable")
    return float(np.mean(values))


def stable_ties(task_ids: np.ndarray, occurrences: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            int(hashlib.sha256(f"E216\0{task}\0{int(occ)}".encode()).hexdigest()[:16], 16)
            for task, occ in zip(task_ids.astype(str), occurrences, strict=True)
        ],
        dtype=np.uint64,
    )


def review_metrics(
    score: np.ndarray,
    outcome: np.ndarray,
    task_ids: np.ndarray,
    occurrences: np.ndarray | None = None,
) -> dict[str, float]:
    score, outcome = np.asarray(score, float), np.asarray(outcome, float)
    if occurrences is None:
        occurrences = np.zeros(len(score), dtype=int)
    n_select = int(math.ceil(0.20 * len(score)))
    ties = stable_ties(np.asarray(task_ids), np.asarray(occurrences, int))
    selected = np.lexsort((ties, -score))[:n_select]
    oracle = np.lexsort((ties, -outcome))[:n_select]
    selected_mean = float(outcome[selected].mean())
    overall_mean = float(outcome.mean())
    oracle_mean = float(outcome[oracle].mean())
    denominator = oracle_mean - overall_mean
    return {
        "n_selected": n_select,
        "high_error_capture": len(set(selected) & set(oracle)) / n_select,
        "selected_mean_error": selected_mean,
        "overall_mean_error": overall_mean,
        "oracle_mean_error": oracle_mean,
        "oracle_normalized_utility": (
            (selected_mean - overall_mean) / denominator if denominator > 1e-15 else float("nan")
        ),
    }


def bootstrap_increments(frame: pd.DataFrame) -> pd.DataFrame:
    strata = []
    for treatment, treatment_frame in frame.groupby("treatment", sort=True):
        condition_values = treatment_frame.condition.astype(str).to_numpy()
        treatment_indices = treatment_frame.index.to_numpy(dtype=int)
        conditions = sorted(np.unique(condition_values))
        members = [
            treatment_indices[condition_values == condition]
            for condition in conditions
        ]
        strata.append((str(treatment), members))
    if len(strata) != 3 or any(len(members) < 2 for _, members in strata):
        raise EvaluationFailure("registered bootstrap strata changed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    attempts = 0
    maximum_attempts = N_BOOTSTRAP * 10
    while len(rows) < N_BOOTSTRAP:
        attempts += 1
        if attempts > maximum_attempts:
            raise EvaluationFailure("unable to obtain the registered evaluable bootstrap draws")
        sampled = []
        for _, members in strata:
            chosen = rng.integers(0, len(members), len(members))
            sampled.extend(members[index] for index in chosen)
        indices = np.concatenate(sampled)
        block = frame.iloc[indices].copy()
        block["occurrence"] = block.groupby("task_id", sort=False).cumcount()
        combined_review = review_metrics(
            block.safeconf_m_4to1.to_numpy(float),
            block.latent_centroid_error.to_numpy(float),
            block.task_id.to_numpy(),
            block.occurrence.to_numpy(int),
        )
        magnitude_review = review_metrics(
            block.predicted_magnitude.to_numpy(float),
            block.latent_centroid_error.to_numpy(float),
            block.task_id.to_numpy(),
            block.occurrence.to_numpy(int),
        )
        try:
            delta_spearman = macro_state_spearman(
                block, "safeconf_m_4to1", "latent_centroid_error"
            ) - macro_state_spearman(
                block, "predicted_magnitude", "latent_centroid_error"
            )
        except EvaluationFailure:
            continue
        delta_utility = (
            combined_review["oracle_normalized_utility"]
            - magnitude_review["oracle_normalized_utility"]
        )
        if not np.isfinite([delta_spearman, delta_utility]).all():
            continue
        rows.append(
            {
                "draw": len(rows),
                "attempt": attempts - 1,
                "delta_macro_spearman": delta_spearman,
                "delta_utility_20": delta_utility,
            }
        )
    return pd.DataFrame(rows)


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "estimate": float(np.mean(values)),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--pretruth-dir", type=Path, required=True)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--resource-split", type=Path, required=True)
    parser.add_argument("--hvg", type=Path, required=True)
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise EvaluationFailure(f"refusing to overwrite: {output_dir}")
    authorization = verify_authorization(
        repo, args.authorization.resolve(), args.pretruth_dir.resolve()
    )
    pretruth = args.pretruth_dir.resolve()
    pretruth_status = load_json(pretruth / "E216_PRETRUTH_STATUS.json")
    if (
        pretruth_status.get("status") not in {"PRETRUTH_COMPLETE_UNPUSHED", "SEALED"}
        or pretruth_status.get("test_perturbed_expression_rows_read") != 0
    ):
        raise EvaluationFailure("pretruth status is not releasable")
    scores = pd.read_csv(pretruth / "E216_PRETRUTH_TASK_SCORES.csv")
    scores = scores.sort_values(list(TASK_KEYS), kind="stable").reset_index(drop=True)
    tasks = pd.read_csv(args.tasks.resolve()).sort_values(list(TASK_KEYS), kind="stable").reset_index(drop=True)
    if not scores.loc[:, TASK_KEYS].astype(str).equals(tasks.loc[:, TASK_KEYS].astype(str)):
        raise EvaluationFailure("score and truth task identities differ")
    hvg = pd.read_csv(args.hvg.resolve(), header=None).iloc[:, 0].astype(str).tolist()
    family = np.load(pretruth / "E216_ARCHITECTURE_BALANCED_FAMILY.npz", allow_pickle=False)
    members = np.asarray(family["member_predictions"], dtype=np.float64)
    weights = np.asarray(family["member_weight"], dtype=np.float64)
    family_centroid = np.asarray(family["family_centroid"], dtype=np.float64)
    lower_bound = np.asarray(family["family_lower_bound"], dtype=np.float64)
    if members.shape != (5, 224, 4000) or not np.isclose(weights.sum(), 1):
        raise EvaluationFailure("pretruth family shape changed")
    truth, selected_obs = truth_centroids(
        args.h5ad.resolve(),
        args.resource_split.resolve(),
        tasks,
        hvg,
        args.perturbench_repo.resolve(),
    )
    latent_centroid = members[:4].mean(axis=0)
    linear_prediction = members[4]

    # Control centroids used by the predictions are recovered from magnitude:
    # the actual vectors remain in each member prediction release.  Authorization
    # records the seed-1 control file, loaded here by its explicit path.
    control_path = Path(authorization["control_centroid_file"]).resolve()
    if sha256(control_path) != authorization["control_centroid_sha256"]:
        raise EvaluationFailure("authorized control centroid changed")
    control_payload = np.load(control_path, allow_pickle=False)
    control_map = {
        (str(cell), str(treatment)): np.asarray(vector, dtype=np.float64)
        for cell, treatment, vector in zip(
            control_payload["cell_type"],
            control_payload["treatment"],
            control_payload["control_centroids"],
            strict=True,
        )
    }
    task_controls = np.stack(
        [control_map[(str(row.cell_type), str(row.treatment))] for row in tasks.itertuples()]
    )

    evaluated = scores.copy()
    evaluated["task_id"] = evaluated.loc[:, TASK_KEYS].astype(str).agg("|".join, axis=1)
    evaluated["latent_centroid_error"] = rmse(latent_centroid, truth)
    evaluated["linear_error"] = rmse(linear_prediction, truth)
    evaluated["no_change_error"] = rmse(task_controls, truth)
    member_error_squared = np.mean(np.square(members - truth[None, :, :]), axis=2)
    evaluated["registered_family_rms_error"] = np.sqrt(
        np.einsum("m,mt->t", weights, member_error_squared)
    )
    evaluated["registered_family_centroid_error"] = rmse(family_centroid, truth)
    evaluated["family_identity_residual"] = (
        np.square(evaluated.registered_family_rms_error)
        - np.square(evaluated.registered_family_centroid_error)
        - np.square(lower_bound)
    )
    tolerance = np.maximum(
        1e-12,
        1e-10
        * np.maximum(
            np.square(evaluated.registered_family_rms_error),
            np.square(evaluated.registered_family_centroid_error) + np.square(lower_bound),
        ),
    )
    evaluated["family_identity_tolerance"] = tolerance
    evaluated["family_lower_tightness"] = lower_bound / evaluated.registered_family_rms_error

    state_rows = []
    for state, block in evaluated.groupby(["cell_type", "treatment"], sort=True):
        state_rows.append(
            {
                "cell_type": state[0],
                "treatment": state[1],
                "n_tasks": len(block),
                "safeconf_m_spearman": spearman(block.safeconf_m_4to1, block.latent_centroid_error),
                "magnitude_spearman": spearman(block.predicted_magnitude, block.latent_centroid_error),
            }
        )
    states = pd.DataFrame(state_rows)
    states["delta_spearman"] = states.safeconf_m_spearman - states.magnitude_spearman
    observed_delta = float(states.delta_spearman.mean())
    combined_review = review_metrics(
        evaluated.safeconf_m_4to1.to_numpy(float),
        evaluated.latent_centroid_error.to_numpy(float),
        evaluated.task_id.to_numpy(),
    )
    magnitude_review = review_metrics(
        evaluated.predicted_magnitude.to_numpy(float),
        evaluated.latent_centroid_error.to_numpy(float),
        evaluated.task_id.to_numpy(),
    )
    observed_utility_delta = (
        combined_review["oracle_normalized_utility"] - magnitude_review["oracle_normalized_utility"]
    )
    observed_capture_delta = (
        combined_review["high_error_capture"] - magnitude_review["high_error_capture"]
    )
    bootstrap = bootstrap_increments(evaluated)
    spearman_interval = interval(bootstrap.delta_macro_spearman.to_numpy(float))
    utility_interval = interval(bootstrap.delta_utility_20.to_numpy(float))

    thresholds = pd.read_csv(pretruth / "E216_CERTIFICATE_THRESHOLDS.csv")
    certificate_rows = []
    family_error = evaluated.registered_family_rms_error.to_numpy(float)
    for row in thresholds.itertuples(index=False):
        certified = lower_bound > float(row.tau)
        true_high = family_error > float(row.tau)
        certificate_rows.append(
            {
                "quantile": float(row.quantile),
                "tau": float(row.tau),
                "n_certified": int(certified.sum()),
                "n_true_high": int(true_high.sum()),
                "n_false_certificates": int((certified & ~true_high).sum()),
                "recall": float((certified & true_high).sum() / true_high.sum()) if true_high.any() else np.nan,
                "coverage": float(certified.mean()),
            }
        )
    certificates = pd.DataFrame(certificate_rows)
    identity_failures = int(
        (evaluated.family_identity_residual.abs() > evaluated.family_identity_tolerance).sum()
    )
    lower_violations = int((lower_bound > family_error + 1e-12).sum())
    useful_certificate = bool(
        (
            certificates.n_true_high.ge(20)
            & certificates.recall.ge(0.05)
            & certificates.n_false_certificates.eq(0)
        ).any()
    )
    certificate_supported = bool(
        identity_failures == 0
        and lower_violations == 0
        and float(evaluated.family_lower_tightness.median()) >= 0.15
        and useful_certificate
    )
    upstream = {
        "latent_mean_rmse": float(evaluated.latent_centroid_error.mean()),
        "linear_mean_rmse": float(evaluated.linear_error.mean()),
        "no_change_mean_rmse": float(evaluated.no_change_error.mean()),
    }
    upstream_competent = bool(
        upstream["latent_mean_rmse"] < upstream["no_change_mean_rmse"]
        and upstream["linear_mean_rmse"] < upstream["no_change_mean_rmse"]
    )
    primary_supported = bool(observed_delta > 0 and spearman_interval["ci95_lower"] > 0)
    practical_supported = bool(
        observed_utility_delta >= 0.02
        and utility_interval["ci95_lower"] > 0
        and observed_capture_delta >= 0.05
        and int(states.delta_spearman.gt(0).sum()) >= 9
    )

    output_dir.mkdir(parents=True)
    truth_path = output_dir / "E216_TEST_TRUTH_CENTROIDS.npz"
    np.savez_compressed(
        truth_path,
        truth_centroids=truth,
        gene_names=np.asarray(hvg, dtype=str),
        condition=tasks.condition.to_numpy(dtype=str),
        cell_type=tasks.cell_type.to_numpy(dtype=str),
        treatment=tasks.treatment.to_numpy(dtype=str),
    )
    files = {
        "E216_TASK_RESULTS.csv": evaluated,
        "E216_STATE_RESULTS.csv": states,
        "E216_BOOTSTRAP_DRAWS.csv": bootstrap,
        "E216_CERTIFICATE_CURVES.csv": certificates,
    }
    for name, frame in files.items():
        atomic_text(output_dir / name, frame.to_csv(index=False))
    status = {
        "experiment": "E216_jiang24_resource_bounded_confirmation",
        "stage": "D4_AUTHORIZED_FORMAL_EVALUATION",
        "status": "PASS",
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "test_truth_released_after_risk_seal": True,
        "risk_seal_commit": authorization["risk_seal_commit"],
        "test_perturbed_expression_rows_read": int(len(selected_obs)),
        "registered_tasks": 224,
        "primary_macro_spearman_delta": observed_delta,
        "primary_macro_spearman_bootstrap": spearman_interval,
        "primary_external_confirmation": "SUPPORTED" if primary_supported else "NOT_SUPPORTED",
        "utility_20_delta": observed_utility_delta,
        "utility_20_bootstrap": utility_interval,
        "high_error_capture_delta": observed_capture_delta,
        "positive_states": int(states.delta_spearman.gt(0).sum()),
        "practical_effect_gate": "SUPPORTED" if practical_supported else "NOT_SUPPORTED",
        "upstream": upstream,
        "upstream_competence": "SUPPORTED" if upstream_competent else "NOT_SUPPORTED",
        "certificate_identity_failures": identity_failures,
        "certificate_lower_violations": lower_violations,
        "certificate_median_tightness": float(evaluated.family_lower_tightness.median()),
        "registered_family_certificate": "SUPPORTED" if certificate_supported else "NOT_SUPPORTED",
    }
    atomic_text(
        output_dir / "E216_FORMAL_EVALUATION_STATUS.json",
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
