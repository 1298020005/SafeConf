#!/usr/bin/env python3
"""Independently reconstruct E208 truth errors and audit formal result tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata


EXPECTED_H5_BYTES = 93_532_364_449
EXPECTED_SPLIT_SHA256 = "5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d"
JOBS = (("latent", 1), ("latent", 2), ("latent", 3), ("latent", 4), ("linear", 1))
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 208_202_609
BUDGET = 0.20
TOLERANCE = 1e-10


class AuditFailure(RuntimeError):
    """Formal results differ from an independent truth reconstruction."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([value.decode() if isinstance(value, bytes) else str(value) for value in values], dtype=str)


def categorical(group: h5py.Group) -> np.ndarray:
    categories = decode(group["categories"][:])
    codes = group["codes"][:].astype(np.int64)
    if np.any(codes < 0) or np.any(codes >= len(categories)):
        raise AuditFailure(f"invalid categorical codes: {group.name}")
    return categories[codes]


def runs(rows: np.ndarray) -> list[tuple[int, int]]:
    cuts = np.flatnonzero(np.diff(rows) != 1) + 1
    return [(int(block[0]), int(block[-1]) + 1) for block in np.split(rows, cuts)] if len(rows) else []


def sparse_mean(rows: np.ndarray, indptr: np.ndarray, data: h5py.Dataset, indices: h5py.Dataset) -> np.ndarray:
    total = np.zeros(15473, dtype=np.float64)
    for start_row, stop_row in runs(rows):
        start, stop = int(indptr[start_row]), int(indptr[stop_row])
        total += np.bincount(
            indices[start:stop].astype(np.int64, copy=False),
            weights=data[start:stop].astype(np.float64, copy=False),
            minlength=15473,
        )
    # Match the registered centroid storage contract: float32 on disk, float64 for metrics.
    return (total / len(rows)).astype(np.float32).astype(np.float64)


def extract_truth(h5ad: Path, split_path: Path, tasks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    split = pd.read_csv(split_path, header=None, names=["cell_id", "split"])
    with h5py.File(h5ad, "r") as handle:
        if not np.array_equal(decode(handle["obs/_index"][:]), split.cell_id.astype(str).to_numpy()):
            raise AuditFailure("split/H5 row alignment changed")
        condition = categorical(handle["obs/condition"])
        cell_type = categorical(handle["obs/cell_type"])
        treatment = categorical(handle["obs/treatment"])
        split_values = split.split.astype(str).to_numpy()
        matrix = handle["X"]
        indptr = matrix["indptr"][:].astype(np.int64)
        truth = np.empty((len(tasks), 15473), dtype=np.float64)
        counts = np.empty(len(tasks), dtype=np.int64)
        for index, task in enumerate(tasks.itertuples(index=False)):
            rows = np.flatnonzero(
                (split_values == "test")
                & (cell_type == str(task.cell_type))
                & (treatment == str(task.treatment))
                & (condition == str(task.condition))
            )
            if len(rows) < 30:
                raise AuditFailure(f"fewer than 30 truth cells for {task.task_id}")
            truth[index] = sparse_mean(rows, indptr, matrix["data"], matrix["indices"])
            counts[index] = len(rows)
    if int(counts.sum()) != 214_901 or not np.isfinite(truth).all():
        raise AuditFailure("independent truth extraction count/finite gate failed")
    return truth, counts


def stable_ties(task_ids: np.ndarray, occurrences: np.ndarray) -> np.ndarray:
    return np.asarray(
        [int(hashlib.sha256(f"E208\0{task}\0{int(occ)}".encode()).hexdigest()[:16], 16)
         for task, occ in zip(map(str, task_ids), occurrences, strict=True)],
        dtype=np.uint64,
    )


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr, yr = rankdata(x, method="average"), rankdata(y, method="average")
    return float(np.corrcoef(xr, yr)[0, 1]) if np.std(xr) and np.std(yr) else float("nan")


def context_stat(frame: pd.DataFrame, score: str, outcome: str, occurrences: np.ndarray | None = None) -> dict[str, float]:
    block_frame = frame.copy()
    block_frame["_occurrence"] = np.zeros(len(frame), dtype=int) if occurrences is None else occurrences
    rhos, utilities, captures = [], [], []
    for _, block in block_frame.groupby(["cell_type", "treatment"], sort=True):
        values = block[outcome].to_numpy(float)
        scores = block[score].to_numpy(float)
        ties = stable_ties(block.task_id.to_numpy(), block._occurrence.to_numpy(int))
        count = math.ceil(BUDGET * len(block))
        selected = np.lexsort((ties, -scores))[:count]
        oracle = np.lexsort((ties, -values))[:count]
        denominator = float(values[oracle].mean() - values.mean())
        rhos.append(spearman(scores, values))
        utilities.append((float(values[selected].mean()) - float(values.mean())) / denominator)
        captures.append(float(values[selected].sum() / values.sum()))
    return {"spearman": float(np.mean(rhos)), "utility": float(np.mean(utilities)), "capture": float(np.mean(captures))}


def max_abs(left: np.ndarray, right: np.ndarray, label: str, tolerance: float = TOLERANCE) -> float:
    if left.shape != right.shape:
        raise AuditFailure(f"shape mismatch for {label}")
    value = float(np.max(np.abs(left.astype(float) - right.astype(float))))
    if not np.isfinite(value) or value > tolerance:
        raise AuditFailure(f"{label} maximum absolute difference {value} exceeds {tolerance}")
    return value


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    return (
        0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
        1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--risk-table", type=Path, required=True)
    parser.add_argument("--progeny", type=Path, required=True)
    parser.add_argument("--gene-axis", type=Path, required=True)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("h5ad", "split", "prediction_root", "risk_table", "progeny", "gene_axis", "formal_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    if args.h5ad.stat().st_size != EXPECTED_H5_BYTES or sha256_file(args.split) != EXPECTED_SPLIT_SHA256:
        raise AuditFailure("frozen Jiang24 source gate failed")

    status_path = args.formal_dir / "E208_FORMAL_EVALUATION_STATUS.json"
    task_path = args.formal_dir / "E208_FORMAL_TASK_RESULTS.csv"
    summary_path = args.formal_dir / "E208_OUTPUT_SPACE_SUMMARY.csv"
    context_path = args.formal_dir / "E208_CONTEXT_RESULTS.csv"
    bootstrap_path = args.formal_dir / "E208_GENE_CLUSTER_BOOTSTRAP.csv.gz"
    certificate_path = args.formal_dir / "E208_CERTIFICATE_RESULTS.csv"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE" or int(status.get("n_tasks", -1)) != 224:
        raise AuditFailure("formal evaluation status is incomplete")
    for path in (task_path, summary_path, context_path, bootstrap_path, certificate_path):
        if sha256_file(path) != status["files"][path.name]["sha256"]:
            raise AuditFailure(f"formal result hash differs: {path.name}")

    result = pd.read_csv(task_path)
    risk = pd.read_csv(args.risk_table)
    shared = list(risk.columns)
    if len(result) != 224 or list(result.columns[:len(shared)]) != shared:
        raise AuditFailure("formal table does not preserve the sealed risk-table prefix")
    for column in shared:
        if pd.api.types.is_numeric_dtype(risk[column]):
            max_abs(result[column].to_numpy(), risk[column].to_numpy(), f"sealed column {column}")
        elif not result[column].astype(str).equals(risk[column].astype(str)):
            raise AuditFailure(f"sealed text column differs: {column}")

    members, controls, tasks = [], None, None
    for architecture, seed in JOBS:
        directory = args.prediction_root / "test_pretruth" / architecture / f"seed_{seed}"
        prediction = np.load(directory / "E208_PREDICTION_CENTROIDS.npy", allow_pickle=False).astype(float)
        control = np.load(directory / "E208_CONTROL_CENTROIDS.npy", allow_pickle=False).astype(float)
        task = pd.read_csv(directory / "E208_PREDICTION_TASKS.csv")
        prediction_status = json.loads((directory / "E208_PREDICTION_STATUS.json").read_text(encoding="utf-8"))
        if prediction_status.get("status") != "PASS" or int(prediction_status.get("test_perturbed_expression_rows_read", -1)) != 0:
            raise AuditFailure(f"prediction status failed: {architecture}/{seed}")
        if controls is None:
            controls, tasks = control, task
        elif not np.array_equal(control, controls) or not task.equals(tasks):
            raise AuditFailure("model prediction task/control inputs differ")
        members.append(prediction)
    assert controls is not None and tasks is not None
    if not tasks.task_id.equals(result.task_id):
        raise AuditFailure("prediction/result task order differs")

    truth, counts = extract_truth(args.h5ad, args.split, tasks)
    latent = sum(members[:4]) / 4.0
    family_centroid = 0.5 * latent + 0.5 * members[4]
    family_squared = sum(
        weight * np.mean((prediction - truth) ** 2, axis=1)
        for weight, prediction in zip((0.125, 0.125, 0.125, 0.125, 0.5), members, strict=True)
    )
    recomputed = {
        "n_truth_cells": counts,
        "full_gene_rmse": np.sqrt(np.mean((latent - truth) ** 2, axis=1)),
        "no_change_rmse": np.sqrt(np.mean((controls - truth) ** 2, axis=1)),
        "linear_rmse": np.sqrt(np.mean((members[4] - truth) ** 2, axis=1)),
        "registered_family_centroid_rmse": np.sqrt(np.mean((family_centroid - truth) ** 2, axis=1)),
        "registered_family_rms_error": np.sqrt(family_squared),
    }
    true_effect, predicted_effect = truth - controls, latent - controls
    genes_index = np.arange(15473)
    deg = np.empty(224)
    for index in range(224):
        top = np.lexsort((genes_index, -np.abs(true_effect[index])))[:100]
        deg[index] = np.sqrt(np.mean((latent[index, top] - truth[index, top]) ** 2))
    recomputed["top100_deg_rmse"] = deg

    genes = pd.read_csv(args.gene_axis).sort_values("gene_index").gene_name.astype(str).to_numpy()
    resource = pd.read_csv(args.progeny)
    pathways = sorted(resource.pathway.astype(str).unique())
    lookup = {gene: index for index, gene in enumerate(genes)}
    weights = np.zeros((15473, len(pathways)))
    for pathway_index, pathway in enumerate(pathways):
        for row in resource.loc[resource.pathway.astype(str).eq(pathway)].itertuples(index=False):
            if str(row.gene) in lookup:
                weights[lookup[str(row.gene)], pathway_index] = float(row.weight)
        weights[:, pathway_index] /= np.linalg.norm(weights[:, pathway_index])
    recomputed["progeny14_rmse"] = np.sqrt(
        np.mean(((predicted_effect @ weights) - (true_effect @ weights)) ** 2, axis=1)
    )
    differences = {column: max_abs(np.asarray(values), result[column].to_numpy(), column) for column, values in recomputed.items()}

    lower = result.registered_family_lower_bound.to_numpy(float)
    identity = family_squared - (recomputed["registered_family_centroid_rmse"] ** 2 + lower**2)
    differences["family_identity_residual"] = max_abs(identity, result.family_identity_residual.to_numpy(), "family identity")
    violation_values = result.family_lower_bound_violation
    violations = (
        violation_values.to_numpy(bool)
        if pd.api.types.is_bool_dtype(violation_values)
        else violation_values.astype(str).str.lower().eq("true").to_numpy()
    )
    if violations.any():
        raise AuditFailure("formal result contains a lower-bound violation")

    spaces = {"full_gene": "full_gene_rmse", "top100_deg": "top100_deg_rmse", "progeny14": "progeny14_rmse"}
    summary_rows, context_rows = [], []
    for space, outcome in spaces.items():
        magnitude = context_stat(result, "predicted_magnitude", outcome)
        combined = context_stat(result, "safeconf_m_primary", outcome)
        summary_rows.append({
            "output_space": space, "outcome": outcome,
            **{f"magnitude_{key}": value for key, value in magnitude.items()},
            **{f"safeconf_m_{key}": value for key, value in combined.items()},
            "delta_spearman": combined["spearman"] - magnitude["spearman"],
            "delta_utility_20": combined["utility"] - magnitude["utility"],
            "relative_capture_improvement_20": (combined["capture"] - magnitude["capture"]) / magnitude["capture"],
        })
        for (cell, treatment), block in result.groupby(["cell_type", "treatment"], sort=True):
            context_rows.append({
                "output_space": space, "cell_type": cell, "treatment": treatment, "n_tasks": len(block),
                "magnitude_spearman": spearman(block.predicted_magnitude.to_numpy(float), block[outcome].to_numpy(float)),
                "safeconf_m_spearman": spearman(block.safeconf_m_primary.to_numpy(float), block[outcome].to_numpy(float)),
            })
    rebuilt_summary = pd.DataFrame(summary_rows)
    rebuilt_context = pd.DataFrame(context_rows)
    rebuilt_context["delta_spearman"] = rebuilt_context.safeconf_m_spearman - rebuilt_context.magnitude_spearman
    sealed_summary, sealed_context = pd.read_csv(summary_path), pd.read_csv(context_path)
    for column in rebuilt_summary.select_dtypes(include=[np.number]).columns:
        differences[f"summary_{column}"] = max_abs(rebuilt_summary[column].to_numpy(), sealed_summary[column].to_numpy(), f"summary {column}")
    for column in rebuilt_context.select_dtypes(include=[np.number]).columns:
        differences[f"context_{column}"] = max_abs(rebuilt_context[column].to_numpy(), sealed_context[column].to_numpy(), f"context {column}")

    clusters = sorted(result.condition.astype(str).unique())
    members_by_cluster = [np.flatnonzero(result.condition.astype(str).to_numpy() == cluster) for cluster in clusters]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for draw in range(N_BOOTSTRAP):
        chosen = rng.integers(0, len(clusters), len(clusters))
        indices = np.concatenate([members_by_cluster[int(index)] for index in chosen])
        occurrences = np.concatenate([np.full(len(members_by_cluster[int(index)]), occurrence) for occurrence, index in enumerate(chosen)])
        block = result.iloc[indices]
        magnitude = context_stat(block, "predicted_magnitude", "full_gene_rmse", occurrences)
        combined = context_stat(block, "safeconf_m_primary", "full_gene_rmse", occurrences)
        draws.append((draw, combined["spearman"] - magnitude["spearman"], combined["utility"] - magnitude["utility"], (combined["capture"] - magnitude["capture"]) / magnitude["capture"]))
    rebuilt_bootstrap = pd.DataFrame(draws, columns=["draw", "delta_spearman", "delta_utility_20", "relative_capture_improvement_20"])
    sealed_bootstrap = pd.read_csv(bootstrap_path)
    for column in rebuilt_bootstrap.columns:
        differences[f"bootstrap_{column}"] = max_abs(rebuilt_bootstrap[column].to_numpy(), sealed_bootstrap[column].to_numpy(), f"bootstrap {column}")

    sealed_certificates = pd.read_csv(certificate_path)
    certificate_rows = []
    family_error = recomputed["registered_family_rms_error"]
    upper = result.registered_family_conformal_upper.to_numpy(float)
    for threshold in sealed_certificates[["quantile", "tau"]].itertuples(index=False):
        tau = float(threshold.tau)
        high = lower > tau
        low = (~high) & (upper <= tau)
        true_high = family_error > tau
        certificate_rows.append({
            "quantile": float(threshold.quantile),
            "tau": tau,
            "certified_high": int(high.sum()),
            "conformal_low": int(low.sum()),
            "unknown": int((~high & ~low).sum()),
            "true_high": int(true_high.sum()),
            "high_recall": float((high & true_high).sum() / true_high.sum()) if true_high.sum() else float("nan"),
            "false_high_certificates": int((high & ~true_high).sum()),
            "false_low_releases": int((low & true_high).sum()),
        })
    rebuilt_certificates = pd.DataFrame(certificate_rows)
    for column in rebuilt_certificates.select_dtypes(include=[np.number]).columns:
        differences[f"certificate_{column}"] = max_abs(
            rebuilt_certificates[column].to_numpy(),
            sealed_certificates[column].to_numpy(),
            f"certificate {column}",
        )

    primary = rebuilt_summary.loc[rebuilt_summary.output_space.eq("full_gene")].iloc[0]
    valid = rebuilt_bootstrap.dropna()
    delta_ci = [float(valid.delta_spearman.quantile(0.025)), float(valid.delta_spearman.quantile(0.975))]
    utility_ci = [float(valid.delta_utility_20.quantile(0.025)), float(valid.delta_utility_20.quantile(0.975))]
    differences["status_primary_delta"] = abs(float(primary.delta_spearman) - float(status["primary_delta_spearman"]))
    differences["status_primary_delta_ci"] = max_abs(np.asarray(delta_ci), np.asarray(status["primary_delta_spearman_ci95"]), "status primary delta CI")
    differences["status_utility_delta"] = abs(float(primary.delta_utility_20) - float(status["primary_delta_utility_20"]))
    differences["status_utility_delta_ci"] = max_abs(np.asarray(utility_ci), np.asarray(status["primary_delta_utility_20_ci95"]), "status utility delta CI")
    differences["status_relative_capture"] = abs(
        float(primary.relative_capture_improvement_20)
        - float(status["primary_relative_capture_improvement_20"])
    )
    target_coverage = (
        result.assign(_covered=family_error <= upper)
        .groupby("condition", sort=True)._covered.all()
    )
    covered = int(target_coverage.sum())
    coverage_ci = clopper_pearson(covered, len(target_coverage))
    differences["status_target_coverage"] = abs(
        covered / len(target_coverage) - float(status["target_conformal_coverage"])
    )
    differences["status_target_coverage_ci"] = max_abs(
        np.asarray(coverage_ci), np.asarray(status["target_conformal_coverage_ci95"]), "status target coverage CI"
    )
    capture_ci = [
        float(valid.relative_capture_improvement_20.quantile(0.025)),
        float(valid.relative_capture_improvement_20.quantile(0.975)),
    ]
    gates = {
        "primary_delta_spearman_ci_lower_gt_zero": delta_ci[0] > 0,
        "utility_delta_at_least_0_02": float(primary.delta_utility_20) >= 0.02,
        "utility_delta_ci_lower_gt_zero": utility_ci[0] > 0,
        "capture_relative_improvement_at_least_0_05": float(primary.relative_capture_improvement_20) >= 0.05,
        "at_least_9_of_12_context_deltas_positive": int(
            (rebuilt_context.loc[rebuilt_context.output_space.eq("full_gene"), "delta_spearman"] > 0).sum()
        ) >= 9,
        "at_least_2_of_3_output_spaces_positive": int((rebuilt_summary.delta_spearman > 0).sum()) >= 2,
    }
    if gates != status["gates"]:
        raise AuditFailure(f"formal gate decisions differ: {gates} != {status['gates']}")
    if bool(status["external_confirmation"]) != gates["primary_delta_spearman_ci_lower_gt_zero"]:
        raise AuditFailure("external-confirmation decision differs")
    if bool(status["all_practical_effect_gates"]) != all(gates.values()):
        raise AuditFailure("all-practical-gates decision differs")
    if not np.isfinite(capture_ci).all():
        raise AuditFailure("relative-capture bootstrap interval is non-finite")
    tightness = lower / family_error
    differences["status_family_tightness_median"] = abs(
        float(np.median(tightness)) - float(status["family_tightness_median"])
    )
    certificate_operational = (
        float(np.median(tightness)) >= 0.15
        and bool(((rebuilt_certificates.true_high >= 20) & (rebuilt_certificates.high_recall >= 0.05)).any())
    )
    if bool(status["certificate_operational"]) != certificate_operational:
        raise AuditFailure("certificate operational decision differs")
    if int(status["family_identity_failures"]) != int(
        (np.abs(identity) > result.family_identity_tolerance.to_numpy(float)).sum()
    ) or int(status["family_lower_bound_violations"]) != int(violations.sum()):
        raise AuditFailure("family identity/lower-bound failure counts differ")
    if max(differences.values()) > TOLERANCE:
        raise AuditFailure("post-truth numerical audit exceeded tolerance")

    audit = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D2_INDEPENDENT_FORMAL_RESULT_AUDIT",
        "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_tasks": len(result),
        "n_truth_cells_reread": int(counts.sum()),
        "n_bootstrap_draws_recomputed": len(rebuilt_bootstrap),
        "maximum_absolute_differences": differences,
        "maximum_over_all_numeric_checks": float(max(differences.values())),
        "tolerance": TOLERANCE,
        "formal_status_sha256": sha256_file(status_path),
        "formal_task_results_sha256": sha256_file(task_path),
        "test_truth_access": "AUTHORIZED_POST_SEAL",
    }
    atomic_json(args.output_dir / "E208_FORMAL_RESULTS_INDEPENDENT_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
