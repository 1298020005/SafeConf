#!/usr/bin/env python3
"""One-shot E208 external evaluation after an independently committed truth authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata


EXPECTED_H5_BYTES = 93_532_364_449
EXPECTED_SPLIT_SHA256 = "5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d"
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 208_202_609
BUDGET = 0.20


class EvaluationFailure(RuntimeError):
    """The authorization, sealed inputs, truth extraction, or metric contract failed."""


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


def atomic_csv(path: Path, value: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    value.to_csv(temporary, index=False)
    os.replace(temporary, path)


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values], dtype=str
    )


def categorical(group: h5py.Group) -> np.ndarray:
    categories = decode(group["categories"][:])
    codes = group["codes"][:].astype(np.int64)
    if np.any(codes < 0) or np.any(codes >= len(categories)):
        raise EvaluationFailure(f"invalid categorical codes: {group.name}")
    return categories[codes]


def consecutive_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    if len(rows) == 0:
        return []
    cuts = np.flatnonzero(np.diff(rows) != 1) + 1
    return [(int(block[0]), int(block[-1]) + 1) for block in np.split(rows, cuts)]


def sparse_mean(rows: np.ndarray, indptr: np.ndarray, data: h5py.Dataset, indices: h5py.Dataset, n_genes: int) -> np.ndarray:
    accumulator = np.zeros(n_genes, dtype=np.float64)
    for row_start, row_stop in consecutive_runs(rows):
        start, stop = int(indptr[row_start]), int(indptr[row_stop])
        accumulator += np.bincount(
            indices[start:stop].astype(np.int64, copy=False),
            weights=data[start:stop].astype(np.float64, copy=False),
            minlength=n_genes,
        )
    return (accumulator / len(rows)).astype(np.float32)


def git_output(repo: Path, *arguments: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=text)


def verify_authorization(
    repo: Path,
    authorization_path: Path,
    risk_path: Path,
    risk_status_path: Path,
    thresholds_path: Path,
) -> dict:
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    risk_status = json.loads(risk_status_path.read_text(encoding="utf-8"))
    risk_hash = sha256_file(risk_path)
    risk_status_hash = sha256_file(risk_status_path)
    thresholds_hash = sha256_file(thresholds_path)
    checks = {
        "experiment": authorization.get("experiment") == "E208_jiang24_external_confirmation",
        "stage": authorization.get("stage") == "D2_TEST_TRUTH_AUTHORIZATION",
        "authorized": authorization.get("status") == "AUTHORIZED",
        "risk_status": risk_status.get("status") == "PASS_AWAITING_REMOTE_SEAL",
        "risk_rows": int(risk_status.get("n_tasks", -1)) == 224,
        "risk_hash": authorization.get("risk_table_sha256") == risk_hash == risk_status["risk_table"]["sha256"],
        "risk_status_hash": authorization.get("risk_status_sha256") == risk_status_hash,
        "thresholds_hash": thresholds_hash == risk_status["family_thresholds"]["sha256"],
        "zero_prior_truth": int(risk_status.get("test_perturbed_expression_rows_read", -1)) == 0,
    }
    pretruth_commit = str(authorization.get("pretruth_git_commit", ""))
    remote_ref = str(authorization.get("remote_ref", ""))
    if len(pretruth_commit) != 40 or subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{pretruth_commit}^{{commit}}"], check=False
    ).returncode:
        raise EvaluationFailure("invalid pretruth Git commit in authorization")
    try:
        authorization_relative = authorization_path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as error:
        raise EvaluationFailure("authorization file must live inside the repository") from error
    authorization_commit = git_output(
        repo, "log", "-1", "--format=%H", "--", authorization_relative
    ).strip()
    head_commit = git_output(repo, "rev-parse", "HEAD").strip()
    if len(authorization_commit) != 40:
        raise EvaluationFailure("authorization file is not committed")
    checks["authorization_descends_from_pretruth"] = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", pretruth_commit, authorization_commit], check=False
    ).returncode == 0
    checks["head_is_authorization_commit"] = head_commit == authorization_commit
    checks["worktree_is_clean"] = not git_output(
        repo, "status", "--porcelain", "--untracked-files=all"
    ).strip()
    risk_repo_path = str(authorization.get("risk_table_repo_path", ""))
    status_repo_path = str(authorization.get("risk_status_repo_path", ""))
    for label, repo_path, expected_hash in (
        ("risk table", risk_repo_path, risk_hash),
        ("risk status", status_repo_path, risk_status_hash),
    ):
        if not repo_path or repo_path.startswith("/") or ".." in Path(repo_path).parts:
            raise EvaluationFailure(f"invalid {label} repository path in authorization")
        try:
            sealed_bytes = git_output(repo, "show", f"{pretruth_commit}:{repo_path}", text=False)
        except subprocess.CalledProcessError as error:
            raise EvaluationFailure(f"cannot read sealed {label} from pretruth commit") from error
        checks[f"sealed_{label.replace(' ', '_')}_hash"] = hashlib.sha256(sealed_bytes).hexdigest() == expected_hash
    if not remote_ref.startswith("refs/heads/"):
        raise EvaluationFailure("authorization remote_ref is not a branch ref")
    for remote in ("origin", "github"):
        output = subprocess.check_output(
            ["git", "-C", str(repo), "ls-remote", remote, remote_ref], text=True
        ).strip().split()
        observed = output[0] if output else ""
        checks[f"{remote}_remote_head"] = observed == authorization_commit
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise EvaluationFailure(f"truth authorization gates failed before H5 access: {failed}")
    return authorization


def stable_ties(task_ids: np.ndarray, occurrences: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            int(hashlib.sha256(f"E208\0{task}\0{int(occ)}".encode()).hexdigest()[:16], 16)
            for task, occ in zip(map(str, task_ids), occurrences, strict=True)
        ],
        dtype=np.uint64,
    )


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr, yr = rankdata(x, method="average"), rankdata(y, method="average")
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def context_stat(frame: pd.DataFrame, score: str, outcome: str, occurrences: np.ndarray | None = None) -> dict[str, float]:
    if occurrences is None:
        occurrences = np.zeros(len(frame), dtype=int)
    frame = frame.copy()
    frame["_occurrence"] = occurrences
    rhos, utilities, captures = [], [], []
    for _, block in frame.groupby(["cell_type", "treatment"], sort=True):
        if len(block) < 5:
            return {"spearman": float("nan"), "utility": float("nan"), "capture": float("nan")}
        values = block[outcome].to_numpy(float)
        scores = block[score].to_numpy(float)
        ties = stable_ties(block.task_id.to_numpy(), block._occurrence.to_numpy(int))
        n_select = math.ceil(BUDGET * len(block))
        selected = np.lexsort((ties, -scores))[:n_select]
        oracle = np.lexsort((ties, -values))[:n_select]
        overall = float(values.mean())
        denominator = float(values[oracle].mean()) - overall
        rhos.append(spearman(scores, values))
        utilities.append(
            (float(values[selected].mean()) - overall) / denominator
            if denominator > 1e-15
            else float("nan")
        )
        captures.append(float(values[selected].sum() / values.sum()))
    return {
        "spearman": float(np.mean(rhos)),
        "utility": float(np.mean(utilities)),
        "capture": float(np.mean(captures)),
    }


def load_test_predictions(prediction_root: Path) -> tuple[list[np.ndarray], np.ndarray, pd.DataFrame]:
    members, controls, tasks = [], None, None
    for architecture, seed in (("latent", 1), ("latent", 2), ("latent", 3), ("latent", 4), ("linear", 1)):
        directory = prediction_root / "test_pretruth" / architecture / f"seed_{seed}"
        p = np.load(directory / "E208_PREDICTION_CENTROIDS.npy", allow_pickle=False)
        c = np.load(directory / "E208_CONTROL_CENTROIDS.npy", allow_pickle=False)
        t = pd.read_csv(directory / "E208_PREDICTION_TASKS.csv")
        s = json.loads((directory / "E208_PREDICTION_STATUS.json").read_text(encoding="utf-8"))
        if s.get("status") != "PASS" or s.get("architecture") != architecture or int(s.get("seed", -1)) != seed:
            raise EvaluationFailure(f"invalid test prediction: {directory}")
        if controls is None:
            controls, tasks = c, t
        elif not np.array_equal(c, controls) or not t.equals(tasks):
            raise EvaluationFailure("test members used different controls/tasks")
        members.append(p.astype(np.float64))
    return members, controls.astype(np.float64), tasks


def extract_truth(h5ad: Path, split_path: Path, tasks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    split = pd.read_csv(split_path, header=None, names=["cell_id", "split"])
    with h5py.File(h5ad, "r") as handle:
        cell_ids = decode(handle["obs/_index"][:])
        if not np.array_equal(cell_ids, split.cell_id.astype(str).to_numpy()):
            raise EvaluationFailure("split/H5 alignment changed")
        condition = categorical(handle["obs/condition"])
        cell_type = categorical(handle["obs/cell_type"])
        treatment = categorical(handle["obs/treatment"])
        split_values = split.split.astype(str).to_numpy()
        matrix = handle["X"]
        indptr = matrix["indptr"][:].astype(np.int64)
        truth = np.empty((len(tasks), 15473), dtype=np.float32)
        counts = np.empty(len(tasks), dtype=np.int64)
        for index, task in enumerate(tasks.itertuples(index=False)):
            rows = np.flatnonzero(
                (split_values == "test")
                & (cell_type == str(task.cell_type))
                & (treatment == str(task.treatment))
                & (condition == str(task.condition))
            )
            if len(rows) < 30:
                raise EvaluationFailure(f"truth cells below frozen minimum: {task.task_id}")
            truth[index] = sparse_mean(rows, indptr, matrix["data"], matrix["indices"], 15473)
            counts[index] = len(rows)
    if int(counts.sum()) != 214_901 or not np.isfinite(truth).all():
        raise EvaluationFailure(f"truth extraction count/finite gate failed: {counts.sum()}")
    return truth.astype(np.float64), counts


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    return (
        0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
        1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--risk-table", type=Path, required=True)
    parser.add_argument("--risk-status", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--progeny", type=Path, required=True)
    parser.add_argument("--gene-axis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("repo", "h5ad", "split", "prediction_root", "risk_table", "risk_status", "thresholds", "authorization", "progeny", "gene_axis", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    if args.h5ad.stat().st_size != EXPECTED_H5_BYTES or sha256_file(args.split) != EXPECTED_SPLIT_SHA256:
        raise EvaluationFailure("Jiang24 frozen file gate failed")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise EvaluationFailure("formal output directory is not empty; one-shot evaluation will not overwrite it")
    authorization = verify_authorization(
        args.repo,
        args.authorization,
        args.risk_table,
        args.risk_status,
        args.thresholds,
    )
    risk = pd.read_csv(args.risk_table)
    members, controls, tasks = load_test_predictions(args.prediction_root)
    if not tasks.task_id.equals(risk.task_id) or len(risk) != 224:
        raise EvaluationFailure("sealed risk and prediction task order differ")

    truth, truth_counts = extract_truth(args.h5ad, args.split, tasks)
    latent_mean = sum(members[:4]) / 4.0
    family_mean = 0.5 * latent_mean + 0.5 * members[4]
    full_error = np.sqrt(np.mean((latent_mean - truth) ** 2, axis=1))
    no_change_error = np.sqrt(np.mean((controls - truth) ** 2, axis=1))
    linear_error = np.sqrt(np.mean((members[4] - truth) ** 2, axis=1))
    family_centroid_error = np.sqrt(np.mean((family_mean - truth) ** 2, axis=1))
    family_error_sq = np.zeros(len(tasks), dtype=np.float64)
    for weight, prediction in zip((0.125, 0.125, 0.125, 0.125, 0.5), members, strict=True):
        family_error_sq += weight * np.mean((prediction - truth) ** 2, axis=1)
    family_error = np.sqrt(family_error_sq)

    true_effect = truth - controls
    predicted_effect = latent_mean - controls
    deg_error = np.empty(len(tasks), dtype=np.float64)
    gene_index = np.arange(truth.shape[1])
    for index in range(len(tasks)):
        top = np.lexsort((gene_index, -np.abs(true_effect[index])))[:100]
        deg_error[index] = np.sqrt(np.mean((latent_mean[index, top] - truth[index, top]) ** 2))

    genes = pd.read_csv(args.gene_axis).sort_values("gene_index").gene_name.astype(str).to_numpy()
    resource = pd.read_csv(args.progeny)
    pathways = sorted(resource.pathway.astype(str).unique())
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    weights = np.zeros((len(genes), len(pathways)), dtype=np.float64)
    overlap_counts = {}
    for pathway_index, pathway in enumerate(pathways):
        block = resource.loc[resource.pathway.astype(str).eq(pathway)]
        used = 0
        for row in block.itertuples(index=False):
            if str(row.gene) in gene_lookup:
                weights[gene_lookup[str(row.gene)], pathway_index] = float(row.weight)
                used += 1
        norm = np.linalg.norm(weights[:, pathway_index])
        if used < 100 or norm == 0:
            raise EvaluationFailure(f"insufficient PROGENy overlap: {pathway}/{used}")
        weights[:, pathway_index] /= norm
        overlap_counts[pathway] = used
    predicted_pathway = predicted_effect @ weights
    true_pathway = true_effect @ weights
    pathway_error = np.sqrt(np.mean((predicted_pathway - true_pathway) ** 2, axis=1))

    result = risk.copy()
    result["n_truth_cells"] = truth_counts
    result["full_gene_rmse"] = full_error
    result["top100_deg_rmse"] = deg_error
    result["progeny14_rmse"] = pathway_error
    result["no_change_rmse"] = no_change_error
    result["linear_rmse"] = linear_error
    result["registered_family_centroid_rmse"] = family_centroid_error
    result["registered_family_rms_error"] = family_error
    lower = result.registered_family_lower_bound.to_numpy(float)
    identity_residual = family_error_sq - (family_centroid_error**2 + lower**2)
    tolerance = np.maximum(1e-12, 1e-10 * np.maximum(family_error_sq, family_centroid_error**2 + lower**2))
    result["family_identity_residual"] = identity_residual
    result["family_identity_tolerance"] = tolerance
    result["family_lower_bound_violation"] = lower**2 > family_error_sq + tolerance
    result["family_lower_bound_tightness"] = lower / family_error
    if result.family_lower_bound_violation.any() or np.any(np.abs(identity_residual) > tolerance):
        raise EvaluationFailure("registered family identity/lower-bound numerical gate failed")

    spaces = {
        "full_gene": "full_gene_rmse",
        "top100_deg": "top100_deg_rmse",
        "progeny14": "progeny14_rmse",
    }
    summary_rows = []
    context_rows = []
    for space, outcome in spaces.items():
        magnitude = context_stat(result, "predicted_magnitude", outcome)
        combined = context_stat(result, "safeconf_m_primary", outcome)
        summary_rows.append(
            {
                "output_space": space,
                "outcome": outcome,
                **{f"magnitude_{k}": v for k, v in magnitude.items()},
                **{f"safeconf_m_{k}": v for k, v in combined.items()},
                "delta_spearman": combined["spearman"] - magnitude["spearman"],
                "delta_utility_20": combined["utility"] - magnitude["utility"],
                "relative_capture_improvement_20": (combined["capture"] - magnitude["capture"]) / magnitude["capture"],
            }
        )
        for (cell, treatment), block in result.groupby(["cell_type", "treatment"], sort=True):
            context_rows.append(
                {
                    "output_space": space,
                    "cell_type": cell,
                    "treatment": treatment,
                    "n_tasks": len(block),
                    "magnitude_spearman": spearman(block.predicted_magnitude.to_numpy(float), block[outcome].to_numpy(float)),
                    "safeconf_m_spearman": spearman(block.safeconf_m_primary.to_numpy(float), block[outcome].to_numpy(float)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    contexts = pd.DataFrame(context_rows)
    contexts["delta_spearman"] = contexts.safeconf_m_spearman - contexts.magnitude_spearman

    clusters = sorted(result.condition.astype(str).unique())
    members_by_cluster = [np.flatnonzero(result.condition.astype(str).to_numpy() == cluster) for cluster in clusters]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for draw in range(N_BOOTSTRAP):
        chosen = rng.integers(0, len(clusters), len(clusters))
        indices = np.concatenate([members_by_cluster[int(index)] for index in chosen])
        occurrences = np.concatenate(
            [np.full(len(members_by_cluster[int(index)]), occurrence, dtype=int) for occurrence, index in enumerate(chosen)]
        )
        block = result.iloc[indices].copy()
        magnitude = context_stat(block, "predicted_magnitude", "full_gene_rmse", occurrences)
        combined = context_stat(block, "safeconf_m_primary", "full_gene_rmse", occurrences)
        draws.append(
            {
                "draw": draw,
                "delta_spearman": combined["spearman"] - magnitude["spearman"],
                "delta_utility_20": combined["utility"] - magnitude["utility"],
                "relative_capture_improvement_20": (combined["capture"] - magnitude["capture"]) / magnitude["capture"],
            }
        )
    bootstrap = pd.DataFrame(draws)
    valid = bootstrap.dropna()
    if len(valid) < 4750:
        raise EvaluationFailure(f"too few valid cluster bootstrap draws: {len(valid)}")
    ci = {
        column: [float(valid[column].quantile(0.025)), float(valid[column].quantile(0.975))]
        for column in ("delta_spearman", "delta_utility_20", "relative_capture_improvement_20")
    }

    target_coverage = (
        result.assign(_covered=result.registered_family_rms_error <= result.registered_family_conformal_upper)
        .groupby("condition", sort=True)._covered.all()
    )
    covered = int(target_coverage.sum())
    coverage_ci = clopper_pearson(covered, len(target_coverage))
    thresholds = pd.read_csv(args.thresholds)
    certificate_rows = []
    for threshold in thresholds.itertuples(index=False):
        tau = float(threshold.tau)
        high = lower > tau
        upper = result.registered_family_conformal_upper.to_numpy(float)
        low = (~high) & (upper <= tau)
        true_high = family_error > tau
        certificate_rows.append(
            {
                "quantile": threshold.quantile,
                "tau": tau,
                "certified_high": int(high.sum()),
                "conformal_low": int(low.sum()),
                "unknown": int((~high & ~low).sum()),
                "true_high": int(true_high.sum()),
                "high_recall": float((high & true_high).sum() / true_high.sum()) if true_high.sum() else float("nan"),
                "false_high_certificates": int((high & ~true_high).sum()),
                "false_low_releases": int((low & true_high).sum()),
            }
        )
    certificates = pd.DataFrame(certificate_rows)
    certificate_operational = (
        float(result.family_lower_bound_tightness.median()) >= 0.15
        and bool(((certificates.true_high >= 20) & (certificates.high_recall >= 0.05)).any())
    )

    primary = summary.loc[summary.output_space.eq("full_gene")].iloc[0]
    gates = {
        "primary_delta_spearman_ci_lower_gt_zero": ci["delta_spearman"][0] > 0,
        "utility_delta_at_least_0_02": float(primary.delta_utility_20) >= 0.02,
        "utility_delta_ci_lower_gt_zero": ci["delta_utility_20"][0] > 0,
        "capture_relative_improvement_at_least_0_05": float(primary.relative_capture_improvement_20) >= 0.05,
        "at_least_9_of_12_context_deltas_positive": int(
            (contexts.loc[contexts.output_space.eq("full_gene"), "delta_spearman"] > 0).sum()
        ) >= 9,
        "at_least_2_of_3_output_spaces_positive": int((summary.delta_spearman > 0).sum()) >= 2,
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    task_path = output / "E208_FORMAL_TASK_RESULTS.csv"
    summary_path = output / "E208_OUTPUT_SPACE_SUMMARY.csv"
    context_path = output / "E208_CONTEXT_RESULTS.csv"
    bootstrap_path = output / "E208_GENE_CLUSTER_BOOTSTRAP.csv.gz"
    certificate_path = output / "E208_CERTIFICATE_RESULTS.csv"
    atomic_csv(task_path, result)
    atomic_csv(summary_path, summary)
    atomic_csv(context_path, contexts)
    temporary_bootstrap = bootstrap_path.with_name(f".{bootstrap_path.name}.tmp")
    bootstrap.to_csv(temporary_bootstrap, index=False, compression="gzip")
    os.replace(temporary_bootstrap, bootstrap_path)
    atomic_csv(certificate_path, certificates)
    status = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D2_ONE_SHOT_FORMAL_EVALUATION",
        "status": "COMPLETE",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "authorization": authorization,
        "n_tasks": len(result),
        "n_truth_cells_read": int(truth_counts.sum()),
        "primary_delta_spearman": float(primary.delta_spearman),
        "primary_delta_spearman_ci95": ci["delta_spearman"],
        "primary_delta_utility_20": float(primary.delta_utility_20),
        "primary_delta_utility_20_ci95": ci["delta_utility_20"],
        "primary_relative_capture_improvement_20": float(primary.relative_capture_improvement_20),
        "gates": gates,
        "external_confirmation": bool(gates["primary_delta_spearman_ci_lower_gt_zero"]),
        "all_practical_effect_gates": bool(all(gates.values())),
        "family_identity_failures": int((np.abs(identity_residual) > tolerance).sum()),
        "family_lower_bound_violations": int(result.family_lower_bound_violation.sum()),
        "family_tightness_median": float(result.family_lower_bound_tightness.median()),
        "certificate_operational": bool(certificate_operational),
        "target_conformal_coverage": float(covered / len(target_coverage)),
        "target_conformal_coverage_ci95": list(coverage_ci),
        "target_conformal_coverage_contains_0_9": bool(coverage_ci[0] <= 0.9 <= coverage_ci[1]),
        "progeny_overlap_counts": overlap_counts,
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (task_path, summary_path, context_path, bootstrap_path, certificate_path)
        },
    }
    atomic_json(output / "E208_FORMAL_EVALUATION_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
