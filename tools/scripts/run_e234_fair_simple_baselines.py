#!/usr/bin/env python3
"""Sealed, same-task retrospective comparison of E232 with simple baselines."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e225_raw_evidence_residual_ranker as scoring
import run_e230_history_capacity as capacity


METHODS = ("M", "M+D", "M+G", "M+H", "M+N", "M+C", "E230", "E232")
SIMPLE = METHODS[:6]
TRUTH_FIELDS = {"family_rms_error", "mse", "error", "truth", "target_expression"}
SEED = 20260923
BOOTSTRAPS = 1000


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def check_unique(frame: pd.DataFrame, name: str) -> None:
    if frame.task_id.isna().any() or frame.task_id.duplicated().any():
        raise ValueError(f"{name}: missing or duplicate task_id")


def rank(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values, dtype=float), method="average") / len(values)


def score(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "PRE_EVALUATION_SEAL.json").exists():
        raise RuntimeError("E234 score seal already exists; will not overwrite")
    feature_file = args.e201 / "tables/E201_PRETRUTH_RISK_FEATURES.csv"
    truth_file = args.e201 / "formal_core_evaluation/tables/E201_TASK_METRICS.csv"
    e231_file = args.e231 / "PREDICTIONS.csv.gz"
    e232_file = args.e232 / "SCORES.csv.gz"
    if sha(feature_file) != capacity.FEATURE_HASH or sha(truth_file) != capacity.METRIC_HASH:
        raise ValueError("E201 immutable input hashes changed")
    seal231 = json.loads((args.e231 / "PRE_EVALUATION_SEAL.json").read_text())
    seal232 = json.loads((args.e232 / "PRE_EVALUATION_SEAL.json").read_text())
    if sha(e231_file) != seal231["files"][e231_file.name] or sha(e232_file) != seal232["score_sha256"]:
        raise ValueError("E231/E232 score hashes do not match original seals")

    base = pd.read_csv(feature_file)
    old = pd.read_csv(e231_file)
    old = old.loc[old.variant.eq("clean"), ["task_id", "magnitude", "m_plus_d"]]
    old = old.rename(columns={"magnitude": "magnitude_e231"})
    candidate = pd.read_csv(e232_file)
    candidate = candidate.loc[candidate.scenario.eq("clean"),
                              ["task_id", "candidate", "e230", "magnitude"]]
    candidate = candidate.rename(columns={"magnitude": "magnitude_e232"})
    for frame, name in ((base, "E201"), (old, "E231"), (candidate, "E232")):
        check_unique(frame, name)
    if not (set(base.task_id) == set(old.task_id) == set(candidate.task_id)):
        raise ValueError("Input task universes differ")
    frame = base.merge(old, on="task_id", validate="one_to_one")
    frame = frame.merge(candidate, on="task_id", validate="one_to_one")
    if not np.array_equal(base.task_id.to_numpy(), frame.task_id.to_numpy()):
        raise ValueError("Task order changed")
    output = frame[["task_id", "target", "condition", "gene", "analysis_stratum"]].copy()
    columns = {
        "M+D": "family_disagreement",
        "M+G": "model_source_gap",
        "M+H": "source_delta_dispersion",
        "M+N": "negative_log_source_cells",
        "M+C": "support_context_deficit",
    }
    for target, idx in frame.groupby("target", sort=True).indices.items():
        part = frame.iloc[idx]
        m = rank(part.predicted_magnitude.to_numpy())
        if np.max(np.abs(m - part.magnitude_e231.to_numpy())) > 1e-12:
            raise ValueError(f"E231 magnitude rank mismatch: {target}")
        if np.max(np.abs(m - part.magnitude_e232.to_numpy())) > 1e-12:
            raise ValueError(f"E232 magnitude rank mismatch: {target}")
        output.loc[idx, "M"] = m
        for method, column in columns.items():
            component = part[column].to_numpy(float)
            if not np.isfinite(component).all():
                raise ValueError(f"Non-finite simple component {method}: {target}")
            output.loc[idx, method] = 0.8 * m + 0.2 * rank(component)
        if np.max(np.abs(output.loc[idx, "M+D"].to_numpy() - part.m_plus_d.to_numpy())) > 1e-12:
            raise ValueError(f"M+D independent replication mismatch: {target}")
        output.loc[idx, "E230"] = part.e230.to_numpy(float)
        output.loc[idx, "E232"] = part.candidate.to_numpy(float)
    if set(output.columns) & TRUTH_FIELDS or not np.isfinite(output[list(METHODS)].to_numpy()).all():
        raise ValueError("Sealed score table contains truth or non-finite scores")
    score_file = args.output / "SCORES.csv.gz"
    output.to_csv(score_file, index=False, compression={"method": "gzip", "mtime": 0})
    seal = {
        "status": "SEALED_NOT_EVALUATED",
        "sealed_at": datetime.now().astimezone().isoformat(),
        "score_file_sha256": sha(score_file),
        "script_sha256": sha(Path(__file__)),
        "input_sha256": {p.name: sha(p) for p in (feature_file, e231_file, e232_file)},
        "n_tasks": int(len(output)),
        "primary_tasks": int(output.analysis_stratum.eq("primary_ge30").sum()),
        "targets": sorted(output.target.unique().tolist()),
        "methods": list(METHODS),
        "truth_columns_in_scores": sorted(set(output.columns) & TRUTH_FIELDS),
        "e208_e233_test_perturbed_expression_rows_read": 0,
    }
    write_json(args.output / "PRE_EVALUATION_SEAL.json", seal)
    print(json.dumps(seal, ensure_ascii=False, indent=2))


def utility(score_values: np.ndarray, errors: np.ndarray, budget: float) -> dict:
    result = scoring.metrics(score_values, errors, budget)
    return {key: float(result[key]) for key in ("utility", "capture", "remaining_relative_error")}


def evaluate(args: argparse.Namespace) -> None:
    seal = json.loads((args.output / "PRE_EVALUATION_SEAL.json").read_text())
    scores_file = args.output / "SCORES.csv.gz"
    if sha(scores_file) != seal["score_file_sha256"]:
        raise ValueError("E234 sealed scores changed")
    truth_file = args.e201 / "formal_core_evaluation/tables/E201_TASK_METRICS.csv"
    if sha(truth_file) != capacity.METRIC_HASH:
        raise ValueError("E201 metric hash changed")
    scores = pd.read_csv(scores_file)
    truth = pd.read_csv(truth_file, usecols=["task_id", "family_rms_error"])
    check_unique(scores, "E234 sealed scores")
    check_unique(truth, "E201 truth")
    frame = scores.merge(truth, on="task_id", how="left", validate="one_to_one")
    if frame.family_rms_error.isna().any():
        raise ValueError("Missing truth for a sealed task")
    result_rows = []
    for scope in ("primary", "all"):
        scoped = frame.loc[frame.analysis_stratum.eq("primary_ge30")].copy() if scope == "primary" else frame.copy()
        for target, part in scoped.groupby("target", sort=True):
            y = part.family_rms_error.to_numpy(float)
            for method in METHODS:
                values = part[method].to_numpy(float)
                rho = float(np.corrcoef(rank(values), rank(y))[0, 1])
                for budget in (0.1, 0.2, 0.3):
                    result_rows.append({"scope": scope, "target": target, "method": method,
                                        "budget": budget, "n_tasks": len(part), "spearman": rho,
                                        **utility(values, y, budget)})
    results = pd.DataFrame(result_rows)
    results.to_csv(args.output / "RESULTS.csv", index=False)
    primary = results.loc[results.scope.eq("primary") & results.budget.eq(0.2)]
    summary = primary.groupby("method", sort=False)[["utility", "capture", "spearman"]].mean().reset_index()
    pivot = primary.pivot(index="target", columns="method", values="utility").loc[sorted(frame.target.unique())]

    # Resample genes, preserving all occurrences of each gene across the four fixed backgrounds.
    primary_frame = frame.loc[frame.analysis_stratum.eq("primary_ge30")].copy()
    genes = sorted(primary_frame.gene.unique())
    gene_lookup = {gene: i for i, gene in enumerate(genes)}
    groups = []
    for _, part in primary_frame.groupby("target", sort=True):
        groups.append((part[list(METHODS)].to_numpy(float),
                       part.family_rms_error.to_numpy(float),
                       np.asarray([gene_lookup[g] for g in part.gene], dtype=int)))
    rng = np.random.default_rng(SEED)
    baselines = [method for method in METHODS if method != "E232"]
    draws = {method: [] for method in baselines}
    for _ in range(BOOTSTRAPS):
        counts = rng.multinomial(len(genes), np.full(len(genes), 1 / len(genes)))
        method_utilities = np.zeros((len(groups), len(METHODS)))
        for group_idx, (values, errors, ids) in enumerate(groups):
            selected = np.repeat(np.arange(len(ids)), counts[ids])
            y = errors[selected]
            for method_idx in range(len(METHODS)):
                method_utilities[group_idx, method_idx] = utility(values[selected, method_idx], y, 0.2)["utility"]
        macro = method_utilities.mean(axis=0)
        for method_idx, method in enumerate(METHODS):
            if method != "E232":
                draws[method].append(float(macro[-1] - macro[method_idx]))
    intervals = []
    for method in baselines:
        samples = np.asarray(draws[method], float)
        diffs = pivot["E232"] - pivot[method]
        intervals.append({"baseline": method, "delta_E232_minus_baseline": float(diffs.mean()),
                          "ci95_lower": float(np.quantile(samples, 0.025)),
                          "ci95_upper": float(np.quantile(samples, 0.975)),
                          "positive_targets": int((diffs > 0).sum()),
                          "n_bootstraps": BOOTSTRAPS, "n_unique_genes": len(genes)})
    comparisons = pd.DataFrame(intervals)
    comparisons.to_csv(args.output / "PAIRED_INTERVALS.csv", index=False)
    summary.to_csv(args.output / "SUMMARY.csv", index=False)
    candidate_utility = float(summary.set_index("method").loc["E232", "utility"])
    simple_max = float(summary.set_index("method").loc[list(SIMPLE), "utility"].max())
    m = comparisons.set_index("baseline").loc["M"]
    passed = bool(candidate_utility > simple_max and m.ci95_lower > 0 and m.positive_targets >= 3)
    audit = {
        "status": "COMPLETE", "protocol_commit_required_before_evaluation": True,
        "development_gate": "DEVELOPMENT_SIMPLE_BASELINES_PASS" if passed else "NOT_SUPPORTED",
        "scope": "E201 primary_ge30 gene perturbations; all four targets",
        "primary_tasks": int(len(primary_frame)), "all_tasks": int(len(frame)),
        "n_unique_primary_genes": len(genes),
        "candidate_macro_utility_20": candidate_utility,
        "best_simple_baseline_macro_utility_20": simple_max,
        "best_simple_baselines": summary.loc[summary.method.isin(SIMPLE) &
                                             summary.utility.eq(simple_max), "method"].tolist(),
        "truth_sha256": sha(truth_file), "sealed_scores_sha256": sha(scores_file),
        "e208_e233_test_perturbed_expression_rows_read": 0,
        "independent_new_data_confirmation": False,
    }
    write_json(args.output / "AUDIT.json", audit)
    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("score", "evaluate"))
    parser.add_argument("--e201", type=Path, required=True)
    parser.add_argument("--e231", type=Path, required=True)
    parser.add_argument("--e232", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "score":
        score(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
