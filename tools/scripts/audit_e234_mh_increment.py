#!/usr/bin/env python3
"""Post-hoc descriptive audit for the E234-selected M+H simple candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e225_raw_evidence_residual_ranker as scoring
import run_e230_history_capacity as capacity


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value(score: np.ndarray, truth: np.ndarray, budget: float) -> float:
    return float(scoring.metrics(score, truth, budget)["utility"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e201", type=Path, required=True)
    parser.add_argument("--e234", type=Path, required=True)
    args = parser.parse_args()
    seal = json.loads((args.e234 / "PRE_EVALUATION_SEAL.json").read_text())
    scores_file = args.e234 / "SCORES.csv.gz"
    truth_file = args.e201 / "formal_core_evaluation/tables/E201_TASK_METRICS.csv"
    if sha(scores_file) != seal["score_file_sha256"] or sha(truth_file) != capacity.METRIC_HASH:
        raise ValueError("input hash mismatch")
    scores = pd.read_csv(scores_file)
    truth = pd.read_csv(truth_file, usecols=["task_id", "family_rms_error"])
    frame = scores.merge(truth, on="task_id", how="left", validate="one_to_one")
    if frame.family_rms_error.isna().any():
        raise ValueError("missing task error")
    primary = frame.loc[frame.analysis_stratum.eq("primary_ge30")].copy()
    primary["gene_index"] = pd.Categorical(primary.gene).codes
    n_genes = primary.gene.nunique()
    groups = []
    point = []
    for target, part in primary.groupby("target", sort=True):
        y = part.family_rms_error.to_numpy(float)
        m = part.M.to_numpy(float)
        mh = part["M+H"].to_numpy(float)
        delta = value(mh, y, 0.2) - value(m, y, 0.2)
        point.append({"target": target, "delta_utility_20": delta})
        groups.append((m, mh, y, part.gene_index.to_numpy(int)))
    rng = np.random.default_rng(20260923)
    draws = []
    for _ in range(2000):
        counts = rng.multinomial(n_genes, np.full(n_genes, 1 / n_genes))
        parts = []
        for m, mh, y, gene_index in groups:
            idx = np.repeat(np.arange(len(y)), counts[gene_index])
            parts.append(value(mh[idx], y[idx], 0.2) - value(m[idx], y[idx], 0.2))
        draws.append(float(np.mean(parts)))
    summary = {
        "status": "COMPLETE_POSTHOC_DESCRIPTIVE_ONLY",
        "method_selected_after_E234_truth": True,
        "n_tasks": len(primary), "n_genes": n_genes, "n_targets": len(groups),
        "target_deltas": point,
        "macro_delta_utility_20": float(np.mean([row["delta_utility_20"] for row in point])),
        "gene_cluster_ci95_lower": float(np.quantile(draws, 0.025)),
        "gene_cluster_ci95_upper": float(np.quantile(draws, 0.975)),
        "e208_e233_test_perturbed_expression_rows_read": 0,
    }
    path = args.e234 / "MH_POSTHOC_INCREMENT.json"
    if path.exists():
        raise RuntimeError("refuse overwrite")
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
