#!/usr/bin/env python3
"""Independent audit and compact scientific figure for E232."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def top_weights(values: np.ndarray, fraction: float) -> np.ndarray:
    n = len(values)
    k = max(1, int(np.ceil(fraction * n)))
    threshold = np.partition(values, n - k)[n - k]
    selected = values > threshold
    ties = values == threshold
    weights = selected.astype(float)
    weights[ties] = (k - selected.sum()) / ties.sum()
    return weights


def utility(scores: np.ndarray, errors: np.ndarray, fraction: float) -> float:
    selected = top_weights(scores, fraction)
    oracle = top_weights(errors, fraction)
    average = errors.mean()
    denominator = np.dot(oracle, errors) / oracle.sum() - average
    return float((np.dot(selected, errors) / selected.sum() - average) / denominator)


def make_figure(primary: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    navy, coral, teal, grey = "#315A7D", "#D96C5F", "#208B83", "#7A8288"
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.15))

    clean = primary[primary.scenario.eq("clean")].set_index("target").loc[["K562", "RPE1", "hepg2", "jurkat"]]
    delta = clean.candidate - clean.magnitude
    axes[0].bar(np.arange(4), delta, color=[teal if x > 0 else coral for x in delta], width=0.65)
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_xticks(np.arange(4), ["K562", "RPE1", "HepG2", "Jurkat"], rotation=25)
    axes[0].set_ylabel("Utility difference")
    axes[0].set_title("Clean history vs magnitude", loc="left", fontweight="bold")
    axes[0].text(-0.18, 1.04, "a", transform=axes[0].transAxes, fontsize=11, fontweight="bold")

    records = [{"fraction": 0, "target": r.target, "seed": 0, "utility": r.candidate}
               for r in clean.reset_index().itertuples()]
    corrupt = primary[primary.scenario.str.startswith("corrupt")].copy()
    corrupt["fraction"] = corrupt.scenario.str.extract(r"corrupt_(\d+)").astype(int)
    corrupt["seed"] = corrupt.scenario.str.extract(r"seed(\d+)").astype(int)
    records.extend(corrupt[["fraction", "target", "seed", "candidate"]]
                   .rename(columns={"candidate": "utility"}).to_dict("records"))
    curve = pd.DataFrame(records).groupby("fraction").utility.agg(["mean", "std", "count"]).reset_index()
    error = curve["std"].fillna(0) / np.sqrt(curve["count"])
    axes[1].errorbar(curve.fraction, curve["mean"], yerr=error, color=coral, marker="o",
                     linewidth=1.5, markersize=4, capsize=2)
    axes[1].axhline(float(clean.magnitude.mean()), color=grey, linestyle="--", linewidth=1,
                    label="Magnitude")
    axes[1].set_xlabel("Corrupted history labels (%)")
    axes[1].set_ylabel("20% review utility")
    axes[1].set_title("History-specific information", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].text(-0.18, 1.04, "b", transform=axes[1].transAxes, fontsize=11, fontweight="bold")

    one = primary[primary.scenario.str.startswith("one_source")].candidate
    two = primary[primary.scenario.str.startswith("two_sources")].candidate
    three = clean.candidate
    values = [one.mean(), two.mean(), three.mean()]
    sem = [one.std() / np.sqrt(len(one)), two.std() / np.sqrt(len(two)), three.std() / 2]
    axes[2].bar(np.arange(3), values, yerr=sem, color=["#B8C4CC", navy, teal], width=0.65,
                error_kw={"linewidth": 0.8, "capsize": 2})
    axes[2].axhline(float(clean.magnitude.mean()), color=grey, linestyle="--", linewidth=1)
    axes[2].set_xticks(np.arange(3), ["1", "2", "3"])
    axes[2].set_xlabel("Available source backgrounds")
    axes[2].set_ylabel("20% review utility")
    axes[2].set_title("History coverage", loc="left", fontweight="bold")
    axes[2].text(-0.18, 1.04, "c", transform=axes[2].transAxes, fontsize=11, fontweight="bold")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(width=0.8, length=3)
    figure.tight_layout(w_pad=2.0)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e232", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seal = json.loads((args.e232 / "PRE_EVALUATION_SEAL.json").read_text())
    score_path = args.e232 / "SCORES.csv.gz"
    map_path = args.e232 / "CORRUPTION_MAP.csv.gz"
    if sha256(score_path) != seal["score_sha256"] or sha256(map_path) != seal["corruption_map_sha256"]:
        raise RuntimeError("E232 sealed files changed")
    scores = pd.read_csv(score_path)
    results = pd.read_csv(args.e232 / "RESULTS.csv")
    truth = pd.read_csv(args.truth).set_index("task_id").family_rms_error
    rows = []
    for (target, scenario), frame in scores.groupby(["target", "scenario"], sort=True):
        frame = frame[frame.analysis_stratum.eq("primary_ge30")]
        y = truth.loc[frame.task_id].to_numpy(float)
        for method in ("candidate", "e230", "magnitude"):
            s = frame[method].to_numpy(float)
            rows.append({
                "target": target, "scenario": scenario, "scope": "primary",
                "method": method, "budget": 0.2,
                "utility": utility(s, y, 0.2),
                "spearman": float(np.corrcoef(rankdata(s), rankdata(y))[0, 1]),
            })
    audit = pd.DataFrame(rows)
    stored = results[(results.scope.eq("primary")) & (results.budget.eq(0.2))]
    merged = stored.merge(audit, on=["target", "scenario", "scope", "method", "budget"],
                          suffixes=("_stored", "_audit"), validate="one_to_one")
    max_utility = float(np.abs(merged.utility_stored - merged.utility_audit).max())
    max_rho = float(np.abs(merged.spearman_stored - merged.spearman_audit).max())
    mappings = pd.read_csv(map_path)
    clean = scores[scores.scenario.eq("clean")]
    report = {
        "status": "PASS" if max_utility < 1e-10 and max_rho < 1e-3 else "FAIL",
        "sealed_hashes_match": True,
        "score_rows": int(len(scores)),
        "unique_tasks": int(scores.task_id.nunique()),
        "max_abs_utility_error": max_utility,
        "max_abs_spearman_error": max_rho,
        "truth_fields_in_scores": sorted(set(scores.columns) & {"family_rms_error", "mse", "error", "truth"}),
        "corruption_mapping_rows": int(len(mappings)),
        "identity_corruption_rows": int((mappings.condition == mappings.donor_condition).sum()),
        "clean_score_rows": int(len(clean)),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["status"] != "PASS" or report["truth_fields_in_scores"] or report["identity_corruption_rows"]:
        raise RuntimeError(report)
    make_figure(pd.read_csv(args.e232 / "PRIMARY_UTILITY.csv"), args.e232 / "E232_MECHANISM")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
