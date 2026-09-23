#!/usr/bin/env python3
"""Independent, read-only audit of the sealed E231 outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ranked_fraction(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def utility(scores: np.ndarray, errors: np.ndarray, budget: float = 0.2) -> float:
    def top_weights(values: np.ndarray) -> np.ndarray:
        n = len(values)
        k = max(1, int(np.ceil(budget * n)))
        threshold = np.partition(values, n - k)[n - k]
        selected = values > threshold
        ties = values == threshold
        weights = selected.astype(float)
        weights[ties] = (k - selected.sum()) / ties.sum()
        return weights

    selected = top_weights(scores)
    oracle = top_weights(errors)
    average = float(errors.mean())
    denominator = float(np.dot(oracle, errors) / oracle.sum() - average)
    return float((np.dot(selected, errors) / selected.sum() - average) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e231", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seal = json.loads((args.e231 / "PRE_EVALUATION_SEAL.json").read_text())
    hash_checks = {
        name: {"expected": expected, "observed": sha256(args.e231 / name)}
        for name, expected in seal["files"].items()
    }
    if not all(v["expected"] == v["observed"] for v in hash_checks.values()):
        raise RuntimeError("A sealed E231 file changed")

    predictions = pd.read_csv(args.e231 / "PREDICTIONS.csv.gz")
    results = pd.read_csv(args.e231 / "RESULTS.csv")
    truth = pd.read_csv(args.truth).set_index("task_id")["family_rms_error"]
    methods = [c for c in predictions.columns if c not in {
        "task_id", "target", "condition", "analysis_stratum", "variant"
    }]
    recomputed = []
    for (target, variant), group in predictions.groupby(["target", "variant"], sort=True):
        group = group[group.analysis_stratum.eq("primary_ge30")]
        y = truth.loc[group.task_id].to_numpy(float)
        for method in methods:
            recomputed.append({
                "target": target,
                "variant": variant,
                "scope": "primary",
                "method": method,
                "budget": 0.2,
                "utility": utility(group[method].to_numpy(float), y),
                "spearman": float(np.corrcoef(
                    ranked_fraction(group[method].to_numpy(float)), ranked_fraction(y)
                )[0, 1]),
            })
    recomputed = pd.DataFrame(recomputed)
    stored = results[(results.scope == "primary") & (results.budget == 0.2)]
    merged = stored.merge(
        recomputed,
        on=["target", "variant", "scope", "method", "budget"],
        suffixes=("_stored", "_audit"),
        validate="one_to_one",
    )
    max_utility_error = float(np.max(np.abs(merged.utility_stored - merged.utility_audit)))
    max_spearman_error = float(np.max(np.abs(merged.spearman_stored - merged.spearman_audit)))

    duplicate_scope = []
    for source in seal["source_audits"]:
        target = source["target"]
        h5ad = args.cache / f"E201_blind_{target}" / "de_adata_test.h5ad"
        data = ad.read_h5ad(h5ad, backed="r")
        try:
            obs = data.obs[["cell_line"]].copy()
            within = int(obs.reset_index().duplicated(["cell_line", "index"]).sum())
            duplicate_scope.append({
                "target": target,
                "rows": int(len(obs)),
                "global_duplicate_obs_names": int(obs.index.duplicated().sum()),
                "within_cell_line_duplicate_obs_names": within,
            })
        finally:
            data.file.close()

    # Utility is exactly reproducible. Rank correlation may differ below 1e-4 after
    # decimal CSV round-trip turns machine-epsilon distinctions into ties.
    payload = {
        "status": "PASS" if max_utility_error < 1e-10 and max_spearman_error < 1e-3 else "FAIL",
        "sealed_file_hashes_match": True,
        "prediction_rows": int(len(predictions)),
        "unique_tasks": int(predictions.task_id.nunique()),
        "targets": sorted(predictions.target.unique().tolist()),
        "variants": sorted(predictions.variant.unique().tolist()),
        "max_abs_utility_recompute_error": max_utility_error,
        "max_abs_spearman_recompute_error": max_spearman_error,
        "duplicate_index_scope": duplicate_scope,
        "target_truth_fields_in_sealed_predictions": sorted(set(predictions.columns) & {
            "family_rms_error", "mse", "error", "truth"
        }),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if payload["status"] != "PASS" or payload["target_truth_fields_in_sealed_predictions"]:
        raise RuntimeError(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
