#!/usr/bin/env python3
"""Paired gene-cluster bootstrap for the Source/Target fusion candidates."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def u20(truth, score, ids):
    n = len(truth)
    if n < 20:
        return np.nan
    k = int(np.ceil(.2 * n))
    high = np.lexsort((ids, -score))[:k]
    oracle = np.lexsort((ids, -truth))[:k]
    denom = np.mean(truth[oracle]) - np.mean(truth)
    if denom <= 1e-12:
        return np.nan
    return float((np.mean(truth[high]) - np.mean(truth)) / denom)


def bootstrap(frame: pd.DataFrame, methods: list[str], replicates: int, seed: int):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in frame.groupby("gene", sort=True)]
    rows = []
    for draw in range(replicates):
        picked = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in picked], ignore_index=True)
        truth = sample.true_error_rmse.to_numpy(float)
        ids = np.asarray([f"{i}:{t}" for i, t in enumerate(sample.task_id.astype(str))])
        values = {m: sample[m].to_numpy(float) for m in methods}
        base = u20(truth, values[methods[0]], ids)
        for m in methods:
            score = u20(truth, values[m], ids)
            rows.append({"draw": draw, "method": m, "u20": score,
                         "delta_vs_reference": score - base if np.isfinite(score) and np.isfinite(base) else np.nan,
                         "n_rows": len(sample), "n_gene_clusters": len(groups)})
    out = pd.DataFrame(rows)
    summary = []
    for m, part in out.groupby("method", sort=False):
        score = part.u20.to_numpy(float)
        ref = part.delta_vs_reference.to_numpy(float)
        summary.append({
            "method": m,
            "bootstrap_replicates": int(len(score)),
            "u20_mean": float(np.nanmean(score)),
            "u20_ci95_lower": float(np.nanquantile(score, .025)),
            "u20_ci95_upper": float(np.nanquantile(score, .975)),
            "delta_mean_vs_reference": float(np.nanmean(ref)),
            "delta_ci95_lower": float(np.nanquantile(ref, .025)),
            "delta_ci95_upper": float(np.nanquantile(ref, .975)),
            "valid_draws": int(np.isfinite(score).sum()),
        })
    return out, pd.DataFrame(summary)


def run(input_path: Path, out: Path, reference: str, methods: list[str], replicates: int):
    out.mkdir(parents=False, exist_ok=False)
    frame = pd.read_parquet(input_path)
    methods = [reference] + [m for m in methods if m != reference]
    missing = [m for m in methods if m not in frame]
    if missing:
        raise RuntimeError(f"bootstrap input missing methods: {missing}")
    frame = frame[["task_id", "gene", "true_error_rmse", *methods]].copy()
    if frame[methods + ["true_error_rmse"]].isna().any().any():
        raise RuntimeError("paired bootstrap requires complete method rows")
    draws, summary = bootstrap(frame, methods, replicates, seed=20261004)
    draws.to_parquet(out / "BOOTSTRAP_DRAWS.parquet", index=False)
    summary.to_csv(out / "BOOTSTRAP_SUMMARY.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", default="PublicRule_simple_history")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()
    print(run(args.input, args.out, args.reference, args.methods, args.replicates).to_string(index=False))


if __name__ == "__main__":
    main()
