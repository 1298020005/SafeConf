#!/usr/bin/env python3
"""One preregistered OOF shrinkage repair for McFaline DecoderOnly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_mcfaline_validation_competence import (
    GENES,
    H5AD,
    SPLIT,
    aggregate_truth_and_baselines,
    markdown,
    rmse,
)


ALPHAS = (0.25, 0.50, 0.75, 1.00)
SEED = 20260929


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h5ad", type=Path, default=H5AD)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--genes", type=Path, default=GENES)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def fold_for(perturbation: str) -> int:
    digest = hashlib.sha256(f"SafeConf-McFaline-Decoder-repair-v1\0{perturbation}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 5


def choose_alpha(decoder: np.ndarray, baseline: np.ndarray, truth: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha in ALPHAS:
        error = float(rmse(alpha * decoder + (1 - alpha) * baseline, truth).mean())
        rows.append({"alpha": alpha, "fit_rmse": error})
    table = pd.DataFrame(rows)
    best_error = table.fit_rmse.min()
    # Stable tie handling retains more of the published upstream.
    selected = float(table.loc[np.isclose(table.fit_rmse, best_error, rtol=0, atol=1e-12), "alpha"].max())
    return selected, table


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    payload = json.loads(args.genes.read_text())
    genes = payload["gene_ids"] if isinstance(payload, dict) else payload
    meta, truth, _, state_mean, _ = aggregate_truth_and_baselines(args.h5ad, args.split, genes, 1000)
    recorded = pd.read_csv(args.decoder_result / "TASK_ERRORS.csv.gz")
    order = pd.Series(np.arange(len(meta)), index=meta.task_id).loc[recorded.task_id].to_numpy(int)
    meta, truth, state_mean = meta.iloc[order].reset_index(drop=True), truth[order], state_mean[order]
    decoder = np.load(args.decoder_result / "VALIDATION_PREDICTED_EFFECTS.npy")
    if len(decoder) != len(meta) or not np.array_equal(meta.task_id.to_numpy(str), recorded.task_id.to_numpy(str)):
        raise RuntimeError("Decoder competence artifacts do not align")
    meta["fold"] = meta.perturbation.map(fold_for)
    repaired = np.empty_like(decoder, dtype=np.float64)
    selection_rows = []
    for fold in range(5):
        fit = meta.fold.ne(fold).to_numpy()
        query = ~fit
        alpha, table = choose_alpha(decoder[fit], state_mean[fit], truth[fit])
        repaired[query] = alpha * decoder[query] + (1 - alpha) * state_mean[query]
        table["outer_fold"] = fold
        table["selected"] = table.alpha.eq(alpha)
        selection_rows.append(table)
    selection = pd.concat(selection_rows, ignore_index=True)
    task = meta.copy()
    task["repaired_rmse"] = rmse(repaired, truth)
    task["baseline_rmse"] = rmse(state_mean, truth)
    stratum = task.groupby("stratum", as_index=False).agg(
        n_tasks=("task_id", "size"), repaired_rmse=("repaired_rmse", "mean"), baseline_rmse=("baseline_rmse", "mean")
    )
    relative_gap = float((task.repaired_rmse.mean() - task.baseline_rmse.mean()) / task.baseline_rmse.mean())
    noninferior = float((stratum.repaired_rmse <= 1.02 * stratum.baseline_rmse).mean())
    perturbations = np.asarray(sorted(task.perturbation.unique()))
    indices = {p: np.flatnonzero(task.perturbation.to_numpy(str) == p) for p in perturbations}
    rng = np.random.default_rng(SEED)
    draws = []
    a, b = task.repaired_rmse.to_numpy(float), task.baseline_rmse.to_numpy(float)
    for _ in range(args.bootstrap):
        sampled = rng.integers(0, len(perturbations), len(perturbations))
        idx = np.concatenate([indices[perturbations[i]] for i in sampled])
        draws.append((a[idx].mean() - b[idx].mean()) / b[idx].mean())
    draws = np.asarray(draws)
    final_alpha, final_table = choose_alpha(decoder, state_mean, truth)
    selected_alphas = selection.loc[selection.selected, "alpha"].to_numpy(float)
    passed = bool(
        relative_gap <= 0
        and noninferior >= .60
        and np.quantile(draws, .025) <= .02
        and np.all(selected_alphas >= .25)
    )
    result = pd.DataFrame([{
        "candidate": "PerturBench_DecoderOnly_validation_shrinkage",
        "n_tasks": len(task), "n_perturbation_clusters": len(perturbations), "n_strata": len(stratum),
        "oof_repaired_macro_rmse": float(task.repaired_rmse.mean()),
        "baseline_macro_rmse": float(task.baseline_rmse.mean()), "relative_gap": relative_gap,
        "bootstrap_ci95_lower": float(np.quantile(draws, .025)),
        "bootstrap_ci95_upper": float(np.quantile(draws, .975)),
        "noninferior_strata_fraction": noninferior,
        "selected_fold_alphas": ";".join(f"{x:.2f}" for x in selected_alphas),
        "final_validation_alpha": final_alpha,
        "passes_repair_competence": passed, "test_expression_opened": False,
    }])
    task.to_csv(args.output / "OOF_TASK_ERRORS.csv.gz", index=False)
    stratum.to_csv(args.output / "OOF_STRATUM_ERRORS.csv", index=False)
    selection.to_csv(args.output / "ALPHA_SELECTION.csv", index=False)
    final_table.to_csv(args.output / "FINAL_ALPHA_SELECTION.csv", index=False)
    result.to_csv(args.output / "REPAIR_RESULT.csv", index=False)
    np.save(args.output / "OOF_REPAIRED_EFFECTS.npy", repaired.astype(np.float32))
    (args.output / "REPORT.md").write_text(
        "# McFaline DecoderOnly single shrinkage repair\n\nValidation only; test expression remained sealed.\n\n"
        + markdown(result) + "\n"
    )
    (args.output / "RUN_STATUS.json").write_text(json.dumps({
        "status": "COMPLETE", "repair": "validation-only convex effect shrinkage",
        "alphas": ALPHAS, "outer_folds": 5, "bootstrap_replicates": args.bootstrap,
        "test_expression_opened": False,
    }, indent=2) + "\n")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
