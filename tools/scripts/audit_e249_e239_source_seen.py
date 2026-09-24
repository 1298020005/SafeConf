#!/usr/bin/env python3
"""Post-hoc E239 audit against its strongest one-field context baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_macro_rho(table: pd.DataFrame, score: str, endpoint: str) -> float:
    values = []
    for _, fold in table.groupby("fold_id", sort=True):
        rho = float(spearmanr(fold[score], fold[endpoint]).statistic)
        if not np.isfinite(rho):
            raise ValueError(f"undefined correlation for {score}")
        values.append(rho)
    if len(values) != 2:
        raise ValueError(f"expected two folds, got {len(values)}")
    return float(np.mean(values))


def audit(tasks: Path, top20: Path, draws: int = 2000, seed: int = 20260924) -> dict:
    table = pd.read_csv(tasks)
    utility = pd.read_csv(top20)
    if len(table) != 332 or table.perturbation.nunique() != 128:
        raise ValueError("unexpected E239 task/gene inventory")
    if set(table.setting.unique()) != {
        "context_and_perturbation_unseen", "context_unseen",
        "perturbation_unseen", "random_seen_pair",
    }:
        raise ValueError("unexpected E239 setting inventory")
    endpoint = "direction_error_rank_target"
    risk = fold_macro_rho(table, "directional_risk_frozen", endpoint)
    context = fold_macro_rho(table, "source_seen", endpoint)
    genes = np.asarray(sorted(table.perturbation.unique()))
    by_gene = {gene: table.loc[table.perturbation.eq(gene)] for gene in genes}
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        sampled = pd.concat(
            (by_gene[g] for g in rng.choice(genes, size=len(genes), replace=True)),
            ignore_index=True,
        )
        boot[i] = (
            fold_macro_rho(sampled, "directional_risk_frozen", endpoint)
            - fold_macro_rho(sampled, "source_seen", endpoint)
        )
    # The original top-20 analysis already fixed task count, budget and utility
    # normalization; reuse its formal values rather than silently redefining it.
    piv = utility.pivot(index="fold_id", columns="score", values="normalized_directional_error_capture")
    if set(piv.index) != set(table.fold_id.unique()):
        raise ValueError("top-20 fold inventory mismatch")
    top20_delta = piv["directional_risk_frozen"] - piv["source_seen"]
    per_setting = []
    for setting, subset in table.groupby("setting", sort=True):
        record = {"setting": setting, "n_tasks": int(len(subset)), "n_genes": int(subset.perturbation.nunique())}
        for score in ("directional_risk_frozen", "magnitude"):
            fold_values = []
            for fold_id, fold in subset.groupby("fold_id", sort=True):
                rho = float(spearmanr(fold[score], fold[endpoint]).statistic)
                fold_values.append({"fold_id": fold_id, "rho": rho})
            record[score + "_within_setting_by_fold"] = fold_values
        per_setting.append(record)
    return {
        "status": "POSTHOC_DIAGNOSTIC_NOT_INDEPENDENT_CONFIRMATION",
        "input_tasks_sha256": sha256(tasks),
        "input_top20_sha256": sha256(top20),
        "n_tasks": 332, "n_gene_clusters": 128, "n_folds": 2,
        "risk_fold_macro_spearman": risk,
        "source_seen_fold_macro_spearman": context,
        "risk_minus_source_seen_spearman": risk - context,
        "risk_minus_source_seen_gene_cluster_bootstrap_ci95": [
            float(x) for x in np.quantile(boot, [0.025, 0.975])
        ],
        "bootstrap_draws": draws, "bootstrap_seed": seed,
        "top20_risk_minus_source_seen_by_fold": {
            key: float(value) for key, value in top20_delta.items()
        },
        "top20_risk_minus_source_seen_fold_mean": float(top20_delta.mean()),
        "within_setting": per_setting,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--top20", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.tasks, args.top20)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "within_setting"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
