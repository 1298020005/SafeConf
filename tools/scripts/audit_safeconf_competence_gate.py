#!/usr/bin/env python3
"""Evaluate the preregistered upstream competence gate without SafeConf scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
E201 = ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
E205 = ROOT / "docs/实验结果/E205_cross_family_disagreement_20260830"
E190 = Path("/home/yyf/proj/docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729")
E190_ASSETS = Path("/home/yyf/data/safeconf_e190_adamson_replogle/model_assets")
SEED = 20260929
N_BOOTSTRAP = 5000
MARGIN = 0.02
MIN_FRACTION = 0.60


def bootstrap_ci(frame: pd.DataFrame, model: str, baseline: str) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    genes = np.asarray(sorted(frame.gene.unique()))
    groups = [np.flatnonzero(frame.gene.to_numpy() == gene) for gene in genes]
    values = []
    for _ in range(N_BOOTSTRAP):
        chosen = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in chosen])
        values.append(float(frame.iloc[idx][model].mean() / frame.iloc[idx][baseline].mean() - 1.0))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def txpert_rows() -> list[dict]:
    e201 = pd.read_csv(E201 / "formal_core_evaluation/tables/E201_TASK_METRICS.csv")
    e201 = e201[e201.analysis_stratum.eq("primary_ge30")].copy()
    e205 = pd.read_csv(E205 / "formal_evaluation/E205_TASK_METRICS.csv")
    e205 = e205[e205.analysis_stratum.eq("primary_ge30")].copy()
    rows = []
    for name, data, model_col in [
        ("TxPert_STRING_GAT", e201, "family_centroid_rmse"),
        ("TxPert_Exphormer", e205, "family_centroid_rmse"),
    ]:
        if name == "TxPert_Exphormer":
            data = data.merge(e201[["task_id", "official_general_baseline_error"]], on="task_id", how="inner", validate="one_to_one")
            baseline_col = "official_general_baseline_error"
        else:
            baseline_col = "official_general_baseline_error"
        data = data[["task_id", "target", "gene", model_col, baseline_col]].dropna()
        data["noninferior"] = data[model_col] <= data[baseline_col] * (1.0 + MARGIN)
        by_target = data.groupby("target").agg(model_rmse=(model_col, "mean"), baseline_rmse=(baseline_col, "mean"))
        gap = float(data[model_col].mean() / data[baseline_col].mean() - 1.0)
        low, high = bootstrap_ci(data, model_col, baseline_col)
        stable_disadvantage = low > MARGIN
        rows.append({
            "upstream_model": name,
            "asset": "E201/E205",
            "baseline": "official_general_baseline",
            "n_tasks": len(data),
            "n_strata": len(by_target),
            "relative_macro_error_gap": gap,
            "noninferior_strata_fraction": float(data.groupby("target").noninferior.mean().ge(0.60).mean()),
            "bootstrap_relative_gap_ci95_lower": low,
            "bootstrap_relative_gap_ci95_upper": high,
            "bootstrap_guard": "pass" if not stable_disadvantage else "fail",
            "passes_2pct_competence_gate": bool(gap <= MARGIN and not stable_disadvantage),
            "qualified_formal_family": True,
            "notes": "Same-task official general baseline; competence evaluated before SafeConf comparison.",
        })
    return rows


def e190_rows() -> list[dict]:
    """Apply the frozen competence rule before any SafeConf-v4 E190 run.

    The upstream is the three-seed GEARS centroid defined entirely by the
    pretruth release.  The strongest registered simple baseline is the frozen
    Adamson source effect, which is stronger than zero effect on this target.
    """
    query = pd.read_csv(E190 / "E190_QUERY_MANIFEST.csv", keep_default_na=False)
    with np.load(E190 / "pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False) as archive:
        keys = sorted(key for key in archive.files if key.startswith("GEARS_"))
        if len(keys) != 3:
            raise RuntimeError("E190 GEARS pretruth family must contain exactly three members")
        prediction = np.mean([np.asarray(archive[key], np.float64) for key in keys], axis=0)
    with np.load(E190 / "evaluation_truth/arrays/TARGET_TRUE_EFFECTS.npz", allow_pickle=False) as archive:
        truth = np.stack([np.asarray(archive[task], np.float64) for task in query.task_id.astype(str)])
    with np.load(E190_ASSETS / "SOURCE_GENE_EFFECTS.npz", allow_pickle=False) as archive:
        source = np.stack([np.asarray(archive[gene], np.float64) for gene in query.gene.astype(str)])
    rmse = lambda left, right: np.sqrt(np.mean((left - right) ** 2, axis=1))
    frame = query[["task_id", "batch", "gene"]].copy()
    frame["model_error"] = rmse(prediction, truth)
    frame["baseline_error"] = rmse(source, truth)
    gap = float(frame.model_error.mean() / frame.baseline_error.mean() - 1.0)
    by_batch = frame.groupby("batch", as_index=False).agg(
        model_error=("model_error", "mean"), baseline_error=("baseline_error", "mean")
    )
    noninferior = float((by_batch.model_error <= by_batch.baseline_error * (1.0 + MARGIN)).mean())
    low, high = bootstrap_ci(frame, "model_error", "baseline_error")
    stable_disadvantage = low > MARGIN
    passed = bool(gap <= MARGIN and noninferior >= MIN_FRACTION and not stable_disadvantage)
    return [{
        "upstream_model": "GEARS_3seed_centroid",
        "asset": "E190_Adamson_to_Replogle_K562",
        "baseline": "frozen_Adamson_source_effect",
        "n_tasks": int(len(frame)),
        "n_strata": int(len(by_batch)),
        "relative_macro_error_gap": gap,
        "noninferior_strata_fraction": noninferior,
        "bootstrap_relative_gap_ci95_lower": low,
        "bootstrap_relative_gap_ci95_upper": high,
        "bootstrap_guard": "pass" if not stable_disadvantage else "fail",
        "passes_2pct_competence_gate": passed,
        "qualified_formal_family": passed,
        "notes": "Independent GEARS family and cross-study asset; already opened, so development evidence only and never confirmation.",
    }]


def released_rows() -> list[dict]:
    old = pd.read_csv(STAGE / "UPSTREAM_COMPETENCE.csv")
    rows = []
    for (asset, predictor), group in old.groupby(["asset", "predictor"], sort=True):
        # The released table has no cluster-bootstrap CI for all assets. Keep
        # the asset as pressure/development evidence rather than promote it.
        model = float(group.rmse_mean.mean())
        baseline = float(group.zero_rmse_mean.mean())
        gap = model / baseline - 1.0
        strata = group.copy()
        noninferior = float((strata.rmse_mean <= strata.zero_rmse_mean * (1 + MARGIN)).mean())
        rows.append({
            "upstream_model": predictor,
            "asset": asset,
            "baseline": "released_zero_effect",
            "n_tasks": int(group.n_records.sum()),
            "n_strata": int(len(group)),
            "relative_macro_error_gap": gap,
            "noninferior_strata_fraction": noninferior,
            "bootstrap_relative_gap_ci95_lower": np.nan,
            "bootstrap_relative_gap_ci95_upper": np.nan,
            "bootstrap_guard": "not_available",
            "passes_2pct_competence_gate": False,
            "qualified_formal_family": False,
            "notes": "Retain as development/pressure asset; no complete paired bootstrap guard in released audit.",
        })
    return rows


def main() -> None:
    rows = txpert_rows() + e190_rows() + released_rows()
    output = STAGE / "UPSTREAM_COMPETENCE_V4.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    rule = {
        "rule_version": "competence-v1",
        "strongest_simple_baseline": "registered before new-family SafeConf results",
        "primary_error_metric": "task-level RMSE on aligned effect vector",
        "relative_noninferiority_margin": MARGIN,
        "minimum_noninferior_strata_fraction": MIN_FRACTION,
        "bootstrap_replicates": N_BOOTSTRAP,
        "stable_disadvantage": "bootstrap relative gap lower CI > margin",
        "safeconf_results_used_for_selection": False,
    }
    (STAGE / "UPSTREAM_COMPETENCE_RULE.json").write_text(json.dumps(rule, indent=2) + "\n")
    report = [
        "# Upstream competence gate v1",
        "",
        "The rule was registered independently of SafeConf scores.",
        "",
        "- Non-inferiority margin: 2% relative macro RMSE.",
        "- At least 60% of valid strata must be non-inferior.",
        "- A paired cluster bootstrap lower CI above +2% is a stable disadvantage.",
        "- Released assets without a complete bootstrap guard remain pressure/development assets.",
        "",
        "See `UPSTREAM_COMPETENCE_V4.csv`.",
    ]
    (STAGE / "UPSTREAM_COMPETENCE_V4.md").write_text("\n".join(report) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
