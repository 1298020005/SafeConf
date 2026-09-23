#!/usr/bin/env python3
"""Post-hoc stress test: is Nadig's directional result just source/target mixing?"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E238_context_baseline_audit_20260923"
E237 = ROOT / "docs/实验结果/E237_nadig_disjoint_gene_confirmation_20260923"
E139 = Path("/home/yyf/proj/docs/实验结果/E139_nadig_directional_confirmation_20260714/tables/E139_TASK_AUDIT.csv")
SCORES = ["directional_risk_frozen", "magnitude", "novelty", "disagreement",
          "magnitude_plus_novelty", "source_seen", "source_plus_magnitude_plus_novelty"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rho(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def pct(values: pd.Series):
    return rankdata(values.to_numpy(float)) / len(values)


def prepare() -> tuple[pd.DataFrame, list[dict]]:
    old = pd.read_csv(E139)
    old["cohort"] = "E139_original_96"
    old["magnitude"] = old.baseline_predicted_magnitude
    old["novelty"] = old.perturbation_novelty
    old["disagreement"] = old.risk_model_disagreement
    new_path = E237 / "E237_TASK_AUDIT.csv"
    new = pd.read_csv(new_path)
    score_path = E237 / "E237_SCORES_FIXED_BEFORE_DIRECTIONAL_EVALUATION.csv"
    scores = pd.read_csv(score_path, usecols=["fold_id", "task_id", "context_novelty_scaled"])
    new = new.merge(scores, on=["fold_id", "task_id"], validate="one_to_one")
    new["cohort"] = "E237_disjoint_128"
    for cohort in (old, new):
        cohort["source_seen"] = -cohort.context_novelty_scaled.astype(float)
        cohort["magnitude_plus_novelty"] = (cohort.groupby("fold_id").magnitude.transform(pct)
                                              + cohort.groupby("fold_id").novelty.transform(pct)) / 2
        cohort["source_plus_magnitude_plus_novelty"] = (
            cohort.groupby("fold_id").source_seen.transform(pct)
            + cohort.groupby("fold_id").magnitude.transform(pct)
            + cohort.groupby("fold_id").novelty.transform(pct)) / 3
    return pd.concat([old, new], ignore_index=True, sort=False), [
        {"cohort": "E139_original_96", "task_audit_sha256": sha256(E139)},
        {"cohort": "E237_disjoint_128", "task_audit_sha256": sha256(new_path),
         "fixed_score_sha256": sha256(score_path)},
    ]


def evaluate(frame: pd.DataFrame, draws: int = 1500) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, intervals = [], []
    rng = np.random.default_rng(20260923)
    for cohort, dataset in frame.groupby("cohort", sort=True):
        genes = sorted(dataset.perturbation.astype(str).unique())
        for scope, group in (("all_predeclared_test_tasks", dataset),
                             ("heldout_cell_line_only", dataset[dataset.setting.astype(str).str.startswith("context")])):
            folds = [part for _, part in group.groupby("fold_id", sort=True)]
            point = {}
            for score in SCORES:
                values = [rho(part[score], part.direction_error_rank_target) for part in folds]
                point[score] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
                for part, value in zip(folds, values):
                    rows.append({"cohort": cohort, "scope": scope, "fold_id": part.fold_id.iloc[0],
                                 "score": score, "n_tasks": len(part), "spearman": value})
            samples = {score: [] for score in SCORES}
            for _ in range(draws):
                counts = pd.Series(rng.choice(genes, size=len(genes), replace=True)).value_counts()
                for score in SCORES:
                    values = []
                    for part in folds:
                        weights = part.perturbation.astype(str).map(counts).fillna(0).to_numpy(int)
                        indices = np.repeat(np.arange(len(part)), weights)
                        values.append(rho(part[score].to_numpy(float)[indices],
                                          part.direction_error_rank_target.to_numpy(float)[indices]))
                    samples[score].append(float(np.nanmean(values)) if np.isfinite(values).any() else float("nan"))
            for score in SCORES:
                values = np.asarray(samples[score], float)
                for comparator in ["magnitude", "source_plus_magnitude_plus_novelty"]:
                    difference = values - np.asarray(samples[comparator], float)
                    intervals.append({"cohort": cohort, "scope": scope, "score": score,
                                      "comparator": comparator, "n_gene_clusters": len(genes),
                                      "macro_spearman": point[score],
                                      "delta": point[score] - point[comparator],
                                      "delta_ci95_low": float(np.nanquantile(difference, .025)) if np.isfinite(difference).any() else float("nan"),
                                      "delta_ci95_high": float(np.nanquantile(difference, .975)) if np.isfinite(difference).any() else float("nan")})
    return pd.DataFrame(rows), pd.DataFrame(intervals)


def alternate_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity diagnostics; these are not substituted for the frozen main endpoint."""
    record_paths = {
        "E139_original_96": Path("/home/yyf/proj/docs/实验结果/E138_nadig_formal_dual_models_20260714/Nadig_two_cellline/PREDICTION_RECORDS.csv"),
        "E237_disjoint_128": E237 / "Nadig_E237_disjoint/PREDICTION_RECORDS.csv",
    }
    e237_absolute = pd.read_csv(E237 / "Nadig_E237_disjoint/TASK_RISK_TABLE.csv",
                                usecols=["fold_id", "task_id", "error_two_predictor_mean_rmse"])
    rows = []
    for cohort, group in frame.groupby("cohort", sort=True):
        records = pd.read_csv(record_paths[cohort], usecols=["fold_id", "task_id", "true_error_cosine"])
        cosine = records.groupby(["fold_id", "task_id"], as_index=False).true_error_cosine.mean()
        joined = group.merge(cosine, on=["fold_id", "task_id"], how="inner", validate="one_to_one")
        if cohort == "E237_disjoint_128":
            joined = joined.drop(columns=["error_two_predictor_mean_rmse"]).merge(
                e237_absolute, on=["fold_id", "task_id"],
                                  how="inner", validate="one_to_one")
        if len(joined) != len(group):
            raise RuntimeError(f"alternate endpoint alignment failed for {cohort}")
        for scope, part in (("all_predeclared_test_tasks", joined),
                            ("heldout_cell_line_only", joined[joined.setting.astype(str).str.startswith("context")])):
            for endpoint in ("true_error_cosine", "error_two_predictor_mean_rmse"):
                for score in ("directional_risk_frozen", "magnitude"):
                    values = [rho(fold[score], fold[endpoint]) for _, fold in part.groupby("fold_id")]
                    rows.append({"cohort": cohort, "scope": scope, "endpoint": endpoint,
                                 "score": score, "n_tasks": len(part),
                                 "fold_macro_spearman": float(np.nanmean(values))})
    return pd.DataFrame(rows)


def main() -> None:
    frame, provenance = prepare()
    folds, comparison = evaluate(frame)
    alternate = alternate_endpoints(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT / "FOLD_SCORES.csv", index=False)
    comparison.to_csv(OUT / "COMPARISON.csv", index=False)
    alternate.to_csv(OUT / "ALTERNATE_ENDPOINTS.csv", index=False)
    (OUT / "PROVENANCE.json").write_text(json.dumps({"status": "posthoc_diagnostic", "draws": 1500,
                                                      "sources": provenance}, indent=2) + "\n")
    selected = comparison[(comparison.score == "directional_risk_frozen")
                          & comparison.comparator.isin(["magnitude", "source_plus_magnitude_plus_novelty"])]
    print(selected.to_string(index=False))
    print(alternate.to_string(index=False))


if __name__ == "__main__":
    main()
