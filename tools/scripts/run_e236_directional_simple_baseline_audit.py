#!/usr/bin/env python3
"""E236: retrospective fair-simple-baseline audit of frozen directional risk.

All scores come from the E139/E152 pre-truth files.  The target endpoints come
from archived task audits and are read only after the score keys are verified.
This is an exploratory audit of already unsealed data, NOT a new confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E236_directional_simple_baseline_audit_20260923"
KEYS = ["fold_id", "task_id", "setting", "context", "perturbation"]
ENDPOINTS = ["error_centered_pearson_mean", "error_centered_cosine_mean", "direction_error_rank_target"]
SCORES = ["directional_risk_frozen", "magnitude", "novelty", "disagreement", "magnitude_plus_novelty"]
DATASETS = {
    "Nadig": (
        "E139_nadig_directional_confirmation_20260714",
        "E139_DIRECTIONAL_SCORES_BEFORE_TRUTH.csv",
        "E139_TASK_AUDIT.csv",
        "E138_nadig_formal_dual_models_20260714/Nadig_two_cellline/TASK_RISK_TABLE.csv",
    ),
    "Replogle": (
        "E152_replogle_directional_confirmation_20260714",
        "E152_DIRECTIONAL_SCORES_BEFORE_TRUTH.csv",
        "E152_TASK_AUDIT.csv",
        "E151_replogle_formal_dual_models_20260714/Replogle_two_cellline/PRIMARY_TASK_RISK_TABLE.csv",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def percentile(values: pd.Series) -> np.ndarray:
    return rankdata(values.to_numpy(float), method="average") / len(values)


def load_dataset(name: str, asset_root: Path) -> tuple[pd.DataFrame, dict]:
    directory, score_name, audit_name, risk_name = DATASETS[name]
    score_path = ROOT / "docs/实验结果" / directory / "tables" / score_name
    audit_path = asset_root / "docs/实验结果" / directory / "tables" / audit_name
    risk_path = ROOT / "docs/实验结果" / risk_name
    if not score_path.is_file() or not audit_path.is_file() or not risk_path.is_file():
        raise FileNotFoundError(f"missing archived input: {score_path}, {audit_path}, or {risk_path}")
    scores = pd.read_csv(score_path)
    if scores.duplicated(KEYS).any() or scores.target_truth_used_for_score_or_transform.astype(bool).any():
        raise ValueError(f"{name}: pre-truth score integrity failed")
    for field in ["predicted_magnitude_z", "perturbation_novelty", "risk_disagreement_z", "directional_risk_frozen"]:
        if field not in scores or not np.isfinite(scores[field].to_numpy(float)).all():
            raise ValueError(f"{name}: missing/nonfinite pre-truth field {field}")
    audit = pd.read_csv(audit_path, usecols=KEYS + ENDPOINTS + ["directional_risk_frozen", "baseline_predicted_magnitude", "risk_model_disagreement"])
    deploy = pd.read_csv(risk_path, usecols=KEYS + ["baseline_predicted_magnitude", "risk_model_disagreement"])
    if audit.duplicated(KEYS).any() or len(audit) != len(scores):
        raise ValueError(f"{name}: task keys/count differ")
    frame = scores.merge(audit, on=KEYS, how="inner", validate="one_to_one", suffixes=("", "_audit"))
    frame = frame.merge(deploy, on=KEYS, how="inner", validate="one_to_one", suffixes=("_audit", ""))
    if len(frame) != len(scores) or not np.allclose(
        frame.directional_risk_frozen, frame.directional_risk_frozen_audit, rtol=0, atol=1e-12
    ):
        raise ValueError(f"{name}: archived audit/score mismatch")
    for field in ["baseline_predicted_magnitude", "risk_model_disagreement"]:
        if not np.allclose(frame[field], frame[f"{field}_audit"], rtol=0, atol=1e-12):
            raise ValueError(f"{name}: archived raw deployable {field} mismatch")
    frame["dataset"] = name
    frame["magnitude"] = frame.baseline_predicted_magnitude.astype(float)
    frame["novelty"] = frame.perturbation_novelty.astype(float)
    frame["disagreement"] = frame.risk_model_disagreement.astype(float)
    frame["magnitude_plus_novelty"] = frame.groupby("fold_id", sort=False)["magnitude"].transform(percentile) / 2 + frame.groupby("fold_id", sort=False)["novelty"].transform(percentile) / 2
    for endpoint in ENDPOINTS:
        if not np.isfinite(frame[endpoint].to_numpy(float)).all():
            raise ValueError(f"{name}: nonfinite endpoint {endpoint}")
    return frame, {
        "dataset": name,
        "n_rows": len(frame),
        "n_folds": int(frame.fold_id.nunique()),
        "n_gene_clusters": int(frame.perturbation.nunique()),
        "pretruth_score_sha256": sha256(score_path),
        "predictor_risk_table_sha256": sha256(risk_path),
        "archived_audit_sha256": sha256(audit_path),
    }


def summarize(frame: pd.DataFrame, draws: int, seed: int,
              scores: list[str] | None = None,
              comparators: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = SCORES if scores is None else scores
    comparators = ["magnitude", "novelty", "magnitude_plus_novelty"] if comparators is None else comparators
    rows = []
    for (dataset, fold), part in frame.groupby(["dataset", "fold_id"], sort=True):
        for endpoint in ENDPOINTS:
            target = part[endpoint].to_numpy(float)
            for score in scores:
                rows.append({"dataset": dataset, "fold_id": fold, "endpoint": endpoint, "score": score,
                             "n_tasks": len(part), "spearman": correlation(part[score].to_numpy(float), target)})
    folds = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    output = []
    for (dataset, endpoint), part in folds.groupby(["dataset", "endpoint"], sort=True):
        groups = [group.copy() for _, group in frame[frame.dataset.eq(dataset)].groupby("fold_id", sort=True)]
        genes = sorted(frame.loc[frame.dataset.eq(dataset), "perturbation"].astype(str).unique())
        point = part.groupby("score").spearman.mean().to_dict()
        sample_values = {score: [] for score in scores}
        for _ in range(draws):
            chosen = rng.choice(genes, size=len(genes), replace=True)
            counts = pd.Series(chosen).value_counts()
            for score in scores:
                fold_values = []
                for group in groups:
                    weights = group.perturbation.astype(str).map(counts).fillna(0).to_numpy(int)
                    indices = np.repeat(np.arange(len(group)), weights)
                    fold_values.append(correlation(group[score].to_numpy(float)[indices], group[endpoint].to_numpy(float)[indices]))
                sample_values[score].append(float(np.nanmean(fold_values)))
        for score in scores:
            values = np.asarray(sample_values[score])
            row = {"dataset": dataset, "endpoint": endpoint, "score": score,
                   "n_folds": len(groups), "n_gene_clusters": len(genes),
                   "fold_macro_spearman": point[score],
                   "ci95_low": float(np.nanquantile(values, .025)),
                   "ci95_high": float(np.nanquantile(values, .975))}
            for comparator in comparators:
                difference = values - np.asarray(sample_values[comparator])
                row[f"delta_vs_{comparator}"] = point[score] - point[comparator]
                row[f"delta_vs_{comparator}_ci95_low"] = float(np.nanquantile(difference, .025))
                row[f"delta_vs_{comparator}_ci95_high"] = float(np.nanquantile(difference, .975))
            output.append(row)
    return folds, pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=Path("/home/yyf/proj"))
    parser.add_argument("--draws", type=int, default=1500)
    args = parser.parse_args()
    frames, provenance = [], []
    for name in DATASETS:
        frame, record = load_dataset(name, args.asset_root)
        frames.append(frame)
        provenance.append(record)
    folds, summary = summarize(pd.concat(frames, ignore_index=True), args.draws, 20260923)
    OUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT / "FOLD_SCORES.csv", index=False)
    summary.to_csv(OUT / "COMPARISON.csv", index=False)
    (OUT / "PROVENANCE.json").write_text(json.dumps({"status": "retrospective_exploratory", "bootstrap_draws": args.draws,
                                                      "datasets": provenance}, ensure_ascii=False, indent=2) + "\n")
    print(summary[summary.endpoint.eq("direction_error_rank_target")][
        ["dataset", "score", "fold_macro_spearman", "ci95_low", "ci95_high", "delta_vs_magnitude", "delta_vs_novelty", "delta_vs_magnitude_plus_novelty"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
