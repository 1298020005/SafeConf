#!/usr/bin/env python3
"""E210: audit low-complexity adaptive routing rules on released E153 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "docs/实验结果/E153_eight_study_formal_meta_20260714/tables"
    / "E153_ABSOLUTE_TASK_INPUT.csv"
)
DEFAULT_OUTPUT = ROOT / "docs/实验结果/E210_adaptive_router_development_20260914"
EXPECTED_SHA256 = "b75f5edae0bb585ba5ff18aecafcc2389b0f05fd5cc86b36960afb4b62e4a15a"
EXPECTED_ROWS = 3_465
EXPECTED_STUDIES = 8
EXPECTED_FOLDS = 34
ALPHAS = (0.60, 0.70, 0.80, 0.90, 1.00)
FORMULAS = (
    "magnitude",
    "safeconf",
    "fixed_80_20",
    "one_sided_025",
    "magnitude_quintile_safeconf",
    "setting_adaptive_lodo",
)
MASTER_SEED = 20_260_914


class AnalysisFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def average_rank(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        raise AnalysisFailure("rank input must contain at least two finite values")
    return rankdata(array, method="average") / len(array)


def spearman(score: np.ndarray, outcome: np.ndarray) -> float:
    left = average_rank(score)
    right = average_rank(outcome)
    if np.std(left) <= 0 or np.std(right) <= 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def canonical_setting(value: str) -> str:
    mapping = {
        "random_missing_pair": "random_pair",
        "random_seen_pair": "random_pair",
        "perturbation_unseen_column": "perturbation_unseen",
        "perturbation_unseen": "perturbation_unseen",
        "context_unseen_row": "context_unseen",
        "context_unseen": "context_unseen",
        "context_and_perturbation_unseen": "context_and_perturbation_unseen",
    }
    if value not in mapping:
        raise AnalysisFailure(f"unregistered setting label: {value}")
    return mapping[value]


def validate_input(path: Path) -> pd.DataFrame:
    if sha256_file(path) != EXPECTED_SHA256:
        raise AnalysisFailure("E153 input hash changed")
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "fold_id",
        "task_id",
        "setting",
        "perturbation",
        "error_two_predictor_mean_rmse",
        "safeconf_calibrated_pair_risk",
        "baseline_predicted_magnitude",
        "risk_model_disagreement",
    }
    if not required.issubset(frame.columns):
        raise AnalysisFailure(f"missing columns: {sorted(required-set(frame.columns))}")
    numeric = [
        "error_two_predictor_mean_rmse",
        "safeconf_calibrated_pair_risk",
        "baseline_predicted_magnitude",
        "risk_model_disagreement",
    ]
    if (
        len(frame) != EXPECTED_ROWS
        or frame.dataset.nunique() != EXPECTED_STUDIES
        or frame.groupby(["dataset", "fold_id"]).ngroups != EXPECTED_FOLDS
        or not np.isfinite(frame[numeric].to_numpy(float)).all()
    ):
        raise AnalysisFailure("E153 dimensions or finite-value gate failed")
    frame = frame.copy()
    frame["setting_canonical"] = frame.setting.astype(str).map(canonical_setting)
    blocks = []
    for _, block in frame.groupby(["dataset", "fold_id"], sort=True):
        block = block.copy()
        block["rank_magnitude"] = average_rank(block.baseline_predicted_magnitude)
        block["rank_safeconf"] = average_rank(block.safeconf_calibrated_pair_risk)
        block["rank_disagreement"] = average_rank(block.risk_model_disagreement)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def fixed_scores(block: pd.DataFrame) -> dict[str, np.ndarray]:
    magnitude = block.rank_magnitude.to_numpy(float)
    safeconf = block.rank_safeconf.to_numpy(float)
    magnitude_bin = np.minimum(np.ceil(5.0 * magnitude).astype(int), 5)
    return {
        "magnitude": magnitude,
        "safeconf": safeconf,
        "fixed_80_20": 0.8 * magnitude + 0.2 * safeconf,
        "one_sided_025": magnitude + 0.25 * np.maximum(safeconf - magnitude, 0.0),
        "magnitude_quintile_safeconf": magnitude_bin.astype(float) + safeconf,
    }


def macro_training_spearman(frame: pd.DataFrame, setting: str, alpha: float) -> float:
    values = []
    subset = frame.loc[frame.setting_canonical.eq(setting)]
    for _, block in subset.groupby(["dataset", "fold_id"], sort=True):
        if len(block) < 4:
            continue
        score = alpha * block.rank_magnitude.to_numpy(float) + (1.0 - alpha) * block.rank_safeconf.to_numpy(float)
        value = spearman(score, block.error_two_predictor_mean_rmse.to_numpy(float))
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise AnalysisFailure(f"no training folds for setting {setting}")
    return float(np.mean(values))


def learn_setting_alphas(frame: pd.DataFrame) -> dict[str, float]:
    output = {}
    for setting in sorted(frame.setting_canonical.unique()):
        candidates = [(macro_training_spearman(frame, setting, alpha), alpha) for alpha in ALPHAS]
        output[setting] = max(candidates, key=lambda item: (item[0], item[1]))[1]
    return output


def add_lodo_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = []
    weights = []
    for heldout in sorted(frame.dataset.unique()):
        training = frame.loc[frame.dataset.ne(heldout)]
        held = frame.loc[frame.dataset.eq(heldout)].copy()
        alphas = learn_setting_alphas(training)
        held["setting_adaptive_lodo"] = [
            alphas[setting] * magnitude + (1.0 - alphas[setting]) * safeconf
            for setting, magnitude, safeconf in zip(
                held.setting_canonical,
                held.rank_magnitude,
                held.rank_safeconf,
            )
        ]
        blocks.append(held)
        for setting, alpha in sorted(alphas.items()):
            weights.append(
                {
                    "heldout_dataset": heldout,
                    "setting": setting,
                    "alpha_magnitude": alpha,
                    "n_training_tasks": int(training.setting_canonical.eq(setting).sum()),
                    "heldout_truth_used_for_selection": False,
                }
            )
    return pd.concat(blocks, ignore_index=True), pd.DataFrame(weights)


def top20_metrics(score: np.ndarray, outcome: np.ndarray) -> tuple[float, float]:
    n = len(score)
    k = max(1, int(math.ceil(0.20 * n)))
    score_order = np.argsort(-np.asarray(score), kind="mergesort")[:k]
    truth_order = np.argsort(-np.asarray(outcome), kind="mergesort")[:k]
    overall = float(np.mean(outcome))
    enrichment = float(np.mean(np.asarray(outcome)[score_order]) / overall) if overall > 0 else float("nan")
    capture = float(len(set(score_order) & set(truth_order)) / k)
    return enrichment, capture


def study_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, study in frame.groupby("dataset", sort=True):
        fold_rows = []
        for fold_id, block in study.groupby("fold_id", sort=True):
            outcome = block.error_two_predictor_mean_rmse.to_numpy(float)
            scores = fixed_scores(block)
            scores["setting_adaptive_lodo"] = block.setting_adaptive_lodo.to_numpy(float)
            for formula, score in scores.items():
                enrichment, capture = top20_metrics(score, outcome)
                fold_rows.append(
                    {
                        "fold_id": fold_id,
                        "formula": formula,
                        "spearman": spearman(score, outcome),
                        "top20_enrichment": enrichment,
                        "top20_capture": capture,
                    }
                )
        fold_table = pd.DataFrame(fold_rows)
        for formula, block in fold_table.groupby("formula", sort=True):
            rows.append(
                {
                    "dataset": dataset,
                    "formula": formula,
                    "n_tasks": len(study),
                    "n_folds": study.fold_id.nunique(),
                    "spearman": float(block.spearman.mean()),
                    "top20_enrichment": float(block.top20_enrichment.mean()),
                    "top20_capture": float(block.top20_capture.mean()),
                }
            )
    result = pd.DataFrame(rows)
    magnitude = result.loc[result.formula.eq("magnitude"), ["dataset", "spearman"]].rename(columns={"spearman": "magnitude_spearman"})
    result = result.merge(magnitude, on="dataset", validate="many_to_one")
    result["delta_spearman_vs_magnitude"] = result.spearman - result.magnitude_spearman
    return result


def bootstrap_study(frame: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rng = np.random.default_rng(MASTER_SEED)
    rows = []
    for dataset, study in frame.groupby("dataset", sort=True):
        clusters = sorted(study.perturbation.astype(str).unique())
        members = {cluster: study.index[study.perturbation.astype(str).eq(cluster)].to_numpy() for cluster in clusters}
        for replicate in range(n_bootstrap):
            chosen = rng.choice(clusters, size=len(clusters), replace=True)
            sampled = pd.concat([study.loc[members[cluster]].copy() for cluster in chosen], ignore_index=True)
            metrics = study_metrics(sampled.assign(dataset=dataset))
            mag = float(metrics.loc[metrics.formula.eq("magnitude"), "spearman"].iloc[0])
            for formula in ("one_sided_025", "setting_adaptive_lodo"):
                value = float(metrics.loc[metrics.formula.eq(formula), "spearman"].iloc[0])
                rows.append({"dataset": dataset, "replicate": replicate, "formula": formula, "delta_spearman_vs_magnitude": value - mag})
    return pd.DataFrame(rows)


def hierarchical_summary(studies: pd.DataFrame, draws: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rng = np.random.default_rng(MASTER_SEED + 1)
    datasets = sorted(studies.dataset.unique())
    rows = []
    for formula in FORMULAS:
        block = studies.loc[studies.formula.eq(formula)]
        row = {
            "formula": formula,
            "mean_spearman": float(block.spearman.mean()),
            "mean_top20_enrichment": float(block.top20_enrichment.mean()),
            "mean_top20_capture": float(block.top20_capture.mean()),
            "mean_delta_spearman_vs_magnitude": float(block.delta_spearman_vs_magnitude.mean()),
            "n_positive_studies_vs_magnitude": int((block.delta_spearman_vs_magnitude > 0).sum()),
        }
        if formula in {"one_sided_025", "setting_adaptive_lodo"}:
            values = []
            for _ in range(n_bootstrap):
                selected = rng.choice(datasets, size=len(datasets), replace=True)
                sample = []
                for dataset in selected:
                    choices = draws.loc[(draws.dataset.eq(dataset)) & (draws.formula.eq(formula)), "delta_spearman_vs_magnitude"].to_numpy(float)
                    sample.append(float(rng.choice(choices)))
                values.append(float(np.mean(sample)))
            row["delta_ci95_low"] = float(np.quantile(values, 0.025))
            row["delta_ci95_high"] = float(np.quantile(values, 0.975))
            row["strict_development_gate"] = "PASS" if row["mean_delta_spearman_vs_magnitude"] > 0 and row["delta_ci95_low"] > 0 and row["n_positive_studies_vs_magnitude"] >= 6 else "FAIL"
        else:
            row["delta_ci95_low"] = float("nan")
            row["delta_ci95_high"] = float("nan")
            row["strict_development_gate"] = "NOT_PRIMARY"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        values = np.array([0.1, 0.2, 0.3, 0.4])
        assert np.allclose(average_rank(values), [0.25, 0.5, 0.75, 1.0])
        frame = pd.DataFrame({"rank_magnitude": values, "rank_safeconf": values[::-1]})
        assert np.all(fixed_scores(frame)["one_sided_025"] >= values)
        print("PASS")
        return
    if args.bootstrap < 100:
        raise AnalysisFailure("at least 100 bootstrap replicates are required")
    frame = validate_input(args.input.resolve())
    frame, weights = add_lodo_scores(frame)
    studies = study_metrics(frame)
    draws = bootstrap_study(frame, args.bootstrap)
    summary = hierarchical_summary(studies, draws, args.bootstrap)
    tables = args.output / "tables"
    atomic_csv(tables / "E210_STUDY_RESULTS.csv", studies)
    atomic_csv(tables / "E210_SETTING_WEIGHTS.csv", weights)
    atomic_csv(tables / "E210_BOOTSTRAP_DRAWS.csv", draws)
    atomic_csv(tables / "E210_OVERALL_SUMMARY.csv", summary)
    status = {
        "experiment": "E210_adaptive_router_development",
        "evidence_identity": "released_data_method_development",
        "input_sha256": EXPECTED_SHA256,
        "n_tasks": len(frame),
        "n_studies": frame.dataset.nunique(),
        "n_folds": frame.groupby(["dataset", "fold_id"]).ngroups,
        "bootstrap_replicates": args.bootstrap,
        "primary_formula": "one_sided_025",
        "primary_gate": summary.loc[summary.formula.eq("one_sided_025"), "strict_development_gate"].iloc[0],
        "external_confirmation": False,
        "target_truth_modified": False,
    }
    atomic_text(args.output / "RUN_STATUS.json", json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
