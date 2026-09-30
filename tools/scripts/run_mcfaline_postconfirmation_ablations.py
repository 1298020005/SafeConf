#!/usr/bin/env python3
"""Protocol-specified McFaline ablations after the one-shot confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluate_mcfaline_coldstart_confirmation import macro_metrics
from seal_mcfaline_dual_memory_risk import (
    P,
    PRIOR_FEATURES,
    empirical_rank,
    fit_model,
)


SEED = 20260930


def frozen_predict(frame: pd.DataFrame, payload: dict) -> np.ndarray:
    columns = payload["columns"]
    transform = payload["transform"]
    values = frame[columns].to_numpy(float)
    missing = ~np.isfinite(values)
    medians = np.asarray(transform["median"], dtype=float)
    center = np.asarray(transform["center"], dtype=float)
    scale = np.asarray(transform["scale"], dtype=float)
    values = np.where(missing, medians, values)
    matrix = np.c_[(values - center) / scale, missing]
    return np.clip(payload["model"].predict(matrix), 0, 1)


def add_metrics(
    task: pd.DataFrame, scores: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for method, values in scores.items():
        frame = task.copy()
        frame["risk"] = values
        rows.append({"method": method, **macro_metrics(frame, "risk")})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--risk-seal", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--shuffles", type=int, default=500)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    source = pd.read_csv(args.source_features)
    test = pd.read_csv(args.risk_seal / "TEST_RISK_FEATURES.csv.gz")
    truth = pd.read_csv(args.confirmation / "TEST_TASK_ERRORS.csv.gz")
    truth["stratum"] = truth.context.astype(str) + "::" + truth.treatment.astype(str)
    source_labels = np.zeros(len(source), dtype=float)
    for _, indices in source.groupby(["upstream", "target"]).groups.items():
        use = np.asarray(list(indices), dtype=int)
        source_labels[use] = empirical_rank(source.loc[use, "true_error_rmse"].to_numpy(float))

    learned_columns = P + [f"LearnedHGBRegularized_{x}" for x in PRIOR_FEATURES]
    manual_columns = P + [f"CellWeighted_{x}" for x in PRIOR_FEATURES]
    scores: dict[str, np.ndarray] = {}
    for method, columns in (
        ("ZeroLabelUniversalP_HGB", P),
        ("ZeroLabelManualPublic_HGB", manual_columns),
        ("ZeroLabelLearnedPublic_HGB", learned_columns),
    ):
        values, _, _ = fit_model(source, test, columns, source_labels, "hgb")
        scores[method] = np.clip(values, 0, 1)
    sealed = pd.read_csv(args.risk_seal / "SEALED_TEST_RISK_PREDICTIONS.csv.gz")
    sealed_primary = (
        sealed.loc[sealed.method.eq("ZeroLabelSharedHGB")]
        .set_index("task_id")
        .loc[test.task_id, "risk"]
        .to_numpy(float)
    )
    max_difference = float(
        np.max(np.abs(scores["ZeroLabelLearnedPublic_HGB"] - sealed_primary))
    )
    if max_difference > 1e-12:
        raise RuntimeError(f"sealed primary reproduction failed: {max_difference}")

    task = truth.set_index("task_id").loc[test.task_id].reset_index()
    metrics = add_metrics(task, scores)
    metrics.to_csv(args.output_dir / "ZERO_LABEL_ABLATION_METRICS.csv", index=False)
    predictions = test[["task_id", "perturbation", "context", "treatment"]].copy()
    for method, values in scores.items():
        predictions[method] = values
    predictions.to_csv(args.output_dir / "ZERO_LABEL_ABLATION_PREDICTIONS.csv.gz", index=False)

    perturbations = np.asarray(sorted(task.perturbation.unique()))
    cluster_rows = {
        value: np.flatnonzero(task.perturbation.to_numpy(str) == value)
        for value in perturbations
    }
    comparisons = {
        "Learned_vs_UniversalP": (
            "ZeroLabelLearnedPublic_HGB",
            "ZeroLabelUniversalP_HGB",
        ),
        "Learned_vs_ManualPublic": (
            "ZeroLabelLearnedPublic_HGB",
            "ZeroLabelManualPublic_HGB",
        ),
    }
    rng = np.random.default_rng(SEED)
    draws = {name: [] for name in comparisons}
    for _ in range(args.bootstrap):
        sampled = rng.integers(0, len(perturbations), len(perturbations))
        idx = np.concatenate([cluster_rows[perturbations[i]] for i in sampled])
        part = task.iloc[idx].copy()
        measured = {}
        for method, values in scores.items():
            part["risk"] = values[idx]
            measured[method] = macro_metrics(part, "risk")["utility20"]
        for name, (left, right) in comparisons.items():
            draws[name].append(measured[left] - measured[right])
    bootstrap_rows = []
    metric_index = metrics.set_index("method")
    for name, (left, right) in comparisons.items():
        values = np.asarray(draws[name])
        bootstrap_rows.append(
            {
                "comparison": name,
                "delta_utility20": float(
                    metric_index.loc[left, "utility20"]
                    - metric_index.loc[right, "utility20"]
                ),
                "ci95_lower": float(np.nanquantile(values, 0.025)),
                "ci95_upper": float(np.nanquantile(values, 0.975)),
                "bootstrap_replicates": args.bootstrap,
            }
        )
    pd.DataFrame(bootstrap_rows).to_csv(
        args.output_dir / "ZERO_LABEL_ABLATION_BOOTSTRAP.csv", index=False
    )

    frozen = joblib.load(args.risk_seal / "FROZEN_RISK_MODELS.joblib")
    primary_payload = frozen["ZeroLabelSharedHGB"]
    public_columns = [f"LearnedHGBRegularized_{x}" for x in PRIOR_FEATURES]
    content_columns = [
        column
        for column in public_columns
        if not column.endswith("log_history_support")
    ]
    shuffle_rows = []
    rng = np.random.default_rng(SEED + 1)
    actual = metric_index.loc["ZeroLabelLearnedPublic_HGB", "utility20"]
    for iteration in range(args.shuffles):
        for kind in ("within_stratum", "matched_support"):
            shuffled = test.copy()
            columns = public_columns if kind == "within_stratum" else content_columns
            if kind == "within_stratum":
                group_values = test.context.astype(str) + "::" + test.treatment.astype(str)
            else:
                support = test["LearnedHGBRegularized_log_history_support"]
                bins = pd.qcut(support, q=5, labels=False, duplicates="drop").fillna(-1).astype(int)
                group_values = (
                    test.context.astype(str)
                    + "::"
                    + test.treatment.astype(str)
                    + "::"
                    + bins.astype(str)
                )
            for _, indices in group_values.groupby(group_values).groups.items():
                use = np.asarray(list(indices), dtype=int)
                permuted = rng.permutation(use)
                shuffled.loc[use, columns] = test.loc[permuted, columns].to_numpy()
            shuffled_score = frozen_predict(shuffled, primary_payload)
            part = task.copy()
            part["risk"] = shuffled_score
            shuffle_rows.append(
                {
                    "iteration": iteration,
                    "shuffle": kind,
                    "utility20": macro_metrics(part, "risk")["utility20"],
                }
            )
    shuffles = pd.DataFrame(shuffle_rows)
    shuffles.to_csv(args.output_dir / "PUBLIC_MEMORY_SHUFFLE_DRAWS.csv.gz", index=False)
    shuffle_summary = []
    for kind, group in shuffles.groupby("shuffle"):
        shuffle_summary.append(
            {
                "shuffle": kind,
                "n": len(group),
                "actual_utility20": float(actual),
                "shuffle_mean_utility20": float(group.utility20.mean()),
                "shuffle_ci95_lower": float(group.utility20.quantile(0.025)),
                "shuffle_ci95_upper": float(group.utility20.quantile(0.975)),
                "empirical_p_ge_actual": float(
                    (1 + np.sum(group.utility20.to_numpy(float) >= actual))
                    / (1 + len(group))
                ),
            }
        )
    pd.DataFrame(shuffle_summary).to_csv(
        args.output_dir / "PUBLIC_MEMORY_SHUFFLE_SUMMARY.csv", index=False
    )

    true_effect = np.load(args.confirmation / "TEST_TRUE_EFFECTS.npy")
    candidate = np.load(args.aggregate_dir / "TEST_CALIBRATED_EFFECTS.npy")
    state_mean = np.load(args.aggregate_dir / "TEST_STATE_MEAN_EFFECTS.npy")
    method_effects = {
        "calibrated_decoder": candidate,
        "train_state_mean_effect": state_mean,
        "no_change": np.zeros_like(candidate),
    }
    competence = []
    for method, effect in method_effects.items():
        error = np.sqrt(np.mean(np.square(effect - true_effect), axis=1))
        competence.append(
            {
                "method": method,
                "macro_rmse": float(error.mean()),
                "median_rmse": float(np.median(error)),
            }
        )
    competence = pd.DataFrame(competence)
    competence.to_csv(args.output_dir / "TEST_UPSTREAM_COMPETENCE.csv", index=False)

    status = {
        "status": "COMPLETE",
        "evidence_role": "post_confirmation_protocol_specified_ablation",
        "sealed_primary_reproduction_max_abs_difference": max_difference,
        "bootstrap_replicates": args.bootstrap,
        "shuffle_replicates": args.shuffles,
        "test_truth_was_already_opened_by_confirmation": True,
        "primary_method_changed": False,
    }
    (args.output_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False), flush=True)
    print(pd.DataFrame(bootstrap_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(shuffle_summary).to_string(index=False), flush=True)
    print(competence.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
