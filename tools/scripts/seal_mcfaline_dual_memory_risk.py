#!/usr/bin/env python3
"""Seal McFaline cold-start risk predictions before test treated truth is read.

Two prospectively named estimates are emitted:

* ``ZeroLabelSharedHGB`` is trained only with TxPert upstream error labels.
  McFaline biological validation effects may train the model-independent public
  transfer learner, but no McFaline upstream error labels enter the risk core.
* ``ValidationAdaptedSharedHGB`` is a conventional held-out-test comparator
  trained on McFaline validation upstream errors.

The script never opens the McFaline H5AD and therefore cannot read test truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


SEED = 20260930
P = [
    "predicted_magnitude",
    "prediction_abs_mean",
    "prediction_signed_mean",
    "prediction_std",
    "prediction_abs_q95",
    "prediction_sparsity",
]
PAIR_FEATURES = [
    "same_context",
    "same_condition",
    "control_rmse",
    "control_cosine",
    "log_source_cells",
    "log_source_guides",
    "log_source_plates",
    "guide_reproducibility",
    "plate_reproducibility",
    "split_half_stability",
    "batch_agreement",
    "source_effect_magnitude",
    "source_conflict",
    "support_fraction",
    "quality_missing",
]
PRIOR_FEATURES = [
    "prior_magnitude",
    "prediction_prior_rmse",
    "prediction_prior_cosine",
    "prior_uncertainty",
    "log_history_support",
    "effective_sources",
    "history_conflict",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(a - b), axis=1))


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 1e-12,
    )


def numeric_transform(
    fit: pd.DataFrame, query: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    x = fit[columns].to_numpy(float)
    z = query[columns].to_numpy(float)
    mx, mz = ~np.isfinite(x), ~np.isfinite(z)
    medians = np.asarray(
        [np.nanmedian(x[:, j]) if np.isfinite(x[:, j]).any() else 0.0 for j in range(x.shape[1])]
    )
    x, z = np.where(mx, medians, x), np.where(mz, medians, z)
    center = x.mean(axis=0)
    scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    return (
        np.c_[(x - center) / scale, mx],
        np.c_[(z - center) / scale, mz],
        {"median": medians.tolist(), "center": center.tolist(), "scale": scale.tolist()},
    )


def fit_model(
    fit: pd.DataFrame,
    query: pd.DataFrame,
    columns: list[str],
    target: np.ndarray,
    learner: str,
) -> tuple[np.ndarray, object, dict[str, list[float]]]:
    x, z, transform = numeric_transform(fit, query, columns)
    if learner == "ridge":
        model = Ridge(alpha=10.0).fit(x, target)
    elif learner == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=10.0,
            random_state=SEED,
        ).fit(x, target)
    else:
        raise ValueError(learner)
    return np.asarray(model.predict(z), dtype=float), model, transform


def empirical_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def prediction_features(tasks: pd.DataFrame, effect: np.ndarray) -> pd.DataFrame:
    output = tasks.copy()
    output["predicted_magnitude"] = np.sqrt(np.mean(np.square(effect), axis=1))
    output["prediction_abs_mean"] = np.mean(np.abs(effect), axis=1)
    output["prediction_signed_mean"] = np.mean(effect, axis=1)
    output["prediction_std"] = np.std(effect, axis=1)
    output["prediction_abs_q95"] = np.quantile(np.abs(effect), 0.95, axis=1)
    output["prediction_sparsity"] = np.mean(np.abs(effect) <= 1e-8, axis=1)
    return output


def build_pairs(
    tasks: pd.DataFrame,
    target_controls: np.ndarray,
    memory: pd.DataFrame,
    effects: np.ndarray,
    controls: np.ndarray,
    target_truth: np.ndarray | None,
) -> pd.DataFrame:
    by_target = memory.groupby("perturbation_target").indices
    rows: list[dict[str, object]] = []
    for q, task in enumerate(tasks.itertuples(index=False)):
        indices = np.asarray(by_target.get(str(task.perturbation), []), dtype=int)
        if len(indices):
            exact = (
                memory.iloc[indices].context.astype(str).eq(str(task.context))
                & memory.iloc[indices].condition.astype(str).eq(str(task.treatment))
            ).to_numpy()
            indices = indices[~exact]
        if not len(indices):
            continue
        candidate = np.asarray(effects[indices], dtype=float)
        candidate_controls = np.asarray(controls[indices], dtype=float)
        center = candidate.mean(axis=0)
        conflict = rmse_rows(candidate, np.repeat(center[None, :], len(candidate), axis=0))
        source = memory.iloc[indices]
        support_total = max(float(source.n_cells.sum()), 1.0)
        for local, (memory_row, item) in enumerate(zip(indices, source.itertuples(index=False))):
            row = {
                "task_row": q,
                "task_id": str(task.task_id),
                "perturbation": str(task.perturbation),
                "fold": int(task.fold),
                "memory_row": int(memory_row),
                "same_context": float(str(item.context) == str(task.context)),
                "same_condition": float(str(item.condition) == str(task.treatment)),
                "control_rmse": float(rmse_rows(candidate_controls[local : local + 1], target_controls[q : q + 1])[0]),
                "control_cosine": float(cosine_rows(candidate_controls[local : local + 1], target_controls[q : q + 1])[0]),
                "log_source_cells": math.log1p(float(item.n_cells)),
                "log_source_guides": math.log1p(float(item.n_guides)),
                "log_source_plates": math.log1p(float(item.n_plates)),
                "guide_reproducibility": float(item.guide_reproducibility),
                "plate_reproducibility": float(item.plate_reproducibility),
                "split_half_stability": float(item.split_half_stability),
                "batch_agreement": float(item.batch_agreement),
                "source_effect_magnitude": float(np.sqrt(np.mean(np.square(candidate[local])))),
                "source_conflict": float(conflict[local]),
                "support_fraction": float(item.n_cells / support_total),
                "quality_missing": float(
                    not np.isfinite(
                        [
                            item.guide_reproducibility,
                            item.plate_reproducibility,
                            item.split_half_stability,
                            item.batch_agreement,
                        ]
                    ).all()
                ),
            }
            if target_truth is not None:
                row["transfer_rmse"] = float(rmse_rows(candidate[local : local + 1], target_truth[q : q + 1])[0])
            rows.append(row)
    return pd.DataFrame(rows)


def make_prior(
    pairs: pd.DataFrame, effects: np.ndarray, score: np.ndarray | None
) -> tuple[np.ndarray, float, float, float, float]:
    vectors = np.asarray(effects[pairs.memory_row.to_numpy(int)], dtype=float)
    cells = np.expm1(pairs.log_source_cells.to_numpy(float))
    support_weights = cells / cells.sum()
    if score is None:
        weights = support_weights
    else:
        spread = max(float(np.std(score)), 1e-8)
        learned = np.exp(np.clip(-(score - np.min(score)) / spread, -20, 20))
        learned /= learned.sum()
        weights = 0.5 * learned + 0.5 * support_weights
    prior = np.average(vectors, axis=0, weights=weights)
    uncertainty = float(
        np.sqrt(np.average(np.mean(np.square(vectors - prior[None, :]), axis=1), weights=weights))
    )
    effective = float(1.0 / np.sum(np.square(weights)))
    conflict = float(pairs.source_conflict.mean())
    support = float(cells.sum())
    return prior, uncertainty, effective, conflict, support


def public_priors(
    validation_tasks: pd.DataFrame,
    test_tasks: pd.DataFrame,
    validation_truth: np.ndarray,
    validation_controls: np.ndarray,
    test_controls: np.ndarray,
    memory: pd.DataFrame,
    effects: np.ndarray,
    controls: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    val_pairs = build_pairs(
        validation_tasks, validation_controls, memory, effects, controls, validation_truth
    )
    test_pairs = build_pairs(test_tasks, test_controls, memory, effects, controls, None)
    if val_pairs.empty or test_pairs.empty:
        raise RuntimeError("no eligible McFaline public-history pairs")
    val_pairs["transfer_score"] = np.nan
    for fold in sorted(validation_tasks.fold.unique()):
        fit = val_pairs.fold.ne(fold)
        query = val_pairs.fold.eq(fold)
        score, _, _ = fit_model(
            val_pairs.loc[fit],
            val_pairs.loc[query],
            PAIR_FEATURES,
            val_pairs.loc[fit, "transfer_rmse"].to_numpy(float),
            "hgb",
        )
        val_pairs.loc[query, "transfer_score"] = score
    test_score, public_model, public_transform = fit_model(
        val_pairs,
        test_pairs,
        PAIR_FEATURES,
        val_pairs.transfer_rmse.to_numpy(float),
        "hgb",
    )
    test_pairs["transfer_score"] = test_score

    def assemble(tasks: pd.DataFrame, pairs: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
        manual = np.zeros((len(tasks), effects.shape[1]), dtype=np.float32)
        learned = np.zeros_like(manual)
        summary_rows = []
        groups = pairs.groupby("task_row")
        for q in range(len(tasks)):
            if q not in groups.groups:
                summary_rows.append({
                    "task_id": tasks.iloc[q].task_id,
                    "n_history": 0,
                    "manual_uncertainty": np.nan,
                    "manual_effective_sources": 0.0,
                    "manual_conflict": np.nan,
                    "manual_support": 0.0,
                    "learned_uncertainty": np.nan,
                    "learned_effective_sources": 0.0,
                    "learned_conflict": np.nan,
                    "learned_support": 0.0,
                })
                continue
            part = groups.get_group(q)
            a = make_prior(part, effects, None)
            b = make_prior(part, effects, part.transfer_score.to_numpy(float))
            manual[q], learned[q] = a[0], b[0]
            summary_rows.append({
                "task_id": tasks.iloc[q].task_id,
                "n_history": len(part),
                "manual_uncertainty": a[1],
                "manual_effective_sources": a[2],
                "manual_conflict": a[3],
                "manual_support": a[4],
                "learned_uncertainty": b[1],
                "learned_effective_sources": b[2],
                "learned_conflict": b[3],
                "learned_support": b[4],
            })
        return {"manual": manual, "learned": learned}, pd.DataFrame(summary_rows)

    val_prior, val_summary = assemble(validation_tasks, val_pairs)
    test_prior, test_summary = assemble(test_tasks, test_pairs)
    return val_prior, test_prior, val_summary, test_summary


def add_prior_features(
    frame: pd.DataFrame,
    prediction_effect: np.ndarray,
    priors: dict[str, np.ndarray],
    summary: pd.DataFrame,
) -> pd.DataFrame:
    output = prediction_features(frame, prediction_effect)
    summary = summary.set_index("task_id")
    for key, prefix in (("manual", "CellWeighted"), ("learned", "LearnedHGBRegularized")):
        prior = np.asarray(priors[key], dtype=float)
        output[f"{prefix}_prior_magnitude"] = np.sqrt(np.mean(np.square(prior), axis=1))
        output[f"{prefix}_prediction_prior_rmse"] = rmse_rows(prediction_effect, prior)
        output[f"{prefix}_prediction_prior_cosine"] = cosine_rows(prediction_effect, prior)
        output[f"{prefix}_prior_uncertainty"] = output.task_id.map(summary[f"{key}_uncertainty"])
        output[f"{prefix}_log_history_support"] = np.log1p(output.task_id.map(summary[f"{key}_support"]))
        output[f"{prefix}_effective_sources"] = output.task_id.map(summary[f"{key}_effective_sources"])
        output[f"{prefix}_history_conflict"] = output.task_id.map(summary[f"{key}_conflict"])
    return output


def utility20(ids: np.ndarray, score: np.ndarray, truth: np.ndarray) -> float:
    n = len(truth)
    if n < 20:
        return float("nan")
    k = int(math.ceil(0.2 * n))
    chosen = np.lexsort((ids.astype(str), -score))[:k]
    oracle = np.lexsort((ids.astype(str), -truth))[:k]
    denominator = float(truth[oracle].mean() - truth.mean())
    return float((truth[chosen].mean() - truth.mean()) / denominator) if denominator > 1e-12 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-store", type=Path, required=True)
    parser.add_argument("--validation-result", type=Path, required=True)
    parser.add_argument("--validation-repair", type=Path, required=True)
    parser.add_argument("--test-aggregate", type=Path, required=True)
    parser.add_argument("--txpert-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    memory = pd.read_parquet(args.public_store / "public_memory.parquet")
    effects = np.load(args.public_store / "effect_vectors.npy", mmap_mode="r")
    controls = np.load(args.public_store / "control_vectors.npy", mmap_mode="r")

    validation_tasks = pd.read_csv(args.validation_repair / "OOF_TASK_ERRORS.csv.gz")
    validation_tasks = validation_tasks.rename(columns={"perturbation": "perturbation"})
    validation_tasks["target"] = validation_tasks.context
    validation_effect = np.load(args.validation_repair / "OOF_REPAIRED_EFFECTS.npy")
    validation_truth = np.load(args.validation_result / "VALIDATION_TRUE_EFFECTS.npy")
    recorded = pd.read_csv(args.validation_result / "TASK_ERRORS.csv.gz")
    if not np.array_equal(validation_tasks.task_id.to_numpy(str), recorded.task_id.to_numpy(str)):
        raise RuntimeError("validation repair/result order mismatch")
    validation_tasks["true_error_rmse"] = rmse_rows(validation_effect, validation_truth)

    test_tasks = pd.read_csv(args.test_aggregate / "TEST_TASKS.csv").rename(
        columns={"condition": "perturbation", "cell_type": "context"}
    )
    test_tasks["task_id"] = (
        test_tasks.perturbation.astype(str)
        + "::"
        + test_tasks.context.astype(str)
        + "::"
        + test_tasks.treatment.astype(str)
    )
    test_tasks["target"] = test_tasks.context
    test_tasks["fold"] = -1
    test_effect = np.load(args.test_aggregate / "TEST_CALIBRATED_EFFECTS.npy")
    test_controls = np.load(args.test_aggregate / "TEST_CONTROL_STATES.npy")

    # Validation controls are the public-bank controls for the exact task.
    exact_control = memory.set_index(
        memory.perturbation_target.astype(str)
        + "::"
        + memory.context.astype(str)
        + "::"
        + memory.condition.astype(str)
    ).effect_vector_row.astype(int)
    validation_controls = np.stack(
        [np.asarray(controls[int(exact_control.loc[task])]) for task in validation_tasks.task_id]
    )

    val_priors, test_priors, val_summary, test_summary = public_priors(
        validation_tasks,
        test_tasks,
        validation_truth,
        validation_controls,
        test_controls,
        memory,
        effects,
        controls,
    )
    validation = add_prior_features(
        validation_tasks, validation_effect, val_priors, val_summary
    )
    test = add_prior_features(test_tasks, test_effect, test_priors, test_summary)

    learned_columns = P + [f"LearnedHGBRegularized_{x}" for x in PRIOR_FEATURES]
    manual_columns = P + [f"CellWeighted_{x}" for x in PRIOR_FEATURES]
    validation_rows = []
    for fold in sorted(validation.fold.unique()):
        fit = validation.fold.ne(fold)
        query = validation.fold.eq(fold)
        labels = empirical_rank(validation.loc[fit, "true_error_rmse"].to_numpy(float))
        for name, columns, learner in (
            ("UniversalP_HGB", P, "hgb"),
            ("ManualPublic_HGB", manual_columns, "hgb"),
            ("LearnedPublic_HGB", learned_columns, "hgb"),
            ("LearnedPublic_Ridge", learned_columns, "ridge"),
        ):
            score, _, _ = fit_model(
                validation.loc[fit], validation.loc[query], columns, labels, learner
            )
            for row, risk in zip(validation.loc[query].itertuples(), np.clip(score, 0, 1)):
                validation_rows.append(
                    {
                        "task_id": row.task_id,
                        "perturbation": row.perturbation,
                        "context": row.context,
                        "treatment": row.treatment,
                        "fold": fold,
                        "method": name,
                        "risk": float(risk),
                        "true_error_rmse": float(row.true_error_rmse),
                    }
                )
    validation_oof = pd.DataFrame(validation_rows)
    validation_metrics = []
    for method, group in validation_oof.groupby("method"):
        validation_metrics.append(
            {
                "method": method,
                "utility20": utility20(
                    group.task_id.to_numpy(str),
                    group.risk.to_numpy(float),
                    group.true_error_rmse.to_numpy(float),
                ),
                "spearman": float(spearmanr(group.risk, group.true_error_rmse).statistic),
            }
        )
    validation_metrics = pd.DataFrame(validation_metrics)

    test_predictions: list[pd.DataFrame] = []
    magnitude = test[["task_id", "perturbation", "context", "treatment"]].copy()
    magnitude["method"] = "Magnitude"
    magnitude["risk"] = test.predicted_magnitude.to_numpy(float)
    test_predictions.append(magnitude)

    labels = empirical_rank(validation.true_error_rmse.to_numpy(float))
    fitted_objects = {}
    for name, columns, learner in (
        ("ValidationAdapted_UniversalP_HGB", P, "hgb"),
        ("ValidationAdapted_ManualPublic_HGB", manual_columns, "hgb"),
        ("ValidationAdapted_LearnedPublic_HGB", learned_columns, "hgb"),
        ("ValidationAdapted_LearnedPublic_Ridge", learned_columns, "ridge"),
    ):
        score, model, transform = fit_model(validation, test, columns, labels, learner)
        part = test[["task_id", "perturbation", "context", "treatment"]].copy()
        part["method"] = name
        part["risk"] = np.clip(score, 0, 1)
        test_predictions.append(part)
        fitted_objects[name] = {"model": model, "transform": transform, "columns": columns}

    source = pd.read_csv(args.txpert_features)
    source_labels = np.zeros(len(source), dtype=float)
    for _, idx in source.groupby(["upstream", "target"]).groups.items():
        use = np.asarray(list(idx), dtype=int)
        source_labels[use] = empirical_rank(source.loc[use, "true_error_rmse"].to_numpy(float))
    zero_score, zero_model, zero_transform = fit_model(
        source, test, learned_columns, source_labels, "hgb"
    )
    zero = test[["task_id", "perturbation", "context", "treatment"]].copy()
    zero["method"] = "ZeroLabelSharedHGB"
    zero["risk"] = np.clip(zero_score, 0, 1)
    test_predictions.append(zero)
    fitted_objects["ZeroLabelSharedHGB"] = {
        "model": zero_model,
        "transform": zero_transform,
        "columns": learned_columns,
    }

    predictions = pd.concat(test_predictions, ignore_index=True)
    predictions.to_csv(args.output_dir / "SEALED_TEST_RISK_PREDICTIONS.csv.gz", index=False)
    validation.to_csv(args.output_dir / "VALIDATION_RISK_FEATURES.csv.gz", index=False)
    test.to_csv(args.output_dir / "TEST_RISK_FEATURES.csv.gz", index=False)
    validation_oof.to_csv(args.output_dir / "VALIDATION_RISK_OOF.csv.gz", index=False)
    validation_metrics.to_csv(args.output_dir / "VALIDATION_RISK_METRICS.csv", index=False)
    val_summary.to_csv(args.output_dir / "VALIDATION_PUBLIC_PRIOR_AUDIT.csv.gz", index=False)
    test_summary.to_csv(args.output_dir / "TEST_PUBLIC_PRIOR_AUDIT.csv.gz", index=False)
    np.save(args.output_dir / "TEST_MANUAL_PUBLIC_PRIOR.npy", test_priors["manual"].astype(np.float32))
    np.save(args.output_dir / "TEST_LEARNED_PUBLIC_PRIOR.npy", test_priors["learned"].astype(np.float32))
    joblib.dump(fitted_objects, args.output_dir / "FROZEN_RISK_MODELS.joblib")

    config = {
        "schema": "SafeConf-McFaline-ColdStart-Risk-Seal-v1",
        "status": "SEALED_BEFORE_TEST_TRUTH",
        "primary_external_method": "ZeroLabelSharedHGB",
        "secondary_external_method": "ValidationAdapted_LearnedPublic_HGB",
        "primary_definition": "Shared HGB fit with TxPert error labels only; McFaline public biology learner uses validation biological effects but no McFaline upstream errors",
        "secondary_definition": "Shared HGB fit with McFaline validation upstream errors; held-out test comparison",
        "public_learner": "HGB transfer RMSE; fixed 50:50 learned/support weights",
        "risk_learner": {
            "type": "HistGradientBoostingRegressor",
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_leaf": 20,
            "l2_regularization": 10.0,
            "random_state": SEED,
        },
        "test_treated_expression_opened": False,
        "n_test_tasks": int(len(test)),
        "n_test_clusters": int(test.perturbation.nunique()),
        "n_test_strata": int(test[["context", "treatment"]].drop_duplicates().shape[0]),
        "risk_predictions_sha256": sha256(args.output_dir / "SEALED_TEST_RISK_PREDICTIONS.csv.gz"),
        "risk_models_sha256": sha256(args.output_dir / "FROZEN_RISK_MODELS.joblib"),
        "test_features_sha256": sha256(args.output_dir / "TEST_RISK_FEATURES.csv.gz"),
        "source_txpert_features_sha256": sha256(args.txpert_features),
    }
    (args.output_dir / "FINAL_METHOD_CONFIG.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "RUN_STATUS.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "test_treated_expression_opened": False,
                "prediction_hash": config["risk_predictions_sha256"],
                "method_config_hash": sha256(args.output_dir / "FINAL_METHOD_CONFIG.json"),
            },
            indent=2,
        )
        + "\n"
    )
    print(validation_metrics.to_string(index=False), flush=True)
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
