#!/usr/bin/env python3
"""E273 CPU audit for the SafeConf prediction/history/error-memory design.

This is a development-only experiment.  It uses only rows marked train or val
in the frozen feature matrix and creates a new perturbation-cold split by a
deterministic hash of perturbation.  No final-test rows are read.

The experiment separates:
  M: prediction magnitude anchor;
  P: prediction-side evidence;
  Q: public-history availability/quality proxies;
  H: public-history content proxies;
  E: model-error memory built from fit labels only.

The purpose is source discrimination and feasibility, not model selection by a
single pooled score.  Every dataset/predictor/fold is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


SOURCE_DEFAULT = Path(
    "/home/yyf/safeconf_runtime/outputs/safeconf_lopo_robustness_20260613/"
    "tables/LOPO_FEATURE_MATRIX_PertMeanPredictor.csv"
)

PREDICTORS = ("ContextSimBaseline", "PertMeanPredictor", "V0StrongBaseline")
PREDICTION = (
    "prediction_l2_norm",
    "prediction_abs_mean",
    "prediction_norm_ratio",
    "prediction_magnitude_deviation",
    "model_disagreement_rmse",
    "model_disagreement_cosine",
)
QUALITY = (
    "context_similarity_max",
    "context_similarity_mean",
    "perturbation_support_count",
)
HISTORY = (
    "historical_residual_risk",
    "perturbation_effect_stability",
    "perturbation_effect_variance",
)


def utility(score: np.ndarray, errors: np.ndarray) -> float:
    """Top-20% error capture normalized by the oracle top-20% capture."""
    score = np.asarray(score, dtype=float)
    errors = np.asarray(errors, dtype=float)
    keep = np.isfinite(score) & np.isfinite(errors)
    score, errors = score[keep], errors[keep]
    n = len(score)
    if n < 5:
        return float("nan")
    k = max(1, int(np.ceil(0.2 * n)))

    def selected(values: np.ndarray) -> np.ndarray:
        threshold = np.partition(values, n - k)[n - k]
        higher = values > threshold
        tied = values == threshold
        weights = higher.astype(float)
        if tied.any():
            weights[tied] = (k - int(higher.sum())) / int(tied.sum())
        return weights

    average = float(np.mean(errors))
    optimum = float(np.dot(selected(errors), errors) / k - average)
    if optimum <= 1e-12:
        return float("nan")
    return float((np.dot(selected(score), errors) / k - average) / optimum)


def bucket(value: str, n_buckets: int) -> int:
    digest = hashlib.blake2b(("E273:" + str(value)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % n_buckets


def percentile(reference: pd.Series, values: pd.Series) -> np.ndarray:
    ref = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan)
    val = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = float(ref.median()) if ref.notna().any() else 0.0
    ref_np = np.sort(ref.fillna(fill).to_numpy(float))
    val_np = val.fillna(fill).to_numpy(float)
    return np.searchsorted(ref_np, val_np, side="right") / max(1, len(ref_np))


def error_memory(fit: pd.DataFrame, query: pd.DataFrame, leave_one_out: bool) -> pd.DataFrame:
    """Build model-error memory from fit labels only.

    Perturbation-cold evaluation intentionally gives held perturbations zero
    perturbation-history support.  Context memory can still transfer across
    unseen perturbations.  For fit rows, leave-one-out prevents self-label
    leakage when the learner is trained.
    """
    y = pd.to_numeric(fit.true_error_rmse, errors="coerce").to_numpy(float)
    global_mean = float(np.nanmean(y))
    global_std = float(np.nanstd(y))
    if not np.isfinite(global_std):
        global_std = 0.0

    def aggregate(keys: pd.Series):
        temp = pd.DataFrame({"key": keys.astype(str).to_numpy(), "y": y})
        out = {}
        for key, frame in temp.groupby("key", sort=False):
            vals = frame.y.to_numpy(float)
            out[str(key)] = (
                float(vals.mean()),
                float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                int(len(vals)),
            )
        return out

    pert = aggregate(fit.perturbation)
    context = aggregate(fit.context)
    rows = []
    for _, row in query.iterrows():
        p = pert.get(str(row.perturbation), (global_mean, global_std, 0))
        c = context.get(str(row.context), (global_mean, global_std, 0))
        if leave_one_out:
            yy = float(row.true_error_rmse)
            for key, groups in ((str(row.perturbation), pert), (str(row.context), context)):
                m, s, n = groups.get(key, (global_mean, global_std, 0))
                if n > 1:
                    sumsq = float((n - 1) * s**2 + n * m**2)
                    new_n = n - 1
                    new_m = (n * m - yy) / new_n
                    new_sumsq = max(0.0, sumsq - yy**2)
                    new_s = float(np.sqrt(max(0.0, new_sumsq / new_n - new_m**2)))
                    value = (new_m, new_s, new_n)
                else:
                    value = (global_mean, global_std, 0)
                if key == str(row.perturbation):
                    p = value
                else:
                    c = value
        rows.append(
            {
                "error_hist_pert_mean": p[0],
                "error_hist_pert_std": p[1],
                "error_hist_pert_n": p[2],
                "error_hist_context_mean": c[0],
                "error_hist_context_std": c[1],
                "error_hist_context_n": c[2],
            }
        )
    return pd.DataFrame(rows, index=query.index)


def rank_features(fit: pd.DataFrame, test: pd.DataFrame, features: tuple[str, ...]):
    fit_r, test_r = fit.copy(), test.copy()
    for col in features:
        fit_r["r_" + col] = percentile(fit[col], fit[col])
        test_r["r_" + col] = percentile(fit[col], test[col])
    return fit_r, test_r


def score_learned(fit: pd.DataFrame, test: pd.DataFrame,
                  features: tuple[str, ...], kind: str) -> np.ndarray:
    cols = ["r_" + c for c in features]
    if kind == "ridge":
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            Ridge(alpha=10.0),
        )
    elif kind == "hgb":
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            HistGradientBoostingRegressor(
                max_iter=80,
                max_depth=3,
                learning_rate=0.05,
                min_samples_leaf=30,
                l2_regularization=0.5,
                early_stopping=False,
                random_state=273,
            ),
        )
    else:
        raise ValueError(kind)
    # Rank target within the fit fold so the learner is not dominated by a
    # dataset-specific error scale.  This preserves the ranking objective.
    y_rank = percentile(fit.true_error_rmse, fit.true_error_rmse)
    model.fit(fit[cols], y_rank)
    return np.asarray(model.predict(test[cols]), dtype=float)


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "dataset_name",
        "fold_id",
        "split",
        "task_key",
        "context",
        "perturbation",
        "predictor_name",
        "true_error_rmse",
        *PREDICTION,
        *QUALITY,
        *HISTORY,
    ]
    raw = pd.read_csv(args.source, usecols=list(dict.fromkeys(cols)))
    raw = raw.loc[
        raw.fold_id.eq(0)
        & raw.split.isin(("train", "val"))
        & raw.predictor_name.isin(PREDICTORS)
    ].copy()
    raw["bucket"] = raw.perturbation.map(lambda x: bucket(x, args.n_buckets))
    if raw.empty or raw.split.eq("test").any():
        raise AssertionError("E273 must use only train/val rows")

    groups: dict[str, tuple[str, ...]] = {
        "M": ("prediction_l2_norm",),
        "P": PREDICTION,
        "P_plus_Q": PREDICTION + QUALITY,
        "P_plus_Q_plus_H": PREDICTION + QUALITY + HISTORY,
        "P_plus_Q_plus_E": PREDICTION + QUALITY + (
            "error_hist_pert_mean",
            "error_hist_pert_std",
            "error_hist_pert_n",
            "error_hist_context_mean",
            "error_hist_context_std",
            "error_hist_context_n",
        ),
        "P_plus_Q_plus_H_plus_E": PREDICTION + QUALITY + HISTORY + (
            "error_hist_pert_mean",
            "error_hist_pert_std",
            "error_hist_pert_n",
            "error_hist_context_mean",
            "error_hist_context_std",
            "error_hist_context_n",
        ),
    }
    rows = []
    feature_audit = []
    for (dataset, predictor), task in raw.groupby(["dataset_name", "predictor_name"], sort=True):
        if task.task_key.duplicated().any():
            raise ValueError(f"duplicate task keys: {dataset}/{predictor}")
        for held in range(args.n_buckets):
            fit = task.loc[task.bucket.ne(held)].copy()
            test = task.loc[task.bucket.eq(held)].copy()
            if len(test) < 20 or len(fit) < 50:
                continue
            if set(fit.perturbation) & set(test.perturbation):
                raise AssertionError(f"perturbation overlap: {dataset}/{predictor}/{held}")
            fit_e = error_memory(fit, fit, leave_one_out=True)
            test_e = error_memory(fit, test, leave_one_out=False)
            fit[list(test_e.columns)] = fit_e.to_numpy()
            test[list(test_e.columns)] = test_e.to_numpy()
            all_features = tuple(dict.fromkeys(f for fs in groups.values() for f in fs))
            fit_r, test_r = rank_features(fit, test, all_features)
            y = test.true_error_rmse.to_numpy(float)
            for name, features in groups.items():
                if name == "M":
                    score = test_r.r_prediction_l2_norm.to_numpy(float)
                    learner = "rank_anchor"
                else:
                    learner = "ridge"
                    score = score_learned(fit_r, test_r, features, learner)
                rows.append(
                    {
                        "dataset": dataset,
                        "predictor": predictor,
                        "held_bucket": held,
                        "method": name,
                        "learner": learner,
                        "n_fit": len(fit),
                        "n_test": len(test),
                        "n_fit_perturbations": fit.perturbation.nunique(),
                        "n_test_perturbations": test.perturbation.nunique(),
                        "utility20": utility(score, y),
                        "spearman": float(spearmanr(score, y).statistic),
                    }
                )
            for feature in all_features:
                feature_audit.append(
                    {
                        "dataset": dataset,
                        "predictor": predictor,
                        "held_bucket": held,
                        "feature": feature,
                        "fit_missing_fraction": float(fit[feature].isna().mean()),
                        "test_missing_fraction": float(test[feature].isna().mean()),
                    }
                )
        print(f"{dataset}/{predictor}: completed", flush=True)

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("no fold produced enough rows")
    result.to_csv(args.output_dir / "FOLD_RESULTS.csv", index=False)
    pd.DataFrame(feature_audit).to_csv(args.output_dir / "FEATURE_AUDIT.csv", index=False)

    summary_rows = []
    for (dataset, predictor), frame in result.groupby(["dataset", "predictor"], sort=True):
        wide_u = frame.pivot(index="held_bucket", columns="method", values="utility20")
        wide_r = frame.pivot(index="held_bucket", columns="method", values="spearman")
        for method in [m for m in groups if m != "M"]:
            summary_rows.append(
                {
                    "dataset": dataset,
                    "predictor": predictor,
                    "method": method,
                    "n_folds": int(len(wide_u)),
                    "utility20_mean": float(wide_u[method].mean()),
                    "delta_utility20_vs_M": float((wide_u[method] - wide_u.M).mean()),
                    "positive_utility_folds_vs_M": int((wide_u[method] > wide_u.M).sum()),
                    "spearman_mean": float(wide_r[method].mean()),
                    "delta_spearman_vs_M": float((wide_r[method] - wide_r.M).mean()),
                    "positive_spearman_folds_vs_M": int((wide_r[method] > wide_r.M).sum()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "SUMMARY.csv", index=False)

    macro = (
        summary.groupby("method", as_index=False)
        .agg(
            dataset_predictor_cells=("dataset", "size"),
            utility20_mean=("utility20_mean", "mean"),
            delta_utility20_vs_M=("delta_utility20_vs_M", "mean"),
            positive_cells=("positive_utility_folds_vs_M", lambda s: int((s > 0).sum())),
            spearman_mean=("spearman_mean", "mean"),
            delta_spearman_vs_M=("delta_spearman_vs_M", "mean"),
        )
    )
    macro.to_csv(args.output_dir / "MACRO_SUMMARY.csv", index=False)
    status = {
        "status": "DEVELOPMENT_ONLY_TRAIN_VAL_PERTURBATION_COLD",
        "source": str(args.source),
        "source_rows_read": int(len(raw)),
        "source_splits": sorted(raw.split.unique().tolist()),
        "source_fold_id": 0,
        "n_datasets": int(raw.dataset_name.nunique()),
        "predictors": list(PREDICTORS),
        "n_buckets": args.n_buckets,
        "feature_groups": {
            "M": ["prediction_l2_norm"],
            "P": list(PREDICTION),
            "Q": list(QUALITY),
            "H": list(HISTORY),
            "E": [c for c in groups["P_plus_Q_plus_E"] if c.startswith("error_hist_")],
        },
        "no_final_test_rows_read": True,
        "error_memory": "fit labels only; leave-one-out for fit rows; held perturbations get zero perturbation support",
        "limits": [
            "Feature matrix is a retrospective development table; public truth is not a new blind test.",
            "Q/H provenance must be rebuilt from raw history before a confirmatory claim.",
            "This round tests feasibility and information-source separation, not final model selection.",
        ],
        "threads": {"omp": os.environ.get("OMP_NUM_THREADS"), "mkl": os.environ.get("MKL_NUM_THREADS")},
    }
    (args.output_dir / "STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print("\nMACRO SUMMARY\n" + macro.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-buckets", type=int, default=5)
    run(parser.parse_args())
