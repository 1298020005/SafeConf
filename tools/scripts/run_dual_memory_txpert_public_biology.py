#!/usr/bin/env python3
"""First real Dual-Memory experiment: learned public biology and shared risk.

All biological-task folds are gene-cluster disjoint and shared by GAT and
Exphormer. Public transfer models see target effects only for outer-train
tasks. The learned public bank itself never receives an upstream error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.safeconf_continual import FrozenErrorCDF, PublicMemoryStore


STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
E201 = ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
TASK_PATH = E201 / "tables/E201_PRETRUTH_TASK_BASE.csv"
STORE_ROOT = Path("/home/yyf/data/safeconf_dual_memory_20260929/public_txpert_e201")
VECTOR_ROOT = Path("/home/yyf/data/txpert_official_20260802")
OUT = STAGE / "dual_memory_continual/txpert_public_biology"
DATA_OUT = Path("/home/yyf/data/safeconf_dual_memory_20260929/txpert_public_biology")
SEED = 20260929
N_BOOTSTRAP = 5000
P = [
    "predicted_magnitude", "prediction_abs_mean", "prediction_signed_mean",
    "prediction_std", "prediction_abs_q95", "prediction_sparsity",
]
PAIR_FEATURES = [
    "log_source_cells", "log_source_batches", "control_rmse", "control_cosine",
    "source_effect_magnitude", "source_effect_abs_mean", "source_conflict",
    "support_fraction", "quality_missing",
]
PRIOR_FEATURES = [
    "prior_magnitude", "prediction_prior_rmse", "prediction_prior_cosine",
    "prior_uncertainty", "log_history_support", "effective_sources",
    "history_conflict",
]
TARGETS = ("K562", "RPE1", "hepg2", "jurkat")


class ExperimentFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-store", type=Path, default=STORE_ROOT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--data-output", type=Path, default=DATA_OUT)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    return parser.parse_args()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    compression = {"method": "gzip", "mtime": 0} if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def rmse_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(a - b), axis=-1))


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 1e-12)


def utility20(task_ids: np.ndarray, score: np.ndarray, truth: np.ndarray) -> float:
    n = len(truth)
    if n < 20:
        return float("nan")
    k = int(math.ceil(0.2 * n))
    chosen = np.lexsort((np.asarray(task_ids, str), -np.asarray(score, float)))[:k]
    oracle = np.lexsort((np.asarray(task_ids, str), -np.asarray(truth, float)))[:k]
    denominator = float(truth[oracle].mean() - truth.mean())
    return float((truth[chosen].mean() - truth.mean()) / denominator) if denominator > 1e-12 else float("nan")


def rho(score: np.ndarray, truth: np.ndarray) -> float:
    if len(score) < 3 or np.ptp(score) <= 1e-15 or np.ptp(truth) <= 1e-15:
        return float("nan")
    return float(spearmanr(score, truth).statistic)


def aurc(score: np.ndarray, truth: np.ndarray) -> float:
    order = np.argsort(score, kind="stable")
    return float(np.mean(np.cumsum(truth[order]) / np.arange(1, len(order) + 1)))


def transform_numeric(fit: pd.DataFrame, query: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x, z = fit[columns].to_numpy(float), query[columns].to_numpy(float)
    mx, mz = ~np.isfinite(x), ~np.isfinite(z)
    medians = np.asarray([
        np.nanmedian(x[:, j]) if np.isfinite(x[:, j]).any() else 0.0
        for j in range(x.shape[1])
    ])
    x, z = np.where(mx, medians, x), np.where(mz, medians, z)
    center = x.mean(axis=0)
    scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    return np.c_[(x - center) / scale, mx], np.c_[(z - center) / scale, mz]


def fit_predict_regressor(
    fit: pd.DataFrame,
    query: pd.DataFrame,
    columns: list[str],
    target: np.ndarray,
    learner: str,
) -> np.ndarray:
    x, z = transform_numeric(fit, query, columns)
    if learner == "ridge":
        model = Ridge(alpha=10.0).fit(x, target)
    elif learner == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=3,
            min_samples_leaf=20, l2_regularization=10.0,
            random_state=SEED,
        ).fit(x, target)
    else:
        raise ValueError(learner)
    return np.asarray(model.predict(z), dtype=float)


def fold_assignment(tasks: pd.DataFrame) -> np.ndarray:
    folds = np.full(len(tasks), -1, dtype=int)
    splitter = GroupKFold(5)
    for fold, (_, query) in enumerate(splitter.split(tasks, groups=tasks.gene.astype(str))):
        folds[query] = fold
    if (folds < 0).any():
        raise ExperimentFailure("incomplete biological-cluster folds")
    audit = tasks.assign(fold=folds).groupby("gene").fold.nunique()
    if not audit.eq(1).all():
        raise ExperimentFailure("same perturbation cluster crossed folds")
    return folds


def build_pairs(
    tasks: pd.DataFrame,
    truth_effect: np.ndarray,
    target_controls: np.ndarray,
    memory: pd.DataFrame,
    effects: np.ndarray,
    controls: np.ndarray,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    memory_index = memory.set_index("experiment_id").effect_vector_row.astype(int).to_dict()
    memory_by_row = memory.set_index("effect_vector_row")
    eligible = eligibility.groupby(["target_context", "condition"]).public_experiment_id.apply(list).to_dict()
    rows: list[dict[str, object]] = []
    for q, task in enumerate(tasks.itertuples(index=False)):
        ids = eligible.get((str(task.target), str(task.condition)), [])
        if not ids:
            raise ExperimentFailure(f"task has no eligible public history: {task.task_id}")
        indices = np.asarray([memory_index[item] for item in ids], dtype=int)
        candidates = np.asarray(effects[indices], dtype=float)
        candidate_controls = np.asarray(controls[indices], dtype=float)
        mean_effect = candidates.mean(axis=0)
        conflict = rmse_rows(candidates, mean_effect[None, :])
        support = memory_by_row.loc[indices]
        total_cells = max(float(support.n_cells.sum()), 1.0)
        for local, (memory_id, row_index) in enumerate(zip(ids, indices)):
            source = candidates[local]
            source_control = candidate_controls[local]
            rows.append({
                "task_row": q,
                "task_id": task.task_id,
                "target": task.target,
                "gene": task.gene,
                "fold": int(task.fold),
                "memory_id": memory_id,
                "memory_row": int(row_index),
                "log_source_cells": math.log1p(float(support.loc[row_index, "n_cells"])),
                "log_source_batches": math.log1p(float(support.loc[row_index, "n_batches"])),
                "control_rmse": float(rmse_rows(source_control[None, :], target_controls[q][None, :])[0]),
                "control_cosine": float(cosine_rows(source_control[None, :], target_controls[q][None, :])[0]),
                "source_effect_magnitude": float(np.sqrt(np.mean(np.square(source)))),
                "source_effect_abs_mean": float(np.mean(np.abs(source))),
                "source_conflict": float(conflict[local]),
                "support_fraction": float(support.loc[row_index, "n_cells"] / total_cells),
                "quality_missing": 1.0,
                "transfer_rmse": float(rmse_rows(source[None, :], truth_effect[q][None, :])[0]),
                "transfer_cosine": float(cosine_rows(source[None, :], truth_effect[q][None, :])[0]),
            })
    pairs = pd.DataFrame(rows)
    if pairs.task_id.nunique() != len(tasks) or pairs.groupby("task_id").fold.nunique().max() != 1:
        raise ExperimentFailure("pair/fold contract failed")
    return pairs


def prior_from_scores(
    task_pairs: pd.DataFrame,
    effects: np.ndarray,
    score_column: str,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    indices = task_pairs.memory_row.to_numpy(int)
    vectors = np.asarray(effects[indices], dtype=float)
    if mode == "uniform":
        weights = np.ones(len(vectors))
    elif mode == "cells":
        weights = np.expm1(task_pairs.log_source_cells.to_numpy(float))
    elif mode == "nearest":
        weights = np.zeros(len(vectors)); weights[np.argmin(task_pairs.control_rmse.to_numpy(float))] = 1.0
    elif mode in {"learned", "learned_regularized"}:
        score = task_pairs[score_column].to_numpy(float)
        spread = max(float(np.std(score)), 1e-8)
        logits = -(score - np.min(score)) / spread
        weights = np.exp(np.clip(logits, -20, 20))
        if mode == "learned_regularized":
            learned = weights / weights.sum()
            cells = np.expm1(task_pairs.log_source_cells.to_numpy(float))
            cells = cells / cells.sum()
            # One preregistered repair: fixed equal blend, never tuned on results.
            weights = 0.5 * learned + 0.5 * cells
    else:
        raise ValueError(mode)
    weights = weights / weights.sum()
    prior = np.average(vectors, axis=0, weights=weights)
    uncertainty = float(np.sqrt(np.average(np.mean(np.square(vectors - prior[None, :]), axis=1), weights=weights)))
    effective = float(1.0 / np.sum(np.square(weights)))
    return prior, weights, uncertainty, effective


def public_oof(
    tasks: pd.DataFrame,
    pairs: pd.DataFrame,
    effects: np.ndarray,
    truth_effect: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    methods = (
        "Uniform", "CellWeighted", "NearestControl", "LearnedRidge",
        "LearnedHGB", "LearnedHGBRegularized",
    )
    priors = {method: np.zeros_like(truth_effect, dtype=np.float32) for method in methods}
    summaries: list[dict[str, object]] = []
    pair_predictions: list[pd.DataFrame] = []
    for fold in range(5):
        fit_pairs = pairs[pairs.fold.ne(fold)].copy()
        query_pairs = pairs[pairs.fold.eq(fold)].copy()
        query_pairs["ridge_transfer_score"] = fit_predict_regressor(
            fit_pairs, query_pairs, PAIR_FEATURES,
            fit_pairs.transfer_rmse.to_numpy(float), "ridge",
        )
        query_pairs["hgb_transfer_score"] = fit_predict_regressor(
            fit_pairs, query_pairs, PAIR_FEATURES,
            fit_pairs.transfer_rmse.to_numpy(float), "hgb",
        )
        pair_predictions.append(query_pairs)
        for task_id, group in query_pairs.groupby("task_id", sort=False):
            q = int(group.task_row.iloc[0])
            specs = {
                "Uniform": ("hgb_transfer_score", "uniform"),
                "CellWeighted": ("hgb_transfer_score", "cells"),
                "NearestControl": ("hgb_transfer_score", "nearest"),
                "LearnedRidge": ("ridge_transfer_score", "learned"),
                "LearnedHGB": ("hgb_transfer_score", "learned"),
                "LearnedHGBRegularized": ("hgb_transfer_score", "learned_regularized"),
            }
            for method, (score, mode) in specs.items():
                prior, weights, uncertainty, effective = prior_from_scores(group, effects, score, mode)
                priors[method][q] = prior.astype(np.float32)
                summaries.append({
                    "task_id": task_id, "target": tasks.iloc[q].target,
                    "gene": tasks.iloc[q].gene, "fold": fold, "method": method,
                    "n_history": len(group), "effective_sources": effective,
                    "prior_uncertainty": uncertainty,
                    "history_support": float(np.expm1(group.log_source_cells).sum()),
                    "history_conflict": float(group.source_conflict.mean()),
                    "effect_rmse": float(rmse_rows(prior[None, :], truth_effect[q][None, :])[0]),
                    "effect_cosine": float(cosine_rows(prior[None, :], truth_effect[q][None, :])[0]),
                    "max_weight": float(weights.max()),
                })
        print(f"[PublicBiology] fold {fold} complete", flush=True)
    return priors, pd.DataFrame(summaries), pd.concat(pair_predictions, ignore_index=True)


def risk_features(
    base: pd.DataFrame,
    prediction_effect: np.ndarray,
    priors: dict[str, np.ndarray],
    summaries: pd.DataFrame,
) -> pd.DataFrame:
    output = base.copy()
    for method in ("CellWeighted", "LearnedHGB", "LearnedHGBRegularized"):
        prior = np.asarray(priors[method], dtype=float)
        summary = summaries[summaries.method.eq(method)].set_index("task_id")
        output[f"{method}_prior_magnitude"] = np.sqrt(np.mean(np.square(prior), axis=1))
        output[f"{method}_prediction_prior_rmse"] = rmse_rows(prediction_effect, prior)
        output[f"{method}_prediction_prior_cosine"] = cosine_rows(prediction_effect, prior)
        output[f"{method}_prior_uncertainty"] = output.task_id.map(summary.prior_uncertainty)
        output[f"{method}_log_history_support"] = np.log1p(output.task_id.map(summary.history_support))
        output[f"{method}_effective_sources"] = output.task_id.map(summary.effective_sources)
        output[f"{method}_history_conflict"] = output.task_id.map(summary.history_conflict)
    return output


def rank_training_labels(frame: pd.DataFrame) -> np.ndarray:
    labels = np.zeros(len(frame), dtype=float)
    for (_, target), indices in frame.groupby(["upstream", "target"]).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        labels[idx] = FrozenErrorCDF.fit(frame.loc[idx, "true_error_rmse"]).transform(
            frame.loc[idx, "true_error_rmse"]
        )
    return labels


def shared_risk_oof(frame: pd.DataFrame) -> pd.DataFrame:
    predictions: list[dict[str, object]] = []
    groups = {
        "SharedRidge_P": (P, "ridge"),
        "SharedRidge_Manual": (P + [f"CellWeighted_{x}" for x in PRIOR_FEATURES], "ridge"),
        "SharedRidge_Learned": (P + [f"LearnedHGB_{x}" for x in PRIOR_FEATURES], "ridge"),
        "SharedRidge_Regularized": (P + [f"LearnedHGBRegularized_{x}" for x in PRIOR_FEATURES], "ridge"),
        "SharedHGB_P": (P, "hgb"),
        "SharedHGB_Manual": (P + [f"CellWeighted_{x}" for x in PRIOR_FEATURES], "hgb"),
        "SharedHGB_Learned": (P + [f"LearnedHGB_{x}" for x in PRIOR_FEATURES], "hgb"),
        "SharedHGB_Regularized": (P + [f"LearnedHGBRegularized_{x}" for x in PRIOR_FEATURES], "hgb"),
    }
    for fold in range(5):
        fit = frame[frame.fold.ne(fold)].copy().reset_index(drop=True)
        query = frame[frame.fold.eq(fold)].copy().reset_index(drop=True)
        labels = rank_training_labels(fit)
        for method, (columns, learner) in groups.items():
            score = np.clip(fit_predict_regressor(fit, query, columns, labels, learner), 0, 1)
            for row, value in zip(query.itertuples(), score):
                predictions.append({
                    "task_id": row.task_id, "target": row.target, "gene": row.gene,
                    "fold": fold, "upstream": row.upstream, "method": method,
                    "predicted_risk": float(value), "true_error_rmse": row.true_error_rmse,
                })
        for row in query.itertuples():
            predictions.append({
                "task_id": row.task_id, "target": row.target, "gene": row.gene,
                "fold": fold, "upstream": row.upstream, "method": "Magnitude",
                "predicted_risk": row.predicted_magnitude,
                "true_error_rmse": row.true_error_rmse,
            })
    return pd.DataFrame(predictions)


def cross_architecture_risk(frame: pd.DataFrame) -> pd.DataFrame:
    columns = P + [f"LearnedHGB_{x}" for x in PRIOR_FEATURES]
    rows: list[dict[str, object]] = []
    upstreams = sorted(frame.upstream.unique())
    for train_upstream, test_upstream in ((upstreams[0], upstreams[1]), (upstreams[1], upstreams[0])):
        for fold in range(5):
            fit = frame[frame.upstream.eq(train_upstream) & frame.fold.ne(fold)].copy().reset_index(drop=True)
            query = frame[frame.upstream.eq(test_upstream) & frame.fold.eq(fold)].copy().reset_index(drop=True)
            labels = rank_training_labels(fit)
            score = np.clip(fit_predict_regressor(fit, query, columns, labels, "ridge"), 0, 1)
            for row, value in zip(query.itertuples(), score):
                rows.append({
                    "train_upstream": train_upstream, "test_upstream": test_upstream,
                    "task_id": row.task_id, "target": row.target, "gene": row.gene,
                    "fold": fold, "predicted_risk": float(value),
                    "true_error_rmse": row.true_error_rmse,
                })
    return pd.DataFrame(rows)


def summarize_risk(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, group in predictions.groupby(["upstream", "target", "method"], sort=True):
        upstream, target, method = keys
        rows.append({
            "upstream": upstream, "target": target, "method": method,
            "n_tasks": len(group),
            "utility20": utility20(group.task_id.to_numpy(str), group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
            "spearman": rho(group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
            "aurc": aurc(group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
        })
    strata = pd.DataFrame(rows)
    summary = strata.groupby(["upstream", "method"], as_index=False).agg(
        n_strata=("target", "nunique"), utility20=("utility20", "mean"),
        spearman=("spearman", "mean"), aurc=("aurc", "mean"),
    )
    return strata, summary


def bootstrap(predictions: pd.DataFrame, replicates: int) -> pd.DataFrame:
    comparisons = [
        ("RidgeLearned_vs_RidgeP", "SharedRidge_Learned", "SharedRidge_P"),
        ("RidgeLearned_vs_RidgeManual", "SharedRidge_Learned", "SharedRidge_Manual"),
        ("HGBLearned_vs_HGBP", "SharedHGB_Learned", "SharedHGB_P"),
        ("HGBLearned_vs_HGBManual", "SharedHGB_Learned", "SharedHGB_Manual"),
        ("HGBLearned_vs_Magnitude", "SharedHGB_Learned", "Magnitude"),
        ("HGBLearned_vs_RidgeLearned", "SharedHGB_Learned", "SharedRidge_Learned"),
        ("HGBRegularized_vs_HGBP", "SharedHGB_Regularized", "SharedHGB_P"),
        ("HGBRegularized_vs_HGBManual", "SharedHGB_Regularized", "SharedHGB_Manual"),
        ("HGBRegularized_vs_HGBLearned", "SharedHGB_Regularized", "SharedHGB_Learned"),
        ("HGBRegularized_vs_Magnitude", "SharedHGB_Regularized", "Magnitude"),
    ]
    wide = predictions.pivot(
        index=["upstream", "task_id", "target", "gene", "true_error_rmse"],
        columns="method", values="predicted_risk",
    ).reset_index()
    rng = np.random.default_rng(SEED)
    rows = []
    for upstream, part in wide.groupby("upstream", sort=True):
        genes = np.asarray(sorted(part.gene.astype(str).unique()))
        gene_rows = {gene: np.flatnonzero(part.gene.astype(str).to_numpy() == gene) for gene in genes}
        truth = part.true_error_rmse.to_numpy(float)
        targets = part.target.to_numpy(str)
        ids = part.task_id.to_numpy(str)
        arrays = {method: part[method].to_numpy(float) for method in predictions.method.unique()}
        draws = {name: [] for name, _, _ in comparisons}
        for _ in range(replicates):
            sampled = rng.integers(0, len(genes), len(genes))
            idx = np.concatenate([gene_rows[genes[i]] for i in sampled])
            metrics = {}
            for method, values in arrays.items():
                target_values = []
                for target in TARGETS:
                    use = idx[targets[idx] == target]
                    target_values.append(utility20(ids[use], values[use], truth[use]))
                metrics[method] = float(np.nanmean(target_values))
            for name, a, b in comparisons:
                draws[name].append(metrics[a] - metrics[b])
        for name, a, b in comparisons:
            values = np.asarray(draws[name])
            rows.append({
                "upstream": upstream, "comparison": name, "method_a": a, "method_b": b,
                "delta_utility20": float(np.nanmean(values)),
                "ci95_lower": float(np.nanquantile(values, 0.025)),
                "ci95_upper": float(np.nanquantile(values, 0.975)),
                "bootstrap_replicates": replicates,
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.data_output.exists():
        raise FileExistsError("refusing to overwrite Dual-Memory public-biology result")
    tasks_all = pd.read_csv(TASK_PATH)
    tasks = tasks_all[tasks_all.analysis_stratum.eq("primary_ge30")].copy().reset_index(drop=True)
    if len(tasks) != 1808:
        raise ExperimentFailure("primary task count changed")
    tasks["fold"] = fold_assignment(tasks)
    rows = tasks.source_mean_delta_row.to_numpy(int)
    truth_centroids = np.load(VECTOR_ROOT / "e201/evaluation_vectors/E201_TARGET_TRUTH_CENTROIDS.npy", mmap_mode="r")
    gat_control = np.load(VECTOR_ROOT / "e201/pretruth_vectors/E201_CONTROL_CENTROIDS.npy", mmap_mode="r")
    truth_effect = np.asarray(truth_centroids[rows], dtype=float) - np.asarray(gat_control[rows], dtype=float)
    store = PublicMemoryStore(args.public_store)
    memory, effects, controls, manifest = store.load()
    eligibility = pd.read_parquet(args.public_store / manifest["eligibility"]["path"])
    pairs = build_pairs(
        tasks, truth_effect, np.asarray(gat_control[rows], dtype=float),
        memory, effects, controls, eligibility,
    )
    priors, biology, pair_predictions = public_oof(tasks, pairs, effects, truth_effect)
    upstream_specs = {
        "TxPert_GAT": (
            STAGE / "safeconf_v4_development/txpert_gat/FEATURE_TABLE.csv.gz",
            VECTOR_ROOT / "e201/pretruth_vectors/E201_FAMILY_CENTROIDS.npy",
            VECTOR_ROOT / "e201/pretruth_vectors/E201_CONTROL_CENTROIDS.npy",
        ),
        "TxPert_Exphormer": (
            STAGE / "safeconf_v4_development/txpert_exphormer/FEATURE_TABLE.csv.gz",
            VECTOR_ROOT / "e205/pretruth_vectors/E205_FAMILY_CENTROIDS.npy",
            VECTOR_ROOT / "e205/pretruth_vectors/E205_CONTROL_CENTROIDS.npy",
        ),
    }
    risk_frames = []
    for upstream, (table_path, prediction_path, control_path) in upstream_specs.items():
        table = pd.read_csv(table_path).set_index("task_id").loc[tasks.task_id].reset_index()
        if len(table) != len(tasks):
            raise ExperimentFailure(f"upstream task alignment failed: {upstream}")
        predicted = np.asarray(np.load(prediction_path, mmap_mode="r")[rows], dtype=float)
        control = np.asarray(np.load(control_path, mmap_mode="r")[rows], dtype=float)
        table["fold"] = tasks.fold.to_numpy(int)
        table["upstream"] = upstream
        risk_frames.append(risk_features(table, predicted - control, priors, biology))
    risk_frame = pd.concat(risk_frames, ignore_index=True)
    # Exact biological tasks have one fold across all upstreams.
    if risk_frame.groupby("task_id").fold.nunique().max() != 1:
        raise ExperimentFailure("same biological task crossed upstream folds")
    predictions = shared_risk_oof(risk_frame)
    transfer = cross_architecture_risk(risk_frame)
    transfer_strata = []
    for keys, group in transfer.groupby(["train_upstream", "test_upstream", "target"], sort=True):
        train_upstream, test_upstream, target = keys
        transfer_strata.append({
            "train_upstream": train_upstream, "test_upstream": test_upstream,
            "target": target, "n_tasks": len(group),
            "utility20": utility20(group.task_id.to_numpy(str), group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
            "spearman": rho(group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
            "aurc": aurc(group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
        })
    transfer_strata = pd.DataFrame(transfer_strata)
    transfer_summary = transfer_strata.groupby(["train_upstream", "test_upstream"], as_index=False).agg(
        n_strata=("target", "nunique"), utility20=("utility20", "mean"),
        spearman=("spearman", "mean"), aurc=("aurc", "mean"),
    )
    strata, summary = summarize_risk(predictions)
    intervals = bootstrap(predictions, args.bootstrap)
    biology_summary = biology.groupby("method", as_index=False).agg(
        n_tasks=("task_id", "nunique"), effect_rmse=("effect_rmse", "mean"),
        effect_cosine=("effect_cosine", "mean"),
        effective_sources=("effective_sources", "mean"),
    )
    args.output.mkdir(parents=True)
    args.data_output.mkdir(parents=True)
    atomic_csv(args.output / "BIOLOGY_RECONSTRUCTION.csv", biology_summary)
    atomic_csv(args.output / "BIOLOGY_TASK_RESULTS.csv.gz", biology)
    atomic_csv(args.output / "PAIR_OOF_PREDICTIONS.csv.gz", pair_predictions)
    atomic_csv(args.output / "RISK_STRATUM_RESULTS.csv", strata)
    atomic_csv(args.output / "RISK_SUMMARY.csv", summary)
    atomic_csv(args.output / "RISK_INCREMENT_BOOTSTRAP.csv", intervals)
    atomic_csv(args.output / "SHARED_RISK_OOF.csv.gz", predictions)
    atomic_csv(args.output / "SHARED_RISK_FEATURES.csv.gz", risk_frame)
    atomic_csv(args.output / "CROSS_ARCHITECTURE_PREDICTIONS.csv.gz", transfer)
    atomic_csv(args.output / "CROSS_ARCHITECTURE_STRATA.csv", transfer_strata)
    atomic_csv(args.output / "CROSS_ARCHITECTURE_SUMMARY.csv", transfer_summary)
    atomic_csv(args.output / "SPLIT_MANIFEST.csv", tasks[["task_id", "target", "gene", "fold"]])
    atomic_npy(args.data_output / "PUBLIC_PRIORS.npy", np.stack([priors[k] for k in sorted(priors)]))
    atomic_json(args.data_output / "PUBLIC_PRIORS_MANIFEST.json", {
        "methods": sorted(priors), "shape": list(np.stack([priors[k] for k in sorted(priors)]).shape),
        "task_ids": tasks.task_id.tolist(), "source_public_memory_sha256": manifest["effect_vectors_sha256"],
    })
    def markdown(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in frame.itertuples(index=False, name=None):
            values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    report = [
        "# Dual-Memory Public Biology and Shared Risk", "",
        "This is a DEV/SEEN experiment. McFaline test truth remained sealed.", "",
        "## Biological reconstruction", "", markdown(biology_summary), "",
        "## Shared risk", "", markdown(summary), "",
        "## Cross-architecture transfer", "", markdown(transfer_summary), "",
        "## Paired gene-cluster bootstrap", "", markdown(intervals), "",
        "## Integrity", "",
        "- Public biological retrieval never used an upstream error as a label.",
        "- Every prediction of the same biological task used the same gene-cluster outer fold.",
        "- Error ranks were fitted only from each outer-train partition.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n")
    atomic_json(args.output / "RUN_STATUS.json", {
        "status": "COMPLETE", "evidence_role": "DEV_SEEN",
        "n_tasks": len(tasks), "n_public_items": int(manifest["n_items"]),
        "n_eligibility_edges": int(manifest["eligibility"]["n_rows"]),
        "n_upstreams": 2, "outer_folds": 5, "bootstrap_replicates": args.bootstrap,
        "same_task_all_upstreams_same_fold": True,
        "public_learner_used_upstream_errors": False,
        "mcfaline_test_opened": False,
    })
    print(biology_summary.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(intervals.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
