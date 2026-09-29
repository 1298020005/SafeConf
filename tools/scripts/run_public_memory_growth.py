#!/usr/bin/env python3
"""Measure whether a growing Public Memory improves biology and risk.

Five nested, identity-hash orders are fixed independently of outcomes.  At
every memory fraction and outer fold the public transfer learner is refit using
only revealed memory items and outer-train biological tasks.  The shared risk
core is then refit with the same biological folds across upstreams.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.safeconf_continual import PublicMemoryStore
from tools.scripts.run_dual_memory_txpert_public_biology import (
    DATA_OUT,
    PAIR_FEATURES,
    P,
    PRIOR_FEATURES,
    SEED,
    STAGE,
    STORE_ROOT,
    TASK_PATH,
    TARGETS,
    VECTOR_ROOT,
    aurc,
    cosine_rows,
    fit_predict_regressor,
    prior_from_scores,
    rank_training_labels,
    rho,
    rmse_rows,
    utility20,
)


DEFAULT_SOURCE = STAGE / "dual_memory_continual/txpert_public_biology_repaired"
DEFAULT_OUTPUT = STAGE / "dual_memory_continual/public_memory_growth"
FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)
ORDERS = tuple(range(5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--public-store", type=Path, default=STORE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    compression = {"method": "gzip", "mtime": 0} if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def selected_ids(memory_ids: list[str], order: int, fraction: float) -> set[str]:
    ranked = sorted(
        memory_ids,
        key=lambda value: hashlib.sha256(f"SafeConf-memory-order-{order}::{value}".encode()).hexdigest(),
    )
    return set(ranked[: int(math.ceil(fraction * len(ranked)))])


def partial_public_prior(
    pairs: pd.DataFrame,
    effects: np.ndarray,
    selected: set[str],
    n_tasks: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    priors = np.full((n_tasks, effects.shape[1]), np.nan, dtype=np.float32)
    summaries: list[dict[str, object]] = []
    for fold in sorted(pairs.fold.unique()):
        fit = pairs[pairs.fold.ne(fold) & pairs.memory_id.isin(selected)].copy()
        query = pairs[pairs.fold.eq(fold) & pairs.memory_id.isin(selected)].copy()
        if len(fit) < 40:
            raise RuntimeError(f"insufficient revealed transfer pairs in fold {fold}: {len(fit)}")
        if not query.empty:
            query["hgb_transfer_score"] = fit_predict_regressor(
                fit, query, PAIR_FEATURES, fit.transfer_rmse.to_numpy(float), "hgb"
            )
        for task_id, group in query.groupby("task_id", sort=False):
            q = int(group.task_row.iloc[0])
            prior, weights, uncertainty, effective = prior_from_scores(
                group, effects, "hgb_transfer_score", "learned_regularized"
            )
            priors[q] = prior
            summaries.append({
                "task_id": task_id,
                "task_row": q,
                "fold": int(fold),
                "n_history": len(group),
                "effective_sources": effective,
                "prior_uncertainty": uncertainty,
                "history_support": float(np.expm1(group.log_source_cells).sum()),
                "history_conflict": float(group.source_conflict.mean()),
                "max_weight": float(weights.max()),
            })
    return priors, pd.DataFrame(summaries)


def build_risk_frame(
    base: pd.DataFrame,
    tasks: pd.DataFrame,
    priors: np.ndarray,
    summaries: pd.DataFrame,
) -> pd.DataFrame:
    prefix = "LearnedHGBRegularized"
    summary = summaries.set_index("task_id") if not summaries.empty else pd.DataFrame()
    upstream_vectors = {
        "TxPert_GAT": (
            VECTOR_ROOT / "e201/pretruth_vectors/E201_FAMILY_CENTROIDS.npy",
            VECTOR_ROOT / "e201/pretruth_vectors/E201_CONTROL_CENTROIDS.npy",
        ),
        "TxPert_Exphormer": (
            VECTOR_ROOT / "e205/pretruth_vectors/E205_FAMILY_CENTROIDS.npy",
            VECTOR_ROOT / "e205/pretruth_vectors/E205_CONTROL_CENTROIDS.npy",
        ),
    }
    rows = tasks.source_mean_delta_row.to_numpy(int)
    output = []
    available = np.isfinite(priors).all(axis=1)
    for upstream, (prediction_path, control_path) in upstream_vectors.items():
        part = base[base.upstream.eq(upstream)].set_index("task_id").loc[tasks.task_id].reset_index().copy()
        prediction = np.asarray(np.load(prediction_path, mmap_mode="r")[rows], float) - np.asarray(
            np.load(control_path, mmap_mode="r")[rows], float
        )
        magnitude = np.full(len(tasks), np.nan)
        discrepancy_rmse = np.full(len(tasks), np.nan)
        discrepancy_cosine = np.full(len(tasks), np.nan)
        magnitude[available] = np.sqrt(np.mean(np.square(priors[available]), axis=1))
        discrepancy_rmse[available] = rmse_rows(prediction[available], priors[available])
        discrepancy_cosine[available] = cosine_rows(prediction[available], priors[available])
        part[f"{prefix}_prior_magnitude"] = magnitude
        part[f"{prefix}_prediction_prior_rmse"] = discrepancy_rmse
        part[f"{prefix}_prediction_prior_cosine"] = discrepancy_cosine
        for column, source in [
            ("prior_uncertainty", "prior_uncertainty"),
            ("effective_sources", "effective_sources"),
            ("history_conflict", "history_conflict"),
        ]:
            part[f"{prefix}_{column}"] = part.task_id.map(summary[source]) if len(summary) else np.nan
        support = part.task_id.map(summary.history_support) if len(summary) else pd.Series(np.nan, index=part.index)
        part[f"{prefix}_log_history_support"] = np.log1p(support)
        output.append(part)
    return pd.concat(output, ignore_index=True)


def shared_oof(frame: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "PredictionOnly": P,
        "GrowingPublic": P + [f"LearnedHGBRegularized_{x}" for x in PRIOR_FEATURES],
    }
    rows = []
    for fold in sorted(frame.fold.unique()):
        fit = frame[frame.fold.ne(fold)].copy().reset_index(drop=True)
        query = frame[frame.fold.eq(fold)].copy().reset_index(drop=True)
        labels = rank_training_labels(fit)
        for method, columns in methods.items():
            score = np.clip(fit_predict_regressor(fit, query, columns, labels, "hgb"), 0, 1)
            for row, value in zip(query.itertuples(), score):
                rows.append({
                    "upstream": row.upstream,
                    "task_id": row.task_id,
                    "target": row.target,
                    "gene": row.gene,
                    "fold": int(fold),
                    "method": method,
                    "predicted_risk": float(value),
                    "true_error_rmse": float(row.true_error_rmse),
                })
    return pd.DataFrame(rows)


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["upstream", "target", "method"], sort=True):
        upstream, target, method = keys
        rows.append({
            "upstream": upstream,
            "target": target,
            "method": method,
            "n_tasks": len(group),
            "utility20": utility20(group.task_id.to_numpy(str), group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
            "spearman": rho(group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
            "aurc": aurc(group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
        })
    return pd.DataFrame(rows)


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    tasks_all = pd.read_csv(TASK_PATH)
    tasks = tasks_all[tasks_all.analysis_stratum.eq("primary_ge30")].copy().reset_index(drop=True)
    split = pd.read_csv(args.source / "SPLIT_MANIFEST.csv").set_index("task_id")
    tasks["fold"] = tasks.task_id.map(split.fold).astype(int)
    pairs = pd.read_csv(args.source / "PAIR_OOF_PREDICTIONS.csv.gz")
    base = pd.read_csv(args.source / "SHARED_RISK_FEATURES.csv.gz")
    store = PublicMemoryStore(args.public_store)
    memory, effects, _, manifest = store.load()
    memory_ids = memory.experiment_id.astype(str).tolist()
    truth_centroids = np.load(VECTOR_ROOT / "e201/evaluation_vectors/E201_TARGET_TRUTH_CENTROIDS.npy", mmap_mode="r")
    controls = np.load(VECTOR_ROOT / "e201/pretruth_vectors/E201_CONTROL_CENTROIDS.npy", mmap_mode="r")
    rows = tasks.source_mean_delta_row.to_numpy(int)
    truth_effect = np.asarray(truth_centroids[rows], float) - np.asarray(controls[rows], float)
    all_risk = []
    biology = []
    availability = []
    for order in ORDERS:
        for fraction in FRACTIONS:
            selected = selected_ids(memory_ids, order, fraction)
            priors, summary = partial_public_prior(pairs, effects, selected, len(tasks))
            available = np.isfinite(priors).all(axis=1)
            coverage = float(available.mean())
            availability.append({"order": order, "fraction": fraction, "n_memory_items": len(selected), "task_coverage": coverage})
            if available.any():
                biology.append({
                    "order": order,
                    "fraction": fraction,
                    "n_tasks_with_history": int(available.sum()),
                    "task_coverage": coverage,
                    "effect_rmse": float(rmse_rows(priors[available], truth_effect[available]).mean()),
                    "effect_cosine": float(cosine_rows(priors[available], truth_effect[available]).mean()),
                    "effective_sources": float(summary.effective_sources.mean()),
                })
            risk_frame = build_risk_frame(base, tasks, priors, summary)
            result = metric_rows(shared_oof(risk_frame))
            result["order"] = order
            result["fraction"] = fraction
            result["task_coverage"] = coverage
            all_risk.append(result)
            print(f"[MemoryGrowth] order={order} fraction={fraction:.2f} coverage={coverage:.3f}", flush=True)
    risk = pd.concat(all_risk, ignore_index=True)
    biology_frame = pd.DataFrame(biology)
    availability_frame = pd.DataFrame(availability)
    macro = risk.groupby(["order", "fraction", "upstream", "method"], as_index=False).agg(
        utility20=("utility20", "mean"), spearman=("spearman", "mean"), aurc=("aurc", "mean"), task_coverage=("task_coverage", "first")
    )
    curve = macro.groupby(["fraction", "upstream", "method"], as_index=False).agg(
        mean_utility20=("utility20", "mean"), worst_utility20=("utility20", "min"),
        mean_spearman=("spearman", "mean"), worst_spearman=("spearman", "min"),
        mean_aurc=("aurc", "mean"), worst_aurc=("aurc", "max"),
        mean_task_coverage=("task_coverage", "mean"), worst_task_coverage=("task_coverage", "min"),
    )
    atomic_csv(args.output / "MEMORY_AVAILABILITY.csv", availability_frame)
    atomic_csv(args.output / "BIOLOGY_GROWTH.csv", biology_frame)
    atomic_csv(args.output / "RISK_GROWTH_STRATA.csv", risk)
    atomic_csv(args.output / "RISK_GROWTH_MACRO.csv", macro)
    atomic_csv(args.output / "RISK_GROWTH_CURVE.csv", curve)
    (args.output / "REPORT.md").write_text(
        "# Public Memory growth\n\nFive nested identity-hash orders; all orders are reported. "
        "Transfer and shared-risk learners are refit within each fraction and outer fold.\n\n"
        + markdown(curve) + "\n"
    )
    status = {
        "status": "COMPLETE", "evidence_role": "DEV_SEEN", "fractions": FRACTIONS,
        "hash_orders": len(ORDERS), "n_public_memory_items": len(memory_ids),
        "outer_folds": 5, "transfer_learner_refit_each_fraction": True,
        "shared_core_refit_each_fraction": True, "outcome_selected_order": False,
        "mcfaline_test_opened": False, "source_memory_sha256": manifest["effect_vectors_sha256"],
    }
    (args.output / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(curve.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
