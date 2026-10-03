#!/usr/bin/env python3
"""Run the registered Source-supervised Ridge fusion audit.

This is a CPU-only candidate on frozen assets.  It does not train an upstream
predictor and it never uses target evaluation errors to fit a score.  The
source-risk channel is generated fold-out-of-fold on E201 source errors; the
McFaline external table is used only after the source fusion has been frozen.

Outputs are versioned and separate from paper files.  The script intentionally
keeps the no-error-label public rule beside, rather than inside, the Ridge
candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.safeconf_continual.fusion import (  # noqa: E402
    OOFReadyRidgeFusion,
    TrainOnlyQuantileScale,
    build_fusion_channels,
    rule_score,
)


DEFAULT_SOURCE = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_source_core_20261002_v1")
DEFAULT_TARGET = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/risk_cache/external_Learned.parquet")
DEFAULT_OUT = Path(
    "/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/"
    "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/"
    "research_closure_20261001/data_model_feedback_20261003_v1/unified_fusion_v1"
)

P_COLUMNS = [
    "predicted_magnitude", "prediction_abs_mean", "prediction_signed_mean",
    "prediction_std", "prediction_abs_q95", "prediction_sparsity",
]
PUBLIC_COLUMNS = [
    "prior_magnitude", "prediction_prior_rmse", "prediction_prior_cosine",
    "prior_uncertainty", "log_history_support", "effective_sources", "history_conflict",
]
FEATURE_COLUMNS = P_COLUMNS + PUBLIC_COLUMNS
SOURCE_FOLDS = 5
SEED = 20260930


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inverse_gene_weights(genes: pd.Series) -> np.ndarray:
    counts = genes.astype(str).value_counts()
    w = genes.astype(str).map(lambda g: 1.0 / float(counts[g])).to_numpy(float)
    return w / np.mean(w)


def fit_source_model(x: np.ndarray, y: np.ndarray, train: np.ndarray) -> HistGradientBoostingRegressor:
    """Fit the fixed shared-risk learner on source-only labels."""
    from tools.safeconf_continual.learners import NumericPreprocessor

    pre = NumericPreprocessor().fit(x[train])
    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        l2_regularization=10.0,
        random_state=SEED,
    )
    model.fit(pre.transform(x[train]), y[train])
    model._safeconf_preprocessor = pre  # kept in-memory for deterministic application
    return model


def predict_source_model(model, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(model._safeconf_preprocessor.transform(x)), dtype=float)


def fit_source_oof(x: np.ndarray, labels: np.ndarray, genes: pd.Series, folds: np.ndarray):
    oof = np.full(len(x), np.nan, dtype=float)
    fit_rows = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        query = folds == fold
        model = fit_source_model(x, labels, train)
        oof[query] = predict_source_model(model, x[query])
        fit_rows.append({
            "stage": "source_shared_oof",
            "fold": int(fold),
            "n_fit_rows": int(train.sum()),
            "n_query_rows": int(query.sum()),
            "fit_gene_count": int(genes.iloc[np.flatnonzero(train)].nunique()),
            "query_gene_count": int(genes.iloc[np.flatnonzero(query)].nunique()),
            "gene_disjoint": bool(set(genes.iloc[np.flatnonzero(train)])
                                    .isdisjoint(set(genes.iloc[np.flatnonzero(query)]))),
            "label_source": "SOURCE_RISK_MIDRANK_LABELS.npy",
            "target_error_budget": "none",
        })
    if not np.isfinite(oof).all():
        raise RuntimeError("source OOF shared scores are incomplete")
    return oof, pd.DataFrame(fit_rows)


def safe_cdf(values: np.ndarray, name: str, finite_fit: np.ndarray | None = None):
    fit = np.asarray(values if finite_fit is None else finite_fit, dtype=float)
    scale = TrainOnlyQuantileScale.fit(fit, name)
    return scale, scale.transform(values)


def metric_frame(frame: pd.DataFrame, scores: np.ndarray, method: str) -> dict:
    truth = frame["true_error_rmse"].to_numpy(float)
    task_ids = frame["task_id"].astype(str).to_numpy()
    values = np.asarray(scores, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(values)
    truth, task_ids, values = truth[valid], task_ids[valid], values[valid]
    result = {
        "method": method,
        "n_tasks": int(len(truth)),
        "n_planned": int(len(frame)),
        "coverage": float(len(truth) / max(len(frame), 1)),
        "u20": np.nan,
        "aurc": np.nan,
        "spearman": np.nan,
        "selected_top20": np.nan,
        "true_top20_found": np.nan,
        "remaining_mean_error": np.nan,
    }
    if not len(truth):
        return result
    k = max(1, int(np.ceil(0.2 * len(truth))))
    risky = np.lexsort((task_ids, -values))[:k]
    oracle = np.lexsort((task_ids, -truth))[:k]
    oracle_mean = float(truth[oracle].mean())
    all_mean = float(truth.mean())
    denominator = oracle_mean - all_mean
    if len(truth) >= 20 and denominator > 1e-12:
        result["u20"] = float((truth[risky].mean() - all_mean) / denominator)
    if len(truth) >= 3 and np.ptp(values) > 0 and np.ptp(truth) > 0:
        result["spearman"] = float(spearmanr(values, truth).statistic)
    ascending = np.lexsort((task_ids, values))
    result["aurc"] = float(np.mean(np.cumsum(truth[ascending]) / np.arange(1, len(truth) + 1)))
    result["selected_top20"] = k
    result["true_top20_found"] = int(len(set(risky).intersection(set(oracle))))
    result["remaining_mean_error"] = float(np.delete(truth, risky).mean()) if len(truth) > k else np.nan
    return result


def run(source_root: Path, target_path: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    source_labels_frame = pd.read_parquet(
        Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/common_axis_risk_only_replay_20261002_v1/SOURCE_RECORDS_LABELS_WEIGHTS.parquet")
    )
    # SOURCE_TASKS is the 1,808 biological-task manifest.  The risk feature
    # matrix has one row per upstream/task record, so its aligned manifest is
    # SOURCE_RECORDS_LABELS_WEIGHTS (3,616 rows).
    tasks = source_labels_frame.copy()
    x = np.load(source_root / "SOURCE_Learned_RISK_FEATURES.npy").astype(float)
    labels = np.load(source_root / "SOURCE_RISK_MIDRANK_LABELS.npy").astype(float)
    if not {"task_id", "gene", "target", "fold"}.issubset(tasks.columns):
        raise RuntimeError("unexpected SOURCE_TASKS contract")
    if x.shape != (len(tasks), len(FEATURE_COLUMNS)):
        raise RuntimeError(f"source feature shape {x.shape} does not match {len(FEATURE_COLUMNS)} columns")
    if len(labels) != len(tasks) or not np.isfinite(labels).all():
        raise RuntimeError("source CDF labels are incomplete")

    # Source shared score: every source row receives a fold-out-of-fold value.
    source_oof_shared, split_audit = fit_source_oof(x, labels, tasks["gene"], tasks["fold"].to_numpy())

    amp_scale, source_amp_q = safe_cdf(x[:, 0], "source_amplitude")
    public_scale, source_public_q = safe_cdf(x[:, 7], "source_public_rmse")
    shared_scale, source_shared_q = safe_cdf(source_oof_shared, "source_shared_oof")
    source_channels, channel_names = build_fusion_channels(
        source_amp_q, source_public_q, source_shared_q, None,
        has_public=np.isfinite(x[:, 7]).astype(int),
        has_shared=np.ones(len(x), dtype=int),
        has_target=np.zeros(len(x), dtype=int),
    )
    # This is explicitly Source-supervised: source errors teach the final
    # fusion, while target C errors are not read at any fitting step.
    fusion = OOFReadyRidgeFusion(
        alpha=10.0,
        label_source="E201_SOURCE_RISK_MIDRANK_LABELS",
        target_error_budget="none",
    ).fit(source_channels, labels, channel_names, base_scores_are_oof=True,
          sample_weight=inverse_gene_weights(tasks["gene"]))

    # Freeze the shared model on all legal source records for target inference.
    full_shared = fit_source_model(x, labels, np.ones(len(x), dtype=bool))
    target = pd.read_parquet(target_path).copy()
    missing = [c for c in FEATURE_COLUMNS if c not in target.columns]
    if missing:
        raise RuntimeError(f"target feature contract missing columns: {missing}")
    tx = target[FEATURE_COLUMNS].to_numpy(float)
    target_shared = predict_source_model(full_shared, tx)
    _, target_amp_q = safe_cdf(tx[:, 0], "source_amplitude", x[:, 0])
    _, target_public_q = safe_cdf(tx[:, 7], "source_public_rmse", x[:, 7])
    _, target_shared_q = safe_cdf(target_shared, "source_shared_full", source_oof_shared)
    target_channels, _ = build_fusion_channels(
        target_amp_q, target_public_q, target_shared_q, None,
        has_public=np.isfinite(tx[:, 7]).astype(int),
        has_shared=np.ones(len(target), dtype=int),
        has_target=np.zeros(len(target), dtype=int),
    )
    target_rule, target_has_public = rule_score(target_amp_q, target_public_q)
    target_shared_q_for_eval = target_shared_q
    target_fusion = fusion.predict(target_channels)

    score_table = target[["task_id", "gene", "target", "true_error_rmse", "upstream"]].copy()
    score_table["Amplitude_rule"] = target_amp_q
    score_table["Public_rule_no_error_labels"] = target_rule
    score_table["SharedRisk_source_supervised"] = target_shared_q_for_eval
    score_table["Public_Ridge_source_supervised"] = target_fusion
    score_table["has_public"] = target_has_public
    score_table["has_shared"] = 1

    # Existing locked target comparisons are attached only as evaluation
    # references. They are never used to fit the candidate above.
    matrix_path = Path("/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/MATRIX_TASK_PREDICTIONS.csv.gz")
    legacy_alignment = []
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path)
        matrix = matrix[(matrix["line"] == "TxPert_to_McFaline") &
                        (matrix["seed"] == SEED) &
                        (matrix["method"].isin(["Magnitude", "Learned_WeightedHistoryDistance", "Learned_hgb"]))]
        aligned = target[["task_id", "true_error_rmse"]].merge(
            matrix[["task_id", "true_error_rmse"]].drop_duplicates("task_id"),
            on="task_id", suffixes=("_current", "_legacy")
        )
        max_error_delta = float(np.max(np.abs(
            aligned.true_error_rmse_current.to_numpy(float) - aligned.true_error_rmse_legacy.to_numpy(float)
        ))) if len(aligned) else np.inf
        legacy_alignment.append({
            "asset": str(matrix_path), "rows": len(aligned),
            "max_abs_truth_delta": max_error_delta,
            "same_truth_contract": bool(max_error_delta <= 1e-12),
            "action": "reuse" if max_error_delta <= 1e-12 else "exclude_from_same_task_metrics",
        })
        if max_error_delta <= 1e-12:
            for method, part in matrix.groupby("method"):
                vals = part.drop_duplicates("task_id").set_index("task_id")["risk"]
                score_table[method] = score_table["task_id"].map(vals)

    metrics = []
    for col in [c for c in score_table.columns if c.startswith(("Amplitude", "Public_", "SharedRisk", "Magnitude", "Learned_"))]:
        metrics.append(metric_frame(score_table, score_table[col].to_numpy(float), col))
    metrics = pd.DataFrame(metrics)

    score_table.to_parquet(out / "TARGET_SCORES.parquet", index=False)
    source_truth = tasks["true_error_rmse"].to_numpy(float)
    pd.DataFrame({"task_id": tasks["task_id"], "gene": tasks["gene"], "fold": tasks["fold"],
                  "true_error_rmse": source_truth,
                  "source_label": labels, "source_shared_oof": source_oof_shared,
                  "amplitude_q": source_amp_q, "public_q": source_public_q,
                  "shared_q": source_shared_q}).to_parquet(out / "SOURCE_OOF_CHANNELS.parquet", index=False)
    metrics.to_csv(out / "METRICS.csv", index=False)
    pd.DataFrame(legacy_alignment).to_csv(out / "LEGACY_ALIGNMENT_AUDIT.csv", index=False)
    split_audit.to_csv(out / "SPLIT_AND_FIT_LEDGER.csv", index=False)
    cdf_audit = pd.DataFrame([
        amp_scale.audit(), public_scale.audit(), shared_scale.audit(),
    ])
    cdf_audit.to_csv(out / "CHANNEL_SCORE_CDF_AUDIT.csv", index=False)
    pd.DataFrame([{
        "label_type": "source_error",
        "label_source": "SOURCE_RISK_MIDRANK_LABELS.npy",
        "n_rows": int(len(labels)),
        "n_genes": int(tasks["gene"].nunique()),
        "target_error_used_for_fitting": False,
        "target_feedback_budget": "none",
        "label_scale": "E201 per-upstream/context midrank CDF supplied by frozen source artifact",
    }]).to_csv(out / "ERROR_LABEL_CDF_AUDIT.csv", index=False)
    pd.DataFrame([
        {"information": "source_errors", "purpose": "SharedRisk_and_Ridge_fusion", "n_rows": len(labels), "budget": "source_full"},
        {"information": "target_errors", "purpose": "evaluation_only", "n_rows": 0, "budget": "none"},
    ]).to_csv(out / "INFORMATION_BUDGET.csv", index=False)
    with (out / "FUSION_AUDIT.json").open("w") as f:
        json.dump({
            "status": "COMPLETE",
            "candidate": "Public_Ridge_source_supervised",
            "label_source": "E201_SOURCE_RISK_MIDRANK_LABELS",
            "target_errors_used_for_fit": 0,
            "target_errors_used_for_selection": 0,
            "base_scores_are_oof": True,
            "source_model": "fixed_HGB_source_error_rank",
            "fusion": fusion.audit(),
            "rule_baseline_preserved": True,
            "target_evaluation_asset": str(target_path),
            "target_evaluation_role": "evaluation_only_SEEN",
            "feature_columns": FEATURE_COLUMNS,
            "source_root": str(source_root),
            "source_tasks_sha256": sha256(source_root / "SOURCE_TASKS.parquet"),
            "source_labels_sha256": sha256(source_root / "SOURCE_RISK_MIDRANK_LABELS.npy"),
            "python": platform.python_version(),
        }, f, indent=2)
    with (out / "RUN_STATUS.json").open("w") as f:
        json.dump({"status": "COMPLETE", "n_source": len(tasks), "n_target": len(target),
                   "outputs": [p.name for p in sorted(out.iterdir())]}, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.source_root, args.target, args.out)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
