#!/usr/bin/env python3
"""Continual, model-specific error adaptation on a frozen shared core.

Feedback is allocated by perturbation cluster. Each adapter learns error-rank
residuals for exactly one upstream version. A PertEMA-style direct HGB proxy
uses the same feedback clusters and prediction-only evidence; it is never
reported as the official PertEMA implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.safeconf_continual import (
    ErrorMemoryItem,
    ErrorMemoryRegistry,
    FrozenErrorCDF,
    biological_task_key,
    perturbation_cluster_key,
)


SOURCE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/txpert_public_biology"
OUT = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/error_adaptation"
ERROR_ROOT = Path("/home/yyf/data/safeconf_dual_memory_20260929/error_memory_dev")
SEED = 20260929
BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.00)
KAPPAS = (10.0, 25.0, 50.0, 100.0)
P = [
    "predicted_magnitude", "prediction_abs_mean", "prediction_signed_mean",
    "prediction_std", "prediction_abs_q95", "prediction_sparsity",
]
PUBLIC = [
    "LearnedHGB_prior_magnitude", "LearnedHGB_prediction_prior_rmse",
    "LearnedHGB_prediction_prior_cosine", "LearnedHGB_prior_uncertainty",
    "LearnedHGB_log_history_support", "LearnedHGB_effective_sources",
    "LearnedHGB_history_conflict",
]
MODEL_VERSIONS = {
    "TxPert_GAT": "E201_STRING_GAT_registered_family",
    "TxPert_Exphormer": "E205_Exphormer_registered_family",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--error-root", type=Path, default=ERROR_ROOT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--shared-method", default="SharedHGB_Learned")
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


def stable_gene_order(values: pd.Series, target: str, fold: int) -> list[str]:
    return sorted(
        set(values.astype(str)),
        key=lambda gene: hashlib.sha256(
            f"SafeConf-error-feedback-v1\0{target}\0{fold}\0{gene}".encode()
        ).hexdigest(),
    )


def numeric_transform(fit: pd.DataFrame, query: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
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


def predictor(learner: str, seed: int = SEED):
    if learner == "ridge":
        return Ridge(alpha=10.0)
    if learner == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=3,
            min_samples_leaf=10, l2_regularization=10.0,
            random_state=seed,
        )
    raise ValueError(learner)


def fit_predict(
    fit: pd.DataFrame,
    query: pd.DataFrame,
    columns: list[str],
    labels: np.ndarray,
    learner: str,
) -> np.ndarray:
    x, z = numeric_transform(fit, query, columns)
    return np.asarray(predictor(learner).fit(x, labels).predict(z), dtype=float)


def choose_kappa(
    feedback: pd.DataFrame,
    columns: list[str],
    learner: str,
) -> float:
    genes = feedback.gene.astype(str)
    if genes.nunique() < 6:
        return 100.0
    n_splits = min(3, genes.nunique())
    losses = {kappa: [] for kappa in KAPPAS}
    splitter = GroupKFold(n_splits)
    for train_idx, val_idx in splitter.split(feedback, groups=genes):
        fit = feedback.iloc[train_idx].reset_index(drop=True)
        val = feedback.iloc[val_idx].reset_index(drop=True)
        correction = fit_predict(
            fit, val, columns,
            fit.error_rank.to_numpy(float) - fit.shared_risk.to_numpy(float),
            learner,
        )
        n_clusters = fit.gene.nunique()
        for kappa in KAPPAS:
            weight = n_clusters / (n_clusters + kappa)
            final = np.clip(val.shared_risk.to_numpy(float) + weight * correction, 0, 1)
            losses[kappa].append(float(np.mean(np.abs(final - val.error_rank.to_numpy(float)))))
    # Conservative tie breaking: stronger shrinkage (larger kappa).
    return min(KAPPAS, key=lambda kappa: (np.mean(losses[kappa]), -kappa))


def build_error_banks(frame: pd.DataFrame, root: Path) -> list[dict]:
    registry = ErrorMemoryRegistry(root)
    manifests = []
    for upstream, data in frame.groupby("upstream", sort=True):
        items = []
        for fold in sorted(data.fold.unique()):
            query = data[data.fold.eq(fold)]
            outer = data[data.fold.ne(fold)]
            cdfs = {
                target: FrozenErrorCDF.fit(group.true_error_rmse)
                for target, group in outer.groupby("target")
            }
            for row in query.itertuples():
                rank = float(cdfs[row.target].transform([row.true_error_rmse])[0])
                items.append(ErrorMemoryItem(
                    upstream_model_id=upstream,
                    model_version=MODEL_VERSIONS[upstream],
                    prediction_id=f"{upstream}::{row.task_id}",
                    biological_task_key=biological_task_key(
                        "E201_TxPert", row.target, "genetic_single_gene",
                        row.gene, row.condition, "E201_effect_v1",
                    ),
                    perturbation_cluster_key=perturbation_cluster_key(
                        "E201_TxPert", "genetic_single_gene", row.gene
                    ),
                    shared_risk=float(row.shared_risk),
                    realised_error=float(row.true_error_rmse),
                    error_rank=rank,
                    shared_risk_residual=rank - float(row.shared_risk),
                    feedback_timestamp="2026-09-29T00:00:00+00:00",
                    provenance="gene-disjoint OOF shared-core prediction",
                    is_oof_or_heldout=True,
                ))
        key_dir = registry._key_dir(upstream, MODEL_VERSIONS[upstream])
        if key_dir.exists():
            # Reuse only if exact immutable contents already exist.
            existing = registry.load(upstream, MODEL_VERSIONS[upstream])
            if len(existing) != len(items):
                raise RuntimeError("existing error-memory bank differs from current OOF contract")
            manifests.append(json.loads((key_dir / "manifest.json").read_text()))
        else:
            manifests.append(registry.append(items))
    return manifests


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    feature_public = P + PUBLIC
    feature_residual = feature_public + ["shared_risk"]
    upstreams = sorted(frame.upstream.unique())
    for upstream in upstreams:
        other_upstream = next(value for value in upstreams if value != upstream)
        current = frame[frame.upstream.eq(upstream)]
        wrong = frame[frame.upstream.eq(other_upstream)]
        for (target, fold), query in current.groupby(["target", "fold"], sort=True):
            outer = current[current.target.eq(target) & current.fold.ne(fold)].copy()
            wrong_outer = wrong[wrong.target.eq(target) & wrong.fold.ne(fold)].copy()
            if set(outer.gene) & set(query.gene):
                raise RuntimeError("feedback/query perturbation leakage")
            cdf = FrozenErrorCDF.fit(outer.true_error_rmse)
            outer["error_rank"] = cdf.transform(outer.true_error_rmse)
            wrong_cdf = FrozenErrorCDF.fit(wrong_outer.true_error_rmse)
            wrong_outer["error_rank"] = wrong_cdf.transform(wrong_outer.true_error_rmse)
            order = stable_gene_order(outer.gene, str(target), int(fold))
            for budget in BUDGETS:
                n_genes = max(3, int(math.ceil(budget * len(order))))
                selected = set(order[:n_genes])
                feedback = outer[outer.gene.astype(str).isin(selected)].copy().reset_index(drop=True)
                wrong_feedback = wrong_outer[wrong_outer.gene.astype(str).isin(selected)].copy().reset_index(drop=True)
                if feedback.gene.nunique() != n_genes or wrong_feedback.empty:
                    raise RuntimeError("feedback budget alignment failed")
                predictions = {
                    "SharedCore": query.shared_risk.to_numpy(float),
                    "DirectRidge_PPublic": np.clip(fit_predict(
                        feedback, query, feature_public, feedback.error_rank.to_numpy(float), "ridge"
                    ), 0, 1),
                    "PertEMA_style_HGB_P_proxy": np.clip(fit_predict(
                        feedback, query, P, feedback.error_rank.to_numpy(float), "hgb"
                    ), 0, 1),
                    "DirectHGB_PPublic": np.clip(fit_predict(
                        feedback, query, feature_public, feedback.error_rank.to_numpy(float), "hgb"
                    ), 0, 1),
                }
                kappa_ridge = choose_kappa(feedback, feature_residual, "ridge")
                kappa_hgb = choose_kappa(feedback, feature_residual, "hgb")
                ridge_correction = fit_predict(
                    feedback, query, feature_residual,
                    feedback.error_rank.to_numpy(float) - feedback.shared_risk.to_numpy(float), "ridge",
                )
                hgb_correction = fit_predict(
                    feedback, query, feature_residual,
                    feedback.error_rank.to_numpy(float) - feedback.shared_risk.to_numpy(float), "hgb",
                )
                ridge_weight = n_genes / (n_genes + kappa_ridge)
                hgb_weight = n_genes / (n_genes + kappa_hgb)
                predictions["ResidualRidgeShrink"] = np.clip(query.shared_risk.to_numpy(float) + ridge_weight * ridge_correction, 0, 1)
                predictions["ResidualHGBShrink"] = np.clip(query.shared_risk.to_numpy(float) + hgb_weight * hgb_correction, 0, 1)
                wrong_correction = fit_predict(
                    wrong_feedback, query, feature_residual,
                    wrong_feedback.error_rank.to_numpy(float) - wrong_feedback.shared_risk.to_numpy(float), "hgb",
                )
                predictions["WrongUpstreamResidual"] = np.clip(query.shared_risk.to_numpy(float) + hgb_weight * wrong_correction, 0, 1)
                rng = np.random.default_rng(SEED + int(fold) + n_genes)
                shuffled = feedback.error_rank.to_numpy(float) - feedback.shared_risk.to_numpy(float)
                shuffled = shuffled[rng.permutation(len(shuffled))]
                shuffled_correction = fit_predict(feedback, query, feature_residual, shuffled, "hgb")
                predictions["ShuffledErrorResidual"] = np.clip(query.shared_risk.to_numpy(float) + hgb_weight * shuffled_correction, 0, 1)
                audits.append({
                    "upstream": upstream, "target": target, "fold": int(fold),
                    "budget": budget, "n_available_genes": len(order),
                    "n_feedback_genes": n_genes, "n_feedback_records": len(feedback),
                    "n_query_genes": query.gene.nunique(), "gene_overlap": 0,
                    "kappa_ridge": kappa_ridge, "kappa_hgb": kappa_hgb,
                    "ridge_shrinkage": ridge_weight, "hgb_shrinkage": hgb_weight,
                })
                for method, values in predictions.items():
                    for row, value in zip(query.itertuples(), values):
                        records.append({
                            "upstream": upstream, "target": target, "fold": int(fold),
                            "task_id": row.task_id, "gene": row.gene, "budget": budget,
                            "method": method, "predicted_risk": float(value),
                            "true_error_rmse": float(row.true_error_rmse),
                        })
        print(f"[ErrorAdapter] {upstream} complete", flush=True)
    return pd.DataFrame(records), pd.DataFrame(audits)


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, group in predictions.groupby(["upstream", "budget", "target", "method"], sort=True):
        upstream, budget, target, method = keys
        rows.append({
            "upstream": upstream, "budget": budget, "target": target, "method": method,
            "n_tasks": len(group),
            "utility20": utility20(group.task_id.to_numpy(str), group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
            "spearman": rho(group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
            "aurc": aurc(group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
        })
    strata = pd.DataFrame(rows)
    summary = strata.groupby(["upstream", "budget", "method"], as_index=False).agg(
        n_strata=("target", "nunique"), utility20=("utility20", "mean"),
        spearman=("spearman", "mean"), aurc=("aurc", "mean"),
    )
    return strata, summary


def feedback_auc(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (upstream, method), group in summary.groupby(["upstream", "method"], sort=True):
        ordered = group.sort_values("budget")
        rows.append({
            "upstream": upstream, "method": method,
            "u20_feedback_auc": float(np.trapezoid(ordered.utility20, ordered.budget) / (ordered.budget.max() - ordered.budget.min())),
            "spearman_feedback_auc": float(np.trapezoid(ordered.spearman, ordered.budget) / (ordered.budget.max() - ordered.budget.min())),
            "min_delta_u20_vs_shared": float((ordered.utility20.to_numpy() - summary[
                summary.upstream.eq(upstream) & summary.method.eq("SharedCore")
            ].sort_values("budget").utility20.to_numpy()).min()),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(predictions: pd.DataFrame, replicates: int) -> pd.DataFrame:
    comparisons = [
        ("ResidualHGB_vs_Shared", "ResidualHGBShrink", "SharedCore"),
        ("ResidualHGB_vs_PertEMAProxy", "ResidualHGBShrink", "PertEMA_style_HGB_P_proxy"),
        ("ResidualHGB_vs_DirectPublic", "ResidualHGBShrink", "DirectHGB_PPublic"),
        ("WrongUpstream_vs_Shared", "WrongUpstreamResidual", "SharedCore"),
    ]
    wide = predictions.pivot(
        index=["upstream", "budget", "task_id", "target", "gene", "true_error_rmse"],
        columns="method", values="predicted_risk",
    ).reset_index()
    rng = np.random.default_rng(SEED)
    rows = []
    for (upstream, budget), part in wide.groupby(["upstream", "budget"], sort=True):
        genes = np.asarray(sorted(part.gene.astype(str).unique()))
        gene_rows = {gene: np.flatnonzero(part.gene.astype(str).to_numpy() == gene) for gene in genes}
        targets = part.target.to_numpy(str)
        ids = part.task_id.to_numpy(str)
        truth = part.true_error_rmse.to_numpy(float)
        arrays = {method: part[method].to_numpy(float) for method in predictions.method.unique()}
        draws = {name: [] for name, _, _ in comparisons}
        for _ in range(replicates):
            sampled = rng.integers(0, len(genes), len(genes))
            idx = np.concatenate([gene_rows[genes[i]] for i in sampled])
            metric = {}
            for method, values in arrays.items():
                metric[method] = float(np.nanmean([
                    utility20(ids[use := idx[targets[idx] == target]], values[use], truth[use])
                    for target in sorted(set(targets))
                ]))
            for name, a, b in comparisons:
                draws[name].append(metric[a] - metric[b])
        for name, a, b in comparisons:
            values = np.asarray(draws[name])
            rows.append({
                "upstream": upstream, "budget": budget, "comparison": name,
                "method_a": a, "method_b": b,
                "delta_utility20": float(np.nanmean(values)),
                "ci95_lower": float(np.nanquantile(values, 0.025)),
                "ci95_upper": float(np.nanquantile(values, 0.975)),
                "bootstrap_replicates": replicates,
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite error adaptation: {args.output}")
    features = pd.read_csv(args.source / "SHARED_RISK_FEATURES.csv.gz")
    shared = pd.read_csv(args.source / "SHARED_RISK_OOF.csv.gz")
    shared = shared[shared.method.eq(args.shared_method)][
        ["upstream", "task_id", "fold", "predicted_risk"]
    ].rename(columns={"predicted_risk": "shared_risk"})
    frame = features.merge(shared, on=["upstream", "task_id", "fold"], validate="one_to_one")
    if len(frame) != 3616 or frame.groupby("task_id").fold.nunique().max() != 1:
        raise RuntimeError("shared-core feature/OOF alignment failed")
    manifests = build_error_banks(frame, args.error_root)
    predictions, audit = evaluate(frame)
    strata, summary = summarize(predictions)
    auc = feedback_auc(summary)
    intervals = paired_bootstrap(predictions, args.bootstrap)
    args.output.mkdir(parents=True)
    atomic_csv(args.output / "FEEDBACK_PREDICTIONS.csv.gz", predictions)
    atomic_csv(args.output / "FEEDBACK_BUDGET_AUDIT.csv", audit)
    atomic_csv(args.output / "FEEDBACK_STRATUM_RESULTS.csv", strata)
    atomic_csv(args.output / "FEEDBACK_SUMMARY.csv", summary)
    atomic_csv(args.output / "FEEDBACK_CURVE_AUC.csv", auc)
    atomic_csv(args.output / "FEEDBACK_BOOTSTRAP.csv", intervals)
    atomic_json(args.output / "RUN_STATUS.json", {
        "status": "COMPLETE", "evidence_role": "DEV_SEEN",
        "shared_core_method": args.shared_method, "feedback_budgets": BUDGETS,
        "n_upstreams": frame.upstream.nunique(), "n_tasks_per_upstream": 1808,
        "same_model_error_memory_only": True, "same_task_all_upstreams_same_fold": True,
        "mcfaline_test_opened": False, "bootstrap_replicates": args.bootstrap,
        "error_memory_manifests": manifests,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    report = [
        "# Dual-Memory Error Adaptation", "",
        "DEV/SEEN feedback experiment; McFaline test remained sealed.", "",
        "The shared core is frozen. Every model-specific adapter uses only the same upstream's legal OOF errors. A shallow HGB provides a PertEMA-style proxy under the same feedback budget; it is not presented as the official implementation, whose compatibility is audited separately.", "",
        "## Feedback-curve AUC", "", "```text", auc.to_string(index=False), "```", "",
        "## Paired bootstrap", "", "```text", intervals.to_string(index=False), "```", "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n")
    print(auc.to_string(index=False), flush=True)
    print(intervals.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
