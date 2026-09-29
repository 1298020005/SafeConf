#!/usr/bin/env python3
"""Orthogonally compare public history and same-model error memory.

This is a DEV/SEEN mechanism experiment.  Within every gene-disjoint outer
fold, all four learners receive exactly the same sampled error-label budget.
Public-history fields are available before feedback.  Error-memory fields are
built only from the sampled same-upstream outer-train errors; fit-row memory is
leave-one-out and query genes never contribute labels.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
OUT = STAGE / "public_history_error_memory"
SEED = 20260929
N_BOOTSTRAP = 5000
BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.00)
UPSTREAMS = ("txpert_gat", "txpert_exphormer")

P = [
    "predicted_magnitude", "prediction_abs_mean", "prediction_signed_mean",
    "prediction_std", "prediction_abs_q95", "prediction_sparsity",
]
PUBLIC_HISTORY = [
    "n_source_cells", "n_source_contexts", "n_source_batches",
    "min_source_cells", "prediction_source_cosine", "negative_model_source_gap",
]
ERROR_MEMORY = [
    "error_memory_knn_mean", "error_memory_knn_std",
    "error_memory_weighted_mean", "error_memory_nearest_distance",
    "error_memory_mean_distance",
]
GROUPS = {
    "P": P,
    "P+PublicHistory": P + PUBLIC_HISTORY,
    "P+ErrorMemory": P + ERROR_MEMORY,
    "P+PublicHistory+ErrorMemory": P + PUBLIC_HISTORY + ERROR_MEMORY,
}


def utility20(task_ids: np.ndarray, score: np.ndarray, truth: np.ndarray) -> float:
    n = len(truth)
    if n < 20:
        return float("nan")
    ids = np.asarray(task_ids, str)
    k = int(math.ceil(0.2 * n))
    chosen = np.lexsort((ids, -score))[:k]
    oracle = np.lexsort((ids, -truth))[:k]
    denominator = float(truth[oracle].mean() - truth.mean())
    return float((truth[chosen].mean() - truth.mean()) / denominator) if denominator > 1e-12 else float("nan")


def rho(score: np.ndarray, truth: np.ndarray) -> float:
    if len(score) < 3 or np.ptp(score) <= 1e-15 or np.ptp(truth) <= 1e-15:
        return float("nan")
    return float(spearmanr(score, truth).statistic)


def aurc(score: np.ndarray, truth: np.ndarray) -> float:
    order = np.argsort(score, kind="stable")
    return float(np.mean(np.cumsum(truth[order]) / np.arange(1, len(order) + 1)))


def stable_order(values: np.ndarray, upstream: str, target: str, fold: int) -> list[str]:
    def key(value: str) -> str:
        token = f"SafeConf-E-memory\0{upstream}\0{target}\0{fold}\0{value}"
        return hashlib.sha256(token.encode()).hexdigest()
    return sorted(set(map(str, values)), key=key)


def numeric_transform(fit: pd.DataFrame, query: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = fit[columns].to_numpy(float)
    z = query[columns].to_numpy(float)
    mx, mz = ~np.isfinite(x), ~np.isfinite(z)
    med = np.asarray([
        np.nanmedian(x[:, j]) if np.isfinite(x[:, j]).any() else 0.0
        for j in range(x.shape[1])
    ])
    x, z = np.where(mx, med, x), np.where(mz, med, z)
    center = x.mean(axis=0)
    scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    return np.c_[(x - center) / scale, mx], np.c_[(z - center) / scale, mz]


def error_memory_features(memory: pd.DataFrame, query: pd.DataFrame, leave_self_out: bool) -> pd.DataFrame:
    """Create local same-model error summaries without query-label access."""
    basis, query_basis = numeric_transform(memory, query, P)
    # Missing indicators are excluded from distance; all current Universal-P
    # fields are complete, and this guard prevents future missingness identity.
    basis, query_basis = basis[:, :len(P)], query_basis[:, :len(P)]
    labels = memory.true_error_rmse.to_numpy(float)
    memory_ids = memory.task_id.astype(str).to_numpy()
    query_ids = query.task_id.astype(str).to_numpy()
    rows: list[dict[str, float]] = []
    for i, vector in enumerate(query_basis):
        distances = np.sqrt(np.mean((basis - vector[None, :]) ** 2, axis=1))
        allowed = np.ones(len(memory), dtype=bool)
        if leave_self_out:
            allowed &= memory_ids != query_ids[i]
        candidates = np.flatnonzero(allowed)
        if len(candidates) < 3:
            raise ValueError("error-memory budget has fewer than three eligible records")
        k = min(15, len(candidates))
        selected = candidates[np.argsort(distances[candidates], kind="stable")[:k]]
        d = distances[selected]
        y = labels[selected]
        weights = 1.0 / np.maximum(d, 1e-6)
        rows.append({
            "error_memory_knn_mean": float(y.mean()),
            "error_memory_knn_std": float(y.std()),
            "error_memory_weighted_mean": float(np.average(y, weights=weights)),
            "error_memory_nearest_distance": float(d.min()),
            "error_memory_mean_distance": float(d.mean()),
        })
    return pd.DataFrame(rows, index=query.index)


def fit_ridge(fit: pd.DataFrame, query: pd.DataFrame, columns: list[str]) -> np.ndarray:
    x, z = numeric_transform(fit, query, columns)
    y = fit.true_error_rmse.to_numpy(float)
    center, spread = float(y.mean()), max(float(y.std()), 1e-8)
    model = Ridge(alpha=10.0).fit(x, (y - center) / spread)
    return np.maximum(0.0, model.predict(z) * spread + center)


def load_upstream(upstream: str) -> pd.DataFrame:
    src = STAGE / "safeconf_v4_development" / upstream
    frame = pd.read_csv(src / "FEATURE_TABLE.csv.gz")
    folds = pd.read_csv(src / "OOF_PREDICTIONS.csv.gz")
    folds = folds[folds.method.eq("Ridge_USR")][["task_id", "target", "fold"]]
    frame = frame.merge(folds, on=["task_id", "target"], how="inner", validate="one_to_one")
    if len(frame) != 1808 or frame.task_id.nunique() != 1808:
        raise AssertionError(f"incomplete frozen development table for {upstream}")
    return frame


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, object]] = []
    budget_audit: list[dict[str, object]] = []
    for upstream in UPSTREAMS:
        frame = load_upstream(upstream)
        for (target, fold), query in frame.groupby(["target", "fold"], sort=True):
            outer_train = frame[(frame.target.eq(target)) & (~frame.fold.eq(fold))].copy()
            if set(outer_train.gene) & set(query.gene):
                raise AssertionError("outer gene leakage")
            ordered_genes = stable_order(outer_train.gene.to_numpy(str), upstream, str(target), int(fold))
            for budget in BUDGETS:
                n_genes = max(3, int(math.ceil(budget * len(ordered_genes))))
                selected_genes = set(ordered_genes[:n_genes])
                feedback = outer_train[outer_train.gene.astype(str).isin(selected_genes)].copy()
                if feedback.gene.nunique() != n_genes:
                    raise AssertionError("feedback gene-budget mismatch")
                fit_e = error_memory_features(feedback, feedback, leave_self_out=True)
                query_e = error_memory_features(feedback, query, leave_self_out=False)
                feedback.loc[:, ERROR_MEMORY] = fit_e.to_numpy()
                query_aug = query.copy()
                query_aug.loc[:, ERROR_MEMORY] = query_e.to_numpy()
                budget_audit.append({
                    "upstream": upstream, "target": target, "fold": int(fold),
                    "budget": budget, "n_available_genes": len(ordered_genes),
                    "n_feedback_genes": n_genes, "n_feedback_records": len(feedback),
                    "n_query_genes": int(query.gene.nunique()), "n_query_records": len(query),
                    "gene_overlap": len(set(feedback.gene) & set(query.gene)),
                })
                for method, columns in GROUPS.items():
                    score = fit_ridge(feedback, query_aug, columns)
                    for row, value in zip(query_aug.itertuples(), score):
                        predictions.append({
                            "upstream": upstream, "task_id": row.task_id,
                            "target": target, "gene": row.gene, "fold": int(fold),
                            "budget": budget, "method": method,
                            "predicted_risk": float(value),
                            "true_error_rmse": float(row.true_error_rmse),
                        })
        print(f"[PublicHistory/ErrorMemory] {upstream} complete", flush=True)
    return pd.DataFrame(predictions), pd.DataFrame(budget_audit)


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = []
    for keys, group in predictions.groupby(["upstream", "budget", "target", "method"], sort=True):
        upstream, budget, target, method = keys
        ids = group.task_id.to_numpy(str)
        score = group.predicted_risk.to_numpy(float)
        truth = group.true_error_rmse.to_numpy(float)
        strata.append({
            "upstream": upstream, "budget": budget, "target": target,
            "method": method, "n_tasks": len(group),
            "utility20": utility20(ids, score, truth),
            "spearman": rho(score, truth), "aurc": aurc(score, truth),
        })
    strata = pd.DataFrame(strata)
    summary = strata.groupby(["upstream", "budget", "method"], as_index=False).agg(
        n_strata=("target", "nunique"), utility20=("utility20", "mean"),
        spearman=("spearman", "mean"), aurc=("aurc", "mean"),
    )
    return strata, summary


def paired_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("PublicHistory_given_P", "P+PublicHistory", "P"),
        ("ErrorMemory_given_P", "P+ErrorMemory", "P"),
        ("ErrorMemory_given_PublicHistory", "P+PublicHistory+ErrorMemory", "P+PublicHistory"),
        ("PublicHistory_given_ErrorMemory", "P+PublicHistory+ErrorMemory", "P+ErrorMemory"),
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
        task_ids = part.task_id.to_numpy(str)
        targets = part.target.to_numpy(str)
        truth = part.true_error_rmse.to_numpy(float)
        arrays = {m: part[m].to_numpy(float) for m in GROUPS}
        draws = {name: [] for name, _, _ in comparisons}
        rho_draws = {name: [] for name, _, _ in comparisons}
        for _ in range(N_BOOTSTRAP):
            sampled = rng.integers(0, len(genes), len(genes))
            idx = np.concatenate([gene_rows[genes[i]] for i in sampled])
            scores: dict[str, tuple[float, float]] = {}
            for method in GROUPS:
                us, rs = [], []
                for target in sorted(set(targets)):
                    use = idx[targets[idx] == target]
                    us.append(utility20(task_ids[use], arrays[method][use], truth[use]))
                    rs.append(rho(arrays[method][use], truth[use]))
                scores[method] = (float(np.nanmean(us)), float(np.nanmean(rs)))
            for name, a, b in comparisons:
                draws[name].append(scores[a][0] - scores[b][0])
                rho_draws[name].append(scores[a][1] - scores[b][1])
        for name, a, b in comparisons:
            d, r = np.asarray(draws[name]), np.asarray(rho_draws[name])
            rows.append({
                "upstream": upstream, "budget": budget, "comparison": name,
                "method_a": a, "method_b": b,
                "delta_utility20": float(np.nanmean(d)),
                "utility_ci95_lower": float(np.nanquantile(d, 0.025)),
                "utility_ci95_upper": float(np.nanquantile(d, 0.975)),
                "delta_spearman": float(np.nanmean(r)),
                "spearman_ci95_lower": float(np.nanquantile(r, 0.025)),
                "spearman_ci95_upper": float(np.nanquantile(r, 0.975)),
                "bootstrap_replicates": N_BOOTSTRAP,
            })
    return pd.DataFrame(rows)


def report(summary: pd.DataFrame, intervals: pd.DataFrame) -> str:
    lines = [
        "# Public History vs same-model Error Memory", "",
        "DEV/SEEN mechanism experiment; external confirmation remained sealed.", "",
        "Every method uses the same gene-cluster error-label budget. Error Memory is built only from same-upstream outer-train errors; query labels never enter memory.", "",
        "## Summary", "",
        "| upstream | budget | method | U20 | Spearman | AURC |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.upstream} | {row.budget:.0%} | {row.method} | {row.utility20:.6f} | {row.spearman:.6f} | {row.aurc:.6f} |")
    lines.extend(["", "## Increment intervals", "", "| upstream | budget | comparison | dU20 [95% CI] | dSpearman [95% CI] |", "|---|---:|---|---:|---:|"])
    for row in intervals.itertuples():
        lines.append(
            f"| {row.upstream} | {row.budget:.0%} | {row.comparison} | "
            f"{row.delta_utility20:+.6f} [{row.utility_ci95_lower:+.6f}, {row.utility_ci95_upper:+.6f}] | "
            f"{row.delta_spearman:+.6f} [{row.spearman_ci95_lower:+.6f}, {row.spearman_ci95_upper:+.6f}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "RUN_STATUS.json").exists():
        raise FileExistsError("refusing to overwrite registered mechanism experiment")
    started = time.time()
    predictions, audit = evaluate()
    strata, summary = summarize(predictions)
    intervals = paired_bootstrap(predictions)
    predictions.to_csv(OUT / "OOF_PREDICTIONS.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    audit.to_csv(OUT / "BUDGET_AND_LEAKAGE_AUDIT.csv", index=False)
    strata.to_csv(OUT / "STRATUM_RESULTS.csv", index=False)
    summary.to_csv(OUT / "SUMMARY.csv", index=False)
    intervals.to_csv(OUT / "PAIRED_BOOTSTRAP.csv", index=False)
    (OUT / "REPORT.md").write_text(report(summary, intervals))
    status = {
        "status": "COMPLETE", "elapsed_seconds": time.time() - started,
        "data_role": "DEV_SEEN_ONLY", "sealed_confirmation_opened": False,
        "same_error_label_budget_across_methods": True,
        "outer_gene_disjoint": True, "fit_error_memory_leave_one_out": True,
        "query_error_memory_uses_outer_train_only": True,
        "budgets": list(BUDGETS), "bootstrap_replicates": N_BOOTSTRAP,
        "methods": list(GROUPS),
        "summary_sha256": hashlib.sha256((OUT / "SUMMARY.csv").read_bytes()).hexdigest(),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
