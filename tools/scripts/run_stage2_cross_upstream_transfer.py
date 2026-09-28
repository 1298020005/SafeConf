#!/usr/bin/env python3
"""Cross-upstream, gene-held-out transfer test for the selected SafeConf risk learner."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
SOURCES = {
    "TxPert_GAT": STAGE / "txpert_risk_batch/FEATURE_TABLE.csv.gz",
    "TxPert_Exphormer": STAGE / "txpert_exphormer_risk_batch/FEATURE_TABLE.csv.gz",
}
OUT = STAGE / "cross_upstream_transfer"
P = ["predicted_magnitude", "family_disagreement", "family_radius", "prediction_abs_mean", "prediction_signed_mean", "prediction_std", "prediction_abs_q95"]
Q = ["n_source_cells", "n_source_contexts", "n_source_batches", "min_source_cells"]
H = ["source_transfer_magnitude", "source_delta_dispersion", "model_source_gap", "prediction_source_cosine"]
GROUPS = {"P+Q": P + Q, "P+Q+H": P + Q + H}
TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
SEED = 20260928


def utility(score, truth):
    n = len(score)
    k = int(math.ceil(.2 * n))
    order = np.argsort(-np.asarray(score))[:k]
    oracle = np.argsort(-np.asarray(truth))[:k]
    den = float(np.asarray(truth)[oracle].mean() - np.asarray(truth).mean())
    return float((np.asarray(truth)[order].mean() - np.asarray(truth).mean()) / den) if den > 1e-15 else np.nan


def rho(score, truth):
    return float(spearmanr(score, truth).statistic) if np.ptp(score) and np.ptp(truth) else np.nan


def transform(fit, query, cols):
    x, z = fit[cols].to_numpy(float), query[cols].to_numpy(float)
    mx, mz = ~np.isfinite(x), ~np.isfinite(z)
    med = np.array([np.median(x[~mx[:, j], j]) if (~mx[:, j]).any() else 0 for j in range(x.shape[1])])
    x, z = np.where(mx, med, x), np.where(mz, med, z)
    mu, sd = x.mean(0), np.where(x.std(0) > 1e-8, x.std(0), 1)
    return np.c_[(x-mu)/sd, mx].astype("float32"), np.c_[(z-mu)/sd, mz].astype("float32")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "RUN_STATUS.json").exists(): raise FileExistsError("refusing to overwrite transfer result")
    parts = []
    for name, path in SOURCES.items():
        frame = pd.read_csv(path)
        frame["upstream"] = name
        parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    assert len(frame) == 3616 and frame.task_id.nunique() == 1808
    assert frame.groupby(["target", "task_id"]).upstream.nunique().eq(2).all()
    predictions = []
    splits = []
    for target, data in frame.groupby("target", sort=True):
        data = data.sort_values(["gene", "upstream", "task_id"]).reset_index(drop=True)
        for fold, (fit_idx, eval_idx) in enumerate(GroupKFold(5).split(data, groups=data.gene)):
            fit, query = data.iloc[fit_idx], data.iloc[eval_idx]
            assert not set(fit.gene) & set(query.gene)
            y = fit.family_centroid_rmse.to_numpy(float)
            for r in query.itertuples(): splits.append(dict(target=target, fold=fold, role="eval", task_id=r.task_id, upstream=r.upstream, gene=r.gene))
            for group, cols in GROUPS.items():
                x, z = transform(fit, query, cols)
                center, scale = float(y.mean()), max(float(y.std()), 1e-8)
                model = Ridge(alpha=10.).fit(x, (y-center)/scale)
                risk = np.maximum(0, model.predict(z)*scale+center)
                for r, value in zip(query.itertuples(), risk):
                    predictions.append(dict(target=target, fold=fold, task_id=r.task_id, upstream=r.upstream, gene=r.gene, method=f"Ridge_{group}", true_error_rmse=r.family_centroid_rmse, predicted_risk=float(value)))
        print(f"[cross-upstream] {target} complete", flush=True)
    pred = pd.DataFrame(predictions)
    raw = frame[["target", "task_id", "upstream", "gene", "family_centroid_rmse", "predicted_magnitude"]].rename(columns={"family_centroid_rmse":"true_error_rmse", "predicted_magnitude":"predicted_risk"})
    raw["fold"] = -1; raw["method"] = "Magnitude_raw"
    pred = pd.concat([pred, raw[pred.columns]], ignore_index=True)
    rows = []
    for (upstream, target, method), group in pred.groupby(["upstream", "target", "method"], sort=True):
        rows.append(dict(upstream=upstream, target=target, method=method, n_tasks=len(group), utility20=utility(group.predicted_risk, group.true_error_rmse), spearman=rho(group.predicted_risk, group.true_error_rmse)))
    target = pd.DataFrame(rows)
    summary = target.groupby(["upstream", "method"], as_index=False).agg(n_targets=("target", "nunique"), utility20=("utility20", "mean"), spearman=("spearman", "mean"))
    # A descriptive aggregate across the two upstreams; bootstrap by target gene is kept for the two shared records.
    wide = target.pivot_table(index=["upstream", "target"], columns="method", values="utility20")
    increments = []
    for upstream in sorted(wide.index.get_level_values(0).unique()):
        w = wide.loc[upstream]
        for method in ("Ridge_P+Q", "Ridge_P+Q+H"):
            delta = w[method] - w["Magnitude_raw"]
            increments.append(dict(upstream=upstream, method=method, delta_utility20=float(delta.mean()), positive_targets=int((delta > 0).sum()), target_deltas=";".join(f"{t}:{delta[t]:.6f}" for t in TARGETS)))
    frame.drop(columns=["upstream"]).to_csv(OUT / "FEATURE_TABLE.csv.gz", index=False, compression={"method":"gzip","mtime":0})
    pred.to_csv(OUT / "OOF_PREDICTIONS.csv.gz", index=False, compression={"method":"gzip","mtime":0})
    target.to_csv(OUT / "TARGET_RESULTS.csv", index=False)
    summary.to_csv(OUT / "SUMMARY.csv", index=False)
    pd.DataFrame(increments).to_csv(OUT / "INCREMENTAL_RESULTS.csv", index=False)
    pd.DataFrame(splits).to_csv(OUT / "SPLIT_MANIFEST.csv", index=False)
    status = dict(status="COMPLETE_STOPPED_AFTER_REGISTERED_BATCH", n_records=len(frame), n_tasks=1808, n_upstreams=2, gene_held_out_folds=5, feature_groups=GROUPS, target_truth_used_only_as_label=True, upstream_records_shared_tasks=True, evidence_level="RETROSPECTIVE_CROSS_UPSTREAM_DEVELOPMENT", new_upstream_training_runs=0, python=platform.python_version(), numpy=np.__version__, sklearn=__import__("sklearn").__version__, result_sha256=hashlib.sha256((OUT/"SUMMARY.csv").read_bytes()).hexdigest())
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(increments).to_string(index=False), flush=True)


if __name__ == "__main__": main()
