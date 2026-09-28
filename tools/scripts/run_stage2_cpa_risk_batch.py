#!/usr/bin/env python3
"""Apply the selected lightweight SafeConf risk learner to released CPA tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
PUB = Path("/home/yyf/proj/docs/实验结果")
PARENT = PUB / "E84_cpa_rdkit_cartesian_formal_20260712/manifests"
SPLITS = PUB / "E81_sciplex_cartesian_contract_20260712/tables/E81_SPLIT_MANIFEST.csv"
OUT = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/cpa_risk_batch"
IDS = ["E81_r1_p25", "E81_r1_p50", "E81_r2_p25", "E81_r2_p50", "E81_r2_p75", "E81_r3_p25", "E81_r3_p50", "E81_r3_p75"]
P = ["predicted_magnitude", "prediction_abs_mean", "prediction_std", "prediction_abs_q95"]
Q = ["n_source_cells", "n_source_contexts", "n_source_batches", "min_source_cells"]
H = ["source_magnitude", "source_dispersion", "model_source_gap", "prediction_source_cosine"]
GROUPS = {"M": P[:1], "P": P, "PQ": P + Q, "PQH": P + Q + H}
SEED = 20260928


def utility(score, outcome):
    n = len(score)
    k = int(math.ceil(.2 * n))
    order = np.argsort(-np.asarray(score))[:k]
    oracle = np.argsort(-np.asarray(outcome))[:k]
    den = float(np.asarray(outcome)[oracle].mean() - np.asarray(outcome).mean())
    return float((np.asarray(outcome)[order].mean() - np.asarray(outcome).mean()) / den) if den > 1e-15 else np.nan


def rho(score, outcome):
    return float(spearmanr(score, outcome).statistic) if np.ptp(score) and np.ptp(outcome) else np.nan


def transform(fit, query, cols):
    x = fit[cols].to_numpy(float)
    z = query[cols].to_numpy(float)
    miss_x, miss_z = ~np.isfinite(x), ~np.isfinite(z)
    med = np.array([np.median(x[~miss_x[:, j], j]) if (~miss_x[:, j]).any() else 0 for j in range(x.shape[1])])
    x, z = np.where(miss_x, med, x), np.where(miss_z, med, z)
    mu, sd = x.mean(0), np.where(x.std(0) > 1e-8, x.std(0), 1)
    return np.c_[(x-mu)/sd, miss_x].astype("float32"), np.c_[(z-mu)/sd, miss_z].astype("float32")


def load_truth():
    values = {}
    for ident in IDS:
        with np.load(PARENT / ident / "arrays/true_effects.npz") as archive:
            for key in archive.files:
                task = key.split("::", 2)[2].removesuffix("::true")
                values.setdefault(task, np.asarray(archive[key], dtype=float))
                assert np.allclose(values[task], archive[key], atol=1e-6, rtol=0)
    return values


def source_features(task, train_rows, truth):
    drug = task.perturbation_key
    dose = float(task.dose_key)
    exact = train_rows[(train_rows.perturbation_key == drug) & (train_rows.dose_key.astype(float) == dose)]
    if len(exact) == 0:
        return np.nan, np.nan, 0, 0, 0, np.nan, None
    vectors = np.stack([truth[x] for x in exact.task_key])
    mean = vectors.mean(0)
    return (
        float(np.sqrt(np.mean(mean**2))),
        float(np.sqrt(np.mean((vectors - mean) ** 2))),
        int(exact.n_cells.sum()),
        int(exact.context.nunique()),
        int(exact.n_cells.count()),
        float(exact.n_cells.min()),
        mean,
    )


def load_manifest(ident, truth, predictor):
    split = pd.read_csv(SPLITS)
    split = split[split.manifest_id.eq(ident)].copy()
    records = pd.read_csv(PARENT / ident / "tables/PREDICTION_RECORDS.csv")
    records = records[records.predictor_name.eq(predictor)].copy()
    test = split[split.role.eq("test")].copy()
    train = split[split.role.eq("train")].copy()
    rows = []
    with np.load(PARENT / ident / "arrays/predicted_effects.npz") as predictions:
        for r in records.itertuples():
            assert r.task_id in set(test.task_key)
            query = predictions[r.predicted_effect_key].astype(float)
            y = truth[r.task_id]
            assert abs(np.sqrt(np.mean((query-y)**2)) - r.true_error_rmse) < 1e-6
            key = r.task_id.split("::", 1)[1]
            smag, sdisp, ncells, nctx, nbatch, mincells, source = source_features(
                test[test.task_key.eq(r.task_id)].iloc[0], train, truth
            )
            effect = query
            numerator = np.nan if source is None else float(np.dot(effect, source))
            denominator = np.nan if source is None else float(np.linalg.norm(effect)*np.linalg.norm(source))
            rows.append({
                "manifest": ident, "task_id": r.task_id, "drug": r.perturbation.split("::dose=", 1)[0],
                "context": r.context, "dose": float(r.perturbation.split("::dose=", 1)[1]),
                "true_error_rmse": float(r.true_error_rmse),
                "predicted_magnitude": float(np.sqrt(np.mean(effect**2))),
                "prediction_abs_mean": float(np.mean(np.abs(effect))),
                "prediction_std": float(np.std(effect)),
                "prediction_abs_q95": float(np.quantile(np.abs(effect), .95)),
                "source_magnitude": smag, "source_dispersion": sdisp,
                "n_source_cells": ncells if ncells else np.nan, "n_source_contexts": nctx if nctx else np.nan,
                "n_source_batches": nbatch if nbatch else np.nan, "min_source_cells": mincells,
                "model_source_gap": np.nan if source is None else float(np.sqrt(np.mean((effect-source)**2))),
                "prediction_source_cosine": np.nan if source is None or denominator == 0 else numerator/denominator,
                "source_vector": source,
            })
    result = pd.DataFrame(rows)
    assert len(result) == len(test) and result.task_id.nunique() == len(result)
    return result


def evaluate(frame, predictor):
    pred_rows, splits = [], []
    for ident, data in frame.groupby("manifest", sort=True):
        data = data.reset_index(drop=True)
        groups = data.drug
        if groups.nunique() < 5:
            raise RuntimeError(f"not enough drug groups in {ident}")
        for fold, (ia, ib) in enumerate(GroupKFold(5).split(data, groups=groups)):
            fit, query = data.iloc[ia], data.iloc[ib]
            y = fit.true_error_rmse.to_numpy(float)
            for r in fit.itertuples(): splits.append(dict(manifest=ident, fold=fold, role="fit", task_id=r.task_id))
            for r in query.itertuples(): splits.append(dict(manifest=ident, fold=fold, role="eval", task_id=r.task_id))
            for group, cols in GROUPS.items():
                x, z = transform(fit, query, cols)
                scale, center = max(float(y.std()), 1e-8), float(y.mean())
                model = Ridge(alpha=10.).fit(x, (y-center)/scale)
                risk = np.maximum(0, model.predict(z)*scale+center)
                for r, value in zip(query.itertuples(), risk):
                    pred_rows.append(dict(manifest=ident, fold=fold, task_id=r.task_id, method=f"Ridge_{group}", true_error_rmse=r.true_error_rmse, predicted_risk=float(value)))
    for r in frame.itertuples():
        pred_rows.append(dict(manifest=r.manifest, fold=-1, task_id=r.task_id, method="Magnitude_raw", true_error_rmse=r.true_error_rmse, predicted_risk=r.predicted_magnitude))
    pred = pd.DataFrame(pred_rows)
    # Every test record appears exactly once in each method.
    assert pred.groupby(["manifest", "method"]).task_id.nunique().eq(pred.groupby(["manifest", "method"]).size()).all()
    target_rows = []
    for (ident, method), g in pred.groupby(["manifest", "method"], sort=True):
        target_rows.append(dict(manifest=ident, method=method, n_tasks=len(g), utility20=utility(g.predicted_risk, g.true_error_rmse), spearman=rho(g.predicted_risk, g.true_error_rmse)))
    target = pd.DataFrame(target_rows)
    summary = target.groupby("method", as_index=False).agg(n_manifests=("manifest", "nunique"), utility20=("utility20", "mean"), spearman=("spearman", "mean"))
    return pred, target, summary, pd.DataFrame(splits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", default="CPA_0.8.8_RDKIT_logdose")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "RUN_STATUS.json").exists(): raise FileExistsError("refusing to overwrite CPA result")
    truth = load_truth()
    frames = [load_manifest(ident, truth, args.predictor) for ident in IDS]
    frame = pd.concat(frames, ignore_index=True)
    frame_no_vectors = frame.drop(columns=["source_vector"])
    pred, target, summary, splits = evaluate(frame_no_vectors, args.predictor)
    frame_no_vectors.to_csv(OUT / "FEATURE_TABLE.csv", index=False)
    pred.to_csv(OUT / "OOF_PREDICTIONS.csv", index=False)
    target.to_csv(OUT / "MANIFEST_RESULTS.csv", index=False)
    summary.to_csv(OUT / "SUMMARY.csv", index=False)
    splits.to_csv(OUT / "SPLIT_MANIFEST.csv", index=False)
    status = dict(status="COMPLETE_STOPPED_AFTER_REGISTERED_BATCH", predictor=args.predictor, n_tasks=len(frame), n_manifests=len(IDS), n_methods=summary.method.nunique(), truth_union_tasks=len(truth), new_upstream_training_runs=0, final_test_label_rows_used=0, evidence_level="RELEASED_RETROSPECTIVE_DEVELOPMENT", python=platform.python_version(), numpy=np.__version__, sklearn=__import__("sklearn").__version__, result_sha256=hashlib.sha256((OUT/"SUMMARY.csv").read_bytes()).hexdigest())
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(target.to_string(index=False), flush=True)


if __name__ == "__main__": main()
