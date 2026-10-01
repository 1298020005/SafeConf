"""Budget-audited helpers for the October research closure.

This module leaves the frozen September implementations unchanged. Metadata
defines splits/CDFs only and is never supplied to a numerical learner.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .contracts import FrozenErrorCDF
from .learners import NumericPreprocessor

P = ["predicted_magnitude", "prediction_abs_mean", "prediction_signed_mean",
     "prediction_std", "prediction_abs_q95", "prediction_sparsity"]
PUBLIC = ["prior_magnitude", "prediction_prior_rmse", "prediction_prior_cosine",
          "prior_uncertainty", "log_history_support", "effective_sources", "history_conflict"]
CDF_KEYS = ["dataset_id", "output_contract_id", "upstream", "model_version", "target"]
SEEDS = (20260930, 20261001, 20261002)


def ids_hash(values) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def budget_subset(frame: pd.DataFrame, budget: float, order: int = 0) -> pd.DataFrame:
    if not 0 < budget <= 1:
        raise ValueError("source supervision budget must be in (0,1]")
    clusters = sorted(frame.gene.astype(str).unique(), key=lambda s: hashlib.sha256(
        f"source-budget-v1|{order}|{s}".encode()).hexdigest())
    selected = set(clusters[:max(1, math.ceil(budget * len(clusters)))])
    return frame[frame.gene.astype(str).isin(selected)].copy().reset_index(drop=True)


def rank_labels(frame: pd.DataFrame, run_id: str, budget: float = 1.0):
    """Fit each mid-rank CDF on exactly the records passed by the caller."""
    frame = frame.reset_index(drop=True)
    labels = np.full(len(frame), np.nan)
    audit = []
    for key, group in frame.groupby(CDF_KEYS, dropna=False, sort=True):
        errors = group.true_error_rmse.to_numpy(float)
        unique, counts = np.unique(errors, return_counts=True)
        finite = np.isfinite(errors).all()
        valid = finite and len(errors) >= 2
        if valid:
            labels[group.index] = FrozenErrorCDF.fit(errors).transform(errors)
        audit.append(dict(zip(CDF_KEYS, key)) | {
            "run_id": run_id, "budget": budget, "n_rows": len(group),
            "n_biological_tasks": group.task_id.nunique(), "n_clusters": group.gene.nunique(),
            "n_unique_errors": len(unique), "tie_fraction": 1-len(unique)/len(group),
            "nominal_resolution": 1/len(group), "max_ecdf_jump": float(counts.max()/len(group)),
            "label_min": float(np.nanmin(labels[group.index])) if valid else np.nan,
            "label_max": float(np.nanmax(labels[group.index])) if valid else np.nan,
            "training_records_hash": ids_hash(group.upstream.astype(str)+"::"+group.task_id.astype(str)),
            "training_clusters_hash": ids_hash(group.gene.unique()),
            "status": "CONSTANT_LABEL" if valid and len(unique)==1 else "OK" if valid else "INSUFFICIENT",
        })
    return labels, audit


def cluster_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("gene").gene.transform("size").to_numpy(float)
    weights = 1/counts
    return weights / weights.mean()


@dataclass
class FittedRisk:
    preprocessor: NumericPreprocessor
    model: object
    columns: list[str]

    def predict(self, frame: pd.DataFrame, clip: bool = True):
        out = np.asarray(self.model.predict(self.preprocessor.transform(frame[self.columns].to_numpy(float))))
        return np.clip(out, 0, 1) if clip else out


def fit_risk(frame: pd.DataFrame, labels: np.ndarray, columns: list[str], kind: str,
             seed: int = SEEDS[0], weighted: bool = True) -> FittedRisk:
    labels = np.asarray(labels, float)
    valid = np.isfinite(labels)
    fit = frame.loc[valid].copy()
    if len(fit)<2:
        raise ValueError("no sufficient training-only CDF groups")
    transform = NumericPreprocessor().fit(fit[columns].to_numpy(float))
    model = Ridge(alpha=10) if kind=="ridge" else HistGradientBoostingRegressor(
        max_iter=200, learning_rate=.05, max_depth=3, min_samples_leaf=20,
        l2_regularization=10., random_state=seed)
    model.fit(transform.transform(fit[columns].to_numpy(float)), labels[valid],
              sample_weight=cluster_weights(fit) if weighted else None)
    return FittedRisk(transform, model, columns)


def shuffled_labels(frame: pd.DataFrame, labels: np.ndarray, seed: int):
    """Exchange complete cluster blocks with identical context/source layout."""
    labels = np.asarray(labels, float)
    out = labels.copy()
    signatures = {}
    blocks = {}
    order_cols = CDF_KEYS + ["task_id"]
    for cluster, group in frame.groupby("gene", sort=True):
        ordered = group.sort_values(order_cols)
        # Counts/layout must match; task IDs themselves are deliberately excluded.
        signature = tuple(sorted(tuple(k)+(int(n),) for k,n in group.groupby(CDF_KEYS).size().items()))
        signatures.setdefault(signature, []).append(cluster)
        blocks[cluster] = ordered.index.to_numpy()
    rng = np.random.default_rng(seed)
    moved = 0
    for clusters in signatures.values():
        if len(clusters)<2:
            continue
        shuffled = np.asarray(clusters, object)[rng.permutation(len(clusters))]
        # Cyclic assignment has no fixed point, preserving upstream/context blocks.
        for dest, source in zip(shuffled, np.roll(shuffled, 1)):
            out[blocks[dest]] = labels[blocks[source]]
            moved += 1
    return out, {"shuffle_seed":seed,"moved_clusters":moved,
                 "total_clusters":len(blocks),"moved_fraction":moved/max(len(blocks),1)}


def historical_distance(prediction: np.ndarray, vectors: np.ndarray, weights: np.ndarray):
    weights = np.asarray(weights, float)
    if len(vectors)==0 or weights.sum()<=0 or not np.isfinite(vectors).all():
        raise ValueError("historical distance needs real aligned history")
    weights = weights/weights.sum()
    prior = np.average(vectors, axis=0, weights=weights)
    dispersion = np.average(np.mean((vectors-prior)**2, axis=1), weights=weights)
    distance = np.mean((np.asarray(prediction)-prior)**2)
    return float(np.sqrt(distance+dispersion))


def metrics(frame: pd.DataFrame, scores: np.ndarray):
    truth = frame.true_error_rmse.to_numpy(float)
    ids = frame.task_id.to_numpy(str)
    scores = np.asarray(scores,float)
    valid = np.isfinite(scores) & np.isfinite(truth)
    truth,ids,scores = truth[valid],ids[valid],scores[valid]
    result = {"n_tasks":len(truth),"n_planned":len(frame),"utility20":np.nan,
              "spearman":np.nan,"aurc":np.nan,"high_risk_miss_rate":np.nan}
    if not len(truth): return result
    k = math.ceil(.2*len(truth))
    high = np.lexsort((ids,-scores))[:k]
    oracle = np.lexsort((ids,-truth))[:k]
    denom = truth[oracle].mean()-truth.mean()
    if len(truth)>=20 and denom>1e-12:
        result["utility20"] = float((truth[high].mean()-truth.mean())/denom)
    if len(truth)>=3 and np.ptp(scores)>0 and np.ptp(truth)>0:
        result["spearman"] = float(spearmanr(scores,truth).statistic)
    low = np.lexsort((ids,scores))
    result["aurc"] = float(np.mean(np.cumsum(truth[low])/np.arange(1,len(truth)+1)))
    result["high_risk_miss_rate"] = float(1-len(set(high)&set(oracle))/k)
    for c in (10,20,50):
        result[f"error_at_{c}"] = float(truth[low[:math.ceil(c/100*len(truth))]].mean())
    return result


def summarize(predictions: pd.DataFrame):
    group_cols=[c for c in ["line","seed","method","budget","order"] if c in predictions]
    strata=[]
    for key,part in predictions.groupby(group_cols+["target"],dropna=False,sort=True):
        strata.append(dict(zip(group_cols+["target"],key))|metrics(part,part.risk.to_numpy()))
    strata=pd.DataFrame(strata)
    metric_cols=["utility20","spearman","aurc","high_risk_miss_rate","error_at_10","error_at_20","error_at_50"]
    macro=strata.groupby(group_cols,dropna=False,as_index=False)[metric_cols].mean()
    return strata,macro


def bootstrap_u20(frame: pd.DataFrame, a: np.ndarray, b: np.ndarray, replicates=5000,seed=SEEDS[0]):
    """One paired cluster draw for all contexts; preserve pre-registered macro."""
    frame=frame.reset_index(drop=True)
    a,b=np.asarray(a,float),np.asarray(b,float)
    valid=np.isfinite(a)&np.isfinite(b)&np.isfinite(frame.true_error_rmse.to_numpy())
    frame=frame.loc[valid].reset_index(drop=True);a,b=a[valid],b[valid]
    clusters=sorted(frame.gene.astype(str).unique())
    idx=[np.flatnonzero(frame.gene.astype(str).to_numpy()==g) for g in clusters]
    contexts=frame.target.astype(str).to_numpy()
    truth=frame.true_error_rmse.to_numpy(float);ids=frame.task_id.to_numpy(str)
    def utility(use,score):
        n=len(use)
        if n<20:return np.nan
        k=math.ceil(.2*n);e=truth[use]
        hi=np.lexsort((ids[use],-score[use]))[:k];oracle=np.lexsort((ids[use],-e))[:k]
        den=e[oracle].mean()-e.mean()
        return (e[hi].mean()-e.mean())/den if den>1e-12 else np.nan
    def delta(use):
        ds=[]
        for c in sorted(set(contexts)):
            rows=use[contexts[use]==c]
            ds.append(utility(rows,a)-utility(rows,b))
        return float(np.nanmean(ds)) if np.isfinite(ds).any() else np.nan
    observed=delta(np.arange(len(frame)))
    rng=np.random.default_rng(seed)
    draws=np.asarray([delta(np.concatenate([idx[j] for j in rng.integers(0,len(idx),len(idx))]))
                      for _ in range(replicates)])
    return {"delta_utility20":observed,"bootstrap_mean_delta":float(np.nanmean(draws)),
            "ci95_lower":float(np.nanquantile(draws,.025)),"ci95_upper":float(np.nanquantile(draws,.975)),
            "n_tasks":len(frame),"n_clusters":len(clusters),"bootstrap_replicates":replicates}
