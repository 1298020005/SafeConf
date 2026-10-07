#!/usr/bin/env python3
"""Public-reference stability audit and current-contract feature builder.

The method is intentionally fixed: experiment-record-level weighted
delete-one jackknife for the stability of the historical mean.  It is not
called biological heterogeneity or a prediction-error bound.  E258 is used
only as a legal development validation of the meaning of the statistic; the
current McFaline holdout receives feature values but no selection metric in
this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path("/home/yyf/runtime_artifacts")
IMPL = RUNTIME / "safeconf_impl_20261004_v1/public_reliability"
FEEDBACK = RUNTIME / "safeconf_research_20261003/feedback_v1"
COMMON = RUNTIME / "safeconf_research_20261001/common_gene_axis"
E258 = Path("/home/yyf/data/feng2025_candidate/E258_OFFICIAL_DEV_VIEW.npz")
SEEDS = (20260930, 20261001, 20261002)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.sqrt(np.mean(d * d)))


def jackknife_mean(history: np.ndarray, weights: np.ndarray) -> tuple[float, float, str]:
    """Return weighted mean, delete-one mean stability, and status.

    A task with fewer than three independent experiment records is explicitly
    not estimable.  The output J is a scalar RMS over the fixed output axis.
    """
    history = np.asarray(history, float)
    weights = np.asarray(weights, float)
    if history.ndim != 2 or len(history) != len(weights):
        return np.nan, np.nan, "invalid_shape"
    if len(history) < 3 or not np.isfinite(history).all() or not np.isfinite(weights).all():
        return np.nan, np.nan, "insufficient_or_nonfinite"
    if (weights <= 0).any() or np.count_nonzero(weights) < 2:
        return np.nan, np.nan, "invalid_weights"
    weights = weights / weights.sum()
    mean = weights @ history
    loo = []
    for i in range(len(history)):
        keep = np.ones(len(history), dtype=bool)
        keep[i] = False
        denom = weights[keep].sum()
        if denom <= 0:
            return np.nan, np.nan, "degenerate_leave_one_out"
        loo.append((weights[keep] @ history[keep]) / denom)
    loo = np.asarray(loo)
    # Delete-one jackknife variance is the outer coefficient times the
    # *sum* of leave-one-out deviations.  The previous implementation took
    # an additional mean over records, shrinking J by a factor of m and
    # making tasks with many experiments look spuriously stable.
    j2 = (len(history) - 1) / len(history) * np.sum(np.mean((loo - mean) ** 2, axis=1))
    return float(np.sqrt(np.mean(mean * mean))), float(np.sqrt(max(j2, 0.0))), "ok"


def e258_validation(out: Path) -> dict:
    z = np.load(E258, allow_pickle=True)
    task_line = z["task_line"].astype(str)
    target_gene = z["task_target"].astype(str)
    donor = z["task_donor"].astype(str)
    effects = np.asarray(z["effect"], float)
    axis = z["gene"].astype(str)
    if len(task_line) != effects.shape[0] or len(target_gene) != effects.shape[0]:
        raise RuntimeError("E258 task/effect shape mismatch")
    validation_donors = {"eipl", "oikd"}
    training_donors = {"pipw", "kolf", "paab", "fiaj"}
    records = []
    by_gene_donor: dict[tuple[str, str], list[int]] = {}
    for i, (g, d) in enumerate(zip(target_gene, donor)):
        if d in training_donors:
            by_gene_donor.setdefault((g, d), []).append(i)
    for i, (line, g, d) in enumerate(zip(task_line, target_gene, donor)):
        if d not in validation_donors:
            continue
        donors, vectors = [], []
        for td in sorted(training_donors):
            idx = by_gene_donor.get((g, td), [])
            if not idx:
                continue
            # Multiple lines from one donor are first averaged: they are not
            # independent biological sources for this audit.
            vec = np.mean(effects[np.asarray(idx)], axis=0)
            mask = axis != g  # fixed trans axis; never use the targeted gene
            donors.append(td)
            vectors.append(vec[mask])
        if len(vectors) < 2:
            continue
        hist = np.asarray(vectors)
        w = np.full(len(hist), 1.0 / len(hist))
        mu = w @ hist
        _, j, status = jackknife_mean(hist, w)
        v = float(np.sqrt(np.sum(w * np.mean((hist - mu) ** 2, axis=1))))
        heldout = effects[i][axis != g]
        records.append({
            "task_line": line, "target_gene": g, "target_donor": d,
            "n_history_donors": len(donors), "history_donors": ";".join(donors),
            "reference_holdout_rmse": rmse(mu, heldout),
            "jackknife_mean_stability": j, "new_response_spread": v,
            "support_count": len(donors), "history_status": status,
        })
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("E258 produced no validation reference records")
    rows = []
    for line, part in frame.groupby("task_line", sort=True):
        for feature in ["jackknife_mean_stability", "new_response_spread", "support_count"]:
            valid = np.isfinite(part[feature]) & np.isfinite(part.reference_holdout_rmse)
            rho = spearmanr(part.loc[valid, feature], part.loc[valid, "reference_holdout_rmse"]).statistic if valid.sum() >= 3 else np.nan
            rows.append({"task_line": line, "feature": feature, "n": int(valid.sum()), "spearman": float(rho) if np.isfinite(rho) else np.nan})
    summary = pd.DataFrame(rows)
    line_piv = summary.pivot(index="task_line", columns="feature", values="spearman").reset_index()
    comparable = line_piv.dropna(subset=["jackknife_mean_stability", "new_response_spread", "support_count"])
    j_better = int((comparable.jackknife_mean_stability > comparable.support_count).sum()) if len(comparable) else 0
    valid = bool(len(comparable) >= 3 and j_better >= max(2, math.ceil(0.75 * len(comparable))))
    write_csv(out / "E258_REFERENCE_RELIABILITY.csv", frame)
    write_csv(out / "E258_REFERENCE_RELIABILITY_SUMMARY.csv", summary)
    write_json(out / "E258_VALIDATION_DECISION.json", {
        "status": "PASS" if valid else "FAIL",
        "meaning": "jackknife stability predicts held-out donor reference deviation",
        "n_records": len(frame), "n_lines": int(frame.task_line.nunique()),
        "comparable_lines": int(len(comparable)), "jackknife_better_than_support_lines": j_better,
        "selection_rule": ">=3 comparable lines and jackknife beats support in >=75% of them",
        "test_truth_opened": False, "e258_sha256": sha(E258),
    })
    return {"pass": valid, "records": len(frame), "lines": int(frame.task_line.nunique())}


def task_jackknife_features(frame: pd.DataFrame, out: Path, label: str) -> pd.DataFrame:
    memory_path = COMMON / "public_mcfaline_trainval/public_memory.parquet"
    # Match the exact reference estimand of the cached distance. DEV uses
    # the all-DEV-query-excluded guide-effect bank; the fixed feedback/holdout
    # contract uses the later cell-weighted aligned bank.
    is_fixed_holdout = label == 'MCFALINE_HOLDOUT'
    effects_path = (COMMON / 'reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy'
                    if is_fixed_holdout else COMMON / 'public_mcfaline_trainval/effect_vectors.npy')
    memory = pd.read_parquet(memory_path).sort_values("effect_vector_row").reset_index(drop=True)
    if memory.experiment_id.duplicated().any():
        raise ValueError('duplicate public experiment units')
    effects = np.asarray(np.load(effects_path, mmap_mode="r"), float)
    by_gene = memory.groupby("perturbation_target", sort=False).indices
    prohibited = set('McFaline23::' + frame.task_id.astype(str))
    global_allowed = ~memory.experiment_id.astype(str).isin(prohibited).to_numpy()
    records = []
    for row in frame.itertuples(index=False):
        idx = np.asarray(by_gene.get(str(row.gene), []), int)
        if not len(idx):
            records.append({"task_id": row.task_id, "status": "no_history"})
            continue
        idx = idx[global_allowed[idx]]
        query_id = f"McFaline23::{row.task_id}"
        idx = idx[memory.iloc[idx].experiment_id.astype(str).to_numpy() != query_id]
        if is_fixed_holdout:
            query_condition = str(getattr(row, 'treatment', getattr(row, 'condition', '')))
            mm = memory.iloc[idx]
            idx = idx[~((mm.context.astype(str) == str(row.context)) &
                        (mm.condition.astype(str) == query_condition)).to_numpy()]
        if not len(idx):
            records.append({"task_id": row.task_id, "status": "no_history_after_query_exclusion"})
            continue
        m = memory.iloc[idx]
        same = m.context.astype(str).to_numpy() == str(row.context)
        chosen = idx[same] if same.any() else idx
        m = memory.iloc[chosen]
        h = effects[chosen]
        w = m.n_cells.to_numpy(float)
        ok = np.isfinite(w) & (w > 0) & np.isfinite(h).all(axis=1)
        h, w = h[ok], w[ok]
        m = m.iloc[np.flatnonzero(ok)]
        if len(h) < 1:
            records.append({"task_id": row.task_id, "status": "nonfinite_history"})
            continue
        w = w / w.sum()
        mu = w @ h
        pred = np.asarray([getattr(row, "prediction_prior_rmse", np.nan)])
        d2 = float(getattr(row, "prediction_prior_rmse", np.nan)) ** 2
        v2 = float(np.sum(w * np.mean((h - mu) ** 2, axis=1)))
        if not np.isclose(np.sqrt(v2), float(row.prior_uncertainty), rtol=1e-5, atol=1e-8):
            raise ValueError(f'reliability reference does not match cached distance for {row.task_id}')
        _, j, status = jackknife_mean(h, w)
        records.append({
            "task_id": row.task_id, "status": status, "n_history_records": len(h),
            "effective_sources": float(1.0 / np.sum(w * w)),
            "prediction_to_reference": float(np.sqrt(max(d2, 0.0))),
            "prior_uncertainty": float(np.sqrt(max(v2, 0.0))),
            "jackknife_mean_stability": j,
            "public_rule": float(np.sqrt(max(d2 + v2, 0.0))),
            "public_mean_jackknife": float(np.sqrt(max(d2 + j * j, 0.0))) if np.isfinite(j) else np.nan,
            "history_context_match": bool(same.any()),
            "history_experiment_ids_hash": hashlib.sha256('\n'.join(sorted(m.experiment_id.astype(str))).encode()).hexdigest(),
            "effect_estimand": 'cell-weighted' if is_fixed_holdout else 'registered DEV guide-effect',
        })
    result = pd.DataFrame(records)
    write_csv(out / f"{label}_JACKKNIFE_FEATURES.csv", result)
    return result


def mcfaline_dev(out: Path) -> None:
    dev = pd.read_parquet(FEEDBACK / "DEV_FEATURES.parquet")
    hold = pd.read_parquet(FEEDBACK / "HOLDOUT_FEATURES.parquet")
    dev_j = task_jackknife_features(dev, out, "MCFALINE_DEV")
    hold_j = task_jackknife_features(hold, out, "MCFALINE_HOLDOUT")
    merged = dev[["task_id", "target", "gene", "true_error_rmse", "predicted_magnitude", "prior_uncertainty", "prediction_prior_rmse"]].merge(dev_j, on="task_id", how="left")
    rows = []
    for method in ["predicted_magnitude", "public_rule", "prediction_to_reference", "public_mean_jackknife"]:
        part = merged.dropna(subset=[method, "true_error_rmse"]).copy()
        for target, g in part.groupby("target", sort=True):
            y, s = g.true_error_rmse.to_numpy(float), g[method].to_numpy(float)
            ids = g.task_id.astype(str).to_numpy(); k = max(1, int(math.ceil(.2 * len(g))))
            high = np.lexsort((ids, -s))[:k]; oracle = np.lexsort((ids, -y))[:k]
            den = y[oracle].mean() - y.mean()
            rows.append({"method": method, "target": target, "n": len(g),
                         "u20": float((y[high].mean() - y.mean()) / den) if len(g) >= 20 and den > 1e-12 else np.nan,
                         "spearman": float(spearmanr(s, y).statistic) if np.ptp(s) > 0 and np.ptp(y) > 0 else np.nan})
    write_csv(out / "MCFALINE_DEV_RISK_COMPARISON.csv", pd.DataFrame(rows))
    write_json(out / "MCFALINE_FEATURE_STATUS.json", {
        "status": "COMPLETE", "dev_rows": len(dev), "holdout_rows": len(hold),
        "holdout_scores_only": True, "holdout_metrics_used_for_selection": False,
        "method": "weighted experiment-record delete-one jackknife",
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["preflight", "e258_validation", "mcfaline_dev", "all"], default="all")
    ap.add_argument("--output", type=Path, default=IMPL)
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.phase in {"preflight", "all"}:
        write_json(out / "PUBLIC_RELIABILITY_PREFLIGHT.json", {
            "status": "PASS" if E258.exists() else "FAIL",
            "e258": str(E258), "public_memory": str(COMMON / "public_mcfaline_trainval/public_memory.parquet"),
            "method": "weighted experiment-record delete-one jackknife", "query_truth_used": False,
        })
    if args.phase in {"e258_validation", "all"}:
        e258_validation(out)
    if args.phase in {"mcfaline_dev", "all"}:
        mcfaline_dev(out)
    print(json.dumps({"status": "COMPLETE", "phase": args.phase, "output": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
