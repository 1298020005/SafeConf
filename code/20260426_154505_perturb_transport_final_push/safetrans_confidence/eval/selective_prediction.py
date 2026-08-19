"""Selective-prediction utility metrics with clustered uncertainty.

Risk-coverage at a single 80% point is cherry-pick-prone and non-monotone on
some datasets (e.g. McFarland). This module provides the standard selective
prediction summary used in the literature (Geifman & El-Yaniv 2017; Galil &
El-Yaniv 2021):

- the full risk-coverage curve over all thresholds,
- AURC (area under risk-coverage), oracle-AURC, random-AURC,
- excess-AURC = AURC(scorer) - AURC(oracle),
- AURC reduction vs random = (random - AURC) / random, in [<=0, 1],

all with task-cluster bootstrap confidence intervals that resample whole task
clusters (so the two aligned predictor rows per task stay together and the
intervals are not falsely narrow).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from safetrans_confidence.eval.metrics import (
    compute_aurc,
    compute_excess_aurc,
    compute_oracle_aurc,
    compute_random_aurc,
)


def risk_coverage_curve(errors: np.ndarray, risk: np.ndarray) -> pd.DataFrame:
    """Full risk-coverage curve. ``risk`` is the aligned risk axis (higher = riskier)."""
    errors = np.asarray(errors, dtype=float)
    risk = np.asarray(risk, dtype=float)
    mask = np.isfinite(errors) & np.isfinite(risk)
    errors, risk = errors[mask], risk[mask]
    n = len(errors)
    if n < 2:
        return pd.DataFrame(columns=["coverage", "n_kept", "selective_risk"])
    order = np.argsort(risk)  # keep lowest-risk first
    sorted_errors = errors[order]
    cumrisk = np.cumsum(sorted_errors) / np.arange(1, n + 1)
    coverage = np.arange(1, n + 1) / n
    return pd.DataFrame(
        {"coverage": coverage, "n_kept": np.arange(1, n + 1), "selective_risk": cumrisk}
    )


def selective_prediction_summary(errors: np.ndarray, risk: np.ndarray) -> dict:
    """Scalar AURC family for an aligned risk axis (higher risk = worse)."""
    errors = np.asarray(errors, dtype=float)
    risk = np.asarray(risk, dtype=float)
    aurc = compute_aurc(errors, risk, "risk")
    oracle = compute_oracle_aurc(errors)
    random_a = compute_random_aurc(errors)
    excess = compute_excess_aurc(errors, risk, "risk")
    norm_aurc = aurc / random_a if (np.isfinite(aurc) and np.isfinite(random_a) and random_a > 0) else np.nan
    reduction = (random_a - aurc) / random_a if (np.isfinite(aurc) and np.isfinite(random_a) and random_a > 0) else np.nan
    # excess captured: how much of the avoidable (random-oracle) gap the scorer closes
    avoidable = random_a - oracle
    captured = (random_a - aurc) / avoidable if (np.isfinite(avoidable) and avoidable > 0) else np.nan
    return {
        "n": int(np.isfinite(errors).sum()),
        "aurc": float(aurc) if np.isfinite(aurc) else np.nan,
        "oracle_aurc": float(oracle) if np.isfinite(oracle) else np.nan,
        "random_aurc": float(random_a) if np.isfinite(random_a) else np.nan,
        "excess_aurc": float(excess) if np.isfinite(excess) else np.nan,
        "normalized_aurc": float(norm_aurc) if np.isfinite(norm_aurc) else np.nan,
        "aurc_reduction_vs_random_pct": float(100.0 * reduction) if np.isfinite(reduction) else np.nan,
        "avoidable_gap_captured_pct": float(100.0 * captured) if np.isfinite(captured) else np.nan,
    }


def clustered_bootstrap_aurc(
    df: pd.DataFrame,
    error_col: str,
    risk_col: str,
    cluster_col: str = "task_key",
    n_bootstrap: int = 1000,
    seed: int = 5201,
    metrics: tuple[str, ...] = ("excess_aurc", "aurc_reduction_vs_random_pct"),
) -> dict:
    """Task-cluster bootstrap CIs for AURC-family metrics.

    Resamples whole clusters (e.g. task_key) with replacement so dependent rows
    stay together. Returns {metric: (lo, hi)} 95% intervals plus the point
    estimate on the full data.
    """
    work = df.dropna(subset=[error_col, risk_col]).copy()
    if cluster_col not in work.columns:
        work[cluster_col] = np.arange(len(work))
    point = selective_prediction_summary(
        work[error_col].to_numpy(), work[risk_col].to_numpy()
    )
    out: dict = {f"{m}_point": point.get(m, np.nan) for m in metrics}
    clusters = work[cluster_col].dropna().unique()
    if len(clusters) < 4 or len(work) < 8:
        for m in metrics:
            out[f"{m}_ci_low"] = np.nan
            out[f"{m}_ci_high"] = np.nan
        out["n_clusters"] = int(len(clusters))
        out["n"] = int(len(work))
        return out
    # Pre-split cluster -> numpy arrays so each bootstrap is array concat (fast).
    err_by_cluster: list[np.ndarray] = []
    risk_by_cluster: list[np.ndarray] = []
    for _, g in work.groupby(cluster_col):
        err_by_cluster.append(g[error_col].to_numpy(dtype=float))
        risk_by_cluster.append(g[risk_col].to_numpy(dtype=float))
    n_clusters = len(err_by_cluster)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_bootstrap):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        boot_err = np.concatenate([err_by_cluster[i] for i in pick])
        boot_risk = np.concatenate([risk_by_cluster[i] for i in pick])
        s = selective_prediction_summary(boot_err, boot_risk)
        for m in metrics:
            v = s.get(m, np.nan)
            if np.isfinite(v):
                samples[m].append(float(v))
    for m in metrics:
        vals = samples[m]
        if len(vals) >= max(20, n_bootstrap // 10):
            lo, hi = np.quantile(vals, [0.025, 0.975])
            out[f"{m}_ci_low"], out[f"{m}_ci_high"] = float(lo), float(hi)
        else:
            out[f"{m}_ci_low"], out[f"{m}_ci_high"] = np.nan, np.nan
    out["n_clusters"] = int(len(clusters))
    out["n"] = int(len(work))
    return out


def within_magnitude_stratum_rho(
    df: pd.DataFrame,
    risk_col: str,
    error_col: str,
    magnitude_col: str,
    n_bins: int = 4,
) -> pd.DataFrame:
    """Spearman(risk, error) computed within effect-magnitude strata.

    If the score still ranks error inside narrow magnitude bins, the signal is
    not merely an effect-magnitude proxy.
    """
    work = df.dropna(subset=[risk_col, error_col, magnitude_col]).copy()
    if len(work) < n_bins * 4:
        return pd.DataFrame()
    try:
        work["_mag_bin"] = pd.qcut(
            work[magnitude_col].rank(method="first"), q=n_bins, labels=False
        )
    except ValueError:
        return pd.DataFrame()
    rows = []
    for b, g in work.groupby("_mag_bin"):
        r = pd.to_numeric(g[risk_col], errors="coerce")
        e = pd.to_numeric(g[error_col], errors="coerce")
        m = r.notna() & e.notna()
        rho = float(r[m].corr(e[m], method="spearman")) if int(m.sum()) >= 5 else np.nan
        rows.append(
            {
                "magnitude_bin": int(b),
                "n": int(len(g)),
                "mag_min": float(g[magnitude_col].min()),
                "mag_max": float(g[magnitude_col].max()),
                "within_bin_rho_risk_vs_error": rho,
            }
        )
    return pd.DataFrame(rows)
