#!/usr/bin/env python3
"""Run a same-contract PertEMA official-algorithm adaptation on TxPert.

The released PertEMA demonstration artifact is tied to Rest/Stim8hr/Stim48hr.
We therefore do not apply its frozen weights out of domain. Instead this script
uses the official commit's XGBoost hyperparameters, gene-disjoint OOF error
training, inner isotonic calibration and 90% split-conformal interval on the
same TxPert tasks and label budget as SafeConf. Features are Universal-P only,
so the comparison tests whether historical evidence adds value beyond a
prediction-only PertEMA estimator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
OUT = STAGE / "pertema_fair_comparison"
SEED = 20260929
N_BOOTSTRAP = 5000
OFFICIAL_COMMIT = "43c09a32e23d0ee2ae5dfbab21b2deeab27f1803"
UNIVERSAL_P = [
    "predicted_magnitude", "prediction_abs_mean", "prediction_signed_mean",
    "prediction_std", "prediction_abs_q95", "prediction_sparsity",
]


def utility20(task_ids: np.ndarray, score: np.ndarray, outcome: np.ndarray) -> float:
    n = len(outcome)
    if n < 20:
        return float("nan")
    ids = np.asarray(task_ids, str)
    k = int(math.ceil(0.2 * n))
    chosen = np.lexsort((ids, -score))[:k]
    oracle = np.lexsort((ids, -outcome))[:k]
    denominator = float(outcome[oracle].mean() - outcome.mean())
    return float((outcome[chosen].mean() - outcome.mean()) / denominator) if denominator > 1e-12 else float("nan")


def rho(score: np.ndarray, outcome: np.ndarray) -> float:
    return float(spearmanr(score, outcome).statistic) if np.ptp(score) > 1e-15 else float("nan")


def aurc(score: np.ndarray, outcome: np.ndarray) -> float:
    order = np.argsort(score, kind="stable")  # low predicted error accepted first
    cumulative = np.cumsum(outcome[order]) / np.arange(1, len(order) + 1)
    return float(cumulative.mean())


def official_gbt(seed: int) -> xgb.XGBRegressor:
    # Exact scientific hyperparameters from PertEMA run_estimator.py.
    return xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, n_jobs=8, tree_method="hist", random_state=seed,
    )


def calibration_split(genes: np.ndarray, target: str, fold: int) -> tuple[set[str], set[str]]:
    unique = np.asarray(sorted(set(map(str, genes))))
    def stable(gene: str) -> str:
        return hashlib.sha256(f"PertEMA-v1\0{target}\0{fold}\0{gene}".encode()).hexdigest()
    ordered = sorted(unique, key=stable)
    n_cal = max(5, int(math.ceil(0.2 * len(ordered))))
    calibration = set(ordered[:n_cal])
    fit = set(ordered[n_cal:])
    if not fit or not calibration:
        raise ValueError("invalid calibration split")
    return fit, calibration


def run_one(upstream: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    src = STAGE / "safeconf_v4_development" / upstream
    frame = pd.read_csv(src / "FEATURE_TABLE.csv.gz")
    safe = pd.read_csv(src / "OOF_PREDICTIONS.csv.gz")
    folds = safe[safe.method.eq("Ridge_USR")][["task_id", "target", "fold"]]
    frame = frame.merge(folds, on=["task_id", "target"], how="inner", validate="one_to_one")
    rows, cal_rows = [], []
    for (target, fold), query in frame.groupby(["target", "fold"], sort=True):
        outer_train = frame[(frame.target.eq(target)) & (~frame.fold.eq(fold))].copy()
        fit_genes, cal_genes = calibration_split(outer_train.gene.to_numpy(str), str(target), int(fold))
        fit = outer_train[outer_train.gene.astype(str).isin(fit_genes)].copy()
        calibration = outer_train[outer_train.gene.astype(str).isin(cal_genes)].copy()
        if set(fit.gene) & set(calibration.gene) or set(outer_train.gene) & set(query.gene):
            raise AssertionError("gene leakage")
        model = official_gbt(SEED + int(fold))
        model.fit(fit[UNIVERSAL_P].to_numpy(float), fit.true_error_rmse.to_numpy(float))
        cal_raw = model.predict(calibration[UNIVERSAL_P].to_numpy(float))
        test_raw = model.predict(query[UNIVERSAL_P].to_numpy(float))
        iso = IsotonicRegression(out_of_bounds="clip").fit(cal_raw, calibration.true_error_rmse.to_numpy(float))
        test_cal = iso.predict(test_raw)
        cal_residual = np.abs(calibration.true_error_rmse.to_numpy(float) - iso.predict(cal_raw))
        # Finite-sample split-conformal quantile with the usual ceil rule.
        n_cal = len(cal_residual)
        rank = min(n_cal - 1, int(math.ceil((n_cal + 1) * 0.9)) - 1)
        qhat = float(np.sort(cal_residual)[rank])
        true = query.true_error_rmse.to_numpy(float)
        coverage = np.abs(true - test_cal) <= qhat
        for row, raw, calibrated, covered in zip(query.itertuples(), test_raw, test_cal, coverage):
            rows.append({
                "upstream": upstream, "task_id": row.task_id, "target": target,
                "gene": row.gene, "fold": int(fold), "method": "PertEMA_adapted",
                "predicted_risk": float(raw), "calibrated_risk": float(calibrated),
                "true_error_rmse": row.true_error_rmse, "conformal_qhat": qhat,
                "conformal_covered": bool(covered),
            })
        cal_rows.append({
            "upstream": upstream, "target": target, "fold": int(fold),
            "n_fit": len(fit), "n_calibration": len(calibration), "n_test": len(query),
            "n_fit_genes": len(fit_genes), "n_calibration_genes": len(cal_genes),
            "conformal_qhat": qhat, "conformal_coverage": float(coverage.mean()),
            "calibration_mae": float(np.mean(np.abs(test_cal - true))),
        })
    oof = pd.DataFrame(rows)
    if len(oof) != 1808 or oof.task_id.nunique() != 1808:
        raise AssertionError("incomplete PertEMA OOF output")

    joined = safe[safe.method.isin(["Magnitude_raw", "Ridge_USR", "V2_nested"])].copy()
    joined["upstream"] = upstream
    pertema = oof[["upstream", "task_id", "target", "gene", "fold", "true_error_rmse", "predicted_risk"]].copy()
    pertema["method"] = "PertEMA_adapted"
    all_pred = pd.concat([joined, pertema], ignore_index=True)
    strata = []
    for (target, method), group in all_pred.groupby(["target", "method"], sort=True):
        ids = group.task_id.to_numpy(str)
        risk = group.predicted_risk.to_numpy(float)
        truth = group.true_error_rmse.to_numpy(float)
        strata.append({
            "upstream": upstream, "target": target, "method": method, "n_tasks": len(group),
            "utility20": utility20(ids, risk, truth), "spearman": rho(risk, truth), "aurc": aurc(risk, truth),
        })
    return oof, pd.DataFrame(cal_rows), pd.DataFrame(strata)


def bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("SafeConf_V1_vs_PertEMA", "Ridge_USR", "PertEMA_adapted"),
        ("SafeConf_V2_vs_PertEMA", "V2_nested", "PertEMA_adapted"),
        ("PertEMA_vs_magnitude", "PertEMA_adapted", "Magnitude_raw"),
    ]
    wide = predictions.pivot(
        index=["upstream", "task_id", "target", "gene", "true_error_rmse"],
        columns="method", values="predicted_risk",
    ).reset_index()
    rng = np.random.default_rng(SEED)
    output = []
    for upstream, part in wide.groupby("upstream", sort=True):
        genes = np.asarray(sorted(part.gene.unique()))
        groups = [np.flatnonzero(part.gene.to_numpy() == gene) for gene in genes]
        ids = part.task_id.to_numpy(str)
        targets = part.target.to_numpy(str)
        truth = part.true_error_rmse.to_numpy(float)
        draws = {name: [] for name, _, _ in comparisons}
        rho_draws = {name: [] for name, _, _ in comparisons}
        for _ in range(N_BOOTSTRAP):
            selected = rng.integers(0, len(genes), len(genes))
            idx = np.concatenate([groups[i] for i in selected])
            stats = {}
            for method in sorted({m for _, a, b in comparisons for m in (a, b)}):
                us, rs = [], []
                for target in sorted(set(targets)):
                    use = idx[targets[idx] == target]
                    us.append(utility20(ids[use], part[method].to_numpy(float)[use], truth[use]))
                    rs.append(rho(part[method].to_numpy(float)[use], truth[use]))
                stats[method] = (float(np.nanmean(us)), float(np.nanmean(rs)))
            for name, a, b in comparisons:
                draws[name].append(stats[a][0] - stats[b][0])
                rho_draws[name].append(stats[a][1] - stats[b][1])
        for name, a, b in comparisons:
            d = np.asarray(draws[name]); r = np.asarray(rho_draws[name])
            output.append({
                "upstream": upstream, "comparison": name, "method_a": a, "method_b": b,
                "delta_utility20": float(np.nanmean(d)),
                "utility_ci95_lower": float(np.nanquantile(d, 0.025)),
                "utility_ci95_upper": float(np.nanquantile(d, 0.975)),
                "delta_spearman": float(np.nanmean(r)),
                "spearman_ci95_lower": float(np.nanquantile(r, 0.025)),
                "spearman_ci95_upper": float(np.nanquantile(r, 0.975)),
                "bootstrap_replicates": N_BOOTSTRAP,
            })
    return pd.DataFrame(output)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "RUN_STATUS.json").exists():
        raise FileExistsError("refusing to overwrite registered PertEMA comparison")
    started = time.time()
    oofs, calibrations, strata = [], [], []
    all_predictions = []
    for upstream in ("txpert_gat", "txpert_exphormer"):
        oof, cal, result = run_one(upstream)
        oofs.append(oof); calibrations.append(cal); strata.append(result)
        src = STAGE / "safeconf_v4_development" / upstream / "OOF_PREDICTIONS.csv.gz"
        safe = pd.read_csv(src)
        safe = safe[safe.method.isin(["Magnitude_raw", "Ridge_USR", "V2_nested"])].copy()
        safe["upstream"] = upstream
        pertema = oof[["upstream", "task_id", "target", "gene", "fold", "true_error_rmse", "predicted_risk", "method"]]
        all_predictions.extend([safe, pertema])
        print(f"[PertEMA adapted] {upstream} complete", flush=True)
    oof = pd.concat(oofs, ignore_index=True)
    calibration = pd.concat(calibrations, ignore_index=True)
    stratum = pd.concat(strata, ignore_index=True)
    summary = stratum.groupby(["upstream", "method"], as_index=False).agg(
        n_strata=("target", "nunique"), utility20=("utility20", "mean"),
        spearman=("spearman", "mean"), aurc=("aurc", "mean"),
    )
    all_pred = pd.concat(all_predictions, ignore_index=True)
    intervals = bootstrap(all_pred)
    oof.to_csv(OUT / "PERTEMA_OOF_PREDICTIONS.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    calibration.to_csv(OUT / "CALIBRATION_RESULTS.csv", index=False)
    stratum.to_csv(OUT / "STRATUM_RESULTS.csv", index=False)
    summary.to_csv(OUT / "SUMMARY.csv", index=False)
    intervals.to_csv(OUT / "PAIRED_BOOTSTRAP.csv", index=False)
    status = {
        "status": "COMPLETE", "elapsed_seconds": time.time() - started,
        "official_software_commit": OFFICIAL_COMMIT,
        "implementation_label": "official-algorithm adaptation; not bundled-weight reproduction",
        "official_gbt_hyperparameters": {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
        },
        "same_tasks": True, "same_outer_label_budget": True,
        "outer_gene_disjoint": True, "inner_isotonic_calibration": True,
        "split_conformal_target": 0.90, "features": UNIVERSAL_P,
        "sealed_confirmation_opened": False,
        "summary_sha256": hashlib.sha256((OUT / "SUMMARY.csv").read_bytes()).hexdigest(),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(intervals.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
