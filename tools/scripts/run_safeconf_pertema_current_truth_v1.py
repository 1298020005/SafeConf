#!/usr/bin/env python3
"""Current-truth PertEMA-compatible adaptation.

The official factory and tree parameters are reused, but the CD4 bundled
features/weights are never applied out of domain.  P6 is available for all
current tasks; the 61-dimensional native-control feature adaptation is saved
separately and is only a full-cohort comparison when task IDs are complete.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import P, bootstrap_u20, rank_labels, summarize  # noqa: E402
from tools.scripts import run_safeconf_research_closure as closure  # noqa: E402

POOL = Path("/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1/FIXED_POOL_FEATURES.parquet")
HOLD = Path("/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1/HOLDOUT_FEATURES.parquet")
NATIVE = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/native_control_reference/TASK_FEATURES.parquet")
OFFICIAL_ROOT = Path("/home/yyf/runtime_artifacts/official_pertema_43c09a")
OFFICIAL_COMMIT = "43c09a32e23d0ee2ae5dfbab21b2deeab27f1803"
SEEDS = (20260930, 20261001, 20261002)
BUDGETS = (0.1, 0.25, 0.5, 0.75, 1.0)
P6 = P
NATIVE_FEATURES = [
    "native_prediction_abs_mean", "native_control_baseline", "native_control_dropout",
    "native_control_plate_variance_proxy", "native_state_0", "native_state_1",
    "native_state_2", "native_state_3", "native_state_4", "native_state_5",
] + [f"native_embedding_{i}" for i in range(50)] + ["native_training_similarity"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(b)
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


def official_gbt(seed: int):
    head = subprocess.check_output(["git", "-C", str(OFFICIAL_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if head != OFFICIAL_COMMIT:
        raise RuntimeError(f"official PertEMA source changed: {head}")
    return xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, n_jobs=4, tree_method="hist",
        random_state=seed, objective="reg:squarederror",
    )


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pool = pd.read_parquet(POOL).copy()
    hold = pd.read_parquet(HOLD).copy()
    native = pd.read_parquet(NATIVE).set_index("task_id")
    audit = {"pool_rows": len(pool), "pool_genes": int(pool.gene.nunique()),
             "holdout_rows": len(hold), "holdout_genes": int(hold.gene.nunique()),
             "native_rows": len(native), "native_feature_count": len(NATIVE_FEATURES),
             "native_source_sha256": sha(NATIVE)}
    if len(pool) != 331 or pool.gene.nunique() != 228 or len(hold) != 212 or hold.gene.nunique() != 152:
        raise RuntimeError("current feedback cohort changed")
    for frame in (pool, hold):
        if frame.task_id.isin(native.index).sum() != len(frame):
            audit.setdefault("native_coverage", {})["rows"] = int(frame.task_id.isin(native.index).sum())
    for c in NATIVE_FEATURES:
        if c not in native.columns:
            raise RuntimeError(f"native feature missing: {c}")
    # Add native features by task identity.  Truth from the old native file is
    # intentionally ignored; current pool/holdout truth is authoritative.
    pool = pool.join(native[NATIVE_FEATURES], on="task_id", how="left", rsuffix="_native")
    hold = hold.join(native[NATIVE_FEATURES], on="task_id", how="left", rsuffix="_native")
    audit["native_pool_task_rows_covered"] = int(pool[NATIVE_FEATURES[0]].notna().sum())
    audit["native_holdout_task_rows_covered"] = int(hold[NATIVE_FEATURES[0]].notna().sum())
    audit["native_pool_all_features_finite_rows"] = int(pool[NATIVE_FEATURES].notna().all(axis=1).sum())
    audit["native_holdout_all_features_finite_rows"] = int(hold[NATIVE_FEATURES].notna().all(axis=1).sum())
    audit["native_truth_not_used"] = True
    return pool, hold, audit


def fixed_gene_order(pool: pd.DataFrame) -> list[str]:
    return sorted(pool.gene.astype(str).unique(), key=lambda g: hashlib.sha256(
        f"SafeConf-McFaline-feedback-v1\0{g}".encode()).hexdigest())


def mapped_labels(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    # Map validation errors through the CDF fitted on the inner training rows.
    from tools.safeconf_continual.contracts import FrozenErrorCDF
    from tools.safeconf_continual.research import CDF_KEYS
    out = np.full(len(val), np.nan)
    for key, group in train.groupby(CDF_KEYS, dropna=False, sort=True):
        mask = np.ones(len(val), bool)
        for col, value in zip(CDF_KEYS, key):
            mask &= val[col].eq(value).to_numpy()
        if len(group) >= 2:
            out[mask] = FrozenErrorCDF.fit(group.true_error_rmse.to_numpy(float)).transform(
                val.loc[mask, "true_error_rmse"].to_numpy(float))
    return out


def fit_one(train: pd.DataFrame, labels: np.ndarray, columns: list[str], seed: int):
    # XGBoost's native missing-value path is part of the fixed official
    # implementation; only the target labels must be finite.
    valid = np.isfinite(labels)
    if valid.sum() < 2:
        raise RuntimeError("insufficient finite PertEMA training rows")
    model = official_gbt(seed)
    model.fit(train.loc[valid, columns].to_numpy(float), labels[valid])
    return model


def run(args) -> int:
    pool, hold, audit = load_frames()
    out = args.result_root.resolve(); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "PERTEMA_INPUT_AUDIT.json", {
        "status": "PASS", "official_commit": OFFICIAL_COMMIT,
        "official_source_sha256": sha(OFFICIAL_ROOT / "src/pertema/run_estimator.py"),
        "feature_sets": {"P6": P6, "Native61": NATIVE_FEATURES},
        "pool_path": str(POOL), "holdout_path": str(HOLD), "native_path": str(NATIVE),
        **audit, "current_truth_used": True, "permanent_test_truth_opened": False,
    })
    if args.phase == "preflight":
        write_json(out / "RUN_STATUS.json", {"status": "PREFLIGHT_PASS", **audit})
        print(json.dumps({"status": "PREFLIGHT_PASS", **audit})); return 0
    order = fixed_gene_order(pool)
    frames = {"P6": (pool, hold, P6), "Native61": (pool, hold, NATIVE_FEATURES)}
    records, fits = [], []
    for budget in args.budgets:
        n_genes = int(np.ceil(budget * len(order)))
        fit_pool = pool[pool.gene.isin(set(order[:n_genes]))].reset_index(drop=True)
        labels, cdf_audit = rank_labels(fit_pool, f"pertema-current/{budget}", budget)
        write_csv(out / f"CDF_AUDIT_budget_{budget:.2f}.csv", pd.DataFrame(cdf_audit))
        for feature_name, (train_frame, query_frame, columns) in frames.items():
            for seed in args.seeds:
                inner_raw = np.full(len(fit_pool), np.nan)
                groups = fit_pool.gene.astype(str).to_numpy()
                n_splits = min(3, fit_pool.gene.nunique())
                for inner, (i, j) in enumerate(GroupKFold(n_splits=n_splits).split(fit_pool, groups=groups)):
                    tr = fit_pool.iloc[i].reset_index(drop=True)
                    va = fit_pool.iloc[j].reset_index(drop=True)
                    inner_labels, _ = rank_labels(tr, f"pertema-current/{budget}/{feature_name}/{seed}/inner{inner}", budget)
                    model = fit_one(tr, inner_labels, columns, seed)
                    inner_raw[j] = model.predict(va[columns].to_numpy(float))
                inner_target = mapped_labels(fit_pool, fit_pool)
                finite = np.isfinite(inner_raw) & np.isfinite(inner_target)
                iso = IsotonicRegression(out_of_bounds="clip").fit(inner_raw[finite], inner_target[finite]) if finite.sum() >= 5 and np.ptp(inner_raw[finite]) > 0 else None
                model = fit_one(fit_pool, labels, columns, seed)
                query_x = query_frame[columns].to_numpy(float)
                raw = model.predict(query_x)
                calibrated = iso.predict(raw) if iso is not None else np.full(len(raw), np.nan)
                for score_name, score in [("PertEMA_raw", raw), ("PertEMA_isotonic", calibrated)]:
                    part = query_frame[["task_id", "target", "gene", "true_error_rmse"]].copy()
                    part["budget"] = budget; part["feature_set"] = feature_name; part["seed"] = seed
                    part["method"] = score_name; part["risk"] = np.clip(score, 0, 1)
                    records.append(part)
                fits.append({"budget": budget, "feature_set": feature_name, "seed": seed,
                             "n_fit_rows": len(fit_pool), "n_fit_genes": n_genes,
                             "inner_oof_finite": int(np.isfinite(inner_raw).sum()),
                             "isotonic_fitted": bool(iso is not None), "columns": columns})
    pred = pd.concat(records, ignore_index=True)
    pred.to_parquet(out / "PERTEMA_CURRENT_TASK_PREDICTIONS.parquet", index=False)
    write_csv(out / "PERTEMA_CURRENT_FIT_LEDGER.csv", pd.DataFrame(fits))
    strata, macro = summarize(pred.rename(columns={"risk": "risk"}))
    write_csv(out / "PERTEMA_CURRENT_STRATA.csv", strata)
    write_csv(out / "PERTEMA_CURRENT_MACRO.csv", macro)
    # Holdout comparison is descriptive: configuration was already fixed by
    # the existing DEV contract, and PublicRule remains a separate baseline.
    hold_public = hold.set_index("task_id").simple_history_risk
    comparisons = []
    for (budget, feature, method), part in pred.groupby(["budget", "feature_set", "method"], sort=True):
        q = part.sort_values("task_id")
        if q.risk.notna().sum() < 20:
            continue
        wide = q[["task_id", "target", "gene", "true_error_rmse", "risk"]].copy()
        public = wide.task_id.map(hold_public).to_numpy(float)
        boot = bootstrap_u20(wide, wide.risk.to_numpy(float), public, 5000, 20260930)
        comparisons.append({"budget": budget, "feature_set": feature, "method": method,
                            **boot})
    write_csv(out / "PERTEMA_CURRENT_PAIRED_BOOTSTRAP.csv", pd.DataFrame(comparisons))
    write_json(out / "PERTEMA_CURRENT_RUN_STATUS.json", {
        "status": "COMPLETE", "phase": args.phase, "fit_count": len(fits),
        "budgets": args.budgets, "seeds": args.seeds,
        "primary_score": "PertEMA_raw", "isotonic_secondary": True,
        "current_truth_contract": True, "permanent_test_truth_opened": False,
    })
    print(json.dumps({"status": "COMPLETE", "fits": len(fits), "rows": len(pred)}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["preflight", "fit"], default="preflight")
    ap.add_argument("--budgets", type=float, nargs="+", default=list(BUDGETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--result-root", type=Path, required=True)
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
