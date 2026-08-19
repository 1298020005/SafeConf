#!/usr/bin/env python3
"""SafeConf cross-dataset reliability model and baseline ladder.

This experiment upgrades SafeConf from a frozen per-dataset rule audit to a
transferable reliability model and tests it honestly against the effect-
magnitude confound.

Pipeline:

1. Load corrected per-dataset PredictionRecord/feature runs (no perturbation
   prediction is trained here; predictors are frozen upstream outputs).
2. Attach true-effect L2 magnitude and a within-group normalized RMSE.
3. Quantile-normalize leakage-safe features within (dataset, fold, predictor)
   so they are comparable across datasets.
4. Build a baseline ladder of risk scores:
     - random
     - predicted_magnitude (deployable; uses predicted-effect norm only)
     - oracle_magnitude (diagnostic only; uses TRUE-effect norm = confound ceiling)
     - frozen protocol v0.2 (recomputed)
     - safeconf_perdataset (learned within each dataset, normalized features)
     - safeconf_lodo (learned leave-one-dataset-out, normalized features) <- main
5. Evaluate every score with aligned rho, magnitude-partialled rho, and the
   selective-prediction AURC family with task-cluster bootstrap CIs.
6. Add within-magnitude-stratum rho for the main scores.

The decisive question: does safeconf_lodo generalize to an unseen dataset and
beat the deployable predicted_magnitude baseline on magnitude-partialled rho
and excess-AURC?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.data.records import find_effect_array_files, load_merged_records
from safetrans_confidence.eval.selective_prediction import (
    clustered_bootstrap_aurc,
    selective_prediction_summary,
    within_magnitude_stratum_rho,
)
from safetrans_confidence.features.normalize import (
    QNORM_SUFFIX,
    available_transferable_features,
    normalize_features_within_group,
)
from safetrans_confidence.scoring.error_ranker import _make_model
from safetrans_confidence.scoring.protocol_v0_2 import build_protocol_v0_2_scores

GROUP_COLS = ["dataset_name", "fold_id", "predictor_name"]


# --------------------------------------------------------------------------- #
# loading + magnitude attachment
# --------------------------------------------------------------------------- #
def _load_true_magnitude(records: pd.DataFrame, npz_path: Path) -> pd.DataFrame:
    arrays = np.load(npz_path)
    rows = []
    for _, row in records.iterrows():
        key = str(row["true_effect_key"])
        arr = np.asarray(arrays[key], dtype=float).ravel() if key in arrays.files else None
        rows.append(
            {
                "record_id": row["record_id"],
                "true_effect_l2_norm": float(np.linalg.norm(arr)) if arr is not None else np.nan,
                "effect_scale_rmse": (
                    float(np.linalg.norm(arr) / np.sqrt(max(arr.size, 1))) if arr is not None else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _assign_normalized_rmse(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["normalized_rmse"] = np.nan
    for _, idx_obj in out.groupby(["dataset_name", "predictor_name"], dropna=False).groups.items():
        idx = list(idx_obj)
        sub = out.loc[idx]
        ref = pd.to_numeric(
            sub[sub["split"].isin(["train", "val"])]["effect_scale_rmse"], errors="coerce"
        )
        positive = ref[(ref > 0) & np.isfinite(ref)]
        eps = float(positive.quantile(0.01)) if not positive.empty else 1e-8
        if not np.isfinite(eps) or eps <= 0:
            eps = 1e-8
        denom = pd.to_numeric(sub["effect_scale_rmse"], errors="coerce").clip(lower=eps)
        out.loc[idx, "normalized_rmse"] = pd.to_numeric(sub["true_error_rmse"], errors="coerce") / denom
    return out


def load_corrected_base(run_dirs: list[Path]) -> tuple[pd.DataFrame, list[dict]]:
    frames, status = [], []
    for run_dir in run_dirs:
        try:
            base = load_merged_records(
                run_dir,
                validate_contract=True,
                strict_contract=False,
                require_effect_arrays=True,
            )
            records = pd.read_csv(run_dir / "tables" / "PREDICTION_RECORDS.csv")
            _, true_npz = find_effect_array_files(run_dir)
            mags = _load_true_magnitude(records, true_npz)
            base = base.merge(mags, on="record_id", how="left")
            base["run_dir"] = str(run_dir)
            frames.append(base)
            status.append(
                {"run_dir": str(run_dir), "dataset": sorted(base["dataset_name"].unique().tolist()),
                 "n": int(len(base)), "status": "ok"}
            )
        except Exception as exc:  # pragma: no cover - operational safety
            status.append({"run_dir": str(run_dir), "status": "failed", "error": repr(exc)})
    if not frames:
        raise RuntimeError("no usable run dirs")
    out = pd.concat(frames, ignore_index=True)
    out = _assign_normalized_rmse(out)
    return out, status


# --------------------------------------------------------------------------- #
# targets + learned scores
# --------------------------------------------------------------------------- #
def _train_target_rank(train_df: pd.DataFrame) -> pd.Series:
    """Leakage-free training target: rank of true_error_rmse within (dataset,
    fold, predictor) computed using ONLY the train/val rows passed in.

    Test rows must never enter this function, so the rank scale a training row
    receives never depends on the held-out test distribution.
    """
    out = pd.Series(np.nan, index=train_df.index, dtype=float)
    for _, idx_obj in train_df.groupby(GROUP_COLS, dropna=False).groups.items():
        idx = list(idx_obj)
        vals = pd.to_numeric(train_df.loc[idx, "true_error_rmse"], errors="coerce")
        out.loc[idx] = vals.rank(method="average", pct=True)
    return out


def build_learned_scores(
    base: pd.DataFrame,
    norm_cols: list[str],
    mode: str,
    score_name: str,
    model_type: str = "histgbt",
) -> pd.DataFrame:
    """Learned error-rank scores. mode='perdataset' or 'lodo'.

    Target is the within-(dataset,fold,predictor) rank of true_error_rmse
    computed on the training rows ONLY, so the cross-dataset model learns to
    rank reliability rather than absolute, dataset-specific error scale, and no
    held-out test row contaminates the training target.
    """
    work = base.copy()
    rows: list[dict] = []
    datasets = sorted(work["dataset_name"].dropna().unique().tolist())

    def _fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray | None:
        tr = train_df.copy()
        tr["_err_rank"] = _train_target_rank(tr)
        tr = tr.dropna(subset=["_err_rank"])
        if len(tr) < 24:
            return None
        X_tr = tr[norm_cols].fillna(0.5).to_numpy()
        y_tr = tr["_err_rank"].to_numpy()
        if float(np.std(y_tr)) < 1e-9:
            return None
        model = _make_model(5201, model_type=model_type, n_train_rows=len(tr))
        model.fit(X_tr, y_tr)
        X_te = test_df[norm_cols].fillna(0.5).to_numpy()
        return model.predict(X_te)

    if mode == "perdataset":
        for ds in datasets:
            sub = work[work["dataset_name"] == ds]
            train = sub[sub["split"].isin(["train", "val"])]
            test = sub[sub["split"] == "test"]
            preds = _fit_predict(train, test)
            if preds is None:
                continue
            for (_, row), v in zip(test.iterrows(), preds):
                rows.append(_score_row(row, score_name, float(v)))
    elif mode == "lodo":
        for held in datasets:
            train = work[(work["dataset_name"] != held) & work["split"].isin(["train", "val"])]
            test = work[(work["dataset_name"] == held) & (work["split"] == "test")]
            preds = _fit_predict(train, test)
            if preds is None:
                continue
            for (_, row), v in zip(test.iterrows(), preds):
                r = _score_row(row, score_name, float(v))
                r["heldout_dataset"] = held
                rows.append(r)
    else:
        raise ValueError(mode)
    return pd.DataFrame(rows)


def _score_row(row: pd.Series, score_name: str, value: float, score_type: str = "risk") -> dict:
    return {
        "record_id": row["record_id"],
        "dataset_name": row["dataset_name"],
        "fold_id": int(row["fold_id"]),
        "split": row["split"],
        "context": row.get("context"),
        "perturbation": row.get("perturbation"),
        "predictor_name": row["predictor_name"],
        "task_key": row.get("task_key", row.get("task_id", row["record_id"])),
        "score_name": score_name,
        "score_type": score_type,
        "score_value": value,
        "true_error_rmse": float(row["true_error_rmse"]),
        "true_effect_l2_norm": float(row.get("true_effect_l2_norm", np.nan)),
    }


def build_baseline_scores(base: pd.DataFrame) -> pd.DataFrame:
    """Random / predicted-magnitude / oracle-magnitude risk scores."""
    rng = np.random.default_rng(5201)
    rows = []
    for _, row in base.iterrows():
        rows.append(_score_row(row, "random", float(rng.random())))
        pmag = pd.to_numeric(pd.Series([row.get("prediction_l2_norm")]), errors="coerce").iloc[0]
        rows.append(_score_row(row, "predicted_magnitude", float(pmag) if np.isfinite(pmag) else np.nan))
        omag = pd.to_numeric(pd.Series([row.get("true_effect_l2_norm")]), errors="coerce").iloc[0]
        rows.append(
            _score_row(row, "oracle_magnitude_diagnostic", float(omag) if np.isfinite(omag) else np.nan)
        )
    return pd.DataFrame(rows)


def attach_protocol_v0_2(base: pd.DataFrame) -> pd.DataFrame:
    scores, _ = build_protocol_v0_2_scores(base, include_evaluation_labels=True)
    scores = scores[scores["score_name"] == "protocol_v0_2_family_confidence"].copy()
    keys = base[["record_id", "task_key", "true_effect_l2_norm"]].drop_duplicates("record_id")
    scores = scores.merge(keys, on="record_id", how="left")
    out = []
    for _, r in scores.iterrows():
        out.append(
            {
                "record_id": r["record_id"],
                "dataset_name": r["dataset_name"],
                "fold_id": int(r["fold_id"]),
                "split": r["split"],
                "context": r.get("context"),
                "perturbation": r.get("perturbation"),
                "predictor_name": r["predictor_name"],
                "task_key": r.get("task_key", r["record_id"]),
                "score_name": "protocol_v0_2_family_confidence",
                "score_type": "confidence",
                "score_value": float(r["score_value"]) if pd.notna(r["score_value"]) else np.nan,
                "true_error_rmse": float(r["true_error_rmse"]),
                "true_effect_l2_norm": float(r.get("true_effect_l2_norm", np.nan)),
            }
        )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def _risk_axis(g: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(g["score_value"], errors="coerce")
    return s.where(g["score_type"].astype(str).eq("risk"), -s)


def _partial_rho(risk: pd.Series, err: pd.Series, control: pd.Series) -> float:
    def resid(v, c):
        frame = pd.DataFrame({"v": v, "c": c}).apply(pd.to_numeric, errors="coerce").dropna()
        if len(frame) < 3 or frame["c"].nunique() < 2:
            return pd.Series(np.nan, index=v.index)
        y = frame["v"].rank().to_numpy()
        z = frame["c"].rank().to_numpy()
        design = np.column_stack([np.ones(len(z)), z])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        out = pd.Series(np.nan, index=v.index)
        out.loc[frame.index] = y - design @ beta
        return out
    rr, re = resid(risk, control), resid(err, control)
    m = rr.notna() & re.notna()
    if int(m.sum()) < 3:
        return float("nan")
    return float(rr[m].corr(re[m], method="pearson"))


def evaluate_ladder(all_scores: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    test = all_scores[all_scores["split"] == "test"].dropna(
        subset=["score_value", "true_error_rmse"]
    ).copy()
    test["risk_axis"] = _risk_axis(test)
    rows = []
    for (ds, score), g in test.groupby(["dataset_name", "score_name"], dropna=False):
        g = g.dropna(subset=["risk_axis"])
        if len(g) < 5:
            continue
        err = g["true_error_rmse"]
        aligned = err.corr(g["risk_axis"], method="spearman")
        partial = _partial_rho(g["risk_axis"], err, g["true_effect_l2_norm"])
        sp = selective_prediction_summary(err.to_numpy(), g["risk_axis"].to_numpy())
        boot = clustered_bootstrap_aurc(
            g, "true_error_rmse", "risk_axis", cluster_col="task_key",
            n_bootstrap=n_bootstrap,
        )
        rows.append(
            {
                "dataset_name": ds,
                "score_name": score,
                "n": int(len(g)),
                "aligned_rho": float(aligned) if pd.notna(aligned) else np.nan,
                "partial_rho_control_magnitude": partial,
                "aurc": sp["aurc"],
                "oracle_aurc": sp["oracle_aurc"],
                "random_aurc": sp["random_aurc"],
                "excess_aurc": sp["excess_aurc"],
                "aurc_reduction_vs_random_pct": sp["aurc_reduction_vs_random_pct"],
                "avoidable_gap_captured_pct": sp["avoidable_gap_captured_pct"],
                "excess_aurc_ci_low": boot.get("excess_aurc_ci_low"),
                "excess_aurc_ci_high": boot.get("excess_aurc_ci_high"),
                "reduction_ci_low": boot.get("aurc_reduction_vs_random_pct_ci_low"),
                "reduction_ci_high": boot.get("aurc_reduction_vs_random_pct_ci_high"),
                "n_task_clusters": boot.get("n_clusters"),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_name", "score_name"])


SCORE_ORDER = [
    "random",
    "predicted_magnitude",
    "protocol_v0_2_family_confidence",
    "safeconf_perdataset_risk",
    "safeconf_lodo_risk",
    "safeconf_lodo_linear_risk",
    "oracle_magnitude_diagnostic",
]


def run(run_dirs: list[Path], out_dir: Path, n_bootstrap: int) -> dict:
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    base, load_status = load_corrected_base(run_dirs)
    base, norm_cols = normalize_features_within_group(
        base, available_transferable_features(base)
    )
    pd.DataFrame(load_status).to_csv(out_dir / "tables" / "RELIABILITY_INPUT_STATUS.csv", index=False)

    baseline = build_baseline_scores(base)
    protocol = attach_protocol_v0_2(base)
    perds = build_learned_scores(base, norm_cols, "perdataset", "safeconf_perdataset_risk", "histgbt")
    lodo = build_learned_scores(base, norm_cols, "lodo", "safeconf_lodo_risk", "histgbt")
    lodo_lin = build_learned_scores(base, norm_cols, "lodo", "safeconf_lodo_linear_risk", "elasticnet")

    all_scores = pd.concat(
        [baseline, protocol, perds, lodo, lodo_lin], ignore_index=True, sort=False
    )
    all_scores.to_csv(out_dir / "tables" / "RELIABILITY_ALL_SCORES.csv", index=False)

    ladder = evaluate_ladder(all_scores, n_bootstrap)
    ladder["score_order"] = ladder["score_name"].map(
        {n: i for i, n in enumerate(SCORE_ORDER)}
    ).fillna(99)
    ladder = ladder.sort_values(["dataset_name", "score_order"]).drop(columns=["score_order"])
    ladder.to_csv(out_dir / "tables" / "RELIABILITY_BASELINE_LADDER.csv", index=False)

    # within-magnitude-stratum for the main transferable score
    strat_rows = []
    test = all_scores[all_scores["split"] == "test"].copy()
    test["risk_axis"] = _risk_axis(test)
    for score in ["safeconf_lodo_risk", "predicted_magnitude", "protocol_v0_2_family_confidence"]:
        for ds, g in test[test["score_name"] == score].groupby("dataset_name"):
            s = within_magnitude_stratum_rho(
                g, "risk_axis", "true_error_rmse", "true_effect_l2_norm", n_bins=4
            )
            if not s.empty:
                s.insert(0, "score_name", score)
                s.insert(0, "dataset_name", ds)
                strat_rows.append(s)
    if strat_rows:
        pd.concat(strat_rows, ignore_index=True).to_csv(
            out_dir / "tables" / "RELIABILITY_WITHIN_MAGNITUDE_STRATUM.csv", index=False
        )

    status = {
        "out_dir": str(out_dir),
        "n_run_dirs": len(run_dirs),
        "n_records": int(len(base)),
        "n_datasets": int(base["dataset_name"].nunique()),
        "norm_feature_cols": norm_cols,
        "n_bootstrap": n_bootstrap,
        "status": "ok",
    }
    (out_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    p = argparse.ArgumentParser(description="SafeConf cross-dataset reliability model + ladder.")
    p.add_argument("--run-dir", type=Path, action="append", dest="run_dirs", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    print(json.dumps(run(args.run_dirs, args.out_dir, args.bootstrap), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
