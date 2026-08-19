#!/usr/bin/env python3
"""Formal SafeConf main-table audit across completed single-dataset runs.

This script does not train perturbation predictors.  It re-scores existing
PredictionRecord/feature tables with the frozen protocol v0.2 formula, merges
selected ablation scores from the original run, and reports whether confidence
signals survive effect-magnitude controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.data.records import find_effect_array_files, load_merged_records
from safetrans_confidence.scoring.protocol_v0_2 import build_protocol_v0_2_scores


KEEP_ORIGINAL_SCORES = {
    "simple_combined_confidence",
    "learned_risk_score",
    "model_disagreement_risk",
    "historical_residual_risk",
    "random_score",
}
MAIN_SCORE = "protocol_v0_2_family_confidence"


def _raw_spearman(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(x[mask].corr(y[mask], method="spearman"))


def _rank_residual(values: pd.Series, control: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"v": values, "c": control}).apply(pd.to_numeric, errors="coerce").dropna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if len(frame) < 3 or frame["c"].nunique() < 2:
        return out
    y = frame["v"].rank(method="average").to_numpy(dtype=float)
    z = frame["c"].rank(method="average").to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    out.loc[frame.index] = y - design @ beta
    return out


def _partial_spearman(x: pd.Series, y: pd.Series, control: pd.Series) -> float:
    rx = _rank_residual(x, control)
    ry = _rank_residual(y, control)
    mask = rx.notna() & ry.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(rx[mask].corr(ry[mask], method="pearson"))


def _risk_axis(frame: pd.DataFrame) -> pd.Series:
    score = pd.to_numeric(frame["score_value"], errors="coerce")
    score_type = frame["score_type"].astype(str)
    return score.where(score_type.eq("risk"), -score)


def _load_effect_magnitudes(records: pd.DataFrame, npz_path: Path) -> pd.DataFrame:
    arrays = np.load(npz_path)
    rows: list[dict] = []
    for _, row in records.iterrows():
        key = str(row["true_effect_key"])
        arr = np.asarray(arrays[key], dtype=float).ravel() if key in arrays else None
        rows.append(
            {
                "record_id": row["record_id"],
                "true_effect_l2_norm": float(np.linalg.norm(arr)) if arr is not None else np.nan,
                "true_effect_abs_mean": float(np.mean(np.abs(arr))) if arr is not None else np.nan,
                "effect_scale_rmse": (
                    float(np.linalg.norm(arr) / np.sqrt(max(arr.size, 1))) if arr is not None else np.nan
                ),
                "true_effect_key_found": bool(arr is not None),
            }
        )
    return pd.DataFrame(rows)


def _assign_normalized_rmse(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["normalized_rmse"] = np.nan
    out["normalization_eps"] = np.nan
    for _, idx_obj in out.groupby(["dataset_name", "predictor_name"], dropna=False).groups.items():
        idx = list(idx_obj)
        sub = out.loc[idx]
        ref = pd.to_numeric(sub[sub["split"].isin(["train", "val"])]["effect_scale_rmse"], errors="coerce")
        positive = ref[(ref > 0) & np.isfinite(ref)]
        if positive.empty:
            positive = pd.to_numeric(sub["effect_scale_rmse"], errors="coerce")
            positive = positive[(positive > 0) & np.isfinite(positive)]
        eps = float(positive.quantile(0.01)) if not positive.empty else 1e-8
        if not np.isfinite(eps) or eps <= 0:
            eps = 1e-8
        denom = pd.to_numeric(sub["effect_scale_rmse"], errors="coerce").clip(lower=eps)
        out.loc[idx, "normalization_eps"] = eps
        out.loc[idx, "normalized_rmse"] = pd.to_numeric(sub["true_error_rmse"], errors="coerce") / denom
    return out


def _risk_coverage80(group: pd.DataFrame) -> float:
    valid = group.dropna(subset=["risk_axis", "true_error_rmse"])
    if len(valid) < 3:
        return float("nan")
    full = float(valid["true_error_rmse"].mean())
    keep = max(1, int(np.ceil(0.8 * len(valid))))
    kept = valid.sort_values("risk_axis", ascending=True).head(keep)
    kept_mean = float(kept["true_error_rmse"].mean())
    return 100.0 * (full - kept_mean) / full if full else float("nan")


def _aurc_from_group(group: pd.DataFrame) -> tuple[float, float, float, float]:
    """Compute AURC, oracle AURC, random AURC, and excess AURC for a group."""
    from safetrans_confidence.eval.metrics import (
        compute_aurc,
        compute_excess_aurc,
        compute_oracle_aurc,
        compute_random_aurc,
    )

    valid = group.dropna(subset=["risk_axis", "true_error_rmse"])
    if len(valid) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    err = valid["true_error_rmse"].to_numpy()
    scr = valid["risk_axis"].to_numpy()
    aurc = compute_aurc(err, scr, "risk")
    oracle = compute_oracle_aurc(err)
    random_a = compute_random_aurc(err)
    excess = compute_excess_aurc(err, scr, "risk")
    return aurc, oracle, random_a, excess


def _metric_row(group: pd.DataFrame) -> dict:
    aurc, oracle_aurc, random_aurc, excess_aurc = _aurc_from_group(group)
    return {
        "n": int(len(group)),
        "aligned_rho": _raw_spearman(group["risk_axis"], group["true_error_rmse"]),
        "normalized_rmse_rho": _raw_spearman(group["risk_axis"], group["normalized_rmse"]),
        "magnitude_only_rho": _raw_spearman(group["true_effect_l2_norm"], group["true_error_rmse"]),
        "partial_rho_control_magnitude": _partial_spearman(
            group["risk_axis"], group["true_error_rmse"], group["true_effect_l2_norm"]
        ),
        "risk_coverage80_improve_pct": _risk_coverage80(group),
        "mean_rmse": float(pd.to_numeric(group["true_error_rmse"], errors="coerce").mean()),
        "aurc": aurc,
        "oracle_aurc": oracle_aurc,
        "random_aurc": random_aurc,
        "excess_aurc": excess_aurc,
    }


def _bootstrap_ci(
    group: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    if len(group) < 8 or n_bootstrap <= 0:
        return float("nan"), float("nan")
    values = []
    fold_groups = [g for _, g in group.groupby("fold_id", dropna=False)]
    for _ in range(n_bootstrap):
        pieces = []
        for fold in fold_groups:
            sample_idx = rng.choice(fold.index.to_numpy(), size=len(fold), replace=True)
            pieces.append(group.loc[sample_idx])
        boot = pd.concat(pieces, ignore_index=True)
        if metric == "aligned_rho":
            value = _raw_spearman(boot["risk_axis"], boot["true_error_rmse"])
        elif metric == "partial_rho_control_magnitude":
            value = _partial_spearman(
                boot["risk_axis"], boot["true_error_rmse"], boot["true_effect_l2_norm"]
            )
        else:
            raise ValueError(metric)
        if np.isfinite(value):
            values.append(value)
    if len(values) < 20:
        return float("nan"), float("nan")
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def _load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    records = pd.read_csv(run_dir / "tables" / "PREDICTION_RECORDS.csv")
    base = load_merged_records(
        run_dir,
        validate_contract=True,
        strict_contract=False,
        require_effect_arrays=True,
    )
    protocol_scores, formulas = build_protocol_v0_2_scores(base)
    old_scores_path = run_dir / "tables" / "CONFIDENCE_SCORES.csv"
    old_scores = pd.read_csv(old_scores_path)
    old_scores = old_scores[old_scores["score_name"].isin(KEEP_ORIGINAL_SCORES)].copy()
    if "dataset_family" not in old_scores.columns:
        fam = base[["record_id", "dataset_family"]].drop_duplicates("record_id")
        old_scores = old_scores.merge(fam, on="record_id", how="left")
    scores = pd.concat([protocol_scores, old_scores], ignore_index=True)
    _, true_effect_path = find_effect_array_files(run_dir)
    if true_effect_path is None:
        raise FileNotFoundError(f"true effect arrays not found for {run_dir}")
    magnitudes = _load_effect_magnitudes(records, true_effect_path)
    record_cols = [
        "record_id",
        "task_id",
        "task_key",
        "dataset_name",
        "fold_id",
        "split",
        "context",
        "perturbation",
        "predictor_name",
        "true_effect_key",
    ]
    merged = scores.merge(
        records[record_cols].drop_duplicates("record_id"),
        on=["record_id", "dataset_name", "fold_id", "split", "context", "perturbation", "predictor_name"],
        how="left",
        suffixes=("", "_record"),
    )
    merged = merged.merge(magnitudes, on="record_id", how="left")
    merged["run_dir"] = str(run_dir)
    merged["risk_axis"] = _risk_axis(merged)
    merged = _assign_normalized_rmse(merged)
    status = {
        "run_dir": str(run_dir),
        "dataset_names": sorted(merged["dataset_name"].dropna().astype(str).unique().tolist()),
        "n_records": int(len(records)),
        "n_scores": int(len(scores)),
        "status": "ok",
    }
    return merged, formulas, status


def run_formal_audit(run_dirs: list[Path], out_dir: Path, n_bootstrap: int, seed: int) -> dict:
    for name in ["tables", "reports", "logs"]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    frames = []
    formula_frames = []
    statuses = []
    for run_dir in run_dirs:
        try:
            frame, formulas, status = _load_run(run_dir)
            frames.append(frame)
            formula_frames.append(formulas)
            statuses.append(status)
        except Exception as exc:  # pragma: no cover - operational audit safety.
            statuses.append({"run_dir": str(run_dir), "status": "failed", "error": repr(exc)})
    status_df = pd.DataFrame(statuses)
    status_df.to_csv(out_dir / "tables" / "FORMAL_INPUT_STATUS.csv", index=False)
    if not frames:
        raise RuntimeError("No usable run directories.")
    scored = pd.concat(frames, ignore_index=True)
    scored.to_csv(out_dir / "tables" / "FORMAL_SCORED_RECORDS.csv", index=False)
    formulas = pd.concat(formula_frames, ignore_index=True) if formula_frames else pd.DataFrame()
    formulas.to_csv(out_dir / "tables" / "PROTOCOL_V0_2_FORMULAS.csv", index=False)

    test = scored[scored["split"].eq("test")].dropna(
        subset=["score_value", "true_error_rmse", "true_effect_l2_norm"]
    ).copy()
    rng = np.random.default_rng(seed)
    rows = []
    for (dataset, family, score), group in test.groupby(
        ["dataset_name", "dataset_family", "score_name"], dropna=False
    ):
        row = {
            "dataset_name": dataset,
            "dataset_family": family,
            "score_name": score,
            "score_type": str(group["score_type"].iloc[0]),
            **_metric_row(group),
        }
        a_lo, a_hi = _bootstrap_ci(group, "aligned_rho", rng, n_bootstrap)
        p_lo, p_hi = _bootstrap_ci(group, "partial_rho_control_magnitude", rng, n_bootstrap)
        row.update(
            {
                "aligned_rho_ci_low": a_lo,
                "aligned_rho_ci_high": a_hi,
                "partial_rho_ci_low": p_lo,
                "partial_rho_ci_high": p_hi,
                "n_bootstrap": int(n_bootstrap),
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["dataset_name", "score_name"])
    summary.to_csv(out_dir / "tables" / "FORMAL_SCORE_SUMMARY.csv", index=False)

    fold_rows = []
    for (dataset, score, fold), group in test.groupby(["dataset_name", "score_name", "fold_id"], dropna=False):
        fold_rows.append({"dataset_name": dataset, "score_name": score, "fold_id": int(fold), **_metric_row(group)})
    pd.DataFrame(fold_rows).sort_values(["dataset_name", "score_name", "fold_id"]).to_csv(
        out_dir / "tables" / "FORMAL_PER_FOLD_RHO.csv", index=False
    )

    predictor_rows = []
    for (dataset, score, predictor), group in test.groupby(
        ["dataset_name", "score_name", "predictor_name"], dropna=False
    ):
        predictor_rows.append(
            {"dataset_name": dataset, "score_name": score, "predictor_name": predictor, **_metric_row(group)}
        )
    pd.DataFrame(predictor_rows).sort_values(["dataset_name", "score_name", "predictor_name"]).to_csv(
        out_dir / "tables" / "FORMAL_PER_PREDICTOR_RHO.csv", index=False
    )

    main = summary[summary["score_name"].eq(MAIN_SCORE)].copy()
    main.to_csv(out_dir / "tables" / "FORMAL_MAIN_TABLE.csv", index=False)

    lines = [
        "# SafeConf formal main audit",
        "",
        "本报告把已有 PredictionRecord 重新用冻结 `protocol_v0_2_family_confidence` 打分，",
        "并同时报告 effect magnitude（效应大小）控制后的 partial rho 与 bootstrap CI。",
        "",
        "## Main table preview",
        "",
        "```",
        main[
            [
                "dataset_name",
                "dataset_family",
                "n",
                "aligned_rho",
                "aligned_rho_ci_low",
                "aligned_rho_ci_high",
                "partial_rho_control_magnitude",
                "partial_rho_ci_low",
                "partial_rho_ci_high",
                "magnitude_only_rho",
                "risk_coverage80_improve_pct",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Input status",
        "",
        "```",
        status_df.to_string(index=False),
        "```",
    ]
    (out_dir / "reports" / "FORMAL_MAIN_AUDIT_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    status = {
        "out_dir": str(out_dir),
        "n_input_runs": int(len(run_dirs)),
        "n_usable_runs": int(sum(s.get("status") == "ok" for s in statuses)),
        "n_test_scores": int(len(test)),
        "main_score": MAIN_SCORE,
        "n_bootstrap": int(n_bootstrap),
    }
    (out_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SafeConf formal main-table audit.")
    parser.add_argument("--run-dir", type=Path, action="append", dest="run_dirs", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5201)
    args = parser.parse_args()
    status = run_formal_audit(args.run_dirs, args.out_dir, args.bootstrap, args.seed)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
