#!/usr/bin/env python3
"""Postprocess Tahoe D1 SafeConf scores with task-cluster bootstrap CIs.

This script intentionally does not read Tahoe pseudobulk shards and does not
build new prediction records. It only consumes the small score table produced by
``run_tahoe_pseudobulk_smoke.py --formal-eval`` and recomputes uncertainty by
resampling whole ``task_key`` clusters. That keeps the two predictor rows for the
same biological task together and avoids the row-level bootstrap used in the
early smoke audit.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.eval.selective_prediction import selective_prediction_summary

PRIMARY_METRICS = (
    "aligned_rho",
    "partial_rho_control_magnitude",
    "rc80_improvement_pct",
    "excess_aurc",
    "aurc_reduction_vs_random_pct",
)
REQUIRED_COLUMNS = {
    "task_key",
    "split",
    "predictor_name",
    "score_value",
    "score_type",
    "true_error_rmse",
    "true_effect_l2_norm",
}


def _spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 5 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].corr(frame["y"], method="spearman"))


def _rank_residual(values: pd.Series, control: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"v": values, "c": control}).apply(pd.to_numeric, errors="coerce").dropna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if len(frame) < 5 or frame["c"].nunique() < 2:
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
    if int(mask.sum()) < 5 or rx[mask].nunique() < 2 or ry[mask].nunique() < 2:
        return float("nan")
    return float(rx[mask].corr(ry[mask], method="pearson"))


def _risk_axis(scores: pd.DataFrame) -> pd.Series:
    if "risk_axis_value" in scores.columns:
        return pd.to_numeric(scores["risk_axis_value"], errors="coerce")
    values = pd.to_numeric(scores["score_value"], errors="coerce")
    score_type = scores["score_type"].astype(str)
    return values.where(score_type.eq("risk"), -values)


def _rc80_improvement(errors: pd.Series, risk_axis: pd.Series) -> float:
    frame = (
        pd.DataFrame({"error": errors, "risk": risk_axis})
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .sort_values("risk", ascending=True)
    )
    if len(frame) < 5:
        return float("nan")
    full_mean = float(frame["error"].mean())
    if not np.isfinite(full_mean) or full_mean <= 0:
        return float("nan")
    n_keep = max(1, int(math.ceil(0.8 * len(frame))))
    kept_mean = float(frame.head(n_keep)["error"].mean())
    return 100.0 * (full_mean - kept_mean) / full_mean


def _metric_row(group: pd.DataFrame, level: str, group_id: str) -> dict:
    work = group.dropna(subset=["risk_axis", "true_error_rmse"]).copy()
    errors = pd.to_numeric(work["true_error_rmse"], errors="coerce")
    risk = pd.to_numeric(work["risk_axis"], errors="coerce")
    magnitude = pd.to_numeric(work["true_effect_l2_norm"], errors="coerce")
    sp = selective_prediction_summary(errors.to_numpy(dtype=float), risk.to_numpy(dtype=float))
    return {
        "level": level,
        "group_id": group_id,
        "dataset_name": str(work["dataset_name"].iloc[0]) if "dataset_name" in work.columns and len(work) else "Tahoe100M",
        "score_name": str(work["score_name"].iloc[0]) if "score_name" in work.columns and len(work) else "",
        "n_records": int(len(work)),
        "n_task_clusters": int(work["task_key"].nunique()) if "task_key" in work.columns else int(len(work)),
        "n_predictors": int(work["predictor_name"].nunique()) if "predictor_name" in work.columns else 0,
        "aligned_rho": _spearman(risk, errors),
        "partial_rho_control_magnitude": _partial_spearman(risk, errors, magnitude),
        "rc80_improvement_pct": _rc80_improvement(errors, risk),
        "excess_aurc": sp["excess_aurc"],
        "aurc_reduction_vs_random_pct": sp["aurc_reduction_vs_random_pct"],
        "mean_rmse": float(errors.mean()) if len(errors) else np.nan,
        "median_rmse": float(errors.median()) if len(errors) else np.nan,
        "mean_effect_scale": float(magnitude.mean()) if len(magnitude.dropna()) else np.nan,
    }


def _group_specs(test_scores: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    specs: list[tuple[str, str, pd.DataFrame]] = [("overall", "overall", test_scores)]
    for predictor, group in test_scores.groupby("predictor_name", dropna=False):
        specs.append(("predictor", str(predictor), group))
    return specs


def compute_summary(scores: pd.DataFrame) -> pd.DataFrame:
    test_scores = _prepare_scores(scores)
    rows = [_metric_row(group, level, group_id) for level, group_id, group in _group_specs(test_scores)]
    return pd.DataFrame(rows)


def _bootstrap_draws_for_group(
    group: pd.DataFrame,
    level: str,
    group_id: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    work = group.dropna(subset=["risk_axis", "true_error_rmse", "task_key"]).copy()
    clusters = [g for _, g in work.groupby("task_key", dropna=False)]
    if len(clusters) < 5:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    n_clusters = len(clusters)
    for index in range(n_bootstrap):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        sample = pd.concat([clusters[int(i)] for i in pick], ignore_index=True)
        row = _metric_row(sample, level, group_id)
        row["bootstrap_index"] = int(index)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_task_clusters(
    scores: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 5201,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_scores = _prepare_scores(scores)
    all_draws: list[pd.DataFrame] = []
    for offset, (level, group_id, group) in enumerate(_group_specs(test_scores)):
        draws = _bootstrap_draws_for_group(
            group,
            level,
            group_id,
            n_bootstrap=n_bootstrap,
            seed=seed + 1009 * offset,
        )
        if not draws.empty:
            all_draws.append(draws)
    draw_df = pd.concat(all_draws, ignore_index=True) if all_draws else pd.DataFrame()
    ci_rows: list[dict] = []
    if draw_df.empty:
        return pd.DataFrame(), draw_df
    for (level, group_id), group in draw_df.groupby(["level", "group_id"], dropna=False):
        row = {
            "level": level,
            "group_id": group_id,
            "n_bootstrap_requested": int(n_bootstrap),
            "n_bootstrap_valid": int(len(group)),
        }
        for metric in PRIMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_bootstrap_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_ci_low"] = float(values.quantile(0.025)) if len(values) else np.nan
            row[f"{metric}_ci_high"] = float(values.quantile(0.975)) if len(values) else np.nan
        ci_rows.append(row)
    return pd.DataFrame(ci_rows), draw_df


def _prepare_scores(scores: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(scores.columns))
    if missing:
        raise ValueError(f"Tahoe score table missing required columns: {missing}")
    test_scores = scores[scores["split"].astype(str).eq("test")].copy()
    if test_scores.empty:
        raise ValueError("Tahoe score table has no test rows")
    test_scores["risk_axis"] = _risk_axis(test_scores)
    test_scores["task_key"] = test_scores["task_key"].astype(str)
    return test_scores


def _merge_summary_and_ci(summary: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame:
    if ci.empty:
        return summary
    return summary.merge(ci, on=["level", "group_id"], how="left")


def _primary_gate(summary_with_ci: pd.DataFrame) -> str:
    overall = summary_with_ci[
        summary_with_ci["level"].astype(str).eq("overall")
        & summary_with_ci["group_id"].astype(str).eq("overall")
    ]
    if overall.empty:
        return "FAIL"
    row = overall.iloc[0]
    partial = float(row.get("partial_rho_control_magnitude", np.nan))
    ci_low = float(row.get("partial_rho_control_magnitude_ci_low", np.nan))
    if np.isfinite(partial) and partial > 0.20 and np.isfinite(ci_low) and ci_low > 0:
        return "PASS"
    if np.isfinite(partial) and partial > 0:
        return "PARTIAL"
    return "FAIL"


def _resolve_scores_csv(run_dir: Path | None, scores_csv: Path | None) -> Path:
    if scores_csv is not None:
        return scores_csv
    if run_dir is None:
        raise ValueError("Provide either --run-dir or --scores-csv")
    candidates = [
        run_dir / "tables" / "TAHOE_PROTOCOL_SCORES_D1.csv",
        run_dir / "tables" / "TAHOE_PROTOCOL_SCORES_SMOKE.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find Tahoe score CSV under run dir; expected tables/TAHOE_PROTOCOL_SCORES_D1.csv "
        "or tables/TAHOE_PROTOCOL_SCORES_SMOKE.csv"
    )


def _resolve_records_csv(score_path: Path) -> Path | None:
    tables = score_path.parent
    candidates = [
        tables / "TAHOE_PREDICTION_RECORDS_D1.csv",
        tables / "TAHOE_PREDICTION_RECORDS_SMOKE.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _augment_scores_from_records(scores: pd.DataFrame, score_path: Path) -> pd.DataFrame:
    if "task_key" in scores.columns:
        return scores
    if "record_id" not in scores.columns:
        return scores
    records_path = _resolve_records_csv(score_path)
    if records_path is None:
        return scores
    records = pd.read_csv(records_path, usecols=["record_id", "task_key"])
    records = records.drop_duplicates("record_id")
    return scores.merge(records, on="record_id", how="left")


def _write_report(out_dir: Path, status: dict, summary: pd.DataFrame) -> None:
    overall = summary[
        summary["level"].astype(str).eq("overall")
        & summary["group_id"].astype(str).eq("overall")
    ]
    if overall.empty:
        headline = "- Overall row missing."
    else:
        row = overall.iloc[0]
        headline = "\n".join(
            [
                f"- Gate: `{status['gate']}`",
                f"- aligned rho: {row.get('aligned_rho', np.nan):.4f} "
                f"(CI {row.get('aligned_rho_ci_low', np.nan):.4f} to {row.get('aligned_rho_ci_high', np.nan):.4f})",
                f"- partial rho controlling effect magnitude: {row.get('partial_rho_control_magnitude', np.nan):.4f} "
                f"(CI {row.get('partial_rho_control_magnitude_ci_low', np.nan):.4f} to "
                f"{row.get('partial_rho_control_magnitude_ci_high', np.nan):.4f})",
                f"- RC@80 improvement: {row.get('rc80_improvement_pct', np.nan):.2f}% "
                f"(CI {row.get('rc80_improvement_pct_ci_low', np.nan):.2f} to "
                f"{row.get('rc80_improvement_pct_ci_high', np.nan):.2f})",
                f"- task clusters: {int(row.get('n_task_clusters', 0))}",
            ]
        )
    report = f"""# Tahoe D1 task-cluster bootstrap postprocess

This is a postprocess-only audit. It reads an existing Tahoe SafeConf score
table and does not touch raw Tahoe parquet shards.

## Headline

{headline}

## Files

- `tables/TAHOE_D1_FORMAL_SUMMARY.csv`: point estimates plus task-cluster CI.
- `tables/TAHOE_D1_TASK_CLUSTER_BOOTSTRAP_CI.csv`: bootstrap intervals.
- `tables/TAHOE_D1_TASK_CLUSTER_BOOTSTRAP_DRAWS.csv`: bootstrap draw-level metrics.
- `tables/TAHOE_D1_OVERLAP_AUDIT_MANIFEST.csv`: copied G8 overlap-audit provenance.
- `TAHOE_D1_POSTPROCESS_STATUS.json`: machine-readable status.

## Gate

PASS requires the overall partial rho controlling effect magnitude to be > 0.20
and its task-cluster 95% CI lower bound to be > 0. PARTIAL means the point
estimate remains positive but the PASS rule is not met. FAIL means the point
estimate is non-positive or unusable.
"""
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports" / "TAHOE_D1_POSTPROCESS_REPORT.md").write_text(report, encoding="utf-8")


def _copy_overlap_audits(overlap_audits: list[Path] | None, tables_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for source in overlap_audits or []:
        source = source.resolve()
        target = tables_dir / source.name
        row = {
            "source_path": str(source),
            "target_path": str(target),
            "status": "missing_source",
        }
        if source.exists():
            shutil.copy2(source, target)
            row["status"] = "copied"
        rows.append(row)
    manifest = pd.DataFrame(rows, columns=["source_path", "target_path", "status"])
    manifest.to_csv(tables_dir / "TAHOE_D1_OVERLAP_AUDIT_MANIFEST.csv", index=False)
    return rows


def run(
    run_dir: Path | None,
    scores_csv: Path | None,
    out_dir: Path | None,
    n_bootstrap: int = 1000,
    seed: int = 5201,
    overlap_audits: list[Path] | None = None,
) -> dict:
    score_path = _resolve_scores_csv(run_dir, scores_csv)
    target_dir = out_dir if out_dir is not None else (run_dir if run_dir is not None else score_path.parent.parent)
    tables = target_dir / "tables"
    reports = target_dir / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    scores = _augment_scores_from_records(pd.read_csv(score_path), score_path)
    summary = compute_summary(scores)
    ci, draws = bootstrap_task_clusters(scores, n_bootstrap=n_bootstrap, seed=seed)
    merged = _merge_summary_and_ci(summary, ci)
    gate = _primary_gate(merged)

    summary.to_csv(tables / "TAHOE_D1_FORMAL_POINT_SUMMARY.csv", index=False)
    ci.to_csv(tables / "TAHOE_D1_TASK_CLUSTER_BOOTSTRAP_CI.csv", index=False)
    draws.to_csv(tables / "TAHOE_D1_TASK_CLUSTER_BOOTSTRAP_DRAWS.csv", index=False)
    merged.to_csv(tables / "TAHOE_D1_FORMAL_SUMMARY.csv", index=False)
    overlap_manifest = _copy_overlap_audits(overlap_audits, tables)

    overall = merged[
        merged["level"].astype(str).eq("overall")
        & merged["group_id"].astype(str).eq("overall")
    ]
    status = {
        "status": "ok",
        "gate": gate,
        "scores_csv": str(score_path),
        "out_dir": str(target_dir),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "bootstrap_unit": "task_key",
        "n_rows": int(len(scores)),
        "n_test_rows": int(len(_prepare_scores(scores))),
        "n_task_clusters": int(overall.iloc[0]["n_task_clusters"]) if not overall.empty else 0,
        "n_overlap_audits_requested": int(len(overlap_audits or [])),
        "n_overlap_audits_copied": int(sum(row["status"] == "copied" for row in overlap_manifest)),
        "overall_partial_rho_control_magnitude": (
            float(overall.iloc[0]["partial_rho_control_magnitude"]) if not overall.empty else np.nan
        ),
        "overall_partial_rho_ci_low": (
            float(overall.iloc[0].get("partial_rho_control_magnitude_ci_low", np.nan))
            if not overall.empty
            else np.nan
        ),
    }
    (target_dir / "TAHOE_D1_POSTPROCESS_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    _write_report(target_dir, status, merged)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Postprocess Tahoe D1 SafeConf scores with task-cluster bootstrap CIs."
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--scores-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5201)
    parser.add_argument(
        "--overlap-audit",
        type=Path,
        action="append",
        default=[],
        help="Optional D0/D1 overlap-audit CSV to copy into the Tahoe D1 output tables directory.",
    )
    args = parser.parse_args()
    status = run(
        run_dir=args.run_dir,
        scores_csv=args.scores_csv,
        out_dir=args.out_dir,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        overlap_audits=args.overlap_audit,
    )
    print(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
