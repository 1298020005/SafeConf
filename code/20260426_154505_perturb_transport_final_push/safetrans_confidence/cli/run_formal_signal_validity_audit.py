#!/usr/bin/env python3
"""Signal-validity audit for the formal seven-dataset SafeConf table.

This script intentionally reads ``FORMAL_SCORED_RECORDS.csv`` from the formal
main audit instead of the original per-dataset score files, because the formal
table re-scores records with the frozen protocol v0.2 family formula.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def _within_stratum(df: pd.DataFrame, stratum_col: str) -> dict[str, float | int]:
    rhos: list[float] = []
    weights: list[int] = []
    for _, group in df.groupby(stratum_col, dropna=False):
        if len(group) < 3:
            continue
        rho = _raw_spearman(group["risk_axis"], group["true_error_rmse"])
        if np.isfinite(rho):
            rhos.append(rho)
            weights.append(len(group))
    if not rhos:
        return {
            f"within_{stratum_col}_n_strata": 0,
            f"within_{stratum_col}_mean_rho": np.nan,
            f"within_{stratum_col}_weighted_rho": np.nan,
        }
    return {
        f"within_{stratum_col}_n_strata": int(len(rhos)),
        f"within_{stratum_col}_mean_rho": float(np.mean(rhos)),
        f"within_{stratum_col}_weighted_rho": float(np.average(rhos, weights=weights)),
    }


def _metric_row(group: pd.DataFrame) -> dict[str, float | int]:
    row = {
        "n": int(len(group)),
        "raw_spearman": _raw_spearman(group["risk_axis"], group["true_error_rmse"]),
        "normalized_rmse_spearman": _raw_spearman(group["risk_axis"], group["normalized_rmse"]),
        "magnitude_l2_baseline_rho": _raw_spearman(
            group["true_effect_l2_norm"], group["true_error_rmse"]
        ),
        "magnitude_abs_mean_baseline_rho": _raw_spearman(
            group["true_effect_abs_mean"], group["true_error_rmse"]
        ),
        "magnitude_l2_vs_normalized_rmse_rho": _raw_spearman(
            group["true_effect_l2_norm"], group["normalized_rmse"]
        ),
        "partial_rho_control_magnitude": _partial_spearman(
            group["risk_axis"], group["true_error_rmse"], group["true_effect_l2_norm"]
        ),
        "partial_rho_control_l2_normalized_rmse": _partial_spearman(
            group["risk_axis"], group["normalized_rmse"], group["true_effect_l2_norm"]
        ),
    }
    row.update(_within_stratum(group, "perturbation"))
    row.update(_within_stratum(group, "context"))
    if np.isfinite(row["raw_spearman"]) and np.isfinite(row["magnitude_l2_baseline_rho"]):
        row["rho_minus_magnitude_l2"] = float(row["raw_spearman"] - row["magnitude_l2_baseline_rho"])
    else:
        row["rho_minus_magnitude_l2"] = np.nan
    return row


def _summarize(test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset, family, score), group in test.groupby(
        ["dataset_name", "dataset_family", "score_name"], dropna=False
    ):
        rows.append(
            {
                "level": "dataset_score",
                "dataset_name": dataset,
                "dataset_family": family,
                "predictor_name": "ALL",
                "score_name": score,
                "score_type": str(group["score_type"].iloc[0]),
                **_metric_row(group),
            }
        )
    for (dataset, family, predictor, score), group in test.groupby(
        ["dataset_name", "dataset_family", "predictor_name", "score_name"], dropna=False
    ):
        rows.append(
            {
                "level": "dataset_predictor_score",
                "dataset_name": dataset,
                "dataset_family": family,
                "predictor_name": predictor,
                "score_name": score,
                "score_type": str(group["score_type"].iloc[0]),
                **_metric_row(group),
            }
        )
    return pd.DataFrame(rows).sort_values(["level", "dataset_name", "score_name", "predictor_name"])


def run_audit(scored_records: Path, out_dir: Path, main_score: str = MAIN_SCORE) -> dict:
    for name in ["tables", "reports"]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    scored = pd.read_csv(scored_records)
    required = {
        "dataset_name",
        "dataset_family",
        "predictor_name",
        "score_name",
        "score_type",
        "split",
        "risk_axis",
        "true_error_rmse",
        "normalized_rmse",
        "true_effect_l2_norm",
        "true_effect_abs_mean",
        "perturbation",
        "context",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"Missing required columns in {scored_records}: {missing}")

    test = scored[scored["split"].eq("test")].dropna(
        subset=["risk_axis", "true_error_rmse", "true_effect_l2_norm"]
    ).copy()
    summary = _summarize(test)
    summary.to_csv(out_dir / "tables" / "SIGNAL_VALIDITY_7MAIN_SUMMARY.csv", index=False)

    partial_cols = [
        "level",
        "dataset_name",
        "dataset_family",
        "predictor_name",
        "score_name",
        "n",
        "raw_spearman",
        "normalized_rmse_spearman",
        "magnitude_l2_baseline_rho",
        "magnitude_l2_vs_normalized_rmse_rho",
        "partial_rho_control_magnitude",
        "partial_rho_control_l2_normalized_rmse",
        "within_perturbation_weighted_rho",
        "within_context_weighted_rho",
        "rho_minus_magnitude_l2",
    ]
    summary[partial_cols].to_csv(out_dir / "tables" / "PARTIAL_AND_WITHIN_STRATUM_7MAIN.csv", index=False)

    mag_rows = []
    for dataset, group in test.drop_duplicates("record_id").groupby("dataset_name", dropna=False):
        mag_rows.append(
            {
                "dataset_name": dataset,
                "n_records": int(len(group)),
                "magnitude_l2_vs_rmse_rho": _raw_spearman(
                    group["true_effect_l2_norm"], group["true_error_rmse"]
                ),
                "magnitude_abs_mean_vs_rmse_rho": _raw_spearman(
                    group["true_effect_abs_mean"], group["true_error_rmse"]
                ),
                "magnitude_l2_vs_normalized_rmse_rho": _raw_spearman(
                    group["true_effect_l2_norm"], group["normalized_rmse"]
                ),
            }
        )
    magnitude = pd.DataFrame(mag_rows).sort_values("dataset_name")
    magnitude.to_csv(out_dir / "tables" / "MAGNITUDE_BASELINE_7MAIN.csv", index=False)

    main = summary[
        (summary["level"].eq("dataset_score")) & (summary["score_name"].eq(main_score))
    ].copy()
    main.to_csv(out_dir / "tables" / "SIGNAL_VALIDITY_7MAIN_MAIN_SCORE.csv", index=False)

    lines = [
        "# Signal validity audit for 7 formal main datasets",
        "",
        "本审计只使用 `Formal_main_20260604` 的正式打分结果，不引用旧 v6 signal validity 作为主表证据。",
        "",
        "目的：检查 SafeConf 的 confidence/risk 信号是否只是 effect magnitude（效应大小）伪相关。",
        "",
        "## Main score preview",
        "",
        "```",
        main[
            [
                "dataset_name",
                "dataset_family",
                "n",
                "raw_spearman",
                "magnitude_l2_baseline_rho",
                "partial_rho_control_magnitude",
                "within_perturbation_weighted_rho",
                "within_context_weighted_rho",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Interpretation notes",
        "",
        "- `raw_spearman`: aligned score vs raw RMSE.",
        "- `magnitude_l2_baseline_rho`: true effect magnitude vs raw RMSE; high values indicate possible magnitude confounding.",
        "- `partial_rho_control_magnitude`: score vs RMSE after controlling true effect magnitude ranks.",
        "- `within_perturbation_weighted_rho`: whether the score still ranks errors within the same perturbation.",
        "- `within_context_weighted_rho`: whether the score still ranks errors within the same context.",
        "",
        "## Scope guard",
        "",
        "- Tahoe remains sampled/smoke-tagged external validation, not part of this 7-main formal signal-validity table.",
        "- GEARS alignment is not forced into this table because the context space is mismatched.",
    ]
    (out_dir / "reports" / "SIGNAL_VALIDITY_7MAIN_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    status = {
        "scored_records": str(scored_records),
        "out_dir": str(out_dir),
        "n_datasets": int(test["dataset_name"].nunique()),
        "n_test_scores": int(len(test)),
        "main_score": main_score,
        "outputs": [
            "tables/SIGNAL_VALIDITY_7MAIN_SUMMARY.csv",
            "tables/PARTIAL_AND_WITHIN_STRATUM_7MAIN.csv",
            "tables/MAGNITUDE_BASELINE_7MAIN.csv",
            "tables/SIGNAL_VALIDITY_7MAIN_MAIN_SCORE.csv",
            "reports/SIGNAL_VALIDITY_7MAIN_REPORT.md",
        ],
    }
    (out_dir / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 7-main formal signal-validity audit.")
    parser.add_argument("--scored-records", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--main-score", default=MAIN_SCORE)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.scored_records, args.out_dir, args.main_score), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
