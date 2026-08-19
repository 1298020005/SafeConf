from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SCORE_FILES = [
    "tables/CONFIDENCE_SCORES.csv",
    "tables/SAFECONF_SCORES.csv",
]

DEFAULT_RECORD_FILES = [
    "tables/PREDICTION_RECORDS.csv",
    "input/PREDICTION_RECORDS.csv",
]

DEFAULT_TRUE_EFFECT_FILES = [
    "input/true_effects.npz",
    "arrays/true_effects.npz",
]


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
    x = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    out.loc[frame.index] = y - x @ beta
    return out


def _partial_spearman(x: pd.Series, y: pd.Series, control: pd.Series) -> float:
    rx = _rank_residual(x, control)
    ry = _rank_residual(y, control)
    mask = rx.notna() & ry.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(rx[mask].corr(ry[mask], method="pearson"))


def _find_existing(base: Path, rels: list[str]) -> Path | None:
    for rel in rels:
        p = base / rel
        if p.exists():
            return p
    return None


def _load_true_effect_magnitudes(records: pd.DataFrame, true_effect_npz: Path) -> pd.DataFrame:
    arrays = np.load(true_effect_npz)
    rows: list[dict] = []
    for _, row in records.iterrows():
        key = str(row.get("true_effect_key", ""))
        if key not in arrays:
            rows.append(
                {
                    "record_id": row["record_id"],
                    "true_effect_l2_norm": np.nan,
                    "true_effect_abs_mean": np.nan,
                    "true_effect_std": np.nan,
                    "true_effect_n_genes": np.nan,
                    "effect_scale_rmse": np.nan,
                    "true_effect_key_found": False,
                }
            )
            continue
        arr = np.asarray(arrays[key], dtype=float).ravel()
        rows.append(
            {
                "record_id": row["record_id"],
                "true_effect_l2_norm": float(np.linalg.norm(arr)),
                "true_effect_abs_mean": float(np.mean(np.abs(arr))),
                "true_effect_std": float(np.std(arr)),
                "true_effect_n_genes": int(arr.size),
                "effect_scale_rmse": float(np.linalg.norm(arr) / np.sqrt(max(arr.size, 1))),
                "true_effect_key_found": True,
            }
        )
    return pd.DataFrame(rows)


def _assign_normalized_rmse(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["effect_scale_rmse"] = pd.to_numeric(out["effect_scale_rmse"], errors="coerce")
    out["true_error_rmse"] = pd.to_numeric(out["true_error_rmse"], errors="coerce")
    out["normalization_eps"] = np.nan
    out["normalized_rmse"] = np.nan
    group_cols = ["run_dir", "dataset_name", "predictor_name"]
    for _, idx_obj in out.groupby(group_cols, dropna=False).groups.items():
        idx = list(idx_obj)
        sub = out.loc[idx]
        ref = sub[sub["split"].isin(["train", "val"])]["effect_scale_rmse"]
        ref = pd.to_numeric(ref, errors="coerce")
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


def _risk_axis(scores: pd.DataFrame) -> pd.Series:
    stype = scores["score_type"].astype(str)
    raw = pd.to_numeric(scores["score_value"], errors="coerce")
    return raw.where(stype.eq("risk"), -raw)


def _summarize_within_stratum(df: pd.DataFrame, stratum_col: str) -> dict:
    rhos = []
    weights = []
    usable = 0
    for _, group in df.groupby(stratum_col, dropna=False):
        if len(group) < 3:
            continue
        rho = _raw_spearman(group["risk_axis"], group["true_error_rmse"])
        if np.isfinite(rho):
            rhos.append(rho)
            weights.append(len(group))
            usable += 1
    if not rhos:
        return {
            f"within_{stratum_col}_n_strata": 0,
            f"within_{stratum_col}_mean_rho": np.nan,
            f"within_{stratum_col}_weighted_rho": np.nan,
        }
    return {
        f"within_{stratum_col}_n_strata": int(usable),
        f"within_{stratum_col}_mean_rho": float(np.mean(rhos)),
        f"within_{stratum_col}_weighted_rho": float(np.average(rhos, weights=weights)),
    }


def _eval_group(group: pd.DataFrame) -> dict:
    row = {
        "n": int(len(group)),
        "raw_spearman": _raw_spearman(group["risk_axis"], group["true_error_rmse"]),
        "pooled_risk_spearman": _raw_spearman(group["risk_axis"], group["true_error_rmse"]),
        "normalized_rmse_spearman": _raw_spearman(group["risk_axis"], group["normalized_rmse"]),
        "magnitude_l2_baseline_rho": _raw_spearman(
            group["true_effect_l2_norm"], group["true_error_rmse"]
        ),
        "magnitude_l2_vs_normalized_rmse_rho": _raw_spearman(
            group["true_effect_l2_norm"], group["normalized_rmse"]
        ),
        "magnitude_abs_mean_baseline_rho": _raw_spearman(
            group["true_effect_abs_mean"], group["true_error_rmse"]
        ),
        "partial_rho_control_l2": _partial_spearman(
            group["risk_axis"], group["true_error_rmse"], group["true_effect_l2_norm"]
        ),
        "partial_rho_control_magnitude": _partial_spearman(
            group["risk_axis"], group["true_error_rmse"], group["true_effect_l2_norm"]
        ),
        "partial_rho_control_l2_normalized_rmse": _partial_spearman(
            group["risk_axis"], group["normalized_rmse"], group["true_effect_l2_norm"]
        ),
        "partial_rho_control_abs_mean": _partial_spearman(
            group["risk_axis"], group["true_error_rmse"], group["true_effect_abs_mean"]
        ),
    }
    row.update(_summarize_within_stratum(group, "perturbation"))
    row.update(_summarize_within_stratum(group, "context"))
    if np.isfinite(row["pooled_risk_spearman"]) and np.isfinite(row["magnitude_l2_baseline_rho"]):
        row["rho_minus_magnitude_l2"] = float(
            row["pooled_risk_spearman"] - row["magnitude_l2_baseline_rho"]
        )
    else:
        row["rho_minus_magnitude_l2"] = np.nan
    return row


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    record_path = _find_existing(run_dir, DEFAULT_RECORD_FILES)
    score_path = _find_existing(run_dir, DEFAULT_SCORE_FILES)
    true_npz_path = _find_existing(run_dir, DEFAULT_TRUE_EFFECT_FILES)
    status = {
        "run_dir": str(run_dir),
        "record_path": str(record_path) if record_path else "",
        "score_path": str(score_path) if score_path else "",
        "true_effect_npz": str(true_npz_path) if true_npz_path else "",
        "status": "started",
    }
    if record_path is None or score_path is None or true_npz_path is None:
        status["status"] = "missing_required_files"
        return pd.DataFrame(), status
    records = pd.read_csv(record_path)
    scores = pd.read_csv(score_path)
    magnitudes = _load_true_effect_magnitudes(records, true_npz_path)
    merged = scores.merge(
        records[
            [
                "record_id",
                "task_id",
                "task_key",
                "dataset_name",
                "context",
                "perturbation",
                "predictor_name",
                "true_effect_key",
            ]
        ].drop_duplicates("record_id"),
        on=["record_id", "dataset_name", "context", "perturbation", "predictor_name"],
        how="left",
        suffixes=("", "_record"),
    )
    merged = merged.merge(magnitudes, on="record_id", how="left")
    merged["run_dir"] = str(run_dir)
    merged["risk_axis"] = _risk_axis(merged)
    merged = _assign_normalized_rmse(merged)
    status.update(
        {
            "status": "ok",
            "n_records": int(len(records)),
            "n_scores": int(len(scores)),
            "n_scored_with_effect_magnitude": int(merged["true_effect_l2_norm"].notna().sum()),
        }
    )
    return merged, status


def run_audit(run_dirs: list[Path], out_dir: Path) -> dict:
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    frames = []
    statuses = []
    for run_dir in run_dirs:
        df, status = load_run(run_dir)
        statuses.append(status)
        if not df.empty:
            frames.append(df)
    status_df = pd.DataFrame(statuses)
    status_df.to_csv(out_dir / "tables" / "INPUT_RUN_STATUS.csv", index=False)
    if not frames:
        raise RuntimeError("No usable runs for signal validity audit.")
    all_scores = pd.concat(frames, ignore_index=True)
    all_scores.to_csv(out_dir / "tables" / "SCORES_WITH_EFFECT_MAGNITUDE.csv", index=False)

    test = all_scores[all_scores["split"].eq("test")].dropna(
        subset=["score_value", "true_error_rmse", "true_effect_l2_norm"]
    ).copy()
    group_cols = ["dataset_name", "predictor_name", "score_name"]
    summary_rows = []
    for key, group in test.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))
        row = {**meta, **_eval_group(group)}
        row["score_type"] = str(group["score_type"].iloc[0])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(group_cols)
    summary.to_csv(out_dir / "tables" / "SIGNAL_VALIDITY_SUMMARY.csv", index=False)

    mag_rows = []
    for dataset, group in test.groupby("dataset_name", dropna=False):
        mag_rows.append(
            {
                "dataset_name": dataset,
                "n": int(len(group.drop_duplicates("record_id"))),
                "magnitude_l2_vs_rmse_rho": _raw_spearman(
                    group.drop_duplicates("record_id")["true_effect_l2_norm"],
                    group.drop_duplicates("record_id")["true_error_rmse"],
                ),
                "magnitude_l2_vs_normalized_rmse_rho": _raw_spearman(
                    group.drop_duplicates("record_id")["true_effect_l2_norm"],
                    group.drop_duplicates("record_id")["normalized_rmse"],
                ),
                "magnitude_abs_mean_vs_rmse_rho": _raw_spearman(
                    group.drop_duplicates("record_id")["true_effect_abs_mean"],
                    group.drop_duplicates("record_id")["true_error_rmse"],
                ),
            }
        )
    mag = pd.DataFrame(mag_rows)
    mag.to_csv(out_dir / "tables" / "MAGNITUDE_BASELINE.csv", index=False)

    partial = summary[
        [
            "dataset_name",
            "predictor_name",
            "score_name",
            "n",
            "pooled_risk_spearman",
            "raw_spearman",
            "normalized_rmse_spearman",
            "magnitude_l2_baseline_rho",
            "magnitude_l2_vs_normalized_rmse_rho",
            "partial_rho_control_l2",
            "partial_rho_control_magnitude",
            "partial_rho_control_l2_normalized_rmse",
            "within_perturbation_weighted_rho",
            "within_context_weighted_rho",
            "rho_minus_magnitude_l2",
        ]
    ].copy()
    partial.to_csv(out_dir / "tables" / "PARTIAL_AND_WITHIN_STRATUM.csv", index=False)

    decision_lines = [
        "# Signal validity audit",
        "",
        "This audit checks whether confidence/risk scores are doing more than detecting effect magnitude.",
        "",
        "Key columns:",
        "",
        "- `pooled_risk_spearman`: score vs raw RMSE; higher is better after aligning confidence to risk.",
        "- `normalized_rmse_spearman`: score vs effect-scale-normalized RMSE.",
        "- `magnitude_l2_baseline_rho`: true effect magnitude vs raw RMSE.",
        "- `magnitude_l2_vs_normalized_rmse_rho`: true effect magnitude vs normalized RMSE.",
        "- `partial_rho_control_l2`: score vs RMSE after controlling effect magnitude ranks.",
        "- `within_perturbation_weighted_rho`: score vs RMSE within the same perturbation.",
        "- `rho_minus_magnitude_l2`: pooled rho minus magnitude-only baseline.",
        "",
        "## Highest-risk interpretation rule",
        "",
        "If `partial_rho_control_l2` and within-stratum rho collapse toward zero while magnitude baseline is high, the apparent confidence signal is likely magnitude-confounded.",
        "",
        "## Input runs",
        "",
        "```",
        status_df.to_string(index=False),
        "```",
        "",
        "## Summary preview",
        "",
        "```",
        summary.head(40).to_string(index=False),
        "```",
    ]
    (out_dir / "reports" / "SIGNAL_VALIDITY_DECISION.md").write_text(
        "\n".join(decision_lines) + "\n", encoding="utf-8"
    )
    status = {
        "out_dir": str(out_dir),
        "n_input_runs": int(len(run_dirs)),
        "n_usable_runs": int(sum(s["status"] == "ok" for s in statuses)),
        "n_test_scores": int(len(test)),
    }
    (out_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether confidence signals remain after controlling effect magnitude."
    )
    parser.add_argument("--run-dir", type=Path, action="append", dest="run_dirs", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.run_dirs, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
