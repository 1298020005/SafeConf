#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.eval.metrics import evaluate_scores


def _find_record_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("PREDICTION_RECORDS.csv")})


def _load_pred_norms(record_dir: Path, records: pd.DataFrame) -> pd.Series:
    candidates = [
        record_dir / "gears_predicted_effects.npz",
        record_dir.parent / "arrays" / "gears_predicted_effects.npz",
        record_dir.parent.parent / "arrays" / "gears_predicted_effects.npz",
    ]
    npz_path = next((p for p in candidates if p.exists()), None)
    if npz_path is None:
        return pd.Series(np.nan, index=records.index)
    arrays = np.load(npz_path)
    vals = []
    for key in records["predicted_effect_key"]:
        if key in arrays:
            vals.append(float(np.linalg.norm(arrays[key])))
        else:
            vals.append(np.nan)
    return pd.Series(vals, index=records.index)


def build_gears_scores(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict] = []
    rec_frames: list[pd.DataFrame] = []
    for record_dir in _find_record_dirs(root):
        rec = pd.read_csv(record_dir / "PREDICTION_RECORDS.csv")
        rec["source_record_dir"] = str(record_dir)
        rec_frames.append(rec)
        pred_norm = _load_pred_norms(record_dir, rec)
        for idx, row in rec.iterrows():
            base = {
                "record_id": row["record_id"],
                "dataset_name": row["dataset_name"],
                "dataset_family": "gears_supplement",
                "fold_id": int(row["fold_id"]),
                "split": row["split"],
                "context": row["context"],
                "perturbation": row["perturbation"],
                "predictor_name": row["predictor_name"],
                "true_error_rmse": float(row["true_error_rmse"]),
            }
            if "gears_uncertainty_confidence" in rec.columns and pd.notna(row.get("gears_uncertainty_confidence")):
                score_rows.append(
                    {
                        **base,
                        "score_name": "gears_uncertainty_confidence",
                        "score_type": "confidence",
                        "score_value": float(row["gears_uncertainty_confidence"]),
                    }
                )
            if pd.notna(pred_norm.loc[idx]):
                score_rows.append(
                    {
                        **base,
                        "score_name": "gears_prediction_magnitude_risk",
                        "score_type": "risk",
                        "score_value": float(pred_norm.loc[idx]),
                    }
                )
    records = pd.concat(rec_frames, ignore_index=True) if rec_frames else pd.DataFrame()
    scores = pd.DataFrame(score_rows)
    return records, scores


def run(input_root: Path, out_dir: Path) -> dict:
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    records, scores = build_gears_scores(input_root)
    records.to_csv(out_dir / "tables" / "GEARS_PREDICTION_RECORDS_COMBINED.csv", index=False)
    scores.to_csv(out_dir / "tables" / "GEARS_CONFIDENCE_SCORES.csv", index=False)
    if not scores.empty:
        eval_df, risk_cov = evaluate_scores(scores)
    else:
        eval_df, risk_cov = pd.DataFrame(), pd.DataFrame()
    eval_df.to_csv(out_dir / "tables" / "GEARS_CONFIDENCE_EVAL_SUMMARY.csv", index=False)
    risk_cov.to_csv(out_dir / "tables" / "GEARS_RISK_COVERAGE.csv", index=False)
    report = [
        "# GEARS confidence evaluation",
        "",
        f"- input_root: `{input_root}`",
        f"- records: {len(records)}",
        f"- scores: {len(scores)}",
        "",
    ]
    if not eval_df.empty:
        report.extend(["## Evaluation", "", "```", eval_df.to_string(index=False), "```"])
    else:
        report.append("No usable GEARS confidence scores were available.")
    (out_dir / "reports" / "GEARS_CONFIDENCE_EVAL_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    status = {
        "input_root": str(input_root),
        "out_dir": str(out_dir),
        "n_records": int(len(records)),
        "n_scores": int(len(scores)),
        "has_uncertainty_score": bool((scores.get("score_name", pd.Series(dtype=str)) == "gears_uncertainty_confidence").any()),
    }
    (out_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GEARS per-prediction confidence records.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input_root, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

