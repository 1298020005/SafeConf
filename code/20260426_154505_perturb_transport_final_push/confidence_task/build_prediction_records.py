#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_COLUMNS = [
    "record_id",
    "task_id",
    "dataset_name",
    "fold_id",
    "split",
    "context",
    "perturbation",
    "predictor_name",
    "predicted_effect_key",
    "true_effect_key",
    "true_error_rmse",
    "true_error_cosine",
]


def validate_arrays(records: pd.DataFrame, predicted_npz: Path, true_npz: Path) -> None:
    pred = np.load(predicted_npz)
    true = np.load(true_npz)
    missing_pred = sorted(set(records["predicted_effect_key"].astype(str)) - set(pred.files))
    missing_true = sorted(set(records["true_effect_key"].astype(str)) - set(true.files))
    if missing_pred or missing_true:
        raise RuntimeError(f"Missing arrays: predicted={missing_pred[:5]}, true={missing_true[:5]}")


def write_schema(path: Path) -> None:
    rows = [
        ("record_id", "Unique prediction row id."),
        ("task_id", "Task index from the rebuilt KaggleCrossCell context-perturbation task list."),
        ("dataset_name", "Dataset name, currently KaggleCrossCell."),
        ("fold_id", "Held-out pair fold id."),
        ("split", "Prediction split. MVP step 4 emits val/test predictions."),
        ("context", "Cellular context label from build_effect_tasks()."),
        ("perturbation", "Perturbation label from build_effect_tasks()."),
        ("predictor_name", "V0StrongBaseline or ContextSimBaseline."),
        ("predicted_effect_key", "Key into arrays/kagglecrosscell_predicted_effects.npz."),
        ("true_effect_key", "Key into arrays/kagglecrosscell_true_effects.npz."),
        ("true_error_rmse", "RMSE between predicted_effect and true_effect."),
        ("true_error_cosine", "Cosine distance: 1 - cosine(predicted_effect, true_effect)."),
        ("confidence_score", "Reserved for later confidence scores; empty in Step 4."),
        ("risk_score", "Reserved for later risk scores; empty in Step 4."),
        ("unsafe_flag", "Reserved for later unsafe labels; empty in Step 4."),
    ]
    lines = [
        "# PREDICTION_RECORDS schema",
        "",
        "| column | meaning |",
        "| --- | --- |",
    ]
    lines.extend(f"| `{name}` | {meaning} |" for name, meaning in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unified PredictionRecord CSV from predictor outputs and NPZ arrays.")
    parser.add_argument("--predictions-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp" / "predictions" / "kagglecrosscell_v0_contextsim_predictions.csv"))
    parser.add_argument("--split-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp" / "splits" / "kagglecrosscell_heldout_pair_split.csv"))
    parser.add_argument("--predicted-effects-npz", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp" / "arrays" / "kagglecrosscell_predicted_effects.npz"))
    parser.add_argument("--true-effects-npz", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp" / "arrays" / "kagglecrosscell_true_effects.npz"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(args.predictions_csv)
    split_df = pd.read_csv(args.split_csv)
    missing = [col for col in REQUIRED_COLUMNS if col not in records.columns]
    if missing:
        raise RuntimeError(f"Predictions CSV missing required columns: {missing}")
    validate_arrays(records, Path(args.predicted_effects_npz), Path(args.true_effects_npz))

    legal = set(split_df[["fold_id", "task_id", "split"]].astype(str).agg("::".join, axis=1))
    record_keys = records[["fold_id", "task_id", "split"]].astype(str).agg("::".join, axis=1)
    bad = records.loc[~record_keys.isin(legal), ["record_id", "fold_id", "task_id", "split"]]
    if not bad.empty:
        raise RuntimeError(f"Prediction records not present in split CSV: {bad.head().to_dict('records')}")

    records = records.copy()
    for col in ["confidence_score", "risk_score", "unsafe_flag"]:
        if col not in records.columns:
            records[col] = np.nan
    ordered = REQUIRED_COLUMNS + ["confidence_score", "risk_score", "unsafe_flag"]
    extra = [c for c in records.columns if c not in ordered]
    records = records[ordered + extra]

    out_csv = out_dir / "PREDICTION_RECORDS.csv"
    schema = out_dir / "PREDICTION_RECORDS_SCHEMA.md"
    status = out_dir / "PREDICTION_RECORDS_STATUS.json"
    records.to_csv(out_csv, index=False)
    write_schema(schema)
    status.write_text(
        json.dumps(
            {
                "prediction_records_csv": str(out_csv),
                "schema_md": str(schema),
                "n_records": int(len(records)),
                "predictors": sorted(records["predictor_name"].astype(str).unique().tolist()),
                "splits": sorted(records["split"].astype(str).unique().tolist()),
                "folds": sorted(records["fold_id"].astype(int).unique().tolist()),
                "array_validation": "ok",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"prediction_records_csv": str(out_csv), "schema_md": str(schema), "n_records": int(len(records))}, indent=2))


if __name__ == "__main__":
    main()
