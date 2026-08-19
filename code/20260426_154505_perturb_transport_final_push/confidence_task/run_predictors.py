#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "03_code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_context_splits import build_effect_tasks
from transport_models import ContextSimilarityBaseline, V0StrongBaseline


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cosine_error(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    sim = float(np.dot(a, b) / denom)
    return float(1.0 - sim)


def fit_predict_model(model, tasks: list[dict], train_ids: list[int], eval_ids: list[int]) -> tuple[np.ndarray, pd.DataFrame | None]:
    train_mask = np.zeros(len(tasks), dtype=bool)
    train_mask[np.asarray(train_ids, dtype=int)] = True
    fitted = model.fit(tasks, train_mask)
    indices = np.asarray(eval_ids, dtype=int)
    details = None
    if hasattr(fitted, "predict_details"):
        details_dict = fitted.predict_details(tasks, indices)
        pred = details_dict["prediction"]
        details = details_dict.get("transportability")
    else:
        pred = fitted.predict(tasks, indices)
    return np.asarray(pred, dtype=np.float32), details


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V0StrongBaseline and ContextSimBaseline predictions for held-out pair folds.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--split-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp" / "splits" / "kagglecrosscell_heldout_pair_split.csv"))
    parser.add_argument("--split-summary-json", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp" / "splits" / "kagglecrosscell_heldout_pair_split_summary.json"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp"))
    parser.add_argument("--predict-splits", default="val,test", help="Comma-separated split names to predict. Default: val,test")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    array_dir = out_dir / "arrays"
    report_dir = out_dir / "reports"
    pred_dir.mkdir(parents=True, exist_ok=True)
    array_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(args.split_csv)
    summary = json.loads(Path(args.split_summary_json).read_text(encoding="utf-8"))
    tasks, genes, meta = build_effect_tasks(
        Path(summary["dataset_path"]),
        summary["dataset_name"],
        n_genes=int(summary["n_genes"]),
        min_cells=int(summary["min_cells"]),
        max_cells_per_group=int(summary["max_cells_per_group"]),
        seed=int(summary["task_seed"]),
    )
    if len(tasks) != int(summary["n_tasks"]):
        raise RuntimeError(f"Task count mismatch: split summary has {summary['n_tasks']}, rebuilt {len(tasks)}")

    predict_splits = {x.strip() for x in args.predict_splits.split(",") if x.strip()}
    records = []
    pred_arrays: dict[str, np.ndarray] = {}
    true_arrays: dict[str, np.ndarray] = {}
    status_rows = []
    record_idx = 0

    for fold_id in sorted(split_df["fold_id"].unique()):
        fold = split_df[split_df["fold_id"] == fold_id].copy()
        train_ids = fold.loc[fold["split"] == "train", "task_id"].astype(int).tolist()
        eval_rows = fold[fold["split"].isin(predict_splits)].sort_values(["split", "task_id"])
        eval_ids = eval_rows["task_id"].astype(int).tolist()
        if not eval_ids:
            continue

        models = [
            ("V0StrongBaseline", V0StrongBaseline()),
            ("ContextSimBaseline", ContextSimilarityBaseline()),
        ]
        for predictor_name, model in models:
            try:
                pred, details = fit_predict_model(model, tasks, train_ids, eval_ids)
                status_rows.append({"fold_id": int(fold_id), "predictor_name": predictor_name, "status": "ok", "n_predictions": int(len(eval_ids))})
            except Exception as exc:
                status_rows.append({"fold_id": int(fold_id), "predictor_name": predictor_name, "status": "failed", "error": repr(exc)})
                continue
            for local_pos, (_, row) in enumerate(eval_rows.iterrows()):
                task_id = int(row["task_id"])
                task = tasks[task_id]
                record_id = f"rec_{record_idx:06d}"
                pred_key = f"{record_id}_pred"
                true_key = f"{record_id}_true"
                y_pred = pred[local_pos].astype(np.float32)
                y_true = np.asarray(task["effect"], dtype=np.float32)
                pred_arrays[pred_key] = y_pred
                true_arrays[true_key] = y_true
                rec = {
                    "record_id": record_id,
                    "task_id": task_id,
                    "dataset_name": summary["dataset_name"],
                    "fold_id": int(fold_id),
                    "split": str(row["split"]),
                    "context": str(task["context"]),
                    "perturbation": str(task["perturbation"]),
                    "predictor_name": predictor_name,
                    "predicted_effect_key": pred_key,
                    "true_effect_key": true_key,
                    "true_error_rmse": rmse(y_true, y_pred),
                    "true_error_cosine": cosine_error(y_true, y_pred),
                }
                if details is not None and local_pos < len(details):
                    for col in details.columns:
                        rec[f"predictor_detail_{col}"] = details.iloc[local_pos][col]
                records.append(rec)
                record_idx += 1

    pred_df = pd.DataFrame(records)
    pred_csv = pred_dir / "kagglecrosscell_v0_contextsim_predictions.csv"
    pred_npz = array_dir / "kagglecrosscell_predicted_effects.npz"
    true_npz = array_dir / "kagglecrosscell_true_effects.npz"
    status_csv = report_dir / "predictor_run_status.csv"
    report_md = report_dir / "predictor_run_report.md"
    pred_df.to_csv(pred_csv, index=False)
    np.savez_compressed(pred_npz, **pred_arrays)
    np.savez_compressed(true_npz, **true_arrays)
    pd.DataFrame(status_rows).to_csv(status_csv, index=False)

    lines = [
        "# Predictor run report",
        "",
        f"- records: {len(pred_df)}",
        f"- folds: {sorted(split_df['fold_id'].unique().tolist())}",
        f"- predict_splits: {sorted(predict_splits)}",
        f"- predictions_csv: `{pred_csv}`",
        f"- predicted_effects_npz: `{pred_npz}`",
        f"- true_effects_npz: `{true_npz}`",
        "",
        "## Status",
        "",
        pd.DataFrame(status_rows).to_string(index=False) if status_rows else "No status rows.",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"predictions_csv": str(pred_csv), "predicted_effects_npz": str(pred_npz), "true_effects_npz": str(true_npz), "n_records": int(len(pred_df))}, indent=2))


if __name__ == "__main__":
    main()
