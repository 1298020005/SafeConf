#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "03_code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_context_splits import build_effect_tasks


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fold-safe confidence features for PredictionRecord rows.")
    parser.add_argument("--records-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp" / "PREDICTION_RECORDS.csv"))
    parser.add_argument("--split-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp" / "splits" / "kagglecrosscell_heldout_pair_split.csv"))
    parser.add_argument("--split-summary-json", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp" / "splits" / "kagglecrosscell_heldout_pair_split_summary.json"))
    parser.add_argument("--predicted-effects-npz", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp" / "arrays" / "kagglecrosscell_predicted_effects.npz"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(args.records_csv)
    split_df = pd.read_csv(args.split_csv)
    summary = json.loads(Path(args.split_summary_json).read_text(encoding="utf-8"))
    tasks, _, _ = build_effect_tasks(
        Path(summary["dataset_path"]),
        summary["dataset_name"],
        n_genes=int(summary["n_genes"]),
        min_cells=int(summary["min_cells"]),
        max_cells_per_group=int(summary["max_cells_per_group"]),
        seed=int(summary["task_seed"]),
    )
    pred_npz = np.load(args.predicted_effects_npz)
    by_record = records.set_index("record_id")
    pair_pred = {}
    for (_, fold_id, split, task_id), sub in records.groupby(["predictor_name", "fold_id", "split", "task_id"]):
        pair_pred[(str(sub.iloc[0]["predictor_name"]), int(fold_id), int(task_id))] = pred_npz[str(sub.iloc[0]["predicted_effect_key"])]

    rows = []
    missing_disagreement = 0
    for _, rec in records.iterrows():
        fold_id = int(rec["fold_id"])
        task_id = int(rec["task_id"])
        task = tasks[task_id]
        fold = split_df[split_df["fold_id"] == fold_id]
        train_ids = fold.loc[fold["split"] == "train", "task_id"].astype(int).tolist()
        train_tasks = [tasks[i] for i in train_ids]
        train_controls = np.stack([t["control_mean"] for t in train_tasks], axis=0)
        query = np.asarray(task["control_mean"], dtype=np.float64)
        context_sims = np.asarray([cosine(query, c) for c in train_controls], dtype=np.float64)
        same_pert_effects = [
            np.asarray(t["effect"], dtype=np.float64)
            for t in train_tasks
            if str(t["perturbation"]) == str(task["perturbation"]) and str(t["context"]) != str(task["context"])
        ]
        support_count = len(same_pert_effects)
        if support_count >= 2:
            pairwise = [cosine(a, b) for a, b in combinations(same_pert_effects, 2)]
            stability = float(np.nanmean(pairwise))
            variance = float(np.nanmean(np.var(np.stack(same_pert_effects, axis=0), axis=0)))
        elif support_count == 1:
            stability = 0.5
            variance = 0.0
        else:
            stability = float("nan")
            variance = float("nan")
        pred = pred_npz[str(rec["predicted_effect_key"])]
        other_name = "ContextSimBaseline" if rec["predictor_name"] == "V0StrongBaseline" else "V0StrongBaseline"
        other = pair_pred.get((other_name, fold_id, task_id))
        if other is None:
            disagreement_rmse = float("nan")
            disagreement_cos = float("nan")
            missing_disagreement += 1
        else:
            disagreement_rmse = rmse(pred, other)
            disagreement_cos = 1.0 - cosine(pred, other)
        rows.append(
            {
                "record_id": rec["record_id"],
                "context_similarity_max": float(np.nanmax(context_sims)),
                "context_similarity_mean": float(np.nanmean(context_sims)),
                "perturbation_effect_stability": stability,
                "perturbation_effect_variance": variance,
                "perturbation_support_count": int(support_count),
                "prediction_l2_norm": float(np.linalg.norm(pred)),
                "prediction_abs_mean": float(np.mean(np.abs(pred))),
                "model_disagreement_rmse": disagreement_rmse,
                "model_disagreement_cosine": disagreement_cos,
            }
        )
    out = pd.DataFrame(rows)
    out_path = out_dir / "CONFIDENCE_FEATURES.csv"
    out.to_csv(out_path, index=False)
    report = report_dir / "confidence_features_report.md"
    report.write_text(
        "\n".join(
            [
                "# Confidence features report",
                "",
                f"- records: {len(records)}",
                f"- feature_rows: {len(out)}",
                f"- missing_model_disagreement_rows: {missing_disagreement}",
                "- All train-derived features use only the current fold train rows.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"confidence_features_csv": str(out_path), "n_rows": int(len(out))}, indent=2))


if __name__ == "__main__":
    main()
