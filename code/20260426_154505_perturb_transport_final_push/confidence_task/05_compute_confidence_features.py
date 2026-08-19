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


FEATURE_COLUMNS = [
    "context_similarity_max",
    "context_similarity_mean",
    "perturbation_support_count",
    "perturbation_effect_stability",
    "perturbation_effect_variance",
    "prediction_l2_norm",
    "prediction_abs_mean",
    "model_disagreement_rmse",
    "model_disagreement_cosine",
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def finite_summary(series: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(series, errors="coerce")
    return {
        "missing_n": int(x.isna().sum()),
        "missing_rate": float(x.isna().mean()),
        "mean": float(x.mean()) if x.notna().any() else float("nan"),
        "median": float(x.median()) if x.notna().any() else float("nan"),
        "min": float(x.min()) if x.notna().any() else float("nan"),
        "max": float(x.max()) if x.notna().any() else float("nan"),
    }


def load_tasks(summary_path: Path) -> tuple[list[dict], dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks, _, meta = build_effect_tasks(
        Path(summary["dataset_path"]),
        summary["dataset_name"],
        n_genes=int(summary["n_genes"]),
        min_cells=int(summary["min_cells"]),
        max_cells_per_group=int(summary["max_cells_per_group"]),
        seed=int(summary["task_seed"]),
    )
    if len(tasks) != int(summary["n_tasks"]):
        raise RuntimeError(f"Task count mismatch: summary={summary['n_tasks']} rebuilt={len(tasks)}")
    return tasks, {**summary, **meta}


def compute_features(
    records: pd.DataFrame,
    split_df: pd.DataFrame,
    tasks: list[dict],
    predicted_npz: Path,
) -> tuple[pd.DataFrame, dict]:
    pred_arrays = np.load(predicted_npz)
    pred_lookup: dict[tuple[int, str, int], np.ndarray] = {}
    for _, rec in records.iterrows():
        key = (int(rec["fold_id"]), str(rec["predictor_name"]), int(rec["task_id"]))
        pred_lookup[key] = pred_arrays[str(rec["predicted_effect_key"])]

    rows = []
    n_low_support = 0
    n_missing_disagreement = 0
    for _, rec in records.iterrows():
        fold_id = int(rec["fold_id"])
        task_id = int(rec["task_id"])
        predictor_name = str(rec["predictor_name"])
        task = tasks[task_id]
        fold = split_df[split_df["fold_id"].astype(int) == fold_id]
        train_ids = fold.loc[fold["split"] == "train", "task_id"].astype(int).tolist()
        if not train_ids:
            raise RuntimeError(f"Fold {fold_id} has no train tasks in split CSV.")
        train_tasks = [tasks[i] for i in train_ids]

        target_control = np.asarray(task["control_mean"], dtype=np.float64)
        train_controls = np.stack([np.asarray(t["control_mean"], dtype=np.float64) for t in train_tasks])
        ctx_sims = np.asarray([cosine(target_control, c) for c in train_controls], dtype=np.float64)

        source_effects = [
            np.asarray(t["effect"], dtype=np.float64)
            for t in train_tasks
            if str(t["perturbation"]) == str(task["perturbation"]) and str(t["context"]) != str(task["context"])
        ]
        support_count = len(source_effects)
        if support_count >= 2:
            pairwise = [cosine(a, b) for a, b in combinations(source_effects, 2)]
            perturbation_effect_stability = float(np.nanmean(pairwise))
            effect_stack = np.stack(source_effects, axis=0)
            perturbation_effect_variance = float(np.nanmean(np.var(effect_stack, axis=0)))
        else:
            n_low_support += 1
            perturbation_effect_stability = float("nan")
            perturbation_effect_variance = float("nan")

        pred = pred_arrays[str(rec["predicted_effect_key"])]
        other_predictor = "ContextSimBaseline" if predictor_name == "V0StrongBaseline" else "V0StrongBaseline"
        other = pred_lookup.get((fold_id, other_predictor, task_id))
        if other is None:
            n_missing_disagreement += 1
            disagreement_rmse = float("nan")
            disagreement_cosine = float("nan")
        else:
            disagreement_rmse = rmse(pred, other)
            disagreement_cosine = 1.0 - cosine(pred, other)

        rows.append(
            {
                "record_id": str(rec["record_id"]),
                "task_id": task_id,
                "dataset_name": str(rec["dataset_name"]),
                "fold_id": fold_id,
                "split": str(rec["split"]),
                "context": str(rec["context"]),
                "perturbation": str(rec["perturbation"]),
                "predictor_name": predictor_name,
                "context_similarity_max": float(np.nanmax(ctx_sims)),
                "context_similarity_mean": float(np.nanmean(ctx_sims)),
                "perturbation_support_count": int(support_count),
                "perturbation_effect_stability": perturbation_effect_stability,
                "perturbation_effect_variance": perturbation_effect_variance,
                "prediction_l2_norm": float(np.linalg.norm(pred)),
                "prediction_abs_mean": float(np.mean(np.abs(pred))),
                "model_disagreement_rmse": disagreement_rmse,
                "model_disagreement_cosine": disagreement_cosine,
            }
        )

    features = pd.DataFrame(rows)
    diagnostics = {
        "n_records": int(len(records)),
        "n_feature_rows": int(len(features)),
        "n_low_support_rows": int(n_low_support),
        "n_missing_disagreement_rows": int(n_missing_disagreement),
        "feature_summary": {col: finite_summary(features[col]) for col in FEATURE_COLUMNS},
    }
    return features, diagnostics


def write_report(path: Path, diagnostics: dict) -> None:
    lines = [
        "# Confidence Features Report",
        "",
        f"- PredictionRecord rows: {diagnostics['n_records']}",
        f"- Feature rows: {diagnostics['n_feature_rows']}",
        f"- Rows with <2 same-perturbation source effects: {diagnostics['n_low_support_rows']}",
        f"- Rows missing V0/ContextSim disagreement: {diagnostics['n_missing_disagreement_rows']}",
        "",
        "## Feature Missingness And Summary",
        "",
        "| feature | missing_rate | mean | median | min | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for col, s in diagnostics["feature_summary"].items():
        lines.append(
            f"| `{col}` | {s['missing_rate']:.3f} | {s['mean']:.6g} | {s['median']:.6g} | {s['min']:.6g} | {s['max']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Reliability Notes",
            "",
            "- `context_similarity_max/mean`, `support_count`, and prediction magnitude are mechanically reliable because they are computed for every row from the current fold train set and prediction arrays.",
            "- `perturbation_effect_stability` and `perturbation_effect_variance` are exploratory when the same perturbation has fewer than 2 source contexts; these rows are left as NaN rather than invented.",
            "- `model_disagreement_*` is reliable here because both V0StrongBaseline and ContextSimBaseline are present for the same task/fold/split.",
            "- No feature uses test-set aggregate statistics; all train-derived features are computed inside the current fold.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fold-safe confidence features for the confidence scoring MVP.")
    parser.add_argument("--records-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final" / "input" / "PREDICTION_RECORDS.csv"))
    parser.add_argument("--split-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final" / "input" / "kagglecrosscell_heldout_pair_split.csv"))
    parser.add_argument("--split-summary-json", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final" / "input" / "kagglecrosscell_heldout_pair_split_summary.json"))
    parser.add_argument("--predicted-effects-npz", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final" / "input" / "kagglecrosscell_predicted_effects.npz"))
    parser.add_argument("--out-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final" / "tables" / "CONFIDENCE_FEATURES.csv"))
    parser.add_argument("--report-md", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final" / "reports" / "confidence_features_report.md"))
    args = parser.parse_args()

    records_path = Path(args.records_csv)
    split_path = Path(args.split_csv)
    summary_path = Path(args.split_summary_json)
    predicted_path = Path(args.predicted_effects_npz)
    for path in [records_path, split_path, summary_path, predicted_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    records = pd.read_csv(records_path)
    split_df = pd.read_csv(split_path)
    tasks, _ = load_tasks(summary_path)
    features, diagnostics = compute_features(records, split_df, tasks, predicted_path)

    out_csv = Path(args.out_csv)
    report_md = Path(args.report_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_csv, index=False)
    write_report(report_md, diagnostics)
    print(json.dumps({"out_csv": str(out_csv), "report_md": str(report_md), **diagnostics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
