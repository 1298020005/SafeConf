#!/usr/bin/env python3
"""Evaluate the selected SafeConf risk interface on existing GEARS held-out assets."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor


ROOT = Path(__file__).resolve().parents[2]
GEARS_ROOT = Path("/home/yyf/safeconf_runtime/outputs/gears_prediction_records_formal")
RECORDS_PATH = Path("/home/yyf/safeconf_runtime/outputs/gears_confidence_eval_formal/tables/GEARS_PREDICTION_RECORDS_COMBINED.csv")
STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
OUT = STAGE / "gears_risk_transfer"
SEED = 20260928


def utility(score: np.ndarray, label: np.ndarray) -> float:
    k = max(1, int(math.ceil(0.2 * len(label))))
    selected = np.argsort(-score)[:k]
    oracle = np.argsort(-label)[:k]
    denominator = label[oracle].mean() - label.mean()
    return float((label[selected].mean() - label.mean()) / denominator) if denominator > 1e-12 else np.nan


def load_vector(row: pd.Series, kind: str) -> np.ndarray:
    path = GEARS_ROOT / str(row.dataset_name) / f"seed_{int(row.fold_id)}" / "arrays" / f"gears_{kind}_effects.npz"
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive[str(row[f"{kind}_effect_key"])], dtype=np.float32)


def feature_vector(pred: np.ndarray, n_cells: float) -> dict[str, float]:
    abs_pred = np.abs(pred)
    return {
        "predicted_magnitude": float(np.sqrt(np.mean(pred * pred))),
        "prediction_abs_mean": float(abs_pred.mean()),
        "prediction_signed_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
        "prediction_abs_q95": float(np.quantile(abs_pred, 0.95)),
        "prediction_sparsity": float(np.mean(abs_pred <= 1e-8)),
        "n_cells": float(n_cells),
    }


def score_methods(frame: pd.DataFrame) -> pd.DataFrame:
    features = [
        "predicted_magnitude",
        "prediction_abs_mean",
        "prediction_signed_mean",
        "prediction_std",
        "prediction_abs_q95",
        "prediction_sparsity",
        "n_cells",
    ]
    prediction_features = features[:-1]
    methods = {"Magnitude_raw": [], "Ridge_P+Q": [], "HGB_P+Q": [], "MLP_P+Q": []}
    rows = []
    splitter = GroupKFold(5)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(frame, groups=frame.perturbation)):
        train = frame.iloc[train_idx].copy()
        test = frame.iloc[test_idx].copy()
        x_train = train[features].to_numpy(float)
        x_test = test[features].to_numpy(float)
        mean = x_train.mean(axis=0)
        scale = np.where(x_train.std(axis=0) > 1e-8, x_train.std(axis=0), 1.0)
        xs = (x_train - mean) / scale
        z = (x_test - mean) / scale
        y = train.true_error_rmse.to_numpy(float)
        center, spread = float(y.mean()), max(float(y.std()), 1e-8)
        target = (y - center) / spread
        models = {
            "Ridge_P+Q": Ridge(alpha=10.0),
            "HGB_P+Q": HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=7, learning_rate=0.05, l2_regularization=1.0, random_state=SEED + fold),
            "MLP_P+Q": MLPRegressor(hidden_layer_sizes=(16, 8), alpha=1e-2, max_iter=500, random_state=SEED + fold, early_stopping=False),
        }
        preds = {"Magnitude_raw": test.predicted_magnitude.to_numpy(float)}
        for name, model in models.items():
            model.fit(xs, target)
            preds[name] = np.maximum(0.0, model.predict(z) * spread + center)
        for name, values in preds.items():
            for i, (idx, value) in enumerate(zip(test.index, values)):
                rows.append({"fold": fold, "row_index": int(idx), "method": name, "predicted_risk": float(value), "true_error_rmse": float(test.iloc[i].true_error_rmse)})
    result = pd.DataFrame(rows)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "RUN_STATUS.json").exists():
        raise FileExistsError("refusing overwrite of registered batch")
    records = pd.read_csv(RECORDS_PATH)
    # The formal output has 54 genuine held-out prediction records.
    if len(records) != 54:
        raise ValueError(f"expected 54 GEARS records, got {len(records)}")
    rows = []
    for _, row in records.iterrows():
        pred = load_vector(row, "predicted")
        true = load_vector(row, "true")
        item = row.to_dict()
        item.update(feature_vector(pred, row.n_cells))
        item["vector_rmse_recomputed"] = float(np.sqrt(np.mean((pred - true) ** 2)))
        if abs(item["vector_rmse_recomputed"] - row.true_error_rmse) > 2e-5:
            raise ValueError(f"RMSE mismatch for {row.record_id}")
        rows.append(item)
    frame = pd.DataFrame(rows)
    result = score_methods(frame)
    summary_rows = []
    for method, group in result.groupby("method", sort=True):
        score = group.predicted_risk.to_numpy(float)
        label = group.true_error_rmse.to_numpy(float)
        k = max(1, int(math.ceil(0.2 * len(label))))
        chosen = np.argsort(-score)[:k]
        oracle = np.argsort(-label)[:k]
        summary_rows.append({
            "method": method,
            "n": len(group),
            "utility20": float((label[chosen].mean() - label.mean()) / (label[oracle].mean() - label.mean())),
            "spearman": float(spearmanr(score, label).statistic),
            "mean_top20_error": float(label[chosen].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    frame.to_csv(OUT / "FEATURE_TABLE.csv", index=False)
    result.to_csv(OUT / "OOF_PREDICTIONS.csv", index=False)
    summary.to_csv(OUT / "SUMMARY.csv", index=False)
    status = {
        "status": "COMPLETE_STOPPED_AFTER_REGISTERED_BATCH",
        "records": len(records),
        "datasets": sorted(records.dataset_name.unique().tolist()),
        "grouped_by_perturbation": True,
        "truth_used_only_as_label": True,
        "new_upstream_training_runs": 0,
        "summary_sha256": hashlib.sha256((OUT / "SUMMARY.csv").read_bytes()).hexdigest(),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
