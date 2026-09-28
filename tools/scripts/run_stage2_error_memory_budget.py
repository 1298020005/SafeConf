#!/usr/bin/env python3
"""Estimate how much same-upstream error feedback is useful for SafeConf.

This is a sample-efficiency simulation rather than an online-learning claim:
feedback labels are sampled from training genes only, while evaluation genes
remain completely held out.  The model is deliberately the selected light
Ridge(P+Q) learner so that this experiment measures feedback value, not a
change of model capacity.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
SRC = [
    STAGE / "txpert_risk_batch/FEATURE_TABLE.csv.gz",
    STAGE / "txpert_exphormer_risk_batch/FEATURE_TABLE.csv.gz",
]
OUT = STAGE / "error_memory_budget"
P = [
    "predicted_magnitude",
    "family_disagreement",
    "family_radius",
    "prediction_abs_mean",
    "prediction_signed_mean",
    "prediction_std",
    "prediction_abs_q95",
]
Q = [
    "n_source_cells",
    "n_source_contexts",
    "n_source_batches",
    "min_source_cells",
]
GROUP = P + Q
BUDGETS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
SEED = 20260928


def utility(score: np.ndarray, label: np.ndarray) -> float:
    k = max(1, int(math.ceil(0.2 * len(label))))
    chosen = np.argsort(-score)[:k]
    oracle = np.argsort(-label)[:k]
    denominator = float(label[oracle].mean() - label.mean())
    if denominator <= 1e-15:
        return float("nan")
    return float((label[chosen].mean() - label.mean()) / denominator)


def rho(score: np.ndarray, label: np.ndarray) -> float:
    if np.ptp(score) <= 1e-15 or np.ptp(label) <= 1e-15:
        return float("nan")
    return float(spearmanr(score, label).statistic)


def transform(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = train[GROUP].to_numpy(float)
    z = test[GROUP].to_numpy(float)
    miss_x = ~np.isfinite(x)
    miss_z = ~np.isfinite(z)
    med = np.array(
        [
            np.median(x[~miss_x[:, j], j]) if (~miss_x[:, j]).any() else 0.0
            for j in range(x.shape[1])
        ]
    )
    x = np.where(miss_x, med, x)
    z = np.where(miss_z, med, z)
    mean = x.mean(axis=0)
    scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    return (
        np.c_[(x - mean) / scale, miss_x].astype("float32"),
        np.c_[(z - mean) / scale, miss_z].astype("float32"),
    )


def stable_seed(target: str, upstream: str, fold: int) -> int:
    token = f"{target}:{upstream}:{fold}".encode()
    return SEED + int(hashlib.sha256(token).hexdigest()[:8], 16)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "RUN_STATUS.json").exists():
        raise FileExistsError("refusing overwrite of a registered batch")

    parts = []
    for path in SRC:
        frame = pd.read_csv(path)
        frame["upstream"] = (
            "TxPert_GAT" if "txpert_risk_batch/" in str(path) else "TxPert_Exphormer"
        )
        parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    if len(frame) != 3616:
        raise ValueError(f"expected 3616 combined records, got {len(frame)}")

    rows: list[dict[str, object]] = []
    for target, target_frame in frame.groupby("target", sort=True):
        target_frame = target_frame.sort_values(["gene", "upstream", "task_id"]).reset_index(drop=True)
        splitter = GroupKFold(5)
        for fold, (fit_idx, query_idx) in enumerate(
            splitter.split(target_frame, groups=target_frame["gene"])
        ):
            fit = target_frame.iloc[fit_idx].reset_index(drop=True)
            query = target_frame.iloc[query_idx].reset_index(drop=True)
            if set(fit.gene) & set(query.gene):
                raise AssertionError("gene leakage between feedback and evaluation fold")

            for upstream in ("TxPert_GAT", "TxPert_Exphormer"):
                train_upstream = fit[fit.upstream.eq(upstream)].copy()
                test_upstream = query[query.upstream.eq(upstream)].copy()
                available = np.asarray(sorted(train_upstream.gene.unique()))
                rng = np.random.default_rng(stable_seed(str(target), upstream, fold))
                rng.shuffle(available)

                for budget in BUDGETS:
                    if budget == 0.0:
                        risk = test_upstream.predicted_magnitude.to_numpy(float)
                        n_genes = 0
                    else:
                        n_genes = max(1, int(math.ceil(len(available) * budget)))
                        used = set(available[:n_genes])
                        feedback = train_upstream[train_upstream.gene.isin(used)].copy()
                        x_train, x_test = transform(feedback, test_upstream)
                        y = feedback.family_centroid_rmse.to_numpy(float)
                        center = float(y.mean())
                        spread = max(float(y.std()), 1e-8)
                        model = Ridge(alpha=10.0).fit(x_train, (y - center) / spread)
                        risk = np.maximum(0.0, model.predict(x_test) * spread + center)

                    labels = test_upstream.family_centroid_rmse.to_numpy(float)
                    rows.append(
                        {
                            "target": target,
                            "upstream": upstream,
                            "fold": fold,
                            "budget": budget,
                            "n_feedback_genes": n_genes,
                            "n_feedback_records": n_genes,
                            "utility20": utility(risk, labels),
                            "spearman": rho(risk, labels),
                        }
                    )

    result = pd.DataFrame(rows)
    summary = (
        result.groupby(["upstream", "budget"], as_index=False)
        .agg(
            n_folds=("fold", "size"),
            n_feedback_genes=("n_feedback_genes", "mean"),
            utility20=("utility20", "mean"),
            spearman=("spearman", "mean"),
        )
    )
    result.to_csv(OUT / "FOLD_RESULTS.csv", index=False)
    summary.to_csv(OUT / "SUMMARY.csv", index=False)
    status = {
        "status": "COMPLETE_STOPPED_AFTER_REGISTERED_BATCH",
        "n_records": len(frame),
        "budgets": list(BUDGETS),
        "grouped_by_gene": True,
        "feature_group": "P+Q",
        "same_upstream_feedback": True,
        "target_truth_used_only_as_label": True,
        "new_upstream_training_runs": 0,
        "result_sha256": hashlib.sha256((OUT / "SUMMARY.csv").read_bytes()).hexdigest(),
        "python": platform.python_version(),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
