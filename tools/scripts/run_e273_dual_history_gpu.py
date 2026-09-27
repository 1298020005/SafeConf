#!/usr/bin/env python3
"""E273 GPU tabular risk learner.

This runs the same perturbation-cold, train/val-only contract as the CPU audit
but uses a small MLP as a feasibility check for the proposed adaptive risk
learner.  It is intentionally not a new perturbation predictor: the target is
held-out prediction error and the inputs are deployment-time evidence only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn

from run_e273_dual_history_cpu import (
    HISTORY,
    PREDICTION,
    PREDICTORS,
    QUALITY,
    SOURCE_DEFAULT,
    bucket,
    error_memory,
    percentile,
    utility,
)


GROUPS = {
    "P": PREDICTION,
    "P_plus_Q": PREDICTION + QUALITY,
    "P_plus_Q_plus_H_plus_E": PREDICTION
    + QUALITY
    + HISTORY
    + (
        "error_hist_pert_mean",
        "error_hist_pert_std",
        "error_hist_pert_n",
        "error_hist_context_mean",
        "error_hist_context_std",
        "error_hist_context_n",
    ),
}


class RiskMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def matrix(fit: pd.DataFrame, test: pd.DataFrame, features: tuple[str, ...]):
    tr, te = [], []
    for col in features:
        ref = percentile(fit[col], fit[col])
        val = percentile(fit[col], test[col])
        tr.append(ref)
        te.append(val)
    xtr = np.asarray(tr, dtype=np.float32).T
    xte = np.asarray(te, dtype=np.float32).T
    # Include explicit missingness flags.  The feature matrix contains real
    # missing history fields in some studies; median imputation alone would
    # erase the distinction between "absent" and "measured at the median".
    raw_tr = fit[list(features)].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    raw_te = test[list(features)].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    miss_tr = ~np.isfinite(raw_tr)
    miss_te = ~np.isfinite(raw_te)
    xtr = np.nan_to_num(xtr, nan=0.5)
    xte = np.nan_to_num(xte, nan=0.5)
    xtr = np.concatenate([xtr, miss_tr.astype(np.float32)], axis=1)
    xte = np.concatenate([xte, miss_te.astype(np.float32)], axis=1)
    return xtr, xte


def train_score(xtr, ytr, xte, device: torch.device, seed: int, epochs: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RiskMLP(xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    loss_fn = nn.SmoothL1Loss()
    tx = torch.tensor(xtr, dtype=torch.float32, device=device)
    ty = torch.tensor(ytr, dtype=torch.float32, device=device)
    model.train()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred = model(tx)
        loss = loss_fn(pred, ty)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(xte, dtype=torch.float32, device=device)).detach().cpu().numpy()


def run(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use the cu124 research environment")
    device = torch.device(args.device)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use = [
        "dataset_name",
        "fold_id",
        "split",
        "task_key",
        "context",
        "perturbation",
        "predictor_name",
        "true_error_rmse",
        *PREDICTION,
        *QUALITY,
        *HISTORY,
    ]
    raw = pd.read_csv(args.source, usecols=list(dict.fromkeys(use)))
    raw = raw.loc[
        raw.fold_id.eq(0)
        & raw.split.isin(("train", "val"))
        & raw.predictor_name.isin(PREDICTORS)
    ].copy()
    raw["bucket"] = raw.perturbation.map(lambda x: bucket(x, args.n_buckets))
    rows = []
    for (dataset, predictor), task in raw.groupby(["dataset_name", "predictor_name"], sort=True):
        for held in range(args.n_buckets):
            fit = task.loc[task.bucket.ne(held)].copy()
            test = task.loc[task.bucket.eq(held)].copy()
            if len(test) < 20 or len(fit) < 50:
                continue
            if set(fit.perturbation) & set(test.perturbation):
                raise AssertionError(f"perturbation overlap: {dataset}/{predictor}/{held}")
            fit_e = error_memory(fit, fit, leave_one_out=True)
            test_e = error_memory(fit, test, leave_one_out=False)
            fit[list(test_e.columns)] = fit_e.to_numpy()
            test[list(test_e.columns)] = test_e.to_numpy()
            ytr = percentile(fit.true_error_rmse, fit.true_error_rmse).astype(np.float32)
            yte = test.true_error_rmse.to_numpy(float)
            for method, features in GROUPS.items():
                xtr, xte = matrix(fit, test, features)
                scores = []
                for seed in range(args.seeds):
                    scores.append(train_score(xtr, ytr, xte, device, args.seed + seed, args.epochs))
                score = np.mean(np.vstack(scores), axis=0)
                rows.append(
                    {
                        "dataset": dataset,
                        "predictor": predictor,
                        "held_bucket": held,
                        "method": method + "_MLP",
                        "device": str(device),
                        "seeds": args.seeds,
                        "epochs": args.epochs,
                        "n_fit": len(fit),
                        "n_test": len(test),
                        "utility20": utility(score, yte),
                        "spearman": float(spearmanr(score, yte).statistic),
                    }
                )
            print(f"{dataset}/{predictor}/bucket={held}: completed on {device}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "FOLD_RESULTS.csv", index=False)
    summary_rows = []
    for (dataset, predictor), frame in result.groupby(["dataset", "predictor"], sort=True):
        for method, group in frame.groupby("method", sort=True):
            summary_rows.append(
                {
                    "dataset": dataset,
                    "predictor": predictor,
                    "method": method,
                    "n_folds": int(group.held_bucket.nunique()),
                    "utility20_mean": float(group.utility20.mean()),
                    "spearman_mean": float(group.spearman.mean()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "SUMMARY.csv", index=False)
    status = {
        "status": "DEVELOPMENT_ONLY_TRAIN_VAL_PERTURBATION_COLD_GPU_MLP",
        "source": str(args.source),
        "source_splits": ["train", "val"],
        "source_fold_id": 0,
        "no_final_test_rows_read": True,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "groups": {k: list(v) for k, v in GROUPS.items()},
        "limits": [
            "MLP is a feasibility comparator, not a new perturbation predictor.",
            "Confirmatory claims require a frozen raw-history reconstruction and a blind external panel.",
        ],
    }
    (args.output_dir / "STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-buckets", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=273)
    run(parser.parse_args())
