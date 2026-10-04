#!/usr/bin/env python3
"""One fixed TabPFN V2 target-feedback audit on the 542-task DEV contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.learners import NumericPreprocessor  # noqa: E402
from tools.safeconf_continual.research import P, PUBLIC, SEEDS, cluster_weights  # noqa: E402

DEV = Path("/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1/DEV_FEATURES.parquet")
CHECKPOINT = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/tabpfn_v2_probe/tabpfn-v2-regressor.ckpt")
CHECKPOINT_SHA = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
SEED = 20260930
FEATURES = P + PUBLIC


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def labels_by_context(frame: pd.DataFrame) -> np.ndarray:
    y = np.full(len(frame), np.nan)
    for _, g in frame.groupby("target", sort=True):
        vals = g.true_error_rmse.to_numpy(float)
        y[g.index.to_numpy()] = (pd.Series(vals).rank(method="average").to_numpy() - 0.5) / len(vals)
    return y


def metric(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    vals = []
    for target, g in frame.assign(_score=scores).groupby("target", sort=True):
        y = g.true_error_rmse.to_numpy(float); s = g._score.to_numpy(float)
        ids = g.task_id.astype(str).to_numpy(); k = max(1, int(np.ceil(.2 * len(g))))
        hi = np.lexsort((ids, -s))[:k]; oracle = np.lexsort((ids, -y))[:k]
        den = y[oracle].mean() - y.mean()
        u = (y[hi].mean() - y.mean()) / den if len(g) >= 20 and den > 1e-12 else np.nan
        lo = np.lexsort((ids, s))
        vals.append((u, float(np.mean(np.cumsum(y[lo]) / np.arange(1, len(y) + 1))))
                    )
    arr = np.asarray(vals, float)
    return {"utility20": float(np.nanmean(arr[:, 0])), "aurc": float(np.nanmean(arr[:, 1])),
            "valid_contexts": int(np.isfinite(arr[:, 0]).sum()), "contexts": len(arr)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["preflight", "dev", "fixed"], default="preflight")
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    dev = pd.read_parquet(DEV)
    if len(dev) != 542 or dev.gene.nunique() != 377:
        raise RuntimeError("DEV contract changed")
    audit = {"status": "PASS", "dev_rows": len(dev), "dev_genes": dev.gene.nunique(),
             "budget": args.budget, "budget_genes": 114, "features": FEATURES,
             "checkpoint_sha256": sha(CHECKPOINT), "expected_checkpoint_sha256": CHECKPOINT_SHA,
             "sample_weight_supported": False, "truth_used_for_fit": False,
             "permanent_test_truth_opened": False}
    if audit["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("fixed TabPFN checkpoint hash changed")
    write_json(out / "TABPFN_TARGET50_INPUT_AUDIT.json", audit)
    if args.phase == "preflight":
        print(json.dumps(audit)); return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    import torch
    import tabpfn
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    if args.phase == "fixed":
        pool = pd.read_parquet("/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1/FIXED_POOL_FEATURES.parquet")
        query = pd.read_parquet("/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1/HOLDOUT_FEATURES.parquet")
        if len(pool) != 331 or pool.gene.nunique() != 228 or len(query) != 212 or query.gene.nunique() != 152:
            raise RuntimeError("fixed target cohort changed")
        order = sorted(pool.gene.astype(str).unique(), key=lambda g: hashlib.sha256(
            f"SafeConf-McFaline-feedback-v1\0{g}".encode()).hexdigest())
        fit = pool[pool.gene.astype(str).isin(set(order[:114]))].reset_index(drop=True)
        labels = labels_by_context(fit); valid = np.isfinite(labels)
        transform = NumericPreprocessor().fit(fit[FEATURES].to_numpy(float)[valid])
        x_train = transform.transform(fit[FEATURES].to_numpy(float)); x_query = transform.transform(query[FEATURES].to_numpy(float))
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        learner = TabPFNRegressor.create_default_for_version(ModelVersion.V2, model_path=str(CHECKPOINT), random_state=args.seed, device=device)
        start = time.perf_counter(); learner.fit(x_train[valid], labels[valid]);
        if torch.cuda.is_available(): torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter(); score = np.clip(np.asarray(learner.predict(x_query), float), 0, 1)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        predict_seconds = time.perf_counter() - start
        result = query[["task_id", "target", "gene", "true_error_rmse"]].copy()
        result["method"] = "TabPFN_V2_F1_50"; result["budget"] = 0.5; result["seed"] = args.seed; result["risk"] = score
        write_csv(out / "TABPFN_TARGET50_HOLDOUT_PREDICTIONS.csv", result)
        write_csv(out / "TABPFN_TARGET50_HOLDOUT_RESOURCE_COSTS.csv", pd.DataFrame([{
            "fit_rows": len(fit), "fit_genes": fit.gene.nunique(), "query_rows": len(query),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "gpu_hours": (fit_seconds + predict_seconds) / 3600 if device.startswith("cuda") else 0,
            "device": device, "seed": args.seed,
        }]))
        write_json(out / "TABPFN_TARGET50_HOLDOUT_STATUS.json", {
            "status": "COMPLETE", "role": "FROZEN_POST_DEV_GATE_HOLDOUT_SCORE",
            "fit_rows": len(fit), "fit_genes": fit.gene.nunique(), "query_rows": len(query),
            "query_genes": query.gene.nunique(), "holdout_used_for_fit": False,
            "dev_gate": "TabPFN_V2_F1_50_vs_H1_F1 point +0.0314, CI [-0.00438,+0.09005]",
            "permanent_test_truth_opened": False,
        })
        print(json.dumps({"status": "COMPLETE", "phase": "fixed", "rows": len(result), "fit_genes": fit.gene.nunique()}))
        return 0

    order = sorted(dev.gene.astype(str).unique(), key=lambda g: hashlib.sha256(
        f"SafeConf-target-dev-v1|{g}".encode()).hexdigest())
    records, costs = [], []
    for held in sorted(dev.dev_fold.unique()):
        train_all = dev[dev.dev_fold.ne(held)].reset_index(drop=True)
        query = dev[dev.dev_fold.eq(held)].reset_index(drop=True)
        allowed = [g for g in order if g in set(train_all.gene.astype(str))]
        fit = train_all[train_all.gene.astype(str).isin(set(allowed[:114]))].reset_index(drop=True)
        labels = labels_by_context(fit)
        valid = np.isfinite(labels)
        x_train = fit[FEATURES].to_numpy(float); x_query = query[FEATURES].to_numpy(float)
        transform = NumericPreprocessor().fit(x_train[valid])
        x_train = transform.transform(x_train); x_query = transform.transform(x_query)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        learner = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, model_path=str(CHECKPOINT), random_state=args.seed, device=device)
        start = time.perf_counter(); learner.fit(x_train[valid], labels[valid]);
        if torch.cuda.is_available(): torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter(); score = np.clip(np.asarray(learner.predict(x_query), float), 0, 1)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        predict_seconds = time.perf_counter() - start
        part = query[["task_id", "target", "gene", "true_error_rmse"]].copy()
        part["method"] = "TabPFN_V2_F1_50"; part["heldout_fold"] = held; part["risk"] = score
        records.append(part)
        costs.append({"heldout_fold": int(held), "fit_rows": len(fit), "fit_genes": fit.gene.nunique(),
                      "query_rows": len(query), "fit_seconds": fit_seconds,
                      "predict_seconds": predict_seconds,
                      "gpu_hours": (fit_seconds + predict_seconds) / 3600 if device.startswith("cuda") else 0,
                      "package_version": tabpfn.__version__, "torch_version": torch.__version__,
                      "device": device})
    pred = pd.concat(records, ignore_index=True)
    write_csv(out / "TABPFN_TARGET50_DEV_PREDICTIONS.csv", pred)
    write_csv(out / "TABPFN_TARGET50_RESOURCE_COSTS.csv", pd.DataFrame(costs))
    rows = [{"method": "TabPFN_V2_F1_50", **metric(pred, pred.risk.to_numpy(float))}]
    write_csv(out / "TABPFN_TARGET50_RESULTS.csv", pd.DataFrame(rows))
    write_json(out / "TABPFN_TARGET50_STATUS.json", {
        "status": "COMPLETE", "folds": sorted(map(int, dev.dev_fold.unique())),
        "rows": len(pred), "genes": pred.gene.nunique(), "gpu_hours": float(sum(x["gpu_hours"] for x in costs)),
        "permanent_test_truth_opened": False, "parameter_search": False,
    })
    print(json.dumps({"status": "COMPLETE", "rows": len(pred), "gpu_hours": sum(x["gpu_hours"] for x in costs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
