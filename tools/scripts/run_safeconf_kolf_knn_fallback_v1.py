#!/usr/bin/env python3
"""Bounded upstream-predictor fallback for the KOLF confirmation study.

This is deliberately a *predictor* fallback, not a new risk model.  It is
allowed only after both pre-registered Ridge and MLP predictors fail the
competence gate, before any confirmation expression is read.  The fallback
uses the same frozen gene representation and the same predictor-train genes;
feedback and confirmation responses are never used to choose ``k`` or fit
the neighbour regressor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.submission_evidence import digest_ids, sha, write_json
from tools.scripts import run_safeconf_gladstone_v21 as base
from tools.scripts import run_safeconf_kolf_panel_v21 as kolf

RUN = base.RUN
OUT = RUN / "external_kolf_panel1400_v1"
KS = (5, 15, 30)
SEED = 20261009


def _normalised(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 1e-12, n, 1.0)


def _stable_holdout(genes: pd.Series) -> np.ndarray:
    order = np.argsort([
        hashlib.sha256(f"SafeConf-KOLF-KNN-upstream-holdout|{g}".encode()).hexdigest()
        for g in genes.astype(str)
    ], kind="stable")
    n = max(1, len(order) // 10)
    return np.asarray(order[:n], dtype=int)


def _fit_predict(train_x, train_y, query_x, k, leave_ids=None, train_ids=None):
    """Deterministic cosine-neighbour mean; optionally leave the own gene out."""
    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    query_x = np.asarray(query_x, dtype=np.float64)
    out = np.empty((len(query_x), train_y.shape[1]), dtype=np.float64)
    train_ids = np.asarray(train_ids, dtype=object) if train_ids is not None else None
    leave_ids = np.asarray(leave_ids, dtype=object) if leave_ids is not None else None
    for i, q in enumerate(query_x):
        similarity = train_x @ q
        allowed = np.ones(len(train_x), dtype=bool)
        if leave_ids is not None and train_ids is not None:
            allowed &= train_ids != leave_ids[i]
        candidates = np.flatnonzero(allowed)
        if not len(candidates):
            out[i] = train_y.mean(axis=0)
            continue
        kk = min(int(k), len(candidates))
        order = candidates[np.argsort(-similarity[candidates], kind="stable")[:kk]]
        out[i] = train_y[order].mean(axis=0)
    return out.astype(np.float32)


def _require_preconditions():
    if (OUT / "CONFIRMATION_EVALUATION_OPEN_EVENT.json").exists():
        raise RuntimeError("KNN fallback is forbidden after confirmation truth access")
    if (OUT / "RISK_FREEZE.json").exists():
        raise RuntimeError("risk scores already frozen; fallback cannot replace a frozen system")
    decision_path = OUT / "EXTERNAL_DECISION.json"
    if not decision_path.exists():
        raise RuntimeError("original predictor qualification decision is missing")
    decision = json.loads(decision_path.read_text())
    if decision.get("passed_predictors"):
        raise RuntimeError("fallback is only allowed when all original predictors fail")
    for p in [OUT / "PREDICTOR_FREEZE.json", OUT / "PREDICTOR_FREEZE_PRE_KNN.json"]:
        if p.exists() and p.name == "PREDICTOR_FREEZE_PRE_KNN.json":
            raise RuntimeError("KNN fallback has already been applied")
    required = [OUT / "CONTROL_FEATURES.npz", OUT / "TRAIN_FEEDBACK_TASKS.parquet",
                OUT / "TRAIN_FEEDBACK_EFFECTS.npy", OUT / "PREDICTION_TASKS.parquet",
                OUT / "PREDICTOR_FREEZE.json"]
    for p in required:
        if not p.exists():
            raise RuntimeError(f"missing fallback input: {p}")


def run():
    # Rebind all shared globals to the KOLF root without touching raw data.
    kolf.configure()
    _require_preconditions()
    write_json(OUT / "KNN_FALLBACK_CONTINUATION_CARD.json", {
        "run_id": OUT.name,
        "status": "AUTHORIZED_BOUNDED_ALTERNATE_PREDICTOR",
        "trigger": "ridge_and_mlp_both_failed_pre_registered_competence_gate",
        "representation": "frozen CONTROL_FEATURES embedding from predictor-train-only PCA",
        "candidate_k": list(KS), "selection": "upstream-only 10-percent gene holdout",
        "distance": "cosine similarity after row L2 normalization",
        "fit_roles": ["predictor_train"], "feedback_truth_used_for_fit": False,
        "confirmation_truth_read_before_fit": False, "new_gpu_hours": 0,
        "new_download_bytes": 0, "max_models": 1,
        "fallback_does_not_modify_original_predictors": True,
    })
    pre = OUT / "PREDICTOR_FREEZE_PRE_KNN.json"
    shutil.copy2(OUT / "PREDICTOR_FREEZE.json", pre)

    roles = pd.read_parquet(OUT / "TASK_ROLES.parquet")
    fit_tasks = pd.read_parquet(OUT / "TRAIN_FEEDBACK_TASKS.parquet").reset_index(drop=True)
    y_all = np.load(OUT / "TRAIN_FEEDBACK_EFFECTS.npy").astype(np.float64)
    z = np.load(OUT / "CONTROL_FEATURES.npz")
    emb = {str(g): np.asarray(v, dtype=np.float64) for g, v in zip(z["genes"], z["embedding"])}
    train_mask = fit_tasks.role.astype(str).eq("predictor_train").to_numpy()
    if not train_mask.any():
        raise RuntimeError("no predictor-train rows available for KNN fallback")
    train = fit_tasks.loc[train_mask].reset_index(drop=True)
    y = y_all[train_mask]
    known = np.asarray([g in emb for g in train.gene.astype(str)], dtype=bool)
    train = train.loc[known].reset_index(drop=True)
    y = y[known]
    if len(train) < max(KS):
        raise RuntimeError(f"insufficient known predictor-train genes for KNN: {len(train)}")
    x = _normalised(np.asarray([emb[g] for g in train.gene.astype(str)], dtype=np.float64))

    hold = _stable_holdout(train.gene)
    keep = np.ones(len(train), dtype=bool); keep[hold] = False
    if not keep.any():
        raise RuntimeError("KNN upstream holdout removed all training genes")
    val_scores = {}
    for k in KS:
        pred = _fit_predict(x[keep], y[keep], x[hold], k)
        val_scores[k] = float(np.sqrt(np.mean((pred.astype(float) - y[hold]) ** 2)))
    min_score = min(val_scores.values())
    chosen = max(k for k, score in val_scores.items() if score <= min_score + 1e-12)

    # Fit all permitted upstream genes.  Training rows are leave-one-gene-out
    # to prevent a trivial identity lookup in the saved train predictions.
    all_tasks = pd.read_parquet(OUT / "PREDICTION_TASKS.parquet").reset_index(drop=True)
    qknown = np.asarray([g in emb for g in all_tasks.gene.astype(str)], dtype=bool)
    qx = np.asarray([emb[g] for g in all_tasks.loc[qknown, "gene"].astype(str)], dtype=np.float64)
    qx = _normalised(qx) if len(qx) else np.empty((0, x.shape[1]))
    preds = np.repeat(y.mean(axis=0, keepdims=True).astype(np.float32), len(all_tasks), axis=0)
    qgenes = all_tasks.loc[qknown, "gene"].astype(str).to_numpy()
    fitted = _fit_predict(x, y, qx, chosen, leave_ids=qgenes, train_ids=train.gene.astype(str).to_numpy())
    preds[qknown] = fitted
    np.save(OUT / "knn_ALL_PREDICTIONS.npy", preds.astype(np.float32))
    model_payload = {
        "genes": train.gene.astype(str).to_numpy(), "embedding": x.astype(np.float32),
        "responses": y.astype(np.float32), "k": np.asarray([chosen], dtype=np.int64),
    }
    np.savez(OUT / "KNN_MODEL.npz", **model_payload)
    write_json(OUT / "KNN_SELECTION_LEDGER.json", {
        "candidate_k": list(KS), "validation_rmse": {str(k): v for k, v in val_scores.items()},
        "chosen_k": int(chosen), "holdout_gene_hash": digest_ids(train.gene.iloc[hold].astype(str)),
        "holdout_genes": int(len(hold)), "fit_genes": int(len(train)),
        "fit_gene_hash": digest_ids(train.gene.astype(str)), "query_rows": int(len(all_tasks)),
        "known_query_rows": int(qknown.sum()), "unknown_query_rows": int((~qknown).sum()),
        "leave_own_gene_out": True, "feedback_truth_used_for_k": False,
        "confirmation_truth_read": False, "representation_file": str(OUT / "CONTROL_FEATURES.npz"),
    })

    original = json.loads((OUT / "PREDICTOR_FREEZE.json").read_text())
    original["prediction_sha256"]["knn"] = sha(OUT / "knn_ALL_PREDICTIONS.npy")
    original["predictor_artifact_sha256"]["KNN_MODEL.npz"] = sha(OUT / "KNN_MODEL.npz")
    original["fallback_predictor"] = {
        "name": "knn", "selection_ledger_sha256": sha(OUT / "KNN_SELECTION_LEDGER.json"),
        "continuation_card_sha256": sha(OUT / "KNN_FALLBACK_CONTINUATION_CARD.json"),
        "original_predictors_preserved": True,
    }
    original["status"] = "FROZEN_WITH_BOUNDED_KNN_FALLBACK"
    write_json(OUT / "PREDICTOR_FREEZE.json", original)
    write_json(OUT / "KNN_FALLBACK_RESOURCE_COST.json", {
        "new_gpu_hours": 0, "new_download_bytes": 0, "cpu_threads": 4,
        "fit_gene_count": int(len(train)), "additional_predictor_count": 1,
        "original_freeze_preserved": True,
    })
    base.qualification()
    decision = json.loads((OUT / "EXTERNAL_DECISION.json").read_text())
    write_json(OUT / "KNN_FALLBACK_DECISION.json", {
        "original_decision": "ALL_ORIGINAL_PREDICTORS_FAILED",
        "knn_status": "PASS" if "knn" in decision.get("passed_predictors", []) else "FAIL",
        "passed_predictors_after_fallback": decision.get("passed_predictors", []),
        "confirmation_truth_read": False,
        "selection_cost_separate": True,
    })
    if not decision.get("passed_predictors"):
        base.stress_test_development()
        write_json(OUT / "PIPELINE_STATUS.json", {
            "status": "COMPETENCE_FAILED_AFTER_BOUNDED_KNN",
            "confirmation_truth_read": False,
            "research_complete": False,
            "fallback_attempted": True,
        })
        return
    base.risk_freeze()
    if not (OUT / "RISK_FREEZE.json").exists():
        raise RuntimeError("KNN passed qualification but risk freeze was not written")
    # The risk scores are now frozen; only then is confirmation access legal.
    write_json(OUT / "CONFIRMATION_EVALUATION_OPEN_EVENT.json", {
        "authorized_by": "bounded KNN fallback under SafeConf v2.1",
        "risk_freeze_sha256": sha(OUT / "RISK_FREEZE.json"),
        "scope": "registered KOLF1400 panel", "confirmation_truth_read": False,
        "fallback_predictor": "knn",
    })
    kolf.prepare(["confirmation"])
    base.confirmation()


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", type=Path, default=OUT)
    args = ap.parse_args()
    OUT = args.result_root.resolve()
    # The registered run root is the only permitted fallback location.
    if OUT != RUN / "external_kolf_panel1400_v1":
        raise RuntimeError("fallback result root is not the registered KOLF root")
    run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(OUT / "KNN_FALLBACK_FAILURE_RECEIPT.json", {
            "status": "ENGINEERING_OR_RESOURCE_FAILURE", "error_type": type(exc).__name__,
            "error": str(exc), "confirmation_opened": (OUT / "CONFIRMATION_EVALUATION_OPEN_EVENT.json").exists(),
        })
        raise
