#!/usr/bin/env python3
"""Approval-gated Source DEV label-objective diagnostic; no formal score change."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from unittest.mock import patch

# Keep the fixed, small HGB run on one CPU thread.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import (
    CDF_KEYS, P, PUBLIC, SEEDS, cluster_weights, fit_risk, ids_hash,
    metrics, rank_labels, summarize,
)

CACHE = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/risk_cache")
RUNTIME = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/source_objective_probe_v1")
DOC = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/common_gene_axis/results"
ARCHIVE = DOC / "MATRIX_TASK_PREDICTIONS.csv.gz"
PAIRS = (("GAT_to_Exphormer", "TxPert_GAT", "TxPert_Exphormer"),
         ("Exphormer_to_GAT", "TxPert_Exphormer", "TxPert_GAT"))
SOURCE_PATHS = tuple(CACHE / f"nested_{fold}_{upstream}_{ref}.parquet"
                     for fold in range(5)
                     for upstream in ("TxPert_GAT", "TxPert_Exphormer")
                     for ref in ("Learned", "Manual"))
IDENTITY = list(dict.fromkeys(["task_id", "gene", "fold"] + CDF_KEYS))
SOURCE_COLUMNS = IDENTITY + ["true_error_rmse"]
SEED = SEEDS[0]
MAX_FITS = 20
WALL_SECONDS = 1200
SCORE_TOLERANCE = 1e-14
SCOPE = {
    "role": "SOURCE_DEV_LABEL_OBJECTIVE_DIAGNOSTIC",
    "directions": [p[0] for p in PAIRS], "outer_folds": list(range(5)),
    "label_arms": ["rank", "raw_affine"], "seed": SEED,
    "features": P + PUBLIC, "source_files": 20,
    "learned_feature_files": 10, "manual_identity_only_files": 10,
    "cdf_keys": CDF_KEYS, "cluster_weights": "research.cluster_weights",
    "prediction_clip": False, "old_rank_clip_reference": True,
    "zero_fit_rules": ["Magnitude", "Learned_DirectRMSE", "Learned_WeightedHistoryDistance"],
    "save_fit_models": True, "freeze_scores_before_metrics": True,
    "max_fits": MAX_FITS, "bootstrap_replicates": 0, "parameter_search": False,
}
CORE_PATHS = tuple(ROOT / p for p in (
    "tools/safeconf_continual/research.py", "tools/safeconf_continual/learners.py",
    "tools/safeconf_continual/contracts.py"))


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require_approval(path, output):
    """The first operation in run: no Source hash, read, or fit precedes this."""
    if path is None:
        raise PermissionError("Explicit user approval JSON is required before any Source read or fit")
    document = json.loads(Path(path).read_text())
    if document.get("status") != "USER_APPROVED_SOURCE_DEV_OBJECTIVE_PROBE":
        raise PermissionError("Approval status does not authorize this Source DEV exception")
    required = {"scope": SCOPE, "max_fits": MAX_FITS, "core_unchanged": True,
                "no_external": True, "output_dir": str(output),
                "source_input_paths": [str(p) for p in SOURCE_PATHS],
                "source_reference_csv": str(ARCHIVE)}
    for key, value in required.items():
        if type(document.get(key)) is not type(value) or document[key] != value:
            raise PermissionError(f"Approval has a different or missing {key}")
    if document.get("script_sha256") != file_sha(Path(__file__)):
        raise PermissionError("Approval is not bound to this script SHA256")
    if not isinstance(document.get("user_approval_reference"), str) or not document["user_approval_reference"].strip():
        raise PermissionError("Approval must identify the user's explicit exception")
    return document


def read_source(path):
    path = Path(path)
    if path not in SOURCE_PATHS or path.resolve() != path:
        raise PermissionError("Source path is outside the exact non-symlink whitelist")
    columns = SOURCE_COLUMNS + (P + PUBLIC if path.name.endswith("_Learned.parquet") else [])
    return pd.read_parquet(path, columns=columns).reset_index(drop=True)


def read_archive():
    """Discard external/other rows using string metadata BEFORE numeric parsing."""
    records = []
    with gzip.open(ARCHIVE, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row["line"] not in {p[0] for p in PAIRS}
                    or row["seed"] != str(SEED) or row["method"] != "Learned_hgb"):
                continue
            records.append({**{k: row[k] for k in ["line", "task_id", "target", "gene", "upstream"]},
                            "fold": int(row["fold"]), "risk": float(row["risk"]),
                            "true_error_rmse": float(row["true_error_rmse"])})
    if not records:
        raise RuntimeError("No whitelisted Source LearnHGB reference rows")
    frame = pd.DataFrame(records)
    if frame.duplicated(["line", "task_id"]).any():
        raise RuntimeError("Duplicate Source reference task")
    if not np.isfinite(frame[["risk", "true_error_rmse"]].to_numpy(float)).all():
        raise RuntimeError("Nonfinite whitelisted Source reference numeric values")
    return frame


def raw_affine_labels(train, ranks):
    """Use only training errors; match weighted rank mean/std within CDF_KEYS."""
    if not np.isfinite(ranks).all():
        raise RuntimeError("This probe requires all current training CDF groups to be valid")
    weights = cluster_weights(train)
    labels = np.full(len(train), np.nan)
    audit = []
    for key, group in train.groupby(CDF_KEYS, dropna=False, sort=True):
        positions = group.index.to_numpy()
        w = weights[positions]
        error = group.true_error_rmse.to_numpy(float)
        rank = ranks[positions]
        mean_e, mean_r = np.average(error, weights=w), np.average(rank, weights=w)
        std_e = np.sqrt(np.average((error - mean_e) ** 2, weights=w))
        std_r = np.sqrt(np.average((rank - mean_r) ** 2, weights=w))
        # For constant errors, rank and raw targets are the same constant.
        slope = std_r / std_e if std_e > 0 else 1.0
        if slope <= 0 or (std_e == 0 and std_r != 0):
            raise RuntimeError("Cannot define a positive affine training transform")
        intercept = mean_r - slope * mean_e
        labels[positions] = slope * error + intercept
        transformed = labels[positions]
        matched_mean = np.average(transformed, weights=w)
        matched_std = np.sqrt(np.average((transformed - matched_mean) ** 2, weights=w))
        if not np.allclose([matched_mean, matched_std], [mean_r, std_r], rtol=1e-10, atol=1e-12):
            raise RuntimeError("Weighted affine moments do not match training ranks")
        audit.append(dict(zip(CDF_KEYS, key)) | {
            "n_train_rows": len(group), "slope": float(slope), "intercept": float(intercept),
            "weighted_error_mean": float(mean_e), "weighted_error_std": float(std_e),
            "weighted_rank_mean": float(mean_r), "weighted_rank_std": float(std_r),
            "weighted_affine_mean": float(matched_mean), "weighted_affine_std": float(matched_std),
            "affine_min": float(transformed.min()), "affine_max": float(transformed.max()),
            "affine_below_zero": int((transformed < 0).sum()),
            "affine_above_one": int((transformed > 1).sum()), "training_only": True})
    return labels, audit


def tail_audit(scores):
    scores = np.asarray(scores, float)
    clipped = np.clip(scores, 0, 1)
    boundary = np.sort(scores)[-math.ceil(.2 * len(scores))]
    return {"n_rows": len(scores), "min": float(scores.min()), "max": float(scores.max()),
            "q01": float(np.quantile(scores, .01)), "q99": float(np.quantile(scores, .99)),
            "below_zero": int((scores < 0).sum()), "above_one": int((scores > 1).sum()),
            "changed_by_clipping": int((scores != clipped).sum()),
            "unique_unbounded": len(np.unique(scores)), "unique_clipped": len(np.unique(clipped)),
            "top20_boundary_ties": int((scores == boundary).sum())}


def validate_frames(frames):
    for fold in range(5):
        for upstream in ("TxPert_GAT", "TxPert_Exphormer"):
            learned = frames[CACHE / f"nested_{fold}_{upstream}_Learned.parquet"]
            manual = frames[CACHE / f"nested_{fold}_{upstream}_Manual.parquet"]
            if not learned[SOURCE_COLUMNS].equals(manual[SOURCE_COLUMNS]):
                raise RuntimeError("Manual/Learned task order, identity, CDF keys, or truth differs")
            if (learned.empty or learned.task_id.duplicated().any()
                    or set(learned.upstream) != {upstream}
                    or set(learned.fold) != set(range(5))
                    or not np.isfinite(learned.true_error_rmse.to_numpy(float)).all()):
                raise RuntimeError("Source cohort identity, folds, or truth failed")
            if learned.groupby("gene").fold.nunique().max() != 1:
                raise RuntimeError("A Source gene crosses outer folds")


def alarm_timeout(_signum, _frame):
    raise TimeoutError(f"Fixed {WALL_SECONDS}s read/fit/evaluation budget exhausted")


def run(args):
    output = args.output
    approval = require_approval(args.approval, output)
    if output.parent != RUNTIME or output.resolve() != output or output.exists():
        raise RuntimeError("Output must be a fresh, non-symlink child of the isolated runtime root")
    all_inputs = SOURCE_PATHS + (ARCHIVE,) + CORE_PATHS + (Path(__file__).resolve(),)
    if any(p.resolve() != p for p in all_inputs):
        raise PermissionError("An input resolves outside its fixed whitelist path")
    before = {str(p): file_sha(p) for p in all_inputs}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "REGISTRATION.json", {"scope": SCOPE, "approval": approval,
        "input_sha256_before": before, "wall_budget_seconds": WALL_SECONDS,
        "formal_score_promotion": False, "new_upstream_calls": 0,
        "hgb_recipe": {"max_iter": 200, "learning_rate": .05, "max_depth": 3,
                       "min_samples_leaf": 20, "l2_regularization": 10., "random_state": SEED}})
    previous_handler = signal.signal(signal.SIGALRM, alarm_timeout)
    signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS)
    started = time.perf_counter()
    costs, predictions, cdf_rows, affine_rows, replays, tails, model_paths = [], [], [], [], [], [], []
    state, failure = "FAILED", None
    try:
        frames = {p: read_source(p) for p in SOURCE_PATHS}
        validate_frames(frames)
        old = read_archive()
        for line, source, query_upstream in PAIRS:
            for fold in range(5):
                a = frames[CACHE / f"nested_{fold}_{source}_Learned.parquet"]
                b = frames[CACHE / f"nested_{fold}_{query_upstream}_Learned.parquet"]
                train = a[a.fold.ne(fold)].reset_index(drop=True)
                query = b[b.fold.eq(fold)].reset_index(drop=True)
                if train.empty or query.empty or set(train.gene) & set(query.gene):
                    raise RuntimeError("Missing fold or biological cluster overlap")
                ranks, audit = rank_labels(train, f"source-objective/{line}/outer{fold}")
                cdf_rows.extend(audit)
                affine, audit = raw_affine_labels(train, ranks)
                affine_rows.extend({"line": line, "fold": fold, **r} for r in audit)
                for method, risk in (("Magnitude", query.predicted_magnitude.to_numpy(float)),
                        ("Learned_DirectRMSE", query.prediction_prior_rmse.to_numpy(float)),
                        ("Learned_WeightedHistoryDistance", np.sqrt(
                            query.prediction_prior_rmse.to_numpy(float) ** 2 + query.prior_uncertainty.to_numpy(float) ** 2))):
                    if not np.isfinite(risk).all():
                        raise RuntimeError("Nonfinite fixed zero-fit rule predictions")
                    reference = query[["task_id", "target", "gene", "fold", "upstream", "true_error_rmse"]].copy()
                    reference["line"], reference["seed"], reference["method"], reference["risk"] = line, SEED, method, risk
                    predictions.append(reference)
                for arm, labels in (("rank", ranks), ("raw_affine", affine)):
                    if len(costs) >= MAX_FITS:
                        raise RuntimeError("The fixed 20-fit budget is exhausted")
                    tick = time.perf_counter()
                    model = fit_risk(train, labels, P + PUBLIC, "hgb", SEED, weighted=True)
                    risk = model.predict(query, clip=False)
                    if not np.isfinite(risk).all():
                        raise RuntimeError("Nonfinite unbounded diagnostic predictions")
                    costs.append({"line": line, "fold": fold, "arm": arm,
                        "fit_and_predict_seconds": time.perf_counter() - tick,
                        "n_train_rows": len(train), "n_train_clusters": train.gene.nunique(),
                        "training_records_hash": ids_hash(train.upstream + "::" + train.task_id)})
                    model_paths.extend(Path(p) for p in joblib.dump(
                        model, output / f"MODEL_{line}_fold{fold}_{arm}.joblib", compress=3))
                    part = query[["task_id", "target", "gene", "fold", "upstream", "true_error_rmse"]].copy()
                    part["line"], part["seed"], part["method"], part["risk"] = line, SEED, arm + "_unbounded", risk
                    predictions.append(part)
                    for context, group in part.groupby("target", sort=True):
                        tails.append({"line": line, "fold": fold, "target": context,
                                      "method": part.method.iloc[0], **tail_audit(group.risk)})
                    if arm == "rank":
                        clipped = part.copy()
                        clipped["method"], clipped["risk"] = "rank_clipped_old_reference", np.clip(risk, 0, 1)
                        archive = old[old.line.eq(line) & old.fold.eq(fold)]
                        keys = ["task_id", "target", "gene", "fold", "upstream"]
                        paired = clipped.merge(archive, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
                        if len(paired) != len(query) or len(archive) != len(query):
                            raise RuntimeError("Source score replay has a different task identity")
                        difference = float(np.max(np.abs(paired.risk_new - paired.risk_old)))
                        if (difference > SCORE_TOLERANCE or not np.allclose(
                                paired.true_error_rmse_new, paired.true_error_rmse_old, rtol=1e-12, atol=1e-14)):
                            raise RuntimeError("Exact current Source LearnHGB score replay failed; stop before raw arm")
                        replays.append({"line": line, "fold": fold, "n_rows": len(paired),
                                        "max_abs_score_difference": difference, "tolerance": SCORE_TOLERANCE})
                        predictions.append(clipped)
        if len(costs) != MAX_FITS or len(replays) != 10 or sum(r["n_rows"] for r in replays) != len(old):
            raise RuntimeError("Fixed fit or reference replay coverage is incomplete")
        records = pd.concat(predictions, ignore_index=True)
        score_path = output / "TASK_PREDICTIONS.csv.gz"
        records.to_csv(score_path, index=False)
        write_json(output / "SCORE_FREEZE.json", {"status": "FROZEN_BEFORE_METRICS",
            "script_sha256": before[str(Path(__file__).resolve())],
            "task_predictions_sha256": file_sha(score_path), "models_saved": len(model_paths),
            "model_sha256": {str(p): file_sha(p) for p in model_paths},
            "fits": len(costs), "zero_fit_rules": SCOPE["zero_fit_rules"],
            "formal_score_promotion": False})
        contexts, lines = summarize(records)
        contexts.to_csv(output / "CONTEXT_METRICS.csv", index=False)
        lines.to_csv(output / "LINE_MACRO_METRICS.csv", index=False)
        fold_rows = []
        for key, group in records.groupby(["line", "method", "seed", "fold", "target"], sort=True):
            fold_rows.append(dict(zip(["line", "method", "seed", "fold", "target"], key)) | metrics(group, group.risk))
        pd.DataFrame(fold_rows).to_csv(output / "FOLD_CONTEXT_METRICS.csv", index=False)
        state = "COMPLETE_DEV_DIAGNOSTIC"
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        for name, rows in (("FIT_COSTS", costs), ("CDF_AUDIT", cdf_rows), ("AFFINE_TRAINING_AUDIT", affine_rows),
                           ("CURRENT_SCORE_REPLAY", replays), ("TAIL_CLIPPING_AUDIT", tails)):
            pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
        after = {str(p): file_sha(p) for p in all_inputs}
        unchanged = before == after
        write_json(output / "STATUS.json", {"status": state if unchanged else "FAILED_INPUT_MUTATION",
            "failure": failure, "fits_completed": len(costs), "fit_upper_bound": MAX_FITS,
            "elapsed_seconds": time.perf_counter() - started, "input_sha256_after": after,
            "all_inputs_unchanged": unchanged, "external_numeric_rows_parsed": 0,
            "query_errors_used_in_label_transform": False, "formal_score_promotion": False})
        if not unchanged:
            raise RuntimeError("An input or unchanged core file changed during the diagnostic")
    print(json.dumps({"status": state, "output": str(output), "fits": len(costs)}))


def self_check():
    # Synthetic heteroskedastic groups: A has rare huge error, B has steady error.
    errors = np.array([0.] * 9 + [100.] + [2.] * 10)
    toy = pd.DataFrame({"true_error_rmse": errors, "task_id": [f"toy{i}" for i in range(20)],
                        "gene": [f"g{i}" for i in range(20)],
                        **{key: ["synthetic"] * 20 for key in CDF_KEYS}})
    ranks, _ = rank_labels(toy, "synthetic-self-check")
    raw, audit = raw_affine_labels(toy, ranks)
    raw_means = [float(errors[:10].mean()), float(errors[10:].mean())]
    rank_means = [float(ranks[:10].mean()), float(ranks[10:].mean())]
    assert raw_means[0] > raw_means[1] and rank_means[0] < rank_means[1]
    assert raw[:10].mean() > raw[10:].mean() and audit[0]["slope"] > 0
    attempts = []
    def forbidden(*_args, **_kwargs):
        attempts.append("forbidden_source_or_fit_access")
        raise AssertionError("Permission rejection must happen before Source I/O or fitting")
    rejected = False
    with patch(__name__ + ".read_source", forbidden), patch(__name__ + ".read_archive", forbidden), \
            patch(__name__ + ".fit_risk", forbidden), patch(__name__ + ".file_sha", forbidden):
        try:
            run(argparse.Namespace(approval=None, output=RUNTIME / "self_check_never_created"))
        except PermissionError:
            rejected = True
    assert rejected and not attempts
    print(json.dumps({"status": "SELF_CHECK_PASS", "synthetic_raw_mean_A_B": raw_means,
        "synthetic_rank_mean_A_B": rank_means, "positive_affine_slope": audit[0]["slope"],
        "weighted_moments_matched": True, "permission_rejection_verified": rejected,
        "source_read_attempts": 0, "actual_fits": 0, "files_written": 0,
        "hypothesis_is_possible_mismatch_not_established_root_cause": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="Synthetic math and no-read permission rejection only")
    parser.add_argument("--approval", type=Path, help="JSON recording the user's explicit label-rule exception")
    parser.add_argument("--output", type=Path, help=f"Fresh absolute child of {RUNTIME}")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.output is None:
        parser.error("--output is required for an approved real run")
    else:
        run(args)


if __name__ == "__main__":
    main()
