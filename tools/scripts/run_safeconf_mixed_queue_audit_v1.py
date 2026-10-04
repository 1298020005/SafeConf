#!/usr/bin/env python3
"""Audit the already-sealed 2,993-task mixed-history SafeConf ranking.

This is a SEEN retrospective audit.  The score file was frozen before the
cached Orion errors were read; this script never opens the raw truth file and
does not select a method from the mixed-queue scores.  It makes the fallback
contract explicit: expanded public distance on covered tasks and the frozen
amplitude CDF for tasks with no legal history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GWPS = Path("/home/yyf/runtime_artifacts/safeconf_research_20261003/gwps_v1")
DEFAULT_OUT = Path("/home/yyf/runtime_artifacts/safeconf_impl_20261004_v1/mixed_queue")
SCORE_FILE = GWPS / "SCORES_AND_ORIGINAL_ERRORS.parquet"
SEAL_FILE = GWPS / "SCORE_SEAL.json"
EVAL_FILE = GWPS / "EVALUATION_COMPLETE.json"
DRAW_FILE = GWPS / "all_eligible_BOOTSTRAP_DRAWS.npy"
METRIC_NAMES = ("utility20", "spearman", "aurc", "high_risk_miss_rate", "error_at_10", "error_at_20", "error_at_50")
DRAW_METHODS = ("Magnitude", "Old_DistanceCDF_or_Magnitude", "Expanded_DistanceCDF_or_Magnitude", "Old_HGB_or_Magnitude", "Expanded_HGB_or_Magnitude")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def task_metric(part: pd.DataFrame, score: np.ndarray) -> dict:
    y = part.true_error_rmse.to_numpy(float)
    ids = part.query_id.astype(str).to_numpy()
    s = np.asarray(score, float)
    valid = np.isfinite(y) & np.isfinite(s)
    y, ids, s = y[valid], ids[valid], s[valid]
    out = {"n_tasks": int(len(y)), "u20": np.nan, "aurc": np.nan,
           "spearman": np.nan, "high_risk_miss_rate": np.nan,
           "true_top20_found": np.nan, "review_count": int(np.ceil(.2 * len(y))) if len(y) else 0,
           "remaining_mean_error": np.nan}
    if not len(y):
        return out
    k = max(1, int(np.ceil(.2 * len(y))))
    high = np.lexsort((ids, -s))[:k]
    oracle = np.lexsort((ids, -y))[:k]
    denom = y[oracle].mean() - y.mean()
    if len(y) >= 20 and denom > 1e-12:
        out["u20"] = float((y[high].mean() - y.mean()) / denom)
    if len(y) >= 3 and np.ptp(s) > 0 and np.ptp(y) > 0:
        out["spearman"] = float(spearmanr(s, y).statistic)
    low = np.lexsort((ids, s))
    out["aurc"] = float(np.mean(np.cumsum(y[low]) / np.arange(1, len(y) + 1)))
    found = int(len(set(high).intersection(set(oracle))))
    out["true_top20_found"] = found
    out["high_risk_miss_rate"] = float(1.0 - found / k)
    out["remaining_mean_error"] = float(np.delete(y, high).mean())
    return out


def actual_discovery(frame: pd.DataFrame, score_map: dict[str, np.ndarray]) -> pd.DataFrame:
    y = frame.true_error_rmse.to_numpy(float)
    ids = frame.query_id.astype(str).to_numpy()
    k = max(1, int(np.ceil(.2 * len(y))))
    oracle = set(np.lexsort((ids, -y))[:k])
    rows = []
    for method, score in score_map.items():
        selected = np.lexsort((ids, -np.asarray(score, float)))[:k]
        found = int(len(set(selected).intersection(oracle)))
        rows.append({"method": method, "n_tasks": len(frame), "review_count": k,
                     "review_fraction": k / len(frame), "true_high_error_found": found,
                     "true_high_error_hit_rate": found / k,
                     "remaining_mean_error": float(np.delete(y, selected).mean()),
                     "all_task_mean_error": float(y.mean())})
    out = pd.DataFrame(rows)
    amp = out.loc[out.method.eq("Amplitude")].iloc[0]
    public = out.loc[out.method.eq("PublicRule_mixed")].iloc[0]
    out["additional_high_error_found_vs_amplitude"] = out.true_high_error_found - amp.true_high_error_found
    out["remaining_error_reduction_vs_amplitude"] = amp.remaining_mean_error - out.remaining_mean_error
    out["additional_high_error_found_vs_public"] = out.true_high_error_found - public.true_high_error_found
    out["remaining_error_reduction_vs_public"] = public.remaining_mean_error - out.remaining_mean_error
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    seal = json.loads(SEAL_FILE.read_text())
    evaluation = json.loads(EVAL_FILE.read_text())
    if seal.get("status") != "SCORES_FROZEN_BEFORE_CACHED_ERRORS_READ":
        raise RuntimeError("mixed audit requires the pre-error score seal")
    frame = pd.read_parquet(SCORE_FILE).copy()
    if len(frame) != 2993 or frame.query_id.nunique() != 2993:
        raise RuntimeError("mixed queue size or query identity changed")
    if not np.array_equal(frame.context_id.astype(str).unique(), np.array(frame.context_id.astype(str).unique())):
        raise RuntimeError("context identity is not stable")

    # Expanded_DistanceCDF_or_Magnitude is the only frozen no-label public
    # score that is finite for both covered and naturally missing-history tasks.
    frame["history_status"] = np.where(frame.Expanded_n_history.to_numpy(int) > 0, "PUBLIC_HISTORY_AVAILABLE", "NO_LEGAL_PUBLIC_HISTORY")
    frame["evidence_status"] = np.where(frame.Expanded_n_history.to_numpy(int) > 0, "public_history", "amplitude_fallback")
    score_map = {
        "Amplitude": frame.Magnitude.to_numpy(float),
        "PublicRule_mixed": frame.Expanded_DistanceCDF_or_Magnitude.to_numpy(float),
        "SourceHGB_mixed": frame.Expanded_HGB_or_Magnitude.to_numpy(float),
    }
    if any(not np.isfinite(x).all() for x in score_map.values()):
        raise RuntimeError("mixed unified score contains non-finite values")
    for name, score in score_map.items():
        frame[name] = score
    ranking_cols = ["query_id", "target_gene_id", "target_gene_symbol", "context_id", "role", "n_cells",
                    "Old_n_history", "Expanded_n_history", "history_status", "evidence_status",
                    "true_error_rmse", "Amplitude", "PublicRule_mixed", "SourceHGB_mixed"]
    frame[ranking_cols].to_parquet(out / "MIXED_QUEUE_SYSTEM_RANKING.parquet", index=False)

    rows = []
    for method, score in score_map.items():
        for context, part in [("all", frame), *[(str(c), g) for c, g in frame.groupby("context_id", sort=True)]]:
            m = task_metric(part, score[part.index.to_numpy()])
            rows.append({"method": method, "context": context, "history_scope": "mixed_all_2993",
                         "label_role": "SEEN_cached_error_only_for_scoring", **m})
    comp = pd.DataFrame(rows)
    baseline = comp[(comp.method == "Amplitude") & (comp.context == "all")].iloc[0]
    public = comp[(comp.method == "PublicRule_mixed") & (comp.context == "all")].iloc[0]
    comp["delta_u20_vs_amplitude"] = comp.u20 - baseline.u20
    comp["delta_u20_vs_public"] = comp.u20 - public.u20
    write_csv(out / "MIXED_QUEUE_SYSTEM_COMPARISON.csv", comp)
    discovery = actual_discovery(frame, score_map)
    discovery["truth_role"] = "SEEN_cached_error_only_for_scoring"
    write_csv(out / "MIXED_QUEUE_ACTUAL_ERROR_DISCOVERY.csv", discovery)

    coverage = frame.groupby(["context_id", "history_status"], as_index=False).agg(
        n_tasks=("query_id", "size"), n_genes=("target_gene_id", "nunique"),
        mean_source_history=("Expanded_n_history", "mean"))
    coverage["fraction_of_all"] = coverage.n_tasks / len(frame)
    write_csv(out / "MIXED_QUEUE_COVERAGE.csv", coverage)

    # Reuse the 5,000 paired gene-cluster draws already produced by the GWPS
    # score evaluation.  No new truth or method choice is introduced here.
    draws = np.load(DRAW_FILE, mmap_mode="r")
    if draws.shape != (5000, 3, 5, 7):
        raise RuntimeError(f"unexpected sealed bootstrap shape: {draws.shape}")
    pairs = []
    for a, b in [("Expanded_DistanceCDF_or_Magnitude", "Magnitude"),
                 ("Expanded_HGB_or_Magnitude", "Magnitude"),
                 ("Expanded_HGB_or_Magnitude", "Expanded_DistanceCDF_or_Magnitude")]:
        ia, ib = DRAW_METHODS.index(a), DRAW_METHODS.index(b)
        for ci, context in enumerate(("HCT116", "HEK293T", "macro")):
            delta = draws[:, ci, ia, 0] - draws[:, ci, ib, 0]
            pairs.append({"context": context, "method_a": a, "method_b": b,
                          "metric": "utility20", "point_delta": float(delta[0]),
                          "bootstrap_mean_delta": float(np.nanmean(delta)),
                          "ci95_lower": float(np.nanquantile(delta, .025)),
                          "ci95_upper": float(np.nanquantile(delta, .975)),
                          "bootstrap_replicates": len(delta), "source": str(DRAW_FILE)})
    write_csv(out / "MIXED_QUEUE_PAIRED_BOOTSTRAP.csv", pd.DataFrame(pairs))

    audit = {
        "status": "COMPLETE_SEEN_MIXED_QUEUE_AUDIT",
        "n_tasks": len(frame), "n_genes": int(frame.target_gene_id.nunique()),
        "history_available_tasks": int((frame.Expanded_n_history > 0).sum()),
        "no_history_tasks": int((frame.Expanded_n_history == 0).sum()),
        "public_score": "Expanded_DistanceCDF_or_Magnitude; frozen public CDF for covered history and Magnitude_SourceScoreCDF fallback for no history",
        "source_score": "Expanded_HGB_or_Magnitude; frozen source HGB with the same magnitude fallback",
        "truth_role": "SEEN cached errors only; no raw truth read by this script",
        "score_seal": {"path": str(SEAL_FILE), "sha256": sha(SEAL_FILE), "candidate_selection_on_mixed_queue": False},
        "evaluation_receipt": str(EVAL_FILE), "bootstrap_draws": str(DRAW_FILE),
        "permanent_test_truth_opened": False,
        "independent_confirmation": False,
    }
    write_json(out / "MIXED_QUEUE_AUDIT.json", audit)
    (out / "MIXED_QUEUE_CONTRACT.md").write_text(
        "# SafeConf mixed-history ranking\n\n"
        "This is a frozen SEEN audit of the 2,993-task GWPS queue. `PublicRule_mixed` is one unified ranking: "
        "expanded public distance is used where legal history exists, and the pre-sealed amplitude CDF is used "
        "when no legal history exists. `SourceHGB_mixed` uses the corresponding frozen source-error HGB score "
        "with the same fallback. No score, threshold, or metric was chosen from cached errors in this audit; "
        "the cached error file is used only to report retrospective metrics. It is not independent confirmation.\n"
    )
    print(comp[comp.context.eq("all")].to_string(index=False))
    print(discovery.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
