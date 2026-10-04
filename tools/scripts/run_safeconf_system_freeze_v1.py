#!/usr/bin/env python3
"""Freeze one current-contract comparison after development decisions."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.research import bootstrap_u20

POOL = Path("/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1/FIXED_POOL_FEATURES.parquet")
HOLD = Path("/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1/HOLDOUT_FEATURES.parquet")
TARGET_PRED = Path("/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1/TASK_PREDICTIONS.csv.gz")
TABPFN = Path("/home/yyf/runtime_artifacts/safeconf_impl_20261004_v1/tabpfn_target50/TABPFN_TARGET50_HOLDOUT_PREDICTIONS.csv")
PERTEMA = Path("/home/yyf/runtime_artifacts/safeconf_impl_20261004_v1/pertema_current_truth_v2/PERTEMA_CURRENT_TASK_PREDICTIONS.parquet")
RELIABILITY = Path("/home/yyf/runtime_artifacts/safeconf_impl_20261004_v1/public_reliability/MCFALINE_HOLDOUT_JACKKNIFE_FEATURES.csv")
SOURCE_GATE = Path("/home/yyf/runtime_artifacts/safeconf_impl_20261004_v1/source_gate_v3/SOURCE_GATE_METRICS.csv")
SEED = 20260930


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


def macro(frame: pd.DataFrame, score: np.ndarray) -> dict:
    vals = []
    frame = frame.reset_index(drop=True); score = np.asarray(score, float)
    for target, part in frame.assign(_score=score).groupby("target", sort=True):
        y = part.true_error_rmse.to_numpy(float); s = part._score.to_numpy(float)
        ids = part.task_id.astype(str).to_numpy(); k = max(1, int(np.ceil(.2 * len(part))))
        hi = np.lexsort((ids, -s))[:k]; oracle = np.lexsort((ids, -y))[:k]
        den = y[oracle].mean() - y.mean()
        low = np.lexsort((ids, s))
        vals.append({"target": target, "n": len(part),
                     "u20": (y[hi].mean() - y.mean()) / den if len(part) >= 20 and den > 1e-12 else np.nan,
                     "aurc": float(np.mean(np.cumsum(y[low]) / np.arange(1, len(y) + 1))),
                     "true_top20_found": int(len(set(hi) & set(oracle))), "review_count": k,
                     "remaining_mean_error": float(np.delete(y, hi).mean())})
    d = pd.DataFrame(vals)
    return {"u20": float(d.u20.mean()), "aurc": float(d.aurc.mean()),
            "valid_contexts": int(d.u20.notna().sum()), "contexts": len(d),
            "true_top20_found": int(d.true_top20_found.sum()), "review_count": int(d.review_count.sum()),
            "remaining_mean_error": float(np.average(d.remaining_mean_error, weights=d.n))}


def add_candidate(frame: pd.DataFrame, scores: dict[str, np.ndarray], name: str, values, coverage: list[dict]) -> bool:
    arr = np.asarray(values, float)
    if len(arr) != len(frame):
        coverage.append({"method": name, "n_tasks": 0, "coverage": 0.0, "status": "incomplete_length"})
        return False
    finite = np.isfinite(arr)
    if not finite.all():
        coverage.append({"method": name, "n_tasks": int(finite.sum()), "coverage": float(finite.mean()),
                          "status": "not_full_coverage_fallback_public"})
        return False
    scores[name] = arr
    coverage.append({"method": name, "n_tasks": len(arr), "coverage": 1.0, "status": "full"})
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    hold = pd.read_parquet(HOLD).reset_index(drop=True)
    if len(hold) != 212 or hold.gene.nunique() != 152:
        raise RuntimeError("current holdout contract changed")
    frame = hold[["task_id", "target", "gene", "true_error_rmse", "predicted_magnitude", "simple_history_risk"]].copy()
    scores: dict[str, np.ndarray] = {}
    coverage = []
    add_candidate(frame, scores, "Amplitude", frame.predicted_magnitude, coverage)
    add_candidate(frame, scores, "PublicRule", frame.simple_history_risk, coverage)

    target = pd.read_csv(TARGET_PRED)
    for method, name in [("H1_F1", "TargetH1F1_50"), ("X0_F2", "TargetX0F2_50")]:
        q = target[(target.method == method) & (target.budget == 0.5) & (target.seed == SEED)].drop_duplicates("task_id")
        if len(q) != len(frame):
            raise RuntimeError(f"target coverage incomplete: {method}")
        add_candidate(frame, scores, name, frame.task_id.map(q.set_index("task_id").risk).to_numpy(float), coverage)

    tab = pd.read_csv(TABPFN)
    if len(tab) == len(frame):
        add_candidate(frame, scores, "TargetTabPFN50", frame.task_id.map(tab.set_index("task_id").risk).to_numpy(float), coverage)

    rel = pd.read_csv(RELIABILITY)
    if len(rel) == len(frame):
        add_candidate(frame, scores, "PublicMeanJackknife", frame.task_id.map(rel.set_index("task_id").public_mean_jackknife).to_numpy(float), coverage)

    if PERTEMA.exists():
        p = pd.read_parquet(PERTEMA)
        p = p[(p.budget == 0.5) & (p.seed == SEED) & (p.method == "PertEMA_raw")]
        for feature in sorted(p.feature_set.unique()):
            q = p[p.feature_set == feature].drop_duplicates("task_id")
            if len(q) == len(frame):
                add_candidate(frame, scores, f"PertEMA_{feature}_50", frame.task_id.map(q.set_index("task_id").risk).to_numpy(float), coverage)

    rows, paired = [], []
    public = scores["PublicRule"]
    for name, score in scores.items():
        m = macro(frame, score)
        rows.append({"method": name, "method_class": "rule" if name in {"Amplitude", "PublicRule", "PublicMeanJackknife"} else "supervised_candidate", **m,
                     "delta_u20_vs_public": m["u20"] - macro(frame, public)["u20"]})
        if name != "PublicRule":
            b = bootstrap_u20(frame, score, public, 5000, SEED)
            paired.append({"method": name, "comparison": "vs_PublicRule", **b})
    write_csv(out / "SYSTEM_COMPARISON.csv", pd.DataFrame(rows))
    write_csv(out / "SYSTEM_PAIRED_BOOTSTRAP.csv", pd.DataFrame(paired))
    write_csv(out / "SYSTEM_COVERAGE_AUDIT.csv", pd.DataFrame(coverage))
    result = frame.copy()
    for name, score in scores.items(): result[name] = score
    result.to_parquet(out / "SYSTEM_RANKING.parquet", index=False)
    decisions = []
    for row in paired:
        delta = row.get("delta_utility20", -np.inf)
        decision = bool(delta >= 0.005 and row.get("ci95_lower", -np.inf) >= -0.005)
        decisions.append({"method": row["method"], "pass_registered_gate": decision,
                          "delta_u20": delta, "ci95_lower": row.get("ci95_lower"),
                          "ci95_upper": row.get("ci95_upper")})
    selected = "PublicRule"
    passing = [d for d in decisions if d["pass_registered_gate"]]
    if passing:
        selected = max(passing, key=lambda d: d["delta_u20"])["method"]
    write_json(out / "COMPONENT_DECISION.json", {
        "status": "COMPLETE", "default_current": selected,
        "selection_rule": "PublicRule retained unless candidate point delta >= .005 and CI lower >= -.005",
        "candidates": decisions, "source_gate_metrics_path": str(SOURCE_GATE),
        "holdout_is_seen_current_contract": True, "permanent_test_truth_opened": False,
    })
    write_json(out / "RUN_STATUS.json", {"status": "COMPLETE", "rows": len(frame),
                                           "genes": int(frame.gene.nunique()), "methods": list(scores),
                                           "selected": selected})
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
