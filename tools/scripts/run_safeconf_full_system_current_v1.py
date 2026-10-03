#!/usr/bin/env python3
"""Build the first complete current-contract SafeConf ranking table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_HOLDOUT = Path("/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1/HOLDOUT_FEATURES.parquet")
DEFAULT_SOURCE = Path(
    "/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/"
    "data_model_feedback_20261003_v1/unified_fusion_v4/TARGET_SCORES.parquet"
)
DEFAULT_TARGET = Path(
    "/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/"
    "data_model_feedback_20261003_v1/target_ridge_fusion_v8/HOLDOUT_SCORES_B050.parquet"
)
DEFAULT_OUT = Path(
    "/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/"
    "data_model_feedback_20261003_v1/full_system_current_v1"
)


def metric(frame, method, score):
    y = frame.true_error_rmse.to_numpy(float)
    ids = frame.task_id.astype(str).to_numpy()
    s = np.asarray(score, float)
    k = int(np.ceil(.2 * len(y)))
    top = np.lexsort((ids, -s))[:k]
    oracle = np.lexsort((ids, -y))[:k]
    denom = y[oracle].mean() - y.mean()
    return {
        "method": method,
        "n_tasks": len(y),
        "u20": float((y[top].mean() - y.mean()) / denom) if denom > 1e-12 else np.nan,
        "selected_top20": k,
        "true_top20_found": int(len(set(top).intersection(set(oracle)))),
        "remaining_mean_error": float(np.delete(y, top).mean()),
        "mean_error": float(y.mean()),
        "score_min": float(np.min(s)),
        "score_max": float(np.max(s)),
    }


def run(holdout: Path, source_scores: Path, target_scores: Path, out: Path):
    out.mkdir(parents=False, exist_ok=False)
    h = pd.read_parquet(holdout)
    s = pd.read_parquet(source_scores)
    t = pd.read_parquet(target_scores)
    # All score joins are by biological task identity.  Target truth is taken
    # from the current holdout contract and checked against both score assets.
    frame = h[["task_id", "gene", "target", "true_error_rmse", "predicted_magnitude",
               "simple_history_risk", "prediction_prior_rmse"]].copy()
    s = s[s.task_id.isin(frame.task_id)].drop_duplicates("task_id")
    t = t[t.task_id.isin(frame.task_id)].drop_duplicates("task_id")
    if len(s) != len(frame) or len(t) != len(frame):
        raise RuntimeError("source/target score coverage does not cover the same holdout tasks")
    for label, part in [("source", s), ("target", t)]:
        m = frame[["task_id", "true_error_rmse"]].merge(
            part[["task_id", "true_error_rmse"]], on="task_id", suffixes=("_holdout", f"_{label}")
        )
        delta = np.abs(m[f"true_error_rmse_{label}"].to_numpy(float) - m.true_error_rmse_holdout.to_numpy(float))
        if len(delta) == 0 or float(delta.max()) > 1e-12:
            raise RuntimeError(f"{label} score asset uses a different truth contract: max delta {delta.max()}")
    frame["Amplitude"] = frame.predicted_magnitude
    frame["PublicRule"] = frame.simple_history_risk
    frame["SourceSharedRisk"] = frame.task_id.map(s.set_index("task_id").SharedRisk_source_supervised)
    frame["SourceRidge"] = frame.task_id.map(s.set_index("task_id").Public_Ridge_source_supervised)
    frame["TargetRidge50"] = frame.task_id.map(t.set_index("task_id").TargetRidge_fusion)
    frame["TargetH1F1_50"] = frame.task_id.map(t.set_index("task_id").H1_F1)
    frame["SafeConf_no_target_feedback"] = frame["SourceRidge"].fillna(frame["PublicRule"])
    frame["SafeConf_target_feedback_50"] = frame["TargetRidge50"].fillna(frame["SafeConf_no_target_feedback"])
    score_columns = ["Amplitude", "PublicRule", "SourceSharedRisk", "SourceRidge", "TargetRidge50",
                     "TargetH1F1_50", "SafeConf_no_target_feedback", "SafeConf_target_feedback_50"]
    if frame[score_columns].isna().any().any():
        raise RuntimeError("full system score table contains missing scores")
    rows = [metric(frame, c, frame[c].to_numpy(float)) for c in score_columns]
    frame.to_parquet(out / "FULL_SYSTEM_TASK_SCORES.parquet", index=False)
    pd.DataFrame(rows).to_csv(out / "SYSTEM_METRICS.csv", index=False)
    pd.DataFrame({
        "task_id": frame.task_id,
        "gene": frame.gene,
        "true_error_rmse": frame.true_error_rmse,
        **{c: frame[c] for c in score_columns},
    }).to_parquet(out / "PER_QUERY_RESULTS.parquet", index=False)
    pd.DataFrame([
        {"method": "SafeConf_no_target_feedback", "config": "Source-supervised Ridge fallback PublicRule", "label_source": "E201 source errors only", "target_errors_used_for_fit": 0},
        {"method": "SafeConf_target_feedback_50", "config": "Target-supervised Ridge at 50% feedback; fallback SourceRidge/PublicRule", "label_source": "McFaline DEV errors", "target_errors_used_for_fit": 542},
        {"method": "PublicRule", "config": "simple current Public rule", "label_source": "none", "target_errors_used_for_fit": 0},
    ]).to_csv(out / "INFORMATION_BUDGET_LEDGER.csv", index=False)
    (out / "SYSTEM_CONTRACT.md").write_text(
        "# Current-contract complete system\n\n"
        "All rows are the same 212 McFaline holdout tasks and use the holdout truth contract. "
        "Source and Target score assets are joined by task_id and rejected if their stored truth differs. "
        "`SafeConf_no_target_feedback` uses Source-supervised Ridge with Public fallback. "
        "`SafeConf_target_feedback_50` uses the separately trained Target Ridge candidate at 50% feedback. "
        "The original PublicRule remains an independent no-error-label baseline.\n"
    )
    (out / "RUN_STATUS.json").write_text(json.dumps({
        "status": "COMPLETE", "n_tasks": len(frame), "n_genes": int(frame.gene.nunique()),
        "same_truth_contract": True, "score_columns": score_columns,
        "holdout": str(holdout), "source_scores": str(source_scores), "target_scores": str(target_scores),
    }, indent=2))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--source-scores", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-scores", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(args.holdout, args.source_scores, args.target_scores, args.out).to_string(index=False))


if __name__ == "__main__":
    main()
