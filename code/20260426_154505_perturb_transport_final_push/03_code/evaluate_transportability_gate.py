from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return float("nan")


def _safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, score))
    except Exception:
        return float("nan")


def _task_gate_label(df: pd.DataFrame, task_keys: list[str], model_name: str, baseline_name: str = "V2") -> pd.DataFrame:
    cur = df[df["model"] == model_name].copy()
    base = df[df["model"] == baseline_name].copy()
    if cur.empty or base.empty:
        return pd.DataFrame()
    cols = task_keys + ["task_id"]
    cur = cur[cols + ["rmse", "top20_overlap", "deg_precision_top50", "program_shift_consistency", "confidence", "unsafe_flag"]].copy()
    base = base[cols + ["rmse", "top20_overlap", "deg_precision_top50", "program_shift_consistency"]].copy()
    cur = cur.rename(
        columns={
            "rmse": "rmse_cur",
            "top20_overlap": "top20_cur",
            "deg_precision_top50": "deg_cur",
            "program_shift_consistency": "program_cur",
            "confidence": "gate_score",
            "unsafe_flag": "unsafe_flag",
        }
    )
    base = base.rename(
        columns={
            "rmse": "rmse_base",
            "top20_overlap": "top20_base",
            "deg_precision_top50": "deg_base",
            "program_shift_consistency": "program_base",
        }
    )
    merged = cur.merge(base, on=cols, how="inner")
    if merged.empty:
        return merged
    merged["safe_label"] = (
        (merged["top20_cur"] > merged["top20_base"] + 0.002)
        + (merged["deg_cur"] > merged["deg_base"] + 0.002)
        + (merged["program_cur"] > merged["program_base"] + 0.005)
    ) >= 2
    merged["safe_label"] = merged["safe_label"].astype(int)
    merged["rmse_delta"] = merged["rmse_cur"] - merged["rmse_base"]
    merged["effect_gain_dims"] = (
        (merged["top20_cur"] > merged["top20_base"] + 0.002).astype(int)
        + (merged["deg_cur"] > merged["deg_base"] + 0.002).astype(int)
        + (merged["program_cur"] > merged["program_base"] + 0.005).astype(int)
    )
    return merged


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Run root containing results/SAFETY_TASK_METRICS.csv")
    p.add_argument("--baseline", default="V2")
    p.add_argument("--model", default="SafeTransPT")
    args = p.parse_args()

    root = Path(args.root)
    metrics = pd.read_csv(root / "results" / "SAFETY_TASK_METRICS.csv")
    if metrics.empty:
        raise SystemExit("No SAFETY_TASK_METRICS.csv found or file is empty")

    task_keys = ["phase", "dataset", "split_type", "heldout", "seed"]
    merged = _task_gate_label(metrics, task_keys, args.model, args.baseline)
    if merged.empty:
        raise SystemExit("Could not join current model with baseline rows")

    summary_rows = []
    for keys, sub in merged.groupby(["phase", "dataset", "split_type"], dropna=False):
        row = {"phase": keys[0], "dataset": keys[1], "split_type": keys[2], "model": args.model, "baseline": args.baseline, "n": len(sub)}
        row["gate_auc"] = _safe_auc(sub["safe_label"].to_numpy(), sub["gate_score"].to_numpy())
        row["gate_ap"] = _safe_ap(sub["safe_label"].to_numpy(), sub["gate_score"].to_numpy())
        row["gate_score_spearman_rmse_delta"] = float(pd.Series(sub["gate_score"]).corr(pd.Series(sub["rmse_delta"]), method="spearman"))
        row["unsafe_rate"] = float(sub["unsafe_flag"].mean())
        row["mean_rmse_delta"] = float(sub["rmse_delta"].mean())
        row["safe_label_rate"] = float(sub["safe_label"].mean())
        row["effect_gain_dims_mean"] = float(sub["effect_gain_dims"].mean())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(["phase", "dataset", "split_type"])
    summary.to_csv(root / "results" / "TRANSPORT_GATE_SUMMARY.csv", index=False)
    merged.to_csv(root / "results" / "TRANSPORT_GATE_TASKS.csv", index=False)

    lines = [
        "# Transportability Gate Evaluation",
        "",
        f"Model: {args.model}",
        f"Baseline: {args.baseline}",
        "",
        "This file checks whether the learned transportability score can separate safe from unsafe transport tasks.",
        "",
        "## Summary",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        "- gate_auc close to 1 means the learned gate can rank safe tasks above unsafe ones.",
        "- gate_ap is useful when safe tasks are rarer.",
        "- mean_rmse_delta below 0 means the candidate model beats the baseline on average.",
        "- effect_gain_dims_mean tells whether effect-level improvements happen in at least two dimensions.",
    ]
    (root / "results" / "TRANSPORT_GATE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
