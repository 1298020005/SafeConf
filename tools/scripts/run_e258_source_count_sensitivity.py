#!/usr/bin/env python3
"""Controlled E258 history-source count sensitivity on one fixed validation panel.

The upstream prediction and error object stay fixed using all four training
sources. Only the historical evidence available to the risk score is thinned.
Thus the same target tasks and M comparator are used for 1/2/3/4 sources.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from build_e258_official_dev_view import TRAIN, VALIDATION, TEST
from run_e258_official_dev_predictor import baselines, load_view, sha256
from run_e258_history_validation import percentile, metrics


def run(view_path: Path, shrink_path: Path, output: Path) -> dict:
    view = load_view(view_path)
    donors = view["task_donor"].astype(str)
    if set(donors) & set(TEST):
        raise ValueError("held-out target donor entered development view")
    y = view["effect"].astype(np.float32)
    targets = view["task_target"].astype(str)
    genes = view["gene"].astype(str)
    lines = view["task_line"].astype(str)
    history = {}
    for target in sorted(set(targets[np.isin(donors, TRAIN)])):
        for donor in TRAIN:
            idx = (targets == target) & (donors == donor)
            if idx.any():
                history[target, donor] = y[idx].mean(axis=0)
    val_idx = np.asarray([i for i in np.flatnonzero(np.isin(donors, VALIDATION))
                          if all((targets[i], donor) in history for donor in TRAIN)])
    if len(val_idx) < 1500:
        raise ValueError("complete-four-source common validation panel unexpectedly small")
    raw_mean, _, _ = baselines(view)
    shrink = json.loads(shrink_path.read_text())
    if shrink["view_sha256"] != sha256(view_path):
        raise ValueError("shrinkage coefficient view mismatch")
    alpha = float(shrink["alpha_fit_from_train_leave_one_donor_out"])
    prediction = alpha * raw_mean[val_idx]
    val_lines = lines[val_idx]
    val_targets = targets[val_idx]
    mask = genes[None, :] != val_targets[:, None]
    error = np.sqrt(((((prediction - y[val_idx]) ** 2) * mask).sum(axis=1)) /
                    mask.sum(axis=1))
    magnitude = np.sqrt(((prediction ** 2 * mask).sum(axis=1)) / mask.sum(axis=1))
    m = percentile(magnitude, val_lines)
    baseline = metrics(error, m, val_lines)
    rows = []
    for n_sources in (1, 2, 3, 4):
        for subset in itertools.combinations(TRAIN, n_sources):
            moment = np.empty(len(val_idx), dtype=np.float64)
            cache = {}
            for j, target in enumerate(val_targets):
                if target not in cache:
                    vectors = np.stack([history[target, donor] for donor in subset])
                    gene_keep = genes != target
                    mean = vectors[:, gene_keep].mean(axis=0)
                    disp_sq = float(np.var(vectors[:, gene_keep], axis=0).mean())
                    cache[target] = (mean, disp_sq, gene_keep)
                mean, disp_sq, gene_keep = cache[target]
                gap_sq = float(np.mean((prediction[j, gene_keep] - mean) ** 2))
                moment[j] = np.sqrt(gap_sq + disp_sq)
            risk = percentile(moment, val_lines)
            current = metrics(error, risk, val_lines)
            rows.append({
                "source_donors": list(subset), "n_sources": n_sources,
                "n_tasks": len(val_idx),
                "macro_utility20": current["macro_utility20"],
                "delta_utility20_vs_fixed_M":
                    current["macro_utility20"] - baseline["macro_utility20"],
                "macro_spearman": current["macro_spearman"],
                "delta_spearman_vs_fixed_M":
                    current["macro_spearman"] - baseline["macro_spearman"],
                "per_line_delta_utility20": {
                    line: current["by_line"][line]["utility20"] -
                          baseline["by_line"][line]["utility20"]
                    for line in baseline["by_line"]},
            })
    summary = {}
    for n in (1, 2, 3, 4):
        block = [row for row in rows if row["n_sources"] == n]
        utility = [row["delta_utility20_vs_fixed_M"] for row in block]
        summary[str(n)] = {
            "source_combinations": len(block), "n_fixed_tasks": len(val_idx),
            "delta_utility20_mean": float(np.mean(utility)),
            "delta_utility20_min": float(np.min(utility)),
            "delta_utility20_max": float(np.max(utility)),
            "positive_source_combinations": int(np.count_nonzero(np.asarray(utility) > 0)),
        }
    result = {
        "stage": "E258_RAW_VALIDATION_TRAIN_HISTORY_SOURCE_COUNT_SENSITIVITY",
        "test_donor_target_truth_loaded": 0,
        "fixed_prediction": "train-only shrunk mean using all four train donors",
        "variable": "only the number/identity of historical donor effects seen by the risk score",
        "view_sha256": sha256(view_path), "n_common_validation_tasks": len(val_idx),
        "baseline_fixed_M": baseline, "summary": summary, "source_combinations": rows,
        "limits": ["Only four available training donors; identity and sample size remain limited.",
                   "Validation results are development evidence, not independent test confirmation."],
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": result["stage"], "summary": summary,
                      "test_donor_target_truth_loaded": 0},
                     ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("/home/yyf/data/feng2025_candidate")
    parser.add_argument("--view", type=Path,
                        default=root / "raw_dev/E258_RAW_DEV_VIEW.npz")
    parser.add_argument("--shrinkage", type=Path,
                        default=root / "raw_dev_models/shrinkage_status.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.view, args.shrinkage, args.output)
