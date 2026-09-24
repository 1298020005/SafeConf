#!/usr/bin/env python3
"""Train-only shrinkage of historical mean; no validation tuning or test truth.

Fit one global multiplier to leave-one-training-donor-out source predictions.
This strong simple predictor is a comparator, not an invented novel model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_e258_official_dev_view import TRAIN, VALIDATION
from run_e258_official_dev_predictor import (
    baselines, gene_mask, line_mse, load_view, sha256, train_only_shrinkage,
)


def evaluate(view_path: Path, output: Path, mode: str) -> dict:
    if mode not in {"official", "raw"}:
        raise ValueError("unregistered mode")
    view = load_view(view_path)
    y = view["effect"].astype(np.float32)
    base, similar, _ = baselines(view)
    mask = gene_mask(view)
    donor = view["task_donor"].astype(str)
    line = view["task_line"].astype(str)
    training = np.isin(donor, TRAIN)
    validation = np.isin(donor, VALIDATION)
    alpha = train_only_shrinkage(base[training], y[training], mask[training],
                                 line[training])
    methods = {
        "no_change": np.zeros_like(y[validation]),
        "source_mean": base[validation],
        "source_similarity": similar[validation],
        "train_only_shrunk_mean": alpha * base[validation],
        "train_only_shrunk_similarity": alpha * similar[validation],
    }
    per_line = {name: line_mse(pred, y[validation], mask[validation], line[validation])
                for name, pred in methods.items()}
    macro = {name: float(np.mean(list(scores.values())))
             for name, scores in per_line.items()}
    result = {
        "stage": ("E258_OFFICIAL_LFC_SHRINKAGE_ENGINEERING_ONLY" if mode == "official"
                  else "E258_RAW_COUNT_SHRINKAGE_VALIDATION"),
        "alpha_fit_from_train_leave_one_donor_out": alpha,
        "training_donors": list(TRAIN),
        "validation_donors": list(VALIDATION),
        "n_validation_tasks": int(validation.sum()),
        "macro_mse": macro,
        "per_line_mse": per_line,
        "test_target_truth_loaded": 0,
        "view_sha256": sha256(view_path),
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/E258_OFFICIAL_DEV_VIEW.npz"))
    parser.add_argument("--output", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/official_dev_models/shrinkage_status.json"))
    parser.add_argument("--mode", choices=("official", "raw"), default="official")
    args = parser.parse_args()
    evaluate(args.view, args.output, args.mode)
