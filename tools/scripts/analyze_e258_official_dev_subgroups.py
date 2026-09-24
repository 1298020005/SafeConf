#!/usr/bin/env python3
"""Diagnostic only: fixed train-derived signal strata in E258 validation donors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_e258_official_dev_view import TRAIN, VALIDATION
from run_e258_official_dev_predictor import baselines, gene_mask, line_mse, load_view


def analyze(view_path: Path, model_dir: Path, mode: str = "official") -> dict:
    if mode not in {"official", "raw"}:
        raise ValueError("unregistered analysis mode")
    view = load_view(view_path)
    targets = view["task_target"].astype(str)
    donors = view["task_donor"].astype(str)
    lines = view["task_line"].astype(str)
    y = view["effect"].astype(np.float32)
    mask = gene_mask(view)
    base, similar, _ = baselines(view)
    validation = np.isin(donors, VALIDATION)
    predictions = {
        "no_change": np.zeros_like(y[validation]),
        "source_mean": base[validation],
        "source_similarity": similar[validation],
    }
    for variant in ("small", "medium"):
        prefix = "E258_OFFICIAL_DEV" if mode == "official" else "E258_RAW_DEV"
        with np.load(model_dir / f"{prefix}_{variant.upper()}_VAL_PRED.npz",
                     allow_pickle=False) as saved:
            if not np.array_equal(saved["line"].astype(str), lines[validation]) or \
               not np.array_equal(saved["target"].astype(str), targets[validation]):
                raise ValueError("model validation rows do not align")
            predictions[f"mlp_{variant}"] = saved["prediction"].astype(np.float32)
    stats = {}
    for target in sorted(set(targets[np.isin(donors, TRAIN)])):
        source = []
        for donor in TRAIN:
            selected = (donors == donor) & (targets == target)
            if selected.any():
                source.append(y[selected].mean(axis=0))
        if len(source) < 2:
            continue
        effects = np.stack(source)
        center = effects.mean(axis=0)
        signal = float(np.sqrt(np.mean(center ** 2)))
        dispersion = float(np.sqrt(np.mean(effects.var(axis=0))))
        stats[target] = {"signal": signal, "signal_to_dispersion": signal / max(dispersion, 1e-6)}
    target_val = targets[validation]
    truth_val = y[validation]
    mask_val = mask[validation]
    lines_val = lines[validation]
    results = []
    for feature in ("signal", "signal_to_dispersion"):
        values = np.asarray([entry[feature] for entry in stats.values()])
        for top_fraction in (1.0, 0.5, 0.25):
            threshold = float(np.quantile(values, 1.0 - top_fraction))
            keep_targets = {target for target, entry in stats.items()
                            if entry[feature] >= threshold}
            selected = np.isin(target_val, list(keep_targets))
            if min(int(((lines_val == line) & selected).sum()) for line in set(lines_val)) < 30:
                raise ValueError("fewer than 30 validation tasks in a line/stratum")
            row = {
                "train_only_feature": feature,
                "top_fraction": top_fraction,
                "train_threshold": threshold,
                "n_train_selected_targets": len(keep_targets),
                "n_validation_tasks": int(selected.sum()),
                "per_line_tasks": {line: int(((lines_val == line) & selected).sum())
                                   for line in sorted(set(lines_val))},
                "macro_mse": {},
                "per_line_mse": {},
            }
            for name, pred in predictions.items():
                per_line = line_mse(pred[selected], truth_val[selected], mask_val[selected],
                                    lines_val[selected])
                row["per_line_mse"][name] = per_line
                row["macro_mse"][name] = float(np.mean(list(per_line.values())))
            results.append(row)
    result = {"stage": ("E258_OFFICIAL_LFC_VALIDATION_DIAGNOSTIC_ONLY" if mode == "official"
                        else "E258_RAW_COUNT_VALIDATION_SUBGROUP_DIAGNOSTIC"),
              "selection_features_use_train_donors_only": True,
              "test_donor_effect_values_loaded": 0, "rows": results}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/E258_OFFICIAL_DEV_VIEW.npz"))
    parser.add_argument("--model-dir", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/official_dev_models"))
    parser.add_argument("--mode", choices=("official", "raw"), default="official")
    args = parser.parse_args()
    analyze(args.view, args.model_dir, args.mode)
