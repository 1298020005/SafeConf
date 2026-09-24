#!/usr/bin/env python3
"""Train-donor OOF error-label comparators for E258 development only.

Each held-out training donor is predicted using only the other three training
donors; its true effect is used only as an OOF supervised risk label. Validation
donor effects enter evaluation only. Final test targeted effects are absent.
This is a small fixed-label-budget risk estimator, not official PertEMA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from build_e258_official_dev_view import TRAIN, VALIDATION, TEST
from run_e258_official_dev_predictor import (
    baselines, gene_mask, load_view, sha256, train_only_shrinkage,
)
from run_e258_history_validation import history_features, metrics, percentile


def donor_history(view: dict) -> dict[tuple[str, str], np.ndarray]:
    donors = view["task_donor"].astype(str)
    targets = view["task_target"].astype(str)
    effects = view["effect"].astype(np.float32)
    return {
        (target, donor): effects[(targets == target) & (donors == donor)].mean(axis=0)
        for target in sorted(set(targets[np.isin(donors, TRAIN)]))
        for donor in TRAIN if ((targets == target) & (donors == donor)).any()
    }


def source_mean_for_rows(view: dict, hist: dict, rows: np.ndarray,
                         permitted: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    donors = view["task_donor"].astype(str)
    targets = view["task_target"].astype(str)
    result = np.zeros((len(rows), len(view["gene"])), dtype=np.float32)
    enough = np.zeros(len(rows), dtype=bool)
    for j, row in enumerate(rows):
        source = [hist[targets[row], donor] for donor in permitted
                  if donor != donors[row] and (targets[row], donor) in hist]
        if len(source) >= 2:
            result[j] = np.mean(source, axis=0)
            enough[j] = True
    return result, enough


def oof_features(view: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, dict]:
    donors = view["task_donor"].astype(str)
    targets = view["task_target"].astype(str)
    lines = view["task_line"].astype(str)
    effects = view["effect"].astype(np.float32)
    genes = view["gene"].astype(str)
    hist = donor_history(view)
    all_features = []
    all_errors = []
    all_lines = []
    fold_audit = {}
    for held in TRAIN:
        source = tuple(donor for donor in TRAIN if donor != held)
        source_rows = np.flatnonzero(np.isin(donors, source))
        source_base, source_eligible = source_mean_for_rows(
            view, hist, source_rows, source)
        source_rows = source_rows[source_eligible]
        source_base = source_base[source_eligible]
        alpha = train_only_shrinkage(
            source_base, effects[source_rows], gene_mask(view)[source_rows],
            lines[source_rows])
        target_rows = np.flatnonzero(donors == held)
        target_base, target_eligible = source_mean_for_rows(
            view, hist, target_rows, source)
        target_rows = target_rows[target_eligible]
        target_base = target_base[target_eligible]
        if len(target_rows) < 100:
            raise ValueError(f"too few OOF tasks for {held}")
        pred = alpha * target_base
        fold_features = np.zeros((len(target_rows), 5), dtype=np.float64)
        fold_errors = np.zeros(len(target_rows), dtype=np.float64)
        for j, row in enumerate(target_rows):
            target = targets[row]
            source_donors = [donor for donor in source
                             if (target, donor) in hist]
            vectors = np.stack([hist[target, donor] for donor in source_donors])
            keep = genes != target
            mean = vectors[:, keep].mean(axis=0)
            m = float(np.sqrt(np.mean(pred[j, keep] ** 2)))
            disp = float(np.sqrt(np.var(vectors[:, keep], axis=0).mean()))
            gap = float(np.sqrt(np.mean((pred[j, keep] - mean) ** 2)))
            # The final two features audit source support; no held-out truth.
            cells = view["task_cells"].astype(int)
            source_cells = [int(cells[(targets == target) &
                                      (donors == donor)].sum())
                            for donor in source_donors]
            fold_features[j] = (m, disp, gap, len(source_donors),
                                float(np.median(source_cells)))
            fold_errors[j] = np.sqrt(np.mean(
                (pred[j, keep] - effects[row, keep]) ** 2))
        all_features.append(fold_features)
        all_errors.append(fold_errors)
        all_lines.append(lines[target_rows])
        fold_audit[held] = {
            "n_oof_tasks": len(target_rows), "source_donors": list(source),
            "alpha_fit_without_held_out_donor": alpha,
            "n_alpha_fit_rows": len(source_rows),
        }
    return (np.concatenate(all_features), np.concatenate(all_errors),
            np.concatenate(all_lines), genes, fold_audit)


def run(view_path: Path, shrink_path: Path, output: Path) -> dict:
    view = load_view(view_path)
    donors = view["task_donor"].astype(str)
    if set(donors) != set(TRAIN + VALIDATION) or set(donors) & set(TEST):
        raise ValueError("test donor targeted effect in development view")
    oof_x, oof_y, oof_lines, _, fold_audit = oof_features(view)
    val_idx = np.flatnonzero(np.isin(donors, VALIDATION))
    val_lines = view["task_line"][val_idx].astype(str)
    source_mean, _, _ = baselines(view)
    shrink = json.loads(shrink_path.read_text())
    if shrink["view_sha256"] != sha256(view_path):
        raise ValueError("shrinkage coefficient view mismatch")
    alpha = float(shrink["alpha_fit_from_train_leave_one_donor_out"])
    feat, thresholds = history_features(view, alpha * source_mean[val_idx], val_idx)
    val_x = np.column_stack((feat["M"], feat["H_disp"], feat["H_gap"],
                             feat["n_sources"], feat["median_source_cells"]))
    # Per-line percentiles are calculated from prediction-visible quantities.
    # OOF labels are also made line-relative so a supervised fit cannot exploit
    # donor-specific absolute expression scale in the validation cohort.
    def line_rank_matrix(x: np.ndarray, line: np.ndarray) -> np.ndarray:
        return np.column_stack([percentile(x[:, j], line)
                                for j in range(x.shape[1])])
    train_rank = line_rank_matrix(oof_x, oof_lines)
    val_rank = line_rank_matrix(val_x, val_lines)
    y_rank = percentile(oof_y, oof_lines)
    error = feat["error_rmse"]
    candidates = {
        "OOF_ridge_M": (Ridge(alpha=10.0), (0,)),
        "OOF_ridge_M_support": (Ridge(alpha=10.0), (0, 3, 4)),
        "OOF_ridge_M_Hdisp": (Ridge(alpha=10.0), (0, 1)),
        "OOF_ridge_M_Hdisp_support": (Ridge(alpha=10.0), (0, 1, 3, 4)),
        "OOF_HGB_M": (HistGradientBoostingRegressor(
            max_iter=80, max_leaf_nodes=7, min_samples_leaf=50,
            l2_regularization=10.0, random_state=258), (0,)),
        "OOF_HGB_M_support": (HistGradientBoostingRegressor(
            max_iter=80, max_leaf_nodes=7, min_samples_leaf=50,
            l2_regularization=10.0, random_state=258), (0, 3, 4)),
        "OOF_HGB_M_Hdisp_support": (HistGradientBoostingRegressor(
            max_iter=80, max_leaf_nodes=7, min_samples_leaf=50,
            l2_regularization=10.0, random_state=258), (0, 1, 3, 4)),
    }
    results = {"M": metrics(error, val_rank[:, 0], val_lines)}
    score_vectors = {"M": val_rank[:, 0],
                     "history_empirical_error": percentile(
                         np.sqrt(feat["H_gap"] ** 2 + feat["H_disp"] ** 2),
                         val_lines)}
    for name, (model, columns) in candidates.items():
        model.fit(train_rank[:, columns], y_rank)
        score = model.predict(val_rank[:, columns])
        score_vectors[name] = score
        results[name] = metrics(error, score, val_lines)
    high = feat["H_signal_to_dispersion"] >= thresholds[
        "train_median_signal_to_dispersion"]
    scope_results = {
        label: {name: metrics(error, score, val_lines, selected)
                for name, score in score_vectors.items()}
        for label, selected in (("high_signal_to_dispersion", high),
                                ("low_signal_to_dispersion", ~high))
    }
    result = {
        "stage": "E258_TRAIN_OOF_LABEL_BUDGET_VALIDATION_ONLY",
        "test_donor_target_truth_loaded": 0,
        "not_official_PertEMA": True,
        "view_sha256": sha256(view_path),
        "n_oof_train_error_labels": len(oof_y),
        "n_validation_tasks": len(val_idx),
        "oof_folds": fold_audit,
        "training_features": ["M", "H_disp", "H_gap", "n_source_donors",
                              "median_source_cells"],
        "Hgap_used_in_supervised_models": False,
        "fixed_hyperparameters": "Ridge alpha=10; HGB 80/7/50/L2=10, no validation tuning",
        "results": results,
        "train_median_signal_to_dispersion": thresholds[
            "train_median_signal_to_dispersion"],
        "scope_results": scope_results,
        "limits": [
            "OOF risk model is PertEMA-like in use of training error labels, not an official reproduction.",
            "All 4 train donor OOF labels are used; hand-crafted rules also used validation labels for selection.",
            "No test donor targeted result has been opened.",
        ],
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"n_oof_train_error_labels": len(oof_y),
                      "macro_utility20": {k: v["macro_utility20"]
                                          for k, v in results.items()},
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
