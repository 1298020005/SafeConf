#!/usr/bin/env python3
"""Check whether E258 history dispersion is mostly count/split-half noise.

Only train/validation targeted cells and allowed controls enter the split-half
matrix; held-out test targeted numeric tokens remain unparsed. Split-half
replicates approximate technical/sampling variation, not pure biological noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_e258_official_dev_view import TRAIN, VALIDATION, TEST
from run_e258_official_dev_predictor import baselines, load_view, sha256
from run_e258_history_validation import (
    gene_bootstrap, history_features, metrics, partial_rank_adjusted, percentile,
)


def load_halves(manifest: dict, gene_axis: list[str], prefix: Path,
                original_prefix: Path) -> tuple[np.ndarray, dict]:
    groups = manifest["groups"]
    half_genes = Path(f"{prefix}.genes.txt").read_text().splitlines()
    original_genes = Path(f"{original_prefix}.genes.txt").read_text().splitlines()
    if half_genes != original_genes or len(half_genes) != 6517:
        raise ValueError("split and original raw gene rows disagree")
    shape = (len(half_genes), 2 * len(groups))
    half = np.memmap(f"{prefix}.u32", dtype=np.uint32, mode="r", shape=shape)
    full = np.memmap(f"{original_prefix}.u32", dtype=np.uint32, mode="r",
                     shape=(len(half_genes), len(groups)))
    for start in range(0, len(half_genes), 128):
        stop = min(start + 128, len(half_genes))
        if not np.array_equal(half[start:stop, 0::2].astype(np.uint64) +
                              half[start:stop, 1::2], full[start:stop]):
            raise ValueError("split counts do not reconstruct full group sums")
    half_library = np.fromfile(f"{prefix}.library.u64", dtype=np.uint64)
    full_library = np.fromfile(f"{original_prefix}.library.u64", dtype=np.uint64)
    if len(half_library) != 2 * len(groups) or np.any(half_library == 0) or \
       not np.array_equal(half_library[0::2] + half_library[1::2], full_library):
        raise ValueError("split libraries do not reconstruct full libraries")
    index = {gene: i for i, gene in enumerate(gene_axis)}
    symbol_sum = np.zeros((len(gene_axis), shape[1]), dtype=np.uint64)
    for row, raw_gene in enumerate(half_genes):
        symbol = ":".join(raw_gene.split(":")[1:-1])
        if symbol not in index:
            raise ValueError("split row outside fixed axis")
        symbol_sum[index[symbol]] += half[row]
    expression = np.log1p(10000.0 * symbol_sum / half_library[None, :])
    control_count = manifest["control_groups"]
    tasks = groups[control_count:]
    task_halves = np.empty((2, len(tasks), len(gene_axis)), dtype=np.float32)
    for i, task in enumerate(tasks):
        matched = task["matched_control_batches"]
        weights = [item["target_cells"] for item in matched]
        for half_id in (0, 1):
            control_ids = [2 * item["control_group"] + half_id for item in matched]
            control = np.average(expression[:, control_ids], axis=1, weights=weights)
            task_halves[half_id, i] = (
                expression[:, 2 * (control_count + i) + half_id] - control)
    return task_halves, {
        "half_group_count": shape[1],
        "all_gene_group_counts_reconstruct_original": True,
        "all_library_counts_reconstruct_original": True,
        "split_matrix_sha256": sha256(Path(f"{prefix}.u32")),
        "split_library_sha256": sha256(Path(f"{prefix}.library.u64")),
    }


def run(view_path: Path, shrink_path: Path, manifest_path: Path,
        prefix: Path, original_prefix: Path, output: Path) -> dict:
    view = load_view(view_path)
    donors = view["task_donor"].astype(str)
    if set(donors) != set(TRAIN + VALIDATION) or set(donors) & set(TEST):
        raise ValueError("test targeted effects in development view")
    manifest = json.loads(manifest_path.read_text())
    groups = manifest["groups"][manifest["control_groups"]:]
    for col, key in (("task_line", "line"), ("task_target", "target"),
                     ("task_donor", "donor")):
        if view[col].astype(str).tolist() != [g[key] for g in groups]:
            raise ValueError("split group/task order mismatch")
    genes = view["gene"].astype(str)
    halves, raw_audit = load_halves(manifest, genes.tolist(), prefix,
                                   original_prefix)
    effects = view["effect"].astype(np.float32)
    targets = view["task_target"].astype(str)
    hist_full = {}
    hist_half = {}
    for target in sorted(set(targets[np.isin(donors, TRAIN)])):
        for donor in TRAIN:
            idx = (targets == target) & (donors == donor)
            if idx.any():
                hist_full[target, donor] = effects[idx].mean(axis=0)
                hist_half[target, donor] = halves[:, idx].mean(axis=1)
    train_snr = []
    for target in sorted({key[0] for key in hist_full}):
        source = [donor for donor in TRAIN if (target, donor) in hist_full]
        if len(source) < 2:
            continue
        keep = genes != target
        full_values = np.stack([hist_full[target, donor][keep]
                                for donor in source])
        half_values = np.stack([hist_half[target, donor][:, keep]
                                for donor in source])
        signal = float(np.sqrt(np.mean(full_values.mean(axis=0) ** 2)))
        noise = float(np.sqrt(np.mean((half_values[:, 0] -
                                       half_values[:, 1]) ** 2) / 4.0))
        train_snr.append(signal / max(noise, 1e-12))
    train_median_snr = float(np.median(train_snr))
    val_idx = np.flatnonzero(np.isin(donors, VALIDATION))
    val_lines = view["task_line"][val_idx].astype(str)
    val_targets = targets[val_idx]
    source_mean, _, _ = baselines(view)
    shrink = json.loads(shrink_path.read_text())
    if shrink["view_sha256"] != sha256(view_path):
        raise ValueError("upstream shrinkage view mismatch")
    alpha = float(shrink["alpha_fit_from_train_leave_one_donor_out"])
    feat, _ = history_features(view, alpha * source_mean[val_idx], val_idx)
    corrected = np.empty(len(val_idx), dtype=np.float64)
    source_noise = np.empty(len(val_idx), dtype=np.float64)
    history_signal_to_split_noise = np.empty(len(val_idx), dtype=np.float64)
    full_vs_half = np.empty(len(val_idx), dtype=np.float64)
    negative_corrections = 0
    for j, row in enumerate(val_idx):
        target = targets[row]
        source = [donor for donor in TRAIN if (target, donor) in hist_full]
        keep = genes != target
        values = np.stack([hist_full[target, donor][keep] for donor in source])
        half_values = np.stack([hist_half[target, donor][:, keep]
                                for donor in source])
        # If half estimates are independent with equal cell allocation,
        # variance(full donor effect) ~= (half0-half1)^2/4. This is approximate
        # for log-normalized pseudobulk and matched controls.
        donor_noise = np.mean((half_values[:, 0] - half_values[:, 1]) ** 2,
                              axis=1) / 4.0
        mean_noise = float(donor_noise.mean())
        observed = float(np.var(values, axis=0).mean())
        corrected_var = observed - (1.0 - 1.0 / len(source)) * mean_noise
        negative_corrections += corrected_var <= 0
        corrected[j] = np.sqrt(max(corrected_var, 0.0))
        source_noise[j] = np.sqrt(mean_noise)
        history_signal_to_split_noise[j] = np.sqrt(
            np.mean(values.mean(axis=0) ** 2)) / max(source_noise[j], 1e-12)
        full_vs_half[j] = np.sqrt(np.mean((values - half_values.mean(axis=1)) ** 2))
    m = percentile(feat["M"], val_lines)
    scores = {
        "M": m,
        "Hdisp_observed": percentile(feat["H_disp"], val_lines),
        "Hdisp_split_half_noise_corrected": percentile(corrected, val_lines),
        "source_noise_only": percentile(source_noise, val_lines),
        "history_moment_sampling_noise_only": percentile(np.sqrt(
            feat["H_gap"] ** 2 + (1.0 - 1.0 / feat["n_sources"]) *
            source_noise ** 2), val_lines),
        "history_moment_observed": percentile(np.sqrt(
            feat["H_gap"] ** 2 + feat["H_disp"] ** 2), val_lines),
        "history_moment_noise_corrected": percentile(np.sqrt(
            feat["H_gap"] ** 2 + corrected ** 2), val_lines),
    }
    result = {
        "stage": "E258_SPLIT_HALF_HISTORY_MEASUREMENT_NOISE_DEV_AUDIT",
        "test_donor_target_truth_loaded": 0,
        "view_sha256": sha256(view_path),
        "raw_reconstruction_audit": raw_audit,
        "noise_estimator": "mean_gene((half0-half1)^2)/4 per train donor; approximate",
        "n_validation_tasks": len(val_idx),
        "n_negative_corrected_dispersion_clipped_zero": negative_corrections,
        "median_source_noise_rms": float(np.median(source_noise)),
        "train_median_history_signal_to_split_noise": train_median_snr,
        "median_full_vs_halfmean_rms": float(np.median(full_vs_half)),
        "Hdisp_corrected_partial_rank_given_M_and_source_cells":
            partial_rank_adjusted(corrected,
                                  [feat["M"], feat["median_source_cells"]],
                                  feat["error_rmse"], val_lines),
        "methods": {name: metrics(feat["error_rmse"], score, val_lines)
                    for name, score in scores.items()},
        "train_only_split_noise_snr_strata": {
            label: {name: metrics(feat["error_rmse"], score, val_lines,
                                  selected)
                    for name, score in scores.items()}
            for label, selected in (("higher_snr", history_signal_to_split_noise >=
                                    train_median_snr),
                                    ("lower_snr", history_signal_to_split_noise <
                                     train_median_snr))
        },
        "corrected_moment_vs_M_gene_bootstrap_development_only":
            gene_bootstrap(feat["error_rmse"], m,
                           scores["history_moment_noise_corrected"],
                           val_targets, val_lines),
        "observed_moment_vs_sampling_noise_only_gene_bootstrap_development_only":
            gene_bootstrap(feat["error_rmse"],
                           scores["history_moment_sampling_noise_only"],
                           scores["history_moment_observed"],
                           val_targets, val_lines),
        "limits": [
            "Validation data were already inspected; this is a posthoc noise mechanism audit.",
            "Halves have 15+ target cells but may differ in batch mix; split differences include biology/noise.",
            "Only two validation donors; bootstrap conditional on them is not new-donor inference.",
        ],
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"n_validation_tasks": len(val_idx),
                      "median_source_noise_rms": result["median_source_noise_rms"],
                      "macro_utility20": {k: v["macro_utility20"]
                                          for k, v in result["methods"].items()},
                      "test_donor_target_truth_loaded": 0}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("/home/yyf/data/feng2025_candidate/raw_dev")
    parser.add_argument("--view", type=Path, default=root / "E258_RAW_DEV_VIEW.npz")
    parser.add_argument("--shrinkage", type=Path,
                        default=root.parent / "raw_dev_models/shrinkage_status.json")
    parser.add_argument("--manifest", type=Path,
                        default=root / "E258_RAW_DEV_GROUPS.json")
    parser.add_argument("--prefix", type=Path,
                        default=root / "E258_RAW_SPLIT_HALF_GROUP_SUMS")
    parser.add_argument("--original-prefix", type=Path,
                        default=root / "E258_RAW_ALLOWED_GROUP_SUMS")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.view, args.shrinkage, args.manifest,
        args.prefix, args.original_prefix, args.output)
