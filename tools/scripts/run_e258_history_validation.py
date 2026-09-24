#!/usr/bin/env python3
"""Train-only history features; validation-only risk assessment for E258.

The input view contains no test-donor targeted effects. This file must never
open the original raw count matrix or the authors' all-donor LFC table.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from build_e258_official_dev_view import TRAIN, VALIDATION, TEST
from run_e258_official_dev_predictor import baselines, load_view, sha256


def percentile(values: np.ndarray, lines: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    for line in sorted(set(lines)):
        idx = lines == line
        result[idx] = (rankdata(values[idx], method="average") - 0.5) / idx.sum()
    return result


def utility20(error: np.ndarray, score: np.ndarray) -> float:
    if len(error) < 20 or error.sum() <= 0:
        raise ValueError("insufficient positive-error tasks")
    top = np.argsort(-score, kind="stable")[:math.ceil(0.2 * len(error))]
    return float(error[top].sum() / error.sum())


def metrics(error: np.ndarray, score: np.ndarray, lines: np.ndarray,
            subset: np.ndarray | None = None) -> dict:
    if subset is None:
        subset = np.ones(len(error), dtype=bool)
    by_line = {}
    for line in sorted(set(lines)):
        idx = subset & (lines == line)
        if idx.sum() < 20:
            by_line[line] = {"n": int(idx.sum()), "utility20": None,
                             "spearman": None}
            continue
        correlation = float(spearmanr(score[idx], error[idx]).statistic)
        by_line[line] = {
            "n": int(idx.sum()), "utility20": utility20(error[idx], score[idx]),
            "spearman": correlation if np.isfinite(correlation) else None,
        }
    valid = [entry for entry in by_line.values()
             if entry["utility20"] is not None and entry["spearman"] is not None]
    return {
        "n": int(subset.sum()), "by_line": by_line,
        "macro_utility20": (float(np.mean([x["utility20"] for x in valid]))
                            if len(valid) == len(by_line) else None),
        "macro_spearman": (float(np.mean([x["spearman"] for x in valid]))
                           if len(valid) == len(by_line) else None),
    }


def cosine_centered(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - float(a.mean())
    b = b.astype(np.float64) - float(b.mean())
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def partial_rank(h: np.ndarray, m: np.ndarray, error: np.ndarray,
                 lines: np.ndarray) -> dict:
    return partial_rank_adjusted(h, [m], error, lines)


def partial_rank_adjusted(h: np.ndarray, covariates: list[np.ndarray],
                          error: np.ndarray, lines: np.ndarray) -> dict:
    by_line = {}
    for line in sorted(set(lines)):
        idx = lines == line
        hh = rankdata(h[idx]).astype(float)
        yy = rankdata(error[idx]).astype(float)
        design = np.column_stack((np.ones(idx.sum()),
                                  *(rankdata(x[idx]).astype(float)
                                    for x in covariates)))
        rh = hh - design @ np.linalg.lstsq(design, hh, rcond=None)[0]
        ry = yy - design @ np.linalg.lstsq(design, yy, rcond=None)[0]
        value = float(np.corrcoef(rh, ry)[0, 1])
        by_line[line] = {"n": int(idx.sum()), "partial_rank_corr": value}
    return {"by_line": by_line,
            "macro": float(np.mean([x["partial_rank_corr"] for x in by_line.values()]))}


def history_features(view: dict, prediction: np.ndarray,
                     val_idx: np.ndarray) -> tuple[dict, dict]:
    effects = view["effect"].astype(np.float32)
    donors = view["task_donor"].astype(str)
    targets = view["task_target"].astype(str)
    task_lines = view["task_line"].astype(str)
    genes = view["gene"].astype(str)
    task_cells = view["task_cells"].astype(int)
    control = dict(zip(view["line"].astype(str), view["control"].astype(np.float32)))
    train_controls = {
        donor: np.mean([control[line] for line in control
                        if line.split("_")[0] == donor], axis=0)
        for donor in TRAIN
    }
    hist = {}
    support = {}
    for target in sorted(set(targets[np.isin(donors, TRAIN)])):
        for donor in TRAIN:
            idx = (targets == target) & (donors == donor)
            if idx.any():
                hist[target, donor] = effects[idx].mean(axis=0)
                support[target, donor] = int(task_cells[idx].sum())
    train_disp = []
    train_signal = []
    train_signal_to_disp = []
    for target in sorted(set(targets[np.isin(donors, TRAIN)])):
        values = [hist[target, donor] for donor in TRAIN if (target, donor) in hist]
        if len(values) >= 2:
            keep = genes != target
            vectors = np.stack(values)[:, keep]
            disp = float(np.sqrt(np.var(vectors, axis=0).mean()))
            signal = float(np.sqrt(np.mean(vectors.mean(axis=0) ** 2)))
            train_disp.append(disp)
            train_signal.append(signal)
            train_signal_to_disp.append(signal / max(disp, 1e-12))
    train_sim = []
    for line, expression in control.items():
        own = line.split("_")[0]
        if own in TRAIN:
            train_sim.append(max(cosine_centered(expression, train_controls[donor])
                                 for donor in TRAIN if donor != own))
    thresholds = {
        "train_median_history_dispersion": float(np.median(train_disp)),
        "train_median_history_signal": float(np.median(train_signal)),
        "train_median_signal_to_dispersion": float(np.median(train_signal_to_disp)),
        "train_median_background_similarity": float(np.median(train_sim)),
        "support_cell_cutoff": 60, "support_donor_cutoff": 3,
    }
    out = {name: np.empty(len(val_idx), dtype=np.float64)
           for name in ("M", "H_disp", "H_signal", "H_signal_to_dispersion",
                        "H_gap", "error_rmse", "n_sources",
                        "median_source_cells", "q_support", "background_similarity")}
    for j, row in enumerate(val_idx):
        target = targets[row]
        source = [donor for donor in TRAIN if (target, donor) in hist]
        if len(source) < 2:
            raise ValueError("validation task without two independent train donors")
        vectors = np.stack([hist[target, donor] for donor in source])
        keep = genes != target
        pred = prediction[j, keep]
        mean = vectors[:, keep].mean(axis=0)
        out["M"][j] = np.sqrt(np.mean(pred ** 2))
        out["H_disp"][j] = np.sqrt(np.var(vectors[:, keep], axis=0).mean())
        out["H_signal"][j] = np.sqrt(np.mean(mean ** 2))
        out["H_signal_to_dispersion"][j] = out["H_signal"][j] / max(
            out["H_disp"][j], 1e-12)
        out["H_gap"][j] = np.sqrt(np.mean((pred - mean) ** 2))
        out["error_rmse"][j] = np.sqrt(np.mean((pred - effects[row, keep]) ** 2))
        n_cells = np.median([support[target, donor] for donor in source])
        out["n_sources"][j] = len(source)
        out["median_source_cells"][j] = n_cells
        out["q_support"][j] = min(1.0, len(source) / 4) * min(1.0, n_cells / 60)
        out["background_similarity"][j] = max(
            cosine_centered(control[task_lines[row]], train_controls[donor])
            for donor in source)
    return out, thresholds


def gene_bootstrap(error: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
                   targets: np.ndarray, lines: np.ndarray, draws: int = 2000) -> dict:
    genes = sorted(set(targets))
    rows = {gene: np.flatnonzero(targets == gene) for gene in genes}
    rng = np.random.default_rng(258)
    results = []
    for _ in range(draws):
        sampled = rng.choice(genes, len(genes), replace=True)
        idx = np.concatenate([rows[gene] for gene in sampled])
        deltas = []
        for line in sorted(set(lines)):
            block = idx[lines[idx] == line]
            if len(block) >= 20:
                deltas.append(utility20(error[block], candidate[block]) -
                              utility20(error[block], baseline[block]))
        if len(deltas) == len(set(lines)):
            results.append(float(np.mean(deltas)))
    if len(results) < 0.95 * draws:
        raise ValueError("too many empty bootstrap line samples")
    return {"scope": "gene-cluster CI conditional on the same two validation donors",
            "draws": len(results),
            "ci95": np.quantile(results, [0.025, 0.975]).tolist(),
            "fraction_positive": float(np.mean(np.asarray(results) > 0))}


def stratum_interaction_bootstrap(error: np.ndarray, baseline: np.ndarray,
                                  candidate: np.ndarray, targets: np.ndarray,
                                  lines: np.ndarray, high: np.ndarray,
                                  draws: int = 2000) -> dict:
    """Same-gene resamples for the high-minus-low utility improvement."""
    genes = sorted(set(targets))
    rows = {gene: np.flatnonzero(targets == gene) for gene in genes}
    rng = np.random.default_rng(2581)
    differences = []
    for _ in range(draws):
        sampled = rng.choice(genes, len(genes), replace=True)
        idx = np.concatenate([rows[gene] for gene in sampled])
        deltas = []
        for selected in (high, ~high):
            per_line = []
            for line in sorted(set(lines)):
                block = idx[(lines[idx] == line) & selected[idx]]
                if len(block) >= 20:
                    per_line.append(utility20(error[block], candidate[block]) -
                                    utility20(error[block], baseline[block]))
            if len(per_line) != len(set(lines)):
                break
            deltas.append(float(np.mean(per_line)))
        if len(deltas) == 2:
            differences.append(deltas[0] - deltas[1])
    if len(differences) < 0.95 * draws:
        raise ValueError("too many empty paired stratum bootstrap samples")
    return {
        "scope": "gene-cluster high-minus-low CI conditional on two validation donors",
        "draws": len(differences),
        "ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "fraction_positive": float(np.mean(np.asarray(differences) > 0)),
    }


def analyze(view_path: Path, shrinkage_path: Path, mlp_path: Path,
            output: Path, upstream: str) -> dict:
    view = load_view(view_path)
    donors = view["task_donor"].astype(str)
    if set(donors) != set(TRAIN + VALIDATION) or set(donors) & set(TEST):
        raise ValueError("held-out test donor targeted effect in development view")
    val_idx = np.flatnonzero(np.isin(donors, VALIDATION))
    lines = view["task_line"][val_idx].astype(str)
    targets = view["task_target"][val_idx].astype(str)
    source_mean, _, _ = baselines(view)
    shrink = json.loads(shrinkage_path.read_text())
    if shrink["view_sha256"] != sha256(view_path):
        raise ValueError("shrinkage coefficient trained on another view")
    alpha = float(shrink["alpha_fit_from_train_leave_one_donor_out"])
    if upstream == "shrunk_history_mean":
        prediction = alpha * source_mean[val_idx]
        redundant_gap = True
    elif upstream == "shrunk_mlp_small":
        with np.load(mlp_path, allow_pickle=False) as file:
            if not np.array_equal(file["line"].astype(str), lines) or \
               not np.array_equal(file["target"].astype(str), targets):
                raise ValueError("MLP prediction task alignment mismatch")
            prediction = file["prediction"].astype(np.float32)
        redundant_gap = False
    else:
        raise ValueError("unknown upstream")
    feat, thresholds = history_features(view, prediction, val_idx)
    error = feat["error_rmse"]
    m = percentile(feat["M"], lines)
    h = percentile(feat["H_disp"], lines)
    gap = percentile(feat["H_gap"], lines)
    # Exact finite-sample identity across available historical donors:
    # mean_d ||prediction - historical_effect_d||^2 = H_gap^2 + H_disp^2.
    # Its transfer to a new donor is a testable assumption, not a theorem.
    historical_error = np.sqrt(feat["H_gap"] ** 2 + feat["H_disp"] ** 2)
    scores = {"M": m, "H_disp_only": h,
              "history_empirical_error": percentile(historical_error, lines)}
    for weight in (0.125, 0.25, 0.5):
        scores[f"M_plus_Hdisp_{weight}"] = m + weight * np.maximum(h - m, 0)
        scores[f"M_plus_qHdisp_{weight}"] = m + weight * feat["q_support"] * \
            np.maximum(h - m, 0)
        if not redundant_gap:
            q_gap = feat["q_support"] * (1 - h)
            scores[f"M_plus_qHgap_{weight}"] = m + weight * q_gap * \
                np.maximum(gap - m, 0)
    full = {name: metrics(error, score, lines) for name, score in scores.items()}
    scopes = {
        "full": np.ones(len(error), dtype=bool),
        "sources_2": feat["n_sources"] == 2,
        "sources_3plus": feat["n_sources"] >= 3,
        "source_cells_lt60": feat["median_source_cells"] < 60,
        "source_cells_ge60": feat["median_source_cells"] >= 60,
        "history_consistent": feat["H_disp"] <= thresholds["train_median_history_dispersion"],
        "history_heterogeneous": feat["H_disp"] > thresholds["train_median_history_dispersion"],
        "history_signal_weak": feat["H_signal"] < thresholds["train_median_history_signal"],
        "history_signal_strong": feat["H_signal"] >= thresholds["train_median_history_signal"],
        "history_signal_to_dispersion_low": feat["H_signal_to_dispersion"] <
            thresholds["train_median_signal_to_dispersion"],
        "history_signal_to_dispersion_high": feat["H_signal_to_dispersion"] >=
            thresholds["train_median_signal_to_dispersion"],
        "background_similarity_low": feat["background_similarity"] <
                                     thresholds["train_median_background_similarity"],
        "background_similarity_high": feat["background_similarity"] >=
                                      thresholds["train_median_background_similarity"],
        "trusted_history_composite": (feat["n_sources"] >= 3) &
             (feat["median_source_cells"] >= 60) &
             (feat["H_disp"] <= thresholds["train_median_history_dispersion"]) &
             (feat["background_similarity"] >=
              thresholds["train_median_background_similarity"]),
    }
    strata = {}
    for name, selected in scopes.items():
        base = metrics(error, m, lines, selected)
        methods = {}
        for method, score in scores.items():
            if method == "M":
                continue
            current = metrics(error, score, lines, selected)
            methods[method] = {
                "macro_delta_utility20": (current["macro_utility20"] -
                    base["macro_utility20"] if current["macro_utility20"] is not None
                    and base["macro_utility20"] is not None else None),
                "macro_delta_spearman": (current["macro_spearman"] -
                    base["macro_spearman"] if current["macro_spearman"] is not None
                    and base["macro_spearman"] is not None else None),
                "by_line": {line: {
                    "n": current["by_line"][line]["n"],
                    "delta_utility20": (
                        current["by_line"][line]["utility20"] -
                        base["by_line"][line]["utility20"]
                        if current["by_line"][line]["utility20"] is not None and
                        base["by_line"][line]["utility20"] is not None else None),
                    "delta_spearman": (
                        current["by_line"][line]["spearman"] -
                        base["by_line"][line]["spearman"]
                        if current["by_line"][line]["spearman"] is not None and
                        base["by_line"][line]["spearman"] is not None else None),
                } for line in base["by_line"]},
            }
        strata[name] = {"n": int(selected.sum()), "M": base, "methods": methods}
    main_rules = [name for name in scores if name.startswith("M_plus_qHdisp_")]
    best = max(main_rules, key=lambda name: (full[name]["macro_utility20"],
                                             -float(name.rsplit("_", 1)[1])))
    result = {
        "stage": "E258_RAW_TRAIN_VALIDATION_HISTORY_RISK_DEVELOPMENT_ONLY",
        "upstream": upstream, "upstream_gate": (
            "simple train-only predictor" if redundant_gap else
            "MLP did not pass preregistered 2% strong-baseline gate"),
        "test_donor_target_truth_loaded": 0, "view_sha256": sha256(view_path),
        "n_tasks": len(error), "validation_donors": list(VALIDATION),
        "n_target_clusters": len(set(targets)), "train_only_thresholds": thresholds,
        "n_source_counts": {str(n): int((feat["n_sources"] == n).sum())
                            for n in sorted(set(feat["n_sources"]))},
        "M_Hgap_spearman": float(spearmanr(feat["M"], feat["H_gap"]).statistic),
        "Hgap_redundant_by_design": redundant_gap,
        "Hdisp_partial_rank_given_M": partial_rank(feat["H_disp"], feat["M"],
                                                    error, lines),
        "Hdisp_partial_rank_given_M_and_median_source_cells":
            partial_rank_adjusted(
                feat["H_disp"], [feat["M"], feat["median_source_cells"]],
                error, lines),
        "Hgap_partial_rank_given_M": (None if redundant_gap else
                                      partial_rank(feat["H_gap"], feat["M"],
                                                   error, lines)),
        "full_methods": full, "fixed_train_only_strata": strata,
        "best_quality_Hdisp_development_rule": best,
        "best_rule_gene_bootstrap_development_only":
            gene_bootstrap(error, m, scores[best], targets, lines),
        "history_empirical_error_gene_bootstrap_development_only":
            gene_bootstrap(error, m, scores["history_empirical_error"], targets, lines),
        "history_empirical_error_vs_Hdisp_bootstrap_development_only":
            gene_bootstrap(error, h, scores["history_empirical_error"], targets, lines),
        "history_signal_to_dispersion_interaction_bootstrap_development_only":
            stratum_interaction_bootstrap(
                error, m, scores["history_empirical_error"], targets, lines,
                scopes["history_signal_to_dispersion_high"]),
        "limits": ["Two validation donors are development evidence, not external confirmation.",
                   "Subgroup/rule selection may overfit validation; test truth remains sealed."],
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "stage": result["stage"], "upstream": upstream,
        "Hdisp_partial_rank_given_M": result["Hdisp_partial_rank_given_M"],
        "full_macro_utility20": {key: value["macro_utility20"]
                                 for key, value in full.items()},
        "best_quality_Hdisp_development_rule": best,
        "test_donor_target_truth_loaded": 0,
    }, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("/home/yyf/data/feng2025_candidate")
    parser.add_argument("--view", type=Path,
                        default=root / "raw_dev/E258_RAW_DEV_VIEW.npz")
    parser.add_argument("--shrinkage", type=Path,
                        default=root / "raw_dev_models/shrinkage_status.json")
    parser.add_argument("--mlp-prediction", type=Path, default=root /
                        "raw_dev_models/E258_RAW_DEV_SHRUNK_SMALL_VAL_PRED.npz")
    parser.add_argument("--upstream", choices=("shrunk_history_mean",
                        "shrunk_mlp_small"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.view, args.shrinkage, args.mlp_prediction, args.output, args.upstream)
