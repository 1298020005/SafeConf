#!/usr/bin/env python3
"""Retrospective transfer of frozen E234 M+H to E97 Frangieh E106/E107.

The seal stage loads expression only for E97 training pairs and source controls.
Test errors are read exclusively by the evaluate stage.  Previous E106/E107
truth was already public, so the resulting comparison is retrospective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT if (ROOT / "docs/实验结果/E97_frangieh_gene_cartesian_contract_20260713/manifests/E97_TASK_MANIFEST.csv").exists() else Path("/home/yyf/proj")
MANIFEST = INPUT_ROOT / "docs/实验结果/E97_frangieh_gene_cartesian_contract_20260713/manifests/E97_TASK_MANIFEST.csv"
SOURCE = Path("/home/yyf/data/scgpt_formal_frangieh_fixed_panel_20260711/frangieh_e72_fixed512/perturb_processed.h5ad")
METRICS = {
    "scGPT": INPUT_ROOT / "docs/实验结果/E106_frangieh_context_scgpt_20260713/E106_ALL_TEST_TASK_METRICS.csv",
    "GEARS": INPUT_ROOT / "docs/实验结果/E107_frangieh_context_gears_20260713/E107_ALL_TEST_TASK_METRICS.csv",
}
OUT = ROOT / "docs/实验结果/E250_frangieh_history_transfer_20260924"
PRIMARY_SETTING = "context_unseen_row"
FEATURE_COLUMNS = ["fold_id", "task_id", "split", "context", "perturbation", "setting", "predicted_effect_l2"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def dense_mean(matrix) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0), dtype=np.float64).reshape(-1)


def history_features() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)
    adata = sc.read_h5ad(SOURCE, backed="r")
    try:
        contexts = adata.obs["cell_type"].astype(str).to_numpy()
        conditions = adata.obs["condition"].astype(str).to_numpy()
        genes = adata.n_vars
        rows = []
        for fold, part in manifest.groupby("fold_id", sort=True):
            train = part.loc[part.split.eq("train"), ["context", "perturbation"]].drop_duplicates()
            source_contexts = sorted(train.context.astype(str).unique())
            if len(source_contexts) != 2:
                raise ValueError(f"{fold}: expected two source contexts")
            allowed = set(map(tuple, train.astype(str).itertuples(index=False, name=None)))
            keep = np.array([
                (c in source_contexts and p == "ctrl") or (c, p) in allowed
                for c, p in zip(contexts, conditions)
            ], dtype=bool)
            selected = adata[keep].to_memory()
            selected_contexts = selected.obs["cell_type"].astype(str).to_numpy()
            selected_conditions = selected.obs["condition"].astype(str).to_numpy()
            controls = {}
            for context in source_contexts:
                mask = (selected_contexts == context) & (selected_conditions == "ctrl")
                if not mask.any():
                    raise ValueError(f"{fold}: no source controls for {context}")
                controls[context] = dense_mean(selected.X[mask])
            effects: dict[str, dict[str, np.ndarray]] = {}
            for context, condition in sorted(allowed):
                mask = (selected_contexts == context) & (selected_conditions == condition)
                if not mask.any():
                    raise ValueError(f"{fold}: missing train pair {context}::{condition}")
                effects.setdefault(condition, {})[context] = dense_mean(selected.X[mask]) - controls[context]
            for condition, by_context in effects.items():
                if set(by_context) != set(source_contexts):
                    continue
                values = np.stack([by_context[c] for c in source_contexts])
                dispersion = float(np.sqrt(np.mean((values - values.mean(axis=0)) ** 2)))
                if not np.isfinite(dispersion):
                    raise ValueError(f"{fold} {condition}: nonfinite H")
                rows.append({"fold_id": fold, "perturbation": condition, "h": dispersion,
                             "n_source_contexts": 2, "gene_axis_size": genes,
                             "n_expression_rows_read": int(keep.sum())})
        return pd.DataFrame(rows).sort_values(["fold_id", "perturbation"]).reset_index(drop=True)
    finally:
        adata.file.close()


def score() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seal_path = OUT / "FEATURE_SEAL.json"
    if seal_path.exists():
        raise RuntimeError("E250 score seal exists; will not overwrite")
    hist = history_features()
    hist.to_csv(OUT / "TRAIN_ONLY_HISTORY.csv", index=False)
    pieces = []
    counts = {}
    for predictor, path in METRICS.items():
        frame = pd.read_csv(path, usecols=FEATURE_COLUMNS)
        frame = frame.loc[frame.split.eq("test") & frame.setting.eq(PRIMARY_SETTING)].copy()
        frame = frame.merge(hist[["fold_id", "perturbation", "h"]],
                            on=["fold_id", "perturbation"], how="left", validate="many_to_one")
        counts[predictor] = {"primary_candidates": len(frame), "history_supported": int(frame.h.notna().sum())}
        frame = frame.loc[frame.h.notna()].copy()
        frame["predictor"] = predictor
        for fold, index in frame.groupby("fold_id", sort=True).indices.items():
            part = frame.iloc[index]
            if len(part) < 30:
                raise ValueError(f"{predictor}/{fold}: fewer than 30 supported tasks")
            m = rankdata(part.predicted_effect_l2.to_numpy(float)) / len(part)
            h = rankdata(part.h.to_numpy(float)) / len(part)
            frame.loc[frame.index[index], "M"] = m
            frame.loc[frame.index[index], "H"] = h
            frame.loc[frame.index[index], "M_plus_H"] = 0.8 * m + 0.2 * h
        pieces.append(frame)
    scores = pd.concat(pieces, ignore_index=True)
    if scores[["M", "H", "M_plus_H"]].isna().any().any():
        raise ValueError("missing score")
    score_path = OUT / "SCORES.csv"
    scores.to_csv(score_path, index=False)
    seal = {"status": "SEALED_RETROSPECTIVE", "created_at": datetime.now().astimezone().isoformat(),
            "formula": "M_plus_H = 0.8*rank(M) + 0.2*rank(H)", "scope": PRIMARY_SETTING,
            "n_score_rows": len(scores), "counts": counts, "source_expression_policy": "train pairs and source controls only",
            "known_limitation": "E106/E107 test truth was public before this analysis; not a blind confirmation",
            "input_sha256": {str(p): sha(p) for p in (MANIFEST, SOURCE, *METRICS.values())},
            "score_sha256": sha(score_path), "history_sha256": sha(OUT / "TRAIN_ONLY_HISTORY.csv"),
            "script_sha256": sha(Path(__file__))}
    seal_path.write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(seal, ensure_ascii=False, indent=2), flush=True)


def top_weights(score_values: np.ndarray, fraction: float) -> np.ndarray:
    n = len(score_values)
    k = max(1, int(np.ceil(fraction * n)))
    threshold = np.partition(score_values, n-k)[n-k]
    above = score_values > threshold
    tied = score_values == threshold
    weights = above.astype(float)
    weights[tied] = (k - above.sum()) / tied.sum()
    return weights


def utility(score_values: np.ndarray, errors: np.ndarray) -> float:
    pick = top_weights(score_values, 0.2)
    oracle = top_weights(errors, 0.2)
    average = float(errors.mean())
    denominator = float(np.dot(oracle, errors) / oracle.sum() - average)
    return float((np.dot(pick, errors) / pick.sum() - average) / denominator) if denominator > 1e-15 else float("nan")


def evaluate() -> None:
    seal = json.loads((OUT / "FEATURE_SEAL.json").read_text())
    score_path = OUT / "SCORES.csv"
    if sha(score_path) != seal["score_sha256"]:
        raise ValueError("sealed scores changed")
    scores = pd.read_csv(score_path)
    pieces = []
    for predictor, path in METRICS.items():
        truth = pd.read_csv(path, usecols=["fold_id", "task_id", "true_error_rmse", "true_effect_l2_diagnostic"])
        truth["predictor"] = predictor
        pieces.append(truth)
    truth = pd.concat(pieces, ignore_index=True)
    frame = scores.merge(truth, on=["predictor", "fold_id", "task_id"], validate="one_to_one")
    if len(frame) != len(scores):
        raise ValueError("missing test errors")
    rows = []
    for (predictor, fold), part in frame.groupby(["predictor", "fold_id"], sort=True):
        error = part.true_error_rmse.to_numpy(float)
        no_change = part.true_effect_l2_diagnostic.to_numpy(float) / np.sqrt(part.gene_axis_size.iloc[0] if "gene_axis_size" in part else 512)
        for method in ("M", "H", "M_plus_H"):
            values = part[method].to_numpy(float)
            rho = float(np.corrcoef(rankdata(values), rankdata(error))[0, 1])
            rows.append({"predictor": predictor, "fold_id": fold, "method": method,
                         "n_tasks": len(part), "spearman": rho, "utility_20": utility(values, error),
                         "model_mean_rmse": float(error.mean()), "no_change_mean_rmse": float(no_change.mean()),
                         "model_beats_no_change": bool(error.mean() < no_change.mean())})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "RESULTS.csv", index=False)
    pivot = results.pivot(index=["predictor", "fold_id"], columns="method", values="utility_20")
    pivot["delta_MH_vs_M"] = pivot.M_plus_H - pivot.M
    pivot.to_csv(OUT / "PAIRED.csv")
    summary = {"status": "RETROSPECTIVE_EVALUATED", "n_tasks_per_model": int(len(frame)/len(METRICS)),
               "n_fold_model_pairs": len(pivot), "mean_utility_M": float(pivot.M.mean()),
               "mean_utility_M_plus_H": float(pivot.M_plus_H.mean()),
               "mean_delta": float(pivot.delta_MH_vs_M.mean()),
               "positive_pairs": int((pivot.delta_MH_vs_M > 0).sum()),
               "competent_pairs": int(results.loc[results.method.eq("M"), "model_beats_no_change"].sum()),
               "result_sha256": sha(OUT / "RESULTS.csv"),
               "truth_sha256": {str(p): sha(p) for p in METRICS.values()},
               "warning": "Retrospective; E106/E107 test truth was already public"}
    (OUT / "E250_AUDIT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("seal", "evaluate"))
    args = parser.parse_args()
    if args.stage == "seal":
        score()
    else:
        evaluate()


if __name__ == "__main__":
    main()
