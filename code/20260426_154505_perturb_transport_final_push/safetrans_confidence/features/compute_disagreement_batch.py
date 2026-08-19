"""Batch computation of multi-scale disagreement features."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from safetrans_confidence.features.multiscale_disagreement import (
    MULTISCALE_DISAGREEMENT_COLS,
    compute_multiscale_disagreement,
)

def _find_file(run_dir: Path, candidates: list[str]) -> Path | None:
    for rel in candidates:
        p = run_dir / rel
        if p.exists():
            return p
    return None

def compute_disagreement_for_vectors(features_df, vec_lookup, top_k=50):
    group_cols = ["dataset_name", "fold_id", "context", "perturbation"]
    result_rows = []
    for key, group in features_df.groupby(group_cols, dropna=False):
        dataset, fold, ctx, pert = key
        predictors = sorted(group["predictor_name"].unique())
        if len(predictors) < 2:
            continue
        pred_vecs = {}
        pred_rids = {}
        for pred in predictors:
            sub = group[group["predictor_name"] == pred]
            rid = sub.iloc[0]["record_id"]
            if rid in vec_lookup:
                pred_vecs[pred] = vec_lookup[rid]
                pred_rids[pred] = rid
        if len(pred_vecs) < 2:
            continue
        pred_list = sorted(pred_vecs.keys())
        for i in range(len(pred_list)):
            for j in range(i + 1, len(pred_list)):
                pa, pb = pred_list[i], pred_list[j]
                feats = compute_multiscale_disagreement(pred_vecs[pa], pred_vecs[pb], top_k=top_k)
                row = {"dataset_name": dataset, "fold_id": fold, "context": ctx, "perturbation": pert, "predictor_a": pa, "predictor_b": pb, "record_id_a": pred_rids[pa], "record_id_b": pred_rids[pb]}
                row.update(feats)
                result_rows.append(row)
    return pd.DataFrame(result_rows)

def compute_disagreement_batch(run_dir, top_k=50, feature_file="tables/CONFIDENCE_FEATURES.csv", npz_candidates=None):
    run_dir = Path(run_dir)
    if npz_candidates is None:
        npz_candidates = ["input/predicted_effects.npz", "arrays/predicted_effects.npz"]
    feat_path = run_dir / feature_file
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    features_df = pd.read_csv(feat_path)
    npz_path = _find_file(run_dir, npz_candidates)
    if npz_path is None:
        raise FileNotFoundError(f"predicted_effects.npz not found in {run_dir}")
    arrays = np.load(npz_path)
    vec_lookup = {}
    for key in arrays:
        rid = str(key).split("::")[0]
        vec_lookup[rid] = np.asarray(arrays[key], dtype=np.float64).ravel()
    disagree_df = compute_disagreement_for_vectors(features_df, vec_lookup, top_k)
    if not disagree_df.empty:
        merge_a = disagree_df[["dataset_name","fold_id","context","perturbation","predictor_a"] + MULTISCALE_DISAGREEMENT_COLS].rename(columns={"predictor_a": "predictor_name"})
        merge_b = disagree_df[["dataset_name","fold_id","context","perturbation","predictor_b"] + MULTISCALE_DISAGREEMENT_COLS].rename(columns={"predictor_b": "predictor_name"})
        join_cols = ["dataset_name","fold_id","context","perturbation","predictor_name"]
        features_df = features_df.merge(merge_a, on=join_cols, how="left")
        merge_b_indexed = merge_b.set_index(join_cols)
        for col in MULTISCALE_DISAGREEMENT_COLS:
            mask = features_df[col].isna()
            if mask.any():
                idx_cols = features_df.loc[mask, join_cols]
                idx_tuples = [tuple(r) for r in idx_cols.values]
                mi = pd.MultiIndex.from_tuples(idx_tuples, names=join_cols)
                features_df.loc[mask, col] = merge_b_indexed.reindex(mi)[col].values
    return disagree_df, features_df

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Compute multi-scale disagreement features.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    disagree_df, merged_df = compute_disagreement_batch(args.run_dir, top_k=args.top_k)
    out_dir = Path(args.run_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    disagree_df.to_csv(out_dir / "MULTISCALE_DISAGREEMENT.csv", index=False)
    merged_df.to_csv(out_dir / "CONFIDENCE_FEATURES_MULTISCALE.csv", index=False)
    print(json.dumps({"n_pairs": len(disagree_df), "n_merged": len(merged_df)}, indent=2))
