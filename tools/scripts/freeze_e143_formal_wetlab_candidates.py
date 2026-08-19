#!/usr/bin/env python3
"""Freeze 48 matched E143 candidates from a completed pre-perturbation table.

The private score/stratum codebook is written outside the Git repository.  The
repository receives only a wet-lab handoff manifest and the private file hash.
No post-perturbation expression or phenotype column is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E143_prospective_wetlab_validation_20260714/formal_freeze"
DEFAULT_INPUT = ROOT / "docs/实验结果/E143_prospective_wetlab_validation_20260714/templates/FORMAL_CANDIDATE_INPUT.csv"
DEFAULT_PRIVATE = Path("/home/yyf/private_safeconf/E143_wetlab_blinding")
SEED = 202607143
GROUP_SIZE = 8

NUMERIC = [
    "risk_new_context", "risk_anchor_context", "predicted_magnitude_new", "predicted_magnitude_anchor",
    "model_disagreement_new", "model_disagreement_anchor", "control_expression_new", "control_expression_anchor",
    "string_degree_v12_score700", "depmap_gene_effect_new", "depmap_gene_effect_anchor",
    "guide_1_on_target_score", "guide_2_on_target_score", "maximum_predicted_off_target_score",
]
FORBIDDEN_TOKENS = ["post_perturb", "true_effect", "observed_effect", "prediction_error", "rmse", "pearson_error", "cosine_error"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def standardize(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(float)
    scale = np.nanstd(values, axis=0)
    scale[scale < 1e-12] = 1.0
    return (values - np.nanmean(values, axis=0)) / scale


def matched_pair_selection(left: pd.DataFrame, right: pd.DataFrame, covariates: list[str], n: int):
    if len(left) < n or len(right) < n:
        raise RuntimeError(f"candidate pool too small for matching: left={len(left)}, right={len(right)}, required={n}")
    combined = pd.concat([left[covariates], right[covariates]], ignore_index=True)
    z = standardize(combined)
    left_z, right_z = z[:len(left)], z[len(left):]
    cost = np.sqrt(((left_z[:, None, :] - right_z[None, :, :]) ** 2).sum(axis=2))
    rows, columns = linear_sum_assignment(cost)
    ordering = np.argsort(cost[rows, columns])[:n]
    return left.iloc[rows[ordering]].copy(), right.iloc[columns[ordering]].copy()


def validate(data: pd.DataFrame) -> pd.DataFrame:
    forbidden = [column for column in data.columns if any(token in column.lower() for token in FORBIDDEN_TOKENS)]
    if forbidden:
        raise RuntimeError(f"post-perturbation/truth-like columns are forbidden before freeze: {forbidden}")
    required = {"gene", "new_context_name", "anchor_context_name", "guide_1_sequence", "guide_2_sequence",
                "eligible_before_perturbation_truth", "copy_number_warning", "gene_family_multimapping_warning", *NUMERIC}
    missing = sorted(required - set(data.columns))
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")
    for column in NUMERIC:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    eligible = data[data.eligible_before_perturbation_truth.map(truthy)].copy()
    eligible = eligible[~eligible.copy_number_warning.map(truthy) & ~eligible.gene_family_multimapping_warning.map(truthy)]
    eligible = eligible.dropna(subset=NUMERIC)
    eligible = eligible[(eligible.guide_1_sequence.astype(str).str.len() >= 18) &
                        (eligible.guide_2_sequence.astype(str).str.len() >= 18)]
    eligible = eligible.drop_duplicates("gene", keep=False)
    if len(eligible) < 64:
        raise RuntimeError(f"need at least 64 fully eligible genes before selecting 48; found {len(eligible)}")
    if eligible.new_context_name.nunique() != 1 or eligible.anchor_context_name.nunique() != 1:
        raise RuntimeError("all candidates must use one fixed new context and one fixed anchor context")
    return eligible


def select(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["risk_mean"] = data[["risk_new_context", "risk_anchor_context"]].mean(axis=1)
    data["risk_delta_new_minus_anchor"] = data.risk_new_context - data.risk_anchor_context
    data["risk_mean_rank"] = data.risk_mean.rank(pct=True, method="average")
    data["risk_delta_rank"] = data.risk_delta_new_minus_anchor.rank(pct=True, method="average")
    data["predicted_magnitude_mean"] = data[["predicted_magnitude_new", "predicted_magnitude_anchor"]].mean(axis=1)
    data["model_disagreement_mean"] = data[["model_disagreement_new", "model_disagreement_anchor"]].mean(axis=1)
    data["control_expression_mean"] = data[["control_expression_new", "control_expression_anchor"]].mean(axis=1)
    data["depmap_gene_effect_mean"] = data[["depmap_gene_effect_new", "depmap_gene_effect_anchor"]].mean(axis=1)
    data["guide_on_target_mean"] = data[["guide_1_on_target_score", "guide_2_on_target_score"]].mean(axis=1)
    data["string_log1p_degree"] = np.log1p(data.string_degree_v12_score700)
    covariates = ["predicted_magnitude_mean", "model_disagreement_mean", "control_expression_mean",
                  "depmap_gene_effect_mean", "guide_on_target_mean", "maximum_predicted_off_target_score",
                  "string_log1p_degree"]

    high = data[data.risk_mean_rank >= .70]
    low = data[data.risk_mean_rank <= .30]
    selected_high, selected_low = matched_pair_selection(high, low, covariates, GROUP_SIZE)
    used = set(selected_high.gene) | set(selected_low.gene)

    remaining = data[~data.gene.isin(used)]
    new_high = remaining[(remaining.risk_delta_rank >= .75) & (remaining.risk_new_context >= remaining.risk_new_context.median())]
    anchor_high = remaining[(remaining.risk_delta_rank <= .25) & (remaining.risk_anchor_context >= remaining.risk_anchor_context.median())]
    selected_new, selected_anchor = matched_pair_selection(new_high, anchor_high, covariates, GROUP_SIZE)
    used |= set(selected_new.gene) | set(selected_anchor.gene)

    middle_pool = data[(~data.gene.isin(used)) & data.risk_mean_rank.between(.25, .75)].copy()
    if len(middle_pool) < 16:
        raise RuntimeError(f"middle-risk pool has {len(middle_pool)} genes; 16 required")
    center = np.nanmedian(standardize(data[covariates]), axis=0)
    middle_z = standardize(pd.concat([data[covariates], middle_pool[covariates]], ignore_index=True))[len(data):]
    middle_pool["covariate_center_distance"] = np.sqrt(((middle_z - center) ** 2).sum(axis=1))
    selected_middle = middle_pool.sort_values(["covariate_center_distance", "gene"]).head(16)

    groups = [(selected_high, "both_contexts_high"), (selected_low, "both_contexts_low"),
              (selected_new, "new_context_high_anchor_low"), (selected_anchor, "new_context_low_anchor_high"),
              (selected_middle, "middle_risk_coverage")]
    selected = pd.concat([frame.assign(prefrozen_stratum=stratum) for frame, stratum in groups], ignore_index=True)
    if len(selected) != 48 or selected.gene.nunique() != 48:
        raise RuntimeError("formal selection must contain exactly 48 unique genes")
    return selected.sort_values(["prefrozen_stratum", "gene"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE)
    args = parser.parse_args()
    raw = pd.read_csv(args.input, keep_default_na=False)
    eligible = validate(raw)
    selected = select(eligible)
    rng = np.random.default_rng(SEED)
    codes = [f"G-{value:06d}" for value in rng.choice(np.arange(100000, 999999), len(selected), replace=False)]
    selected.insert(0, "blind_gene_code", codes)
    selected["selection_rank_within_stratum"] = selected.groupby("prefrozen_stratum").cumcount() + 1

    os.umask(0o077)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    private = args.private_dir / "PRIVATE_RISK_CODEBOOK_DO_NOT_SHARE_BEFORE_QC.csv"
    selected.to_csv(private, index=False)
    os.chmod(private, 0o600)
    handoff_columns = ["blind_gene_code", "gene", "new_context_name", "anchor_context_name",
                       "guide_1_sequence", "guide_2_sequence"]
    handoff = selected[handoff_columns].sample(frac=1, random_state=SEED).reset_index(drop=True)
    handoff.to_csv(OUT / "WETLAB_HANDOFF_NO_RISK_LABELS.csv", index=False)
    status = {
        "experiment": "E143_formal_candidate_freeze", "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "formal_candidates_frozen_before_post_perturbation_readout", "n_candidates": len(selected),
        "new_context": str(selected.new_context_name.iloc[0]), "anchor_context": str(selected.anchor_context_name.iloc[0]),
        "input_sha256": sha256(args.input), "wetlab_handoff_sha256": sha256(OUT / "WETLAB_HANDOFF_NO_RISK_LABELS.csv"),
        "private_codebook_sha256": sha256(private), "private_codebook_path_not_committed": str(private),
        "post_perturbation_or_truth_columns_present": False, "seed": SEED,
    }
    (OUT / "FORMAL_FREEZE_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
