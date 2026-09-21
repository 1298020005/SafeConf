#!/usr/bin/env python3
"""Independently audit the released E205 result tables.

This script never loads target expression matrices or model checkpoints.  It
recomputes the released scores and summary statistics from the sealed pretruth
table and the formal post-truth CSV files only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
COMPONENTS = (
    "z_family_disagreement",
    "z_model_source_gap",
    "z_source_delta_dispersion",
    "z_negative_log_source_cells",
    "z_support_context_deficit",
)


class AuditFailure(RuntimeError):
    """A released E205 invariant did not reproduce."""


def percentile_rank(values: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return rankdata(values, method="average") / len(values)


def spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    left_rank, right_rank = percentile_rank(left), percentile_rank(right)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def stable_ties(task_ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            int(hashlib.sha256(f"E205\0{task_id}\00".encode()).hexdigest()[:16], 16)
            for task_id in map(str, task_ids)
        ],
        dtype=np.uint64,
    )


def review_utility(frame: pd.DataFrame, score: str, outcome: str) -> float:
    values = frame[outcome].to_numpy(float)
    scores = frame[score].to_numpy(float)
    task_ids = frame.task_id.to_numpy()
    n_select = int(math.ceil(0.20 * len(frame)))
    ties = stable_ties(task_ids)
    selected = np.lexsort((ties, -scores))[:n_select]
    oracle = np.lexsort((ties, -values))[:n_select]
    overall_mean = float(values.mean())
    denominator = float(values[oracle].mean()) - overall_mean
    return float((float(values[selected].mean()) - overall_mean) / denominator)


def compare_pretruth_columns(pretruth: pd.DataFrame, final: pd.DataFrame) -> list[dict]:
    merged = pretruth.merge(
        final, on="task_id", suffixes=("_pretruth", "_final"), validate="one_to_one"
    )
    mismatches: list[dict] = []
    for column in pretruth.columns:
        if column == "task_id" or column not in final.columns:
            continue
        left = merged[f"{column}_pretruth"]
        right = merged[f"{column}_final"]
        numeric = (
            pd.api.types.is_numeric_dtype(left)
            and pd.api.types.is_numeric_dtype(right)
            and not pd.api.types.is_bool_dtype(left)
            and not pd.api.types.is_bool_dtype(right)
        )
        if numeric:
            left_values = pd.to_numeric(left, errors="coerce").to_numpy(float)
            right_values = pd.to_numeric(right, errors="coerce").to_numpy(float)
            one_nan = np.isnan(left_values) ^ np.isnan(right_values)
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            maximum = (
                float(np.max(np.abs(left_values[finite] - right_values[finite])))
                if finite.any()
                else 0.0
            )
            if one_nan.any() or maximum > 1e-12:
                mismatches.append(
                    {
                        "column": column,
                        "one_sided_nan": int(one_nan.sum()),
                        "max_abs_difference": maximum,
                    }
                )
        else:
            different = (
                left.fillna("<NA>").astype(str).to_numpy()
                != right.fillna("<NA>").astype(str).to_numpy()
            )
            if different.any():
                mismatches.append(
                    {"column": column, "different_rows": int(different.sum())}
                )
    return mismatches


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, value: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    value.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.expanduser().absolute()
    formal = root / "formal_evaluation"
    tasks = pd.read_csv(formal / "E205_TASK_METRICS.csv")
    pretruth = pd.read_csv(root / "tables/E205_PRETRUTH_RISK_FEATURES.csv")
    status = json.loads(
        (formal / "E205_FORMAL_EVALUATION_STATUS.json").read_text(encoding="utf-8")
    )

    primary = tasks.loc[tasks.analysis_stratum.eq("primary_ge30")].copy()
    if len(tasks) != 2008 or len(primary) != 1808:
        raise AuditFailure("released task counts changed")
    if tasks.task_id.nunique() != len(tasks) or primary.task_id.nunique() != len(primary):
        raise AuditFailure("duplicate task identifiers detected")
    if tuple(pd.unique(tasks.target.astype(str))) != TARGETS:
        raise AuditFailure("target order or membership changed")

    pretruth_mismatches = compare_pretruth_columns(pretruth, tasks)
    if pretruth_mismatches:
        raise AuditFailure(f"pretruth columns changed: {pretruth_mismatches[:3]}")

    safeconf_recomputed = tasks[list(COMPONENTS)].mean(axis=1)
    safeconf_error = float(
        np.max(np.abs(safeconf_recomputed - tasks.safeconf_e205_risk))
    )
    fusion_recomputed = pd.Series(index=tasks.index, dtype=float)
    for _, block in tasks.groupby("target", sort=False):
        magnitude_rank = rankdata(block.predicted_magnitude, method="average")
        safeconf_rank = rankdata(block.safeconf_e205_risk, method="average")
        fusion_recomputed.loc[block.index] = (
            4.0 * magnitude_rank + safeconf_rank
        ) / (5.0 * len(block))
    fusion_error = float(
        np.max(np.abs(fusion_recomputed - tasks.safeconf_m_4to1))
    )
    if safeconf_error > 1e-12 or fusion_error > 1e-12:
        raise AuditFailure("released score formula did not reproduce")

    associations = pd.read_csv(formal / "E205_RISK_ASSOCIATIONS.csv")
    utilities = pd.read_csv(formal / "E205_REVIEW_UTILITY.csv")
    published_rho = float(
        associations.loc[
            associations.scope.eq("pooled")
            & associations.predictor.eq("safeconf_m_4to1"),
            "spearman",
        ].iloc[0]
    )
    published_utility = float(
        utilities.loc[
            utilities.scope.eq("pooled")
            & utilities.predictor.eq("safeconf_m_4to1")
            & np.isclose(utilities.budget, 0.20),
            "oracle_normalized_utility",
        ].iloc[0]
    )
    reproduced_rho = spearman(primary.safeconf_m_4to1, primary.family_rms_error)
    reproduced_utility = review_utility(
        primary, "safeconf_m_4to1", "family_rms_error"
    )
    if abs(reproduced_rho - published_rho) > 1e-12:
        raise AuditFailure("pooled SafeConf-M Spearman did not reproduce")
    if abs(reproduced_utility - published_utility) > 1e-12:
        raise AuditFailure("pooled SafeConf-M review utility did not reproduce")

    interval_table = pd.read_csv(formal / "E205_INCREMENTAL_INTERVALS.csv")
    draws = pd.read_csv(formal / "E205_INCREMENTAL_BOOTSTRAP_DRAWS.csv")
    interval_checks = []
    for measure in ("delta_spearman", "delta_utility_20"):
        published = interval_table.loc[interval_table.measure.eq(measure)].iloc[0]
        lower, upper = np.quantile(draws[measure].to_numpy(float), [0.025, 0.975])
        check = {
            "measure": measure,
            "published_lower": float(published.ci95_lower),
            "recomputed_lower": float(lower),
            "published_upper": float(published.ci95_upper),
            "recomputed_upper": float(upper),
        }
        interval_checks.append(check)
        if (
            abs(check["published_lower"] - check["recomputed_lower"]) > 1e-12
            or abs(check["published_upper"] - check["recomputed_upper"]) > 1e-12
        ):
            raise AuditFailure(f"bootstrap interval changed: {measure}")

    rows = []
    for target, block in primary.groupby("target", sort=False):
        magnitude_rho = spearman(block.predicted_magnitude, block.family_rms_error)
        safeconf_m_rho = spearman(block.safeconf_m_4to1, block.family_rms_error)
        magnitude_utility = review_utility(
            block, "predicted_magnitude", "family_rms_error"
        )
        safeconf_m_utility = review_utility(
            block, "safeconf_m_4to1", "family_rms_error"
        )
        registered_magnitude_rho = spearman(
            block.registered_predicted_magnitude,
            block.registered_family_rms_error,
        )
        router_rho = spearman(
            block.context_holdout_router, block.registered_family_rms_error
        )
        registered_magnitude_utility = review_utility(
            block,
            "registered_predicted_magnitude",
            "registered_family_rms_error",
        )
        router_utility = review_utility(
            block, "context_holdout_router", "registered_family_rms_error"
        )
        rows.append(
            {
                "target": target,
                "n_primary_tasks": len(block),
                "safeconf_m_delta_spearman": safeconf_m_rho - magnitude_rho,
                "safeconf_m_delta_utility_20": safeconf_m_utility
                - magnitude_utility,
                "context_router_delta_spearman": router_rho
                - registered_magnitude_rho,
                "context_router_delta_utility_20": router_utility
                - registered_magnitude_utility,
            }
        )
    target_audit = pd.DataFrame(rows)

    audit = {
        "experiment": "E205_cross_family_exphormer",
        "stage": "INDEPENDENT_RELEASE_AUDIT",
        "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_formal_git_head": status.get("git_head"),
        "n_all_tasks": len(tasks),
        "n_primary_tasks": len(primary),
        "n_unique_primary_tasks": int(primary.task_id.nunique()),
        "target_primary_counts": {
            str(key): int(value)
            for key, value in primary.groupby("target").size().items()
        },
        "pretruth_shared_columns_checked": int(
            len(set(pretruth.columns).intersection(tasks.columns)) - 1
        ),
        "pretruth_mismatch_count": 0,
        "safeconf_formula_max_abs_error": safeconf_error,
        "safeconf_m_formula_max_abs_error": fusion_error,
        "published_safeconf_m_spearman": published_rho,
        "recomputed_safeconf_m_spearman": reproduced_rho,
        "published_safeconf_m_utility_20": published_utility,
        "recomputed_safeconf_m_utility_20": reproduced_utility,
        "n_bootstrap_draws": len(draws),
        "bootstrap_interval_checks": interval_checks,
        "scope_note": (
            "SafeConf-M percentiles are computed over all 502 preregistered tasks "
            "within each target; evaluation then retains the 1808 primary tasks."
        ),
        "boundary_note": (
            "All four targets have positive rank-correlation increments. K562 has "
            "an approximately zero, slightly negative 20% utility increment."
        ),
    }
    atomic_csv(formal / "E205_INDEPENDENT_TARGET_AUDIT.csv", target_audit)
    atomic_json(formal / "E205_INDEPENDENT_RESULT_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
