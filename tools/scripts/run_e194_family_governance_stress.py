#!/usr/bin/env python3
"""Run the frozen post-truth E194 model-family governance stress audit."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_e193_multigeometry_certificate_audit import (
    DATASETS,
    GEOMETRIES,
    MODEL_KEYS,
    N_GENES,
    AuditFailure,
    cluster_bootstrap_spearman,
    cluster_groups,
    embed_rows,
    load_dataset,
    markdown_table,
    spearman,
    stable_seed,
    top_indices,
    utility_values,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "docs/实验结果"
OUT = RESULTS_ROOT / "E194_family_governance_stress_20260729"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"

LOWER_TOL = 1e-10
IDENTITY_TOL = 1e-10
INVARIANT_TOL = 1e-10
BOOTSTRAP_SPEARMAN = 5000
BOOTSTRAP_UTILITY = 3000
UTILITY_BUDGET = 0.20

SCGPT = tuple(i for i, key in enumerate(MODEL_KEYS) if key.startswith("scGPT"))
GEARS = tuple(i for i, key in enumerate(MODEL_KEYS) if key.startswith("GEARS"))


@dataclass(frozen=True)
class Scenario:
    """One frozen family definition in a specified Hilbert embedding."""

    scenario_id: str
    group: str
    entries: np.ndarray
    weights: np.ndarray
    lineages: tuple[str, ...]
    entry_roles: tuple[str, ...]
    governance: str
    family_kind: str


def normalize_weights(weights: Iterable[float], n_entries: int) -> np.ndarray:
    result = np.asarray(tuple(weights), dtype=np.float64)
    if result.shape != (n_entries,) or not np.isfinite(result).all():
        raise AuditFailure("invalid scenario weights")
    if (result < 0).any() or not math.isclose(
        float(result.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AuditFailure(f"scenario weights do not sum to one: {result}")
    return result


def array_sha256(values: np.ndarray) -> str:
    """Hash array values together with their canonical dtype and shape."""

    canonical = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def scenario(
    scenario_id: str,
    group: str,
    entries: np.ndarray,
    weights: Iterable[float],
    lineages: Iterable[str],
    entry_roles: Iterable[str],
    governance: str,
    family_kind: str,
) -> Scenario:
    values = np.asarray(entries, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != N_GENES:
        raise AuditFailure(f"{scenario_id}: invalid family array {values.shape}")
    lineage_tuple = tuple(str(value) for value in lineages)
    role_tuple = tuple(str(value) for value in entry_roles)
    if len(lineage_tuple) != len(values) or len(role_tuple) != len(values):
        raise AuditFailure(f"{scenario_id}: member metadata length mismatch")
    return Scenario(
        scenario_id=scenario_id,
        group=group,
        entries=values,
        weights=normalize_weights(weights, len(values)),
        lineages=lineage_tuple,
        entry_roles=role_tuple,
        governance=governance,
        family_kind=family_kind,
    )


def real_subset(
    predictions: np.ndarray,
    indices: Iterable[int],
    scenario_id: str,
    group: str,
    governance: str = "flat",
    family_kind: str = "real_prediction_family",
) -> Scenario:
    take = tuple(int(index) for index in indices)
    return scenario(
        scenario_id,
        group,
        predictions[np.asarray(take)],
        np.repeat(1.0 / len(take), len(take)),
        (MODEL_KEYS[index] for index in take),
        ("real_prediction" for _ in take),
        governance,
        family_kind,
    )


def build_scenarios(
    geometry: str,
    predictions: np.ndarray,
    source: np.ndarray,
) -> list[Scenario]:
    """Construct every frozen A/B/C scenario without looking at target truth."""

    n_tasks = predictions.shape[1]
    result: list[Scenario] = []
    base_lineages = tuple(MODEL_KEYS)
    base_roles = tuple("real_prediction" for _ in MODEL_KEYS)
    base_weights = np.repeat(1.0 / 6.0, 6)

    result.append(
        scenario(
            "A0_primary_balanced_3x3",
            "A0_primary",
            predictions,
            base_weights,
            base_lineages,
            base_roles,
            "architecture_balanced_unique_lineage",
            "primary_real_prediction_family",
        )
    )
    result.append(
        real_subset(
            predictions,
            SCGPT,
            "A1_scgpt_seed_only",
            "A1_scgpt_only",
            family_kind="single_architecture_diagnostic",
        )
    )
    result.append(
        real_subset(
            predictions,
            GEARS,
            "A2_gears_seed_only",
            "A2_gears_only",
            family_kind="single_architecture_diagnostic",
        )
    )
    for seed_position, seed in enumerate((3407, 3408, 3409)):
        indices = (2 * seed_position, 2 * seed_position + 1)
        result.append(
            real_subset(
                predictions,
                indices,
                f"A3_matched_pair_seed{seed}",
                "A3_matched_pair",
                family_kind="matched_seed_diagnostic",
            )
        )

    scgpt_centroid = predictions[np.asarray(SCGPT)].mean(axis=0)
    gears_centroid = predictions[np.asarray(GEARS)].mean(axis=0)
    result.append(
        scenario(
            "A4_architecture_centroids",
            "A4_architecture_centroids",
            np.stack([scgpt_centroid, gears_centroid]),
            (0.5, 0.5),
            ("scGPT_architecture_centroid", "GEARS_architecture_centroid"),
            ("diagnostic_centroid", "diagnostic_centroid"),
            "architecture_balanced",
            "synthetic_diagnostic_family",
        )
    )

    for left in SCGPT:
        for right in GEARS:
            pair_name = f"{MODEL_KEYS[left]}__{MODEL_KEYS[right]}"
            result.append(
                real_subset(
                    predictions,
                    (left, right),
                    f"A5_pair_1x1_{pair_name}",
                    "A5_pair_1x1",
                    family_kind="enumerated_balanced_subfamily",
                )
            )
    for sc_left in range(len(SCGPT)):
        for sc_right in range(sc_left + 1, len(SCGPT)):
            for ge_left in range(len(GEARS)):
                for ge_right in range(ge_left + 1, len(GEARS)):
                    indices = (
                        SCGPT[sc_left],
                        SCGPT[sc_right],
                        GEARS[ge_left],
                        GEARS[ge_right],
                    )
                    name = "__".join(MODEL_KEYS[index] for index in indices)
                    result.append(
                        real_subset(
                            predictions,
                            indices,
                            f"A5_balanced_2x2_{name}",
                            "A5_balanced_2x2",
                            family_kind="enumerated_balanced_subfamily",
                        )
                    )

    doubled = np.concatenate([predictions, predictions], axis=0)
    doubled_lineages = base_lineages + base_lineages
    doubled_roles = base_roles + tuple("duplicate_prediction" for _ in MODEL_KEYS)
    result.append(
        scenario(
            "B1_duplicate_all_flat",
            "B1_duplicate_all_flat",
            doubled,
            np.repeat(1.0 / 12.0, 12),
            doubled_lineages,
            doubled_roles,
            "flat_entry_weight",
            "duplicate_stress",
        )
    )
    result.append(
        scenario(
            "B1_duplicate_all_governed",
            "B1_duplicate_all_governed",
            doubled,
            np.repeat(1.0 / 12.0, 12),
            doubled_lineages,
            doubled_roles,
            "duplicate_splits_lineage_weight",
            "duplicate_stress",
        )
    )

    for member, key in enumerate(MODEL_KEYS):
        duplicated = np.concatenate(
            [predictions, predictions[member : member + 1]], axis=0
        )
        lineages = base_lineages + (key,)
        roles = base_roles + ("duplicate_prediction",)
        result.append(
            scenario(
                f"B2_duplicate_one_flat_{key}",
                "B2_duplicate_one_flat",
                duplicated,
                np.repeat(1.0 / 7.0, 7),
                lineages,
                roles,
                "flat_entry_weight",
                "duplicate_stress",
            )
        )
        governed = np.repeat(1.0 / 6.0, 7)
        governed[member] = 1.0 / 12.0
        governed[-1] = 1.0 / 12.0
        result.append(
            scenario(
                f"B2_duplicate_one_governed_{key}",
                "B2_duplicate_one_governed",
                duplicated,
                governed,
                lineages,
                roles,
                "duplicate_splits_lineage_weight",
                "duplicate_stress",
            )
        )
        keep = tuple(index for index in range(6) if index != member)
        result.append(
            real_subset(
                predictions,
                keep,
                f"B2_leave_one_out_{key}",
                "B2_leave_one_out",
                governance="flat_after_omission",
                family_kind="omission_stress",
            )
        )

    for architecture, indices in (("scgpt", SCGPT), ("gears", GEARS)):
        repeated = np.concatenate(
            [predictions, predictions[np.asarray(indices)]], axis=0
        )
        repeated_lineages = base_lineages + tuple(MODEL_KEYS[index] for index in indices)
        repeated_roles = base_roles + tuple(
            "duplicate_prediction" for _ in indices
        )
        result.append(
            scenario(
                f"B3_overweight_{architecture}_flat",
                f"B3_overweight_{architecture}_flat",
                repeated,
                np.repeat(1.0 / 9.0, 9),
                repeated_lineages,
                repeated_roles,
                "flat_entry_weight",
                "architecture_imbalance_stress",
            )
        )
        governed = np.repeat(1.0 / 6.0, 9)
        for appended, original in enumerate(indices, start=6):
            governed[original] = 1.0 / 12.0
            governed[appended] = 1.0 / 12.0
        result.append(
            scenario(
                f"B3_overweight_{architecture}_governed",
                f"B3_overweight_{architecture}_governed",
                repeated,
                governed,
                repeated_lineages,
                repeated_roles,
                "architecture_and_lineage_weight_governed",
                "architecture_imbalance_stress",
            )
        )

    if geometry == "absolute_rmse":
        zeros = np.zeros((1, n_tasks, N_GENES), dtype=np.float64)
        result.append(
            scenario(
                "C1_add_zero_portfolio",
                "C1_add_zero",
                np.concatenate([predictions, zeros], axis=0),
                np.repeat(1.0 / 7.0, 7),
                base_lineages + ("zero_effect",),
                base_roles + ("negative_control",),
                "flat_entry_weight",
                "portfolio_negative_control",
            )
        )
    result.append(
        scenario(
            "C2_add_source_portfolio",
            "C2_add_source",
            np.concatenate([predictions, source[None, :, :]], axis=0),
            np.repeat(1.0 / 7.0, 7),
            base_lineages + ("source_effect",),
            base_roles + ("baseline_portfolio_member",),
            "flat_entry_weight",
            "portfolio_negative_control",
        )
    )
    if geometry == "absolute_rmse":
        zeros = np.zeros((1, n_tasks, N_GENES), dtype=np.float64)
        result.append(
            scenario(
                "C3_add_zero_source_portfolio",
                "C3_add_zero_source",
                np.concatenate([predictions, zeros, source[None, :, :]], axis=0),
                np.repeat(1.0 / 8.0, 8),
                base_lineages + ("zero_effect", "source_effect"),
                base_roles
                + ("negative_control", "baseline_portfolio_member"),
                "flat_entry_weight",
                "portfolio_negative_control",
            )
        )

    if geometry == "absolute_rmse":
        centroid = predictions.mean(axis=0)
        architecture_difference = scgpt_centroid - gears_centroid
        for attack_lambda in (1, 2, 4):
            positive = centroid + attack_lambda * architecture_difference
            negative = centroid - attack_lambda * architecture_difference
            result.append(
                scenario(
                    f"C4_symmetric_attack_lambda{attack_lambda}",
                    "C4_symmetric_attack",
                    np.concatenate(
                        [
                            predictions,
                            positive[None, :, :],
                            negative[None, :, :],
                        ],
                        axis=0,
                    ),
                    np.repeat(1.0 / 8.0, 8),
                    base_lineages
                    + (
                        f"symmetric_attack_plus_lambda{attack_lambda}",
                        f"symmetric_attack_minus_lambda{attack_lambda}",
                    ),
                    base_roles + ("synthetic_attack", "synthetic_attack"),
                    "flat_entry_weight",
                    "gaming_negative_control",
                )
            )
    ids = [item.scenario_id for item in result]
    if len(ids) != len(set(ids)):
        raise AuditFailure("duplicate scenario id")
    return result


def family_metrics(
    dataset: str,
    geometry: str,
    query: pd.DataFrame,
    target: np.ndarray,
    item: Scenario,
) -> pd.DataFrame:
    entries = item.entries
    weights = item.weights
    centroid = np.einsum("m,mng->ng", weights, entries)
    member_sq = np.sum((entries - target[None, :, :]) ** 2, axis=2)
    member_errors = np.sqrt(member_sq)
    family_sq = np.einsum("m,mn->n", weights, member_sq)
    family_rms = np.sqrt(np.maximum(family_sq, 0.0))
    centroid_error = np.linalg.norm(centroid - target, axis=1)
    diversity_sq = np.einsum(
        "m,mn->n",
        weights,
        np.sum((entries - centroid[None, :, :]) ** 2, axis=2),
    )
    diversity = np.sqrt(np.maximum(diversity_sq, 0.0))
    worst = member_errors.max(axis=0)
    diameter = np.zeros(len(query), dtype=np.float64)
    for left in range(len(entries)):
        for right in range(left + 1, len(entries)):
            diameter = np.maximum(
                diameter,
                np.linalg.norm(entries[left] - entries[right], axis=1),
            )
    diameter_half = diameter / 2.0
    identity_residual = np.abs(
        family_sq - (centroid_error**2 + diversity_sq)
    )
    lineage_weights: dict[str, float] = {}
    for lineage, weight in zip(item.lineages, weights):
        lineage_weights[lineage] = lineage_weights.get(lineage, 0.0) + float(
            weight
        )
    lineage_weight_values = np.asarray(
        tuple(lineage_weights.values()), dtype=np.float64
    )
    scgpt_weight = sum(
        weight
        for lineage, weight in lineage_weights.items()
        if lineage.startswith("scGPT")
    )
    gears_weight = sum(
        weight
        for lineage, weight in lineage_weights.items()
        if lineage.startswith("GEARS")
    )

    frame = query[["task_id", "batch", "gene"]].copy()
    frame.insert(0, "target_family_id", item.scenario_id)
    frame.insert(0, "scenario_group", item.group)
    frame.insert(0, "geometry", geometry)
    frame.insert(0, "dataset", dataset)
    frame["governance"] = item.governance
    frame["family_kind"] = item.family_kind
    frame["n_entries"] = len(entries)
    frame["n_unique_lineages"] = len(set(item.lineages))
    frame["entry_effective_n"] = 1.0 / float(np.sum(weights**2))
    frame["lineage_effective_n"] = 1.0 / float(
        np.sum(lineage_weight_values**2)
    )
    frame["max_entry_weight"] = float(weights.max())
    frame["max_lineage_weight"] = float(lineage_weight_values.max())
    frame["scgpt_total_weight"] = scgpt_weight
    frame["gears_total_weight"] = gears_weight
    frame["auxiliary_total_weight"] = 1.0 - scgpt_weight - gears_weight
    frame["family_rms_error"] = family_rms
    frame["centroid_error"] = centroid_error
    frame["worst_member_error"] = worst
    frame["diversity_lower_bound"] = diversity
    frame["diameter_half_lower_bound"] = diameter_half
    frame["family_identity_residual"] = identity_residual
    frame["family_lower_violation"] = diversity > family_rms + LOWER_TOL
    frame["worst_lower_violation"] = diameter_half > worst + LOWER_TOL
    frame["diversity_over_family_rms"] = np.divide(
        diversity,
        family_rms,
        out=np.full_like(diversity, np.nan),
        where=family_rms > 0,
    )
    frame["diversity_sq_over_family_sq"] = np.divide(
        diversity_sq,
        family_sq,
        out=np.full_like(diversity_sq, np.nan),
        where=family_sq > 0,
    )
    frame["diameter_half_over_worst"] = np.divide(
        diameter_half,
        worst,
        out=np.full_like(diameter_half, np.nan),
        where=worst > 0,
    )
    return frame


def family_member_rows(
    dataset: str,
    geometry: str,
    item: Scenario,
    base_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    lineage_totals: dict[str, float] = {}
    for lineage, weight in zip(item.lineages, item.weights):
        lineage_totals[lineage] = lineage_totals.get(lineage, 0.0) + float(
            weight
        )
    rows: list[dict[str, Any]] = []
    for entry_index, (lineage, role, weight) in enumerate(
        zip(item.lineages, item.entry_roles, item.weights)
    ):
        if lineage.startswith("scGPT"):
            architecture = "scGPT"
        elif lineage.startswith("GEARS"):
            architecture = "GEARS"
        else:
            architecture = "auxiliary_or_synthetic"
        rows.append(
            {
                "dataset": dataset,
                "geometry": geometry,
                "scenario_group": item.group,
                "target_family_id": item.scenario_id,
                "entry_index": entry_index,
                "lineage_id": lineage,
                "entry_role": role,
                "architecture": architecture,
                "entry_weight": float(weight),
                "lineage_total_weight": lineage_totals[lineage],
                "prediction_array_sha256": base_hashes.get(lineage, ""),
                "governance": item.governance,
                "family_kind": item.family_kind,
            }
        )
    return rows


def jaccard_top(
    values: np.ndarray,
    comparator: np.ndarray,
    task_ids: np.ndarray,
    budget: float = UTILITY_BUDGET,
) -> float:
    n_select = int(math.ceil(len(task_ids) * budget))
    first = set(task_ids[top_indices(values, task_ids, n_select)].tolist())
    second = set(task_ids[top_indices(comparator, task_ids, n_select)].tolist())
    return float(len(first & second) / len(first | second))


def gene_macro_spearman(
    frame: pd.DataFrame, predictor: str, outcome: str
) -> float:
    macro = (
        frame.groupby("gene", observed=True, sort=True)[[predictor, outcome]]
        .mean()
        .reset_index(drop=True)
    )
    return spearman(
        macro[predictor].to_numpy(float),
        macro[outcome].to_numpy(float),
    )


def attach_a0_targets(tasks: pd.DataFrame) -> pd.DataFrame:
    reference = tasks.loc[
        tasks.target_family_id.eq("A0_primary_balanced_3x3"),
        [
            "dataset",
            "geometry",
            "task_id",
            "family_rms_error",
            "centroid_error",
            "diversity_lower_bound",
        ],
    ].rename(
        columns={
            "family_rms_error": "a0_family_rms_error",
            "centroid_error": "a0_centroid_error",
            "diversity_lower_bound": "a0_diversity_lower_bound",
        }
    )
    if reference.duplicated(["dataset", "geometry", "task_id"]).any():
        raise AuditFailure("A0 reference contains duplicate tasks")
    result = tasks.merge(
        reference,
        on=["dataset", "geometry", "task_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if result[
        [
            "a0_family_rms_error",
            "a0_centroid_error",
            "a0_diversity_lower_bound",
        ]
    ].isna().any().any():
        raise AuditFailure("A0 fixed outcomes could not be attached")
    return result


def summarize_scenario(frame: pd.DataFrame, a0: pd.DataFrame) -> dict[str, Any]:
    task_ids = frame.task_id.astype(str).to_numpy()
    if task_ids.tolist() != a0.task_id.astype(str).tolist():
        raise AuditFailure("scenario and A0 task order differ")
    utility = utility_values(
        frame.diversity_lower_bound.to_numpy(float),
        frame.family_rms_error.to_numpy(float),
        task_ids,
        UTILITY_BUDGET,
    )
    a0_family_utility = utility_values(
        frame.diversity_lower_bound.to_numpy(float),
        frame.a0_family_rms_error.to_numpy(float),
        task_ids,
        UTILITY_BUDGET,
    )
    a0_centroid_utility = utility_values(
        frame.diversity_lower_bound.to_numpy(float),
        frame.a0_centroid_error.to_numpy(float),
        task_ids,
        UTILITY_BUDGET,
    )
    mean_family = float(frame.family_rms_error.mean())
    mean_diversity = float(frame.diversity_lower_bound.mean())
    mean_a0_family = float(a0.family_rms_error.mean())
    mean_a0_diversity = float(a0.diversity_lower_bound.mean())
    return {
        "dataset": frame.dataset.iloc[0],
        "geometry": frame.geometry.iloc[0],
        "scenario_group": frame.scenario_group.iloc[0],
        "target_family_id": frame.target_family_id.iloc[0],
        "governance": frame.governance.iloc[0],
        "family_kind": frame.family_kind.iloc[0],
        "n_tasks": len(frame),
        "n_entries": int(frame.n_entries.iloc[0]),
        "n_unique_lineages": int(frame.n_unique_lineages.iloc[0]),
        "entry_effective_n": float(frame.entry_effective_n.iloc[0]),
        "lineage_effective_n": float(frame.lineage_effective_n.iloc[0]),
        "max_entry_weight": float(frame.max_entry_weight.iloc[0]),
        "max_lineage_weight": float(frame.max_lineage_weight.iloc[0]),
        "scgpt_total_weight": float(frame.scgpt_total_weight.iloc[0]),
        "gears_total_weight": float(frame.gears_total_weight.iloc[0]),
        "auxiliary_total_weight": float(frame.auxiliary_total_weight.iloc[0]),
        "mean_family_rms_error": mean_family,
        "mean_centroid_error": float(frame.centroid_error.mean()),
        "mean_worst_member_error": float(frame.worst_member_error.mean()),
        "mean_diversity_lower_bound": mean_diversity,
        "mean_diameter_half_lower_bound": float(
            frame.diameter_half_lower_bound.mean()
        ),
        "median_diversity_over_family_rms": float(
            frame.diversity_over_family_rms.median()
        ),
        "median_diversity_sq_over_family_sq": float(
            frame.diversity_sq_over_family_sq.median()
        ),
        "median_diameter_half_over_worst": float(
            frame.diameter_half_over_worst.median()
        ),
        "family_lower_violations": int(frame.family_lower_violation.sum()),
        "worst_lower_violations": int(frame.worst_lower_violation.sum()),
        "max_identity_residual": float(frame.family_identity_residual.max()),
        "diversity_error_spearman": spearman(
            frame.diversity_lower_bound.to_numpy(float),
            frame.family_rms_error.to_numpy(float),
        ),
        "diversity_own_centroid_spearman": spearman(
            frame.diversity_lower_bound.to_numpy(float),
            frame.centroid_error.to_numpy(float),
        ),
        "diversity_a0_family_error_spearman": spearman(
            frame.diversity_lower_bound.to_numpy(float),
            frame.a0_family_rms_error.to_numpy(float),
        ),
        "diversity_a0_centroid_error_spearman": spearman(
            frame.diversity_lower_bound.to_numpy(float),
            frame.a0_centroid_error.to_numpy(float),
        ),
        "macro_diversity_own_family_error_spearman": gene_macro_spearman(
            frame, "diversity_lower_bound", "family_rms_error"
        ),
        "macro_diversity_own_centroid_spearman": gene_macro_spearman(
            frame, "diversity_lower_bound", "centroid_error"
        ),
        "macro_diversity_a0_family_error_spearman": gene_macro_spearman(
            frame, "diversity_lower_bound", "a0_family_rms_error"
        ),
        "macro_diversity_a0_centroid_error_spearman": gene_macro_spearman(
            frame, "diversity_lower_bound", "a0_centroid_error"
        ),
        "capture_at_20pct": float(utility["high_error_capture"]),
        "error_lift_at_20pct": float(utility["error_lift"]),
        "oracle_normalized_utility_at_20pct": float(
            utility["oracle_normalized_utility"]
        ),
        "a0_family_capture_at_20pct": float(
            a0_family_utility["high_error_capture"]
        ),
        "a0_family_error_lift_at_20pct": float(
            a0_family_utility["error_lift"]
        ),
        "a0_family_oracle_normalized_utility_at_20pct": float(
            a0_family_utility["oracle_normalized_utility"]
        ),
        "a0_centroid_capture_at_20pct": float(
            a0_centroid_utility["high_error_capture"]
        ),
        "a0_centroid_error_lift_at_20pct": float(
            a0_centroid_utility["error_lift"]
        ),
        "a0_centroid_oracle_normalized_utility_at_20pct": float(
            a0_centroid_utility["oracle_normalized_utility"]
        ),
        "a0_diversity_top20_jaccard": jaccard_top(
            frame.diversity_lower_bound.to_numpy(float),
            a0.diversity_lower_bound.to_numpy(float),
            task_ids,
        ),
        "mean_family_error_change_vs_a0": mean_family - mean_a0_family,
        "relative_family_error_change_vs_a0": (
            mean_family / mean_a0_family - 1.0
        ),
        "mean_diversity_change_vs_a0": mean_diversity - mean_a0_diversity,
        "relative_diversity_change_vs_a0": (
            mean_diversity / mean_a0_diversity - 1.0
            if mean_a0_diversity > 0
            else float("nan")
        ),
    }


def group_ranges(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "mean_family_rms_error",
        "mean_centroid_error",
        "mean_diversity_lower_bound",
        "median_diversity_sq_over_family_sq",
        "diversity_error_spearman",
        "diversity_own_centroid_spearman",
        "diversity_a0_family_error_spearman",
        "diversity_a0_centroid_error_spearman",
        "macro_diversity_own_family_error_spearman",
        "macro_diversity_own_centroid_spearman",
        "macro_diversity_a0_family_error_spearman",
        "macro_diversity_a0_centroid_error_spearman",
        "capture_at_20pct",
        "error_lift_at_20pct",
        "oracle_normalized_utility_at_20pct",
        "a0_family_oracle_normalized_utility_at_20pct",
        "a0_centroid_oracle_normalized_utility_at_20pct",
        "a0_diversity_top20_jaccard",
        "relative_family_error_change_vs_a0",
        "relative_diversity_change_vs_a0",
    )
    rows: list[dict[str, Any]] = []
    for keys, group in summary.groupby(
        ["dataset", "geometry", "scenario_group"], sort=True, observed=True
    ):
        for metric in metrics:
            values = group[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "dataset": keys[0],
                    "geometry": keys[1],
                    "scenario_group": keys[2],
                    "metric": metric,
                    "n_scenarios": len(group),
                    "n_finite": len(finite),
                    "min": float(np.min(finite)) if len(finite) else np.nan,
                    "median": float(np.median(finite)) if len(finite) else np.nan,
                    "max": float(np.max(finite)) if len(finite) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_utility_20(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
    seed: int,
    n_boot: int = BOOTSTRAP_UTILITY,
) -> dict[str, float | int]:
    work = frame.loc[
        np.isfinite(frame[predictor])
        & np.isfinite(frame[outcome])
    ].reset_index(drop=True)
    genes, groups = cluster_groups(work)
    x = work[predictor].to_numpy(float)
    y = work[outcome].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=np.float64)
    n_valid = 0
    for _ in range(n_boot):
        sampled = rng.choice(genes, size=len(genes), replace=True)
        blocks: list[np.ndarray] = []
        tie_blocks: list[np.ndarray] = []
        offset = 0
        for occurrence, gene in enumerate(sampled):
            block = groups[str(gene)]
            blocks.append(block)
            tie_blocks.append(
                occurrence * (len(work) + 1)
                + np.arange(offset, offset + len(block))
            )
            offset += len(block)
        take = np.concatenate(blocks)
        tie_ids = np.concatenate(tie_blocks)
        value = utility_values(
            x[take], y[take], tie_ids, UTILITY_BUDGET
        )["oracle_normalized_utility"]
        if math.isfinite(float(value)):
            values[n_valid] = float(value)
            n_valid += 1
    if n_valid < int(0.95 * n_boot):
        raise AuditFailure("too few valid E194 utility bootstrap draws")
    values = values[:n_valid]
    point = utility_values(
        x, y, work.task_id.astype(str).to_numpy(), UTILITY_BUDGET
    )
    return {
        "n_tasks": len(work),
        "n_gene_clusters": len(genes),
        "point": float(point["oracle_normalized_utility"]),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "bootstrap_valid": n_valid,
    }


def bootstrap_key_scenarios(tasks: pd.DataFrame) -> pd.DataFrame:
    scenario_ids = (
        "A0_primary_balanced_3x3",
        "A1_scgpt_seed_only",
        "A2_gears_seed_only",
        "A4_architecture_centroids",
    )
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for scenario_id in scenario_ids:
            take = tasks.loc[
                tasks.dataset.eq(dataset)
                & tasks.geometry.eq("absolute_rmse")
                & tasks.target_family_id.eq(scenario_id)
            ].reset_index(drop=True)
            if len(take) != int(DATASETS[dataset]["n_tasks"]):
                raise AuditFailure(f"{dataset} {scenario_id}: bootstrap input missing")
            outcomes = (
                ("family_rms_error", "own_family_rms"),
                ("a0_family_rms_error", "fixed_a0_family_rms"),
                ("a0_centroid_error", "fixed_a0_centroid_error"),
            )
            for outcome, outcome_label in outcomes:
                correlation = cluster_bootstrap_spearman(
                    take,
                    "diversity_lower_bound",
                    outcome,
                    stable_seed(
                        "E194", dataset, scenario_id, outcome_label, "spearman"
                    ),
                    n_boot=BOOTSTRAP_SPEARMAN,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "target_family_id": scenario_id,
                        "outcome": outcome_label,
                        "metric": "diversity_error_spearman",
                        "budget": np.nan,
                        "n_tasks": correlation["n_tasks"],
                        "n_gene_clusters": correlation["n_gene_clusters"],
                        "point": correlation["spearman"],
                        "ci95_lower": correlation["ci95_lower"],
                        "ci95_upper": correlation["ci95_upper"],
                        "bootstrap_valid": correlation["bootstrap_valid"],
                    }
                )
                utility = cluster_bootstrap_utility_20(
                    take,
                    "diversity_lower_bound",
                    outcome,
                    stable_seed(
                        "E194", dataset, scenario_id, outcome_label, "utility20"
                    ),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "target_family_id": scenario_id,
                        "outcome": outcome_label,
                        "metric": "oracle_normalized_utility",
                        "budget": UTILITY_BUDGET,
                        **utility,
                    }
                )
    return pd.DataFrame(rows)


def max_task_difference(
    tasks: pd.DataFrame,
    dataset: str,
    geometry: str,
    scenario_id: str,
    comparator_id: str = "A0_primary_balanced_3x3",
) -> float:
    columns = (
        "family_rms_error",
        "centroid_error",
        "worst_member_error",
        "diversity_lower_bound",
        "diameter_half_lower_bound",
    )
    left = tasks.loc[
        tasks.dataset.eq(dataset)
        & tasks.geometry.eq(geometry)
        & tasks.target_family_id.eq(scenario_id),
        ["task_id", *columns],
    ].sort_values("task_id")
    right = tasks.loc[
        tasks.dataset.eq(dataset)
        & tasks.geometry.eq(geometry)
        & tasks.target_family_id.eq(comparator_id),
        ["task_id", *columns],
    ].sort_values("task_id")
    if (
        len(left) != len(right)
        or left.task_id.astype(str).tolist() != right.task_id.astype(str).tolist()
    ):
        raise AuditFailure("invariant comparison task alignment failed")
    return float(
        np.max(
            np.abs(
                left[list(columns)].to_numpy(float)
                - right[list(columns)].to_numpy(float)
            )
        )
    )


def construction_invariants(
    dataset: str,
    geometry: str,
    scenarios: list[Scenario],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_count = 55 if geometry == "absolute_rmse" else 50
    rows.append(
        {
            "dataset": dataset,
            "geometry": geometry,
            "check": "scenario_count",
            "target_family_id": "ALL",
            "value": len(scenarios),
            "threshold": expected_count,
            "pass": len(scenarios) == expected_count,
        }
    )
    by_id = {item.scenario_id: item for item in scenarios}
    a0 = by_id["A0_primary_balanced_3x3"]
    a4 = by_id["A4_architecture_centroids"]
    a0_centroid = np.einsum("m,mng->ng", a0.weights, a0.entries)
    a4_centroid = np.einsum("m,mng->ng", a4.weights, a4.entries)
    value = float(np.max(np.abs(a0_centroid - a4_centroid)))
    rows.append(
        {
            "dataset": dataset,
            "geometry": geometry,
            "check": "a4_embedding_centroid_equals_a0",
            "target_family_id": a4.scenario_id,
            "value": value,
            "threshold": INVARIANT_TOL,
            "pass": value <= INVARIANT_TOL,
        }
    )
    for item in scenarios:
        finite_nonnegative = bool(
            np.isfinite(item.weights).all() and (item.weights >= 0).all()
        )
        weight_error = abs(float(item.weights.sum()) - 1.0)
        rows.append(
            {
                "dataset": dataset,
                "geometry": geometry,
                "check": "weights_finite_nonnegative_sum_one",
                "target_family_id": item.scenario_id,
                "value": weight_error if finite_nonnegative else float("inf"),
                "threshold": 1e-12,
                "pass": finite_nonnegative and weight_error <= 1e-12,
            }
        )
        if item.scenario_id == "A0_primary_balanced_3x3" or (
            "governed" in item.scenario_id
        ):
            scgpt_weight = sum(
                float(weight)
                for lineage, weight in zip(item.lineages, item.weights)
                if lineage.startswith("scGPT")
            )
            gears_weight = sum(
                float(weight)
                for lineage, weight in zip(item.lineages, item.weights)
                if lineage.startswith("GEARS")
            )
            balance_error = max(
                abs(scgpt_weight - 0.5), abs(gears_weight - 0.5)
            )
            rows.append(
                {
                    "dataset": dataset,
                    "geometry": geometry,
                    "check": "governed_architecture_weight_half_each",
                    "target_family_id": item.scenario_id,
                    "value": balance_error,
                    "threshold": 1e-12,
                    "pass": balance_error <= 1e-12,
                }
            )
    if geometry == "absolute_rmse":
        scgpt_centroid = a0.entries[np.asarray(SCGPT)].mean(axis=0)
        gears_centroid = a0.entries[np.asarray(GEARS)].mean(axis=0)
        architecture_difference = scgpt_centroid - gears_centroid
        a0_diversity_sq = np.einsum(
            "m,mn->n",
            a0.weights,
            np.sum((a0.entries - a0_centroid[None, :, :]) ** 2, axis=2),
        )
        for attack_lambda in (1, 2, 4):
            item = by_id[f"C4_symmetric_attack_lambda{attack_lambda}"]
            centroid = np.einsum("m,mng->ng", item.weights, item.entries)
            centroid_difference = float(
                np.max(np.abs(centroid - a0_centroid))
            )
            rows.append(
                {
                    "dataset": dataset,
                    "geometry": geometry,
                    "check": "symmetric_attack_embedding_centroid_unchanged",
                    "target_family_id": item.scenario_id,
                    "value": centroid_difference,
                    "threshold": INVARIANT_TOL,
                    "pass": centroid_difference <= INVARIANT_TOL,
                }
            )
            observed_diversity_sq = np.einsum(
                "m,mn->n",
                item.weights,
                np.sum(
                    (item.entries - centroid[None, :, :]) ** 2,
                    axis=2,
                ),
            )
            expected_diversity_sq = (
                0.75 * a0_diversity_sq
                + 0.25
                * attack_lambda**2
                * np.sum(architecture_difference**2, axis=1)
            )
            formula_error = float(
                np.max(
                    np.abs(
                        observed_diversity_sq - expected_diversity_sq
                    )
                )
            )
            rows.append(
                {
                    "dataset": dataset,
                    "geometry": geometry,
                    "check": "symmetric_attack_diversity_formula",
                    "target_family_id": item.scenario_id,
                    "value": formula_error,
                    "threshold": INVARIANT_TOL,
                    "pass": formula_error <= INVARIANT_TOL,
                }
            )
    else:
        c4_count = sum(
            item.scenario_id.startswith("C4_") for item in scenarios
        )
        rows.append(
            {
                "dataset": dataset,
                "geometry": geometry,
                "check": "no_directional_symmetric_attack",
                "target_family_id": "C4",
                "value": c4_count,
                "threshold": 0,
                "pass": c4_count == 0,
            }
        )
    return rows


def invariant_audit(
    tasks: pd.DataFrame,
    members: pd.DataFrame,
    construction_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    exact_scenarios = (
        ("B1_duplicate_all_flat", "duplicate_all_flat"),
        ("B1_duplicate_all_governed", "duplicate_all_governed"),
        ("B3_overweight_scgpt_governed", "overweight_scgpt_governed"),
        ("B3_overweight_gears_governed", "overweight_gears_governed"),
    )
    for dataset in DATASETS:
        for geometry in GEOMETRIES:
            for scenario_id, check in exact_scenarios:
                value = max_task_difference(tasks, dataset, geometry, scenario_id)
                rows.append(
                    {
                        "dataset": dataset,
                        "geometry": geometry,
                        "check": check,
                        "target_family_id": scenario_id,
                        "value": value,
                        "threshold": INVARIANT_TOL,
                        "pass": value <= INVARIANT_TOL,
                    }
                )
            for key in MODEL_KEYS:
                scenario_id = f"B2_duplicate_one_governed_{key}"
                value = max_task_difference(tasks, dataset, geometry, scenario_id)
                rows.append(
                    {
                        "dataset": dataset,
                        "geometry": geometry,
                        "check": "duplicate_one_governed",
                        "target_family_id": scenario_id,
                        "value": value,
                        "threshold": INVARIANT_TOL,
                        "pass": value <= INVARIANT_TOL,
                    }
                )
            a0 = tasks.loc[
                tasks.dataset.eq(dataset)
                & tasks.geometry.eq(geometry)
                & tasks.target_family_id.eq("A0_primary_balanced_3x3")
            ].sort_values("task_id")
            a1 = tasks.loc[
                tasks.dataset.eq(dataset)
                & tasks.geometry.eq(geometry)
                & tasks.target_family_id.eq("A1_scgpt_seed_only")
            ].sort_values("task_id")
            a2 = tasks.loc[
                tasks.dataset.eq(dataset)
                & tasks.geometry.eq(geometry)
                & tasks.target_family_id.eq("A2_gears_seed_only")
            ].sort_values("task_id")
            a4 = tasks.loc[
                tasks.dataset.eq(dataset)
                & tasks.geometry.eq(geometry)
                & tasks.target_family_id.eq("A4_architecture_centroids")
            ].sort_values("task_id")
            family_decomposition = float(
                np.max(
                    np.abs(
                        a0.family_rms_error.to_numpy(float) ** 2
                        - 0.5 * a1.family_rms_error.to_numpy(float) ** 2
                        - 0.5 * a2.family_rms_error.to_numpy(float) ** 2
                    )
                )
            )
            diversity_decomposition = float(
                np.max(
                    np.abs(
                        a0.diversity_lower_bound.to_numpy(float) ** 2
                        - 0.5
                        * a1.diversity_lower_bound.to_numpy(float) ** 2
                        - 0.5
                        * a2.diversity_lower_bound.to_numpy(float) ** 2
                        - a4.diversity_lower_bound.to_numpy(float) ** 2
                    )
                )
            )
            for check, value in (
                ("a0_family_error_architecture_decomposition", family_decomposition),
                ("a0_diversity_within_between_decomposition", diversity_decomposition),
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "geometry": geometry,
                        "check": check,
                        "target_family_id": "A0_A1_A2_A4",
                        "value": value,
                        "threshold": INVARIANT_TOL,
                        "pass": value <= INVARIANT_TOL,
                    }
                )
            a4_centroid_error = float(
                np.max(
                    np.abs(
                        a4.centroid_error.to_numpy(float)
                        - a0.centroid_error.to_numpy(float)
                    )
                )
            )
            rows.append(
                {
                    "dataset": dataset,
                    "geometry": geometry,
                    "check": "a4_centroid_error_equals_a0",
                    "target_family_id": "A4_architecture_centroids",
                    "value": a4_centroid_error,
                    "threshold": INVARIANT_TOL,
                    "pass": a4_centroid_error <= INVARIANT_TOL,
                }
            )
            if geometry == "absolute_rmse":
                attack_means = []
                for attack_lambda in (1, 2, 4):
                    scenario_id = f"C4_symmetric_attack_lambda{attack_lambda}"
                    attack = tasks.loc[
                        tasks.dataset.eq(dataset)
                        & tasks.geometry.eq(geometry)
                        & tasks.target_family_id.eq(scenario_id)
                    ].sort_values("task_id")
                    value = float(
                        np.max(
                            np.abs(
                                attack.centroid_error.to_numpy(float)
                                - a0.centroid_error.to_numpy(float)
                            )
                        )
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "geometry": geometry,
                            "check": "symmetric_attack_centroid_error_unchanged",
                            "target_family_id": scenario_id,
                            "value": value,
                            "threshold": INVARIANT_TOL,
                            "pass": value <= INVARIANT_TOL,
                        }
                    )
                    attack_means.append(
                        float(attack.diversity_lower_bound.mean())
                    )
                monotonic = bool(
                    attack_means[0] < attack_means[1] < attack_means[2]
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "geometry": geometry,
                        "check": "symmetric_attack_diversity_strictly_increases",
                        "target_family_id": "C4_lambda1_to_4",
                        "value": float(min(np.diff(attack_means))),
                        "threshold": 0.0,
                        "pass": monotonic,
                    }
                )

    e193 = pd.read_csv(
        RESULTS_ROOT
        / "E193_multigeometry_certificate_robustness_20260729"
        / "tables/E193_TASK_METRICS.csv",
        keep_default_na=False,
    )
    e193_columns = {
        "family_rms_error": "family_rms_error",
        "family_worst_error": "worst_member_error",
        "centroid_error": "centroid_error",
        "diversity_lower_bound": "diversity_lower_bound",
        "diameter_half_lower_bound": "diameter_half_lower_bound",
    }
    for dataset in DATASETS:
        for geometry in GEOMETRIES:
            observed = tasks.loc[
                tasks.dataset.eq(dataset)
                & tasks.geometry.eq(geometry)
                & tasks.target_family_id.eq("A0_primary_balanced_3x3")
            ].sort_values("task_id")
            reference = e193.loc[
                e193.dataset.eq(dataset)
                & e193.geometry.eq(geometry)
                & e193.certificate_valid.astype(str).str.lower().eq("true")
            ].sort_values("task_id")
            if (
                observed.task_id.astype(str).tolist()
                != reference.task_id.astype(str).tolist()
            ):
                raise AuditFailure("E193 replication task alignment failed")
            differences = []
            for reference_column, observed_column in e193_columns.items():
                differences.append(
                    np.max(
                        np.abs(
                            reference[reference_column].to_numpy(float)
                            - observed[observed_column].to_numpy(float)
                        )
                    )
                )
            value = float(np.max(differences))
            rows.append(
                {
                    "dataset": dataset,
                    "geometry": geometry,
                    "check": "a0_reproduces_e193",
                    "target_family_id": "A0_primary_balanced_3x3",
                    "value": value,
                    "threshold": INVARIANT_TOL,
                    "pass": value <= INVARIANT_TOL,
                }
            )

    for dataset in DATASETS:
        base = members.loc[
            members.dataset.eq(dataset)
            & members.geometry.eq("absolute_rmse")
            & members.target_family_id.eq("A0_primary_balanced_3x3")
        ]
        hashes = base.prediction_array_sha256.astype(str)
        unique_hashes = hashes.ne("").all() and hashes.nunique() == len(MODEL_KEYS)
        unique_lineages = base.lineage_id.nunique() == len(MODEL_KEYS)
        rows.append(
            {
                "dataset": dataset,
                "geometry": "all",
                "check": "base_prediction_hash_and_lineage_unique",
                "target_family_id": "A0_primary_balanced_3x3",
                "value": min(hashes.nunique(), base.lineage_id.nunique()),
                "threshold": len(MODEL_KEYS),
                "pass": bool(unique_hashes and unique_lineages),
            }
        )
    rows.extend(construction_rows)
    result = pd.DataFrame(rows)
    if result["pass"].isna().any():
        raise AuditFailure("invariant audit contains missing decisions")
    return result


def make_figure(summary: pd.DataFrame, tasks: pd.DataFrame) -> None:
    dataset_labels = {"E190_K562": "K562", "E192_RPE1": "RPE1"}
    colors = {"E190_K562": "#3C5488", "E192_RPE1": "#00A087"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    selected = (
        "A0_primary_balanced_3x3",
        "A1_scgpt_seed_only",
        "A2_gears_seed_only",
        "A4_architecture_centroids",
    )
    labels = ("A0\n3+3", "A1\nscGPT", "A2\nGEARS", "A4\ncentroids")
    x = np.arange(len(selected))
    width = 0.34
    for offset, dataset in enumerate(DATASETS):
        take = (
            summary.loc[
                summary.dataset.eq(dataset)
                & summary.geometry.eq("absolute_rmse")
                & summary.target_family_id.isin(selected)
            ]
            .set_index("target_family_id")
            .loc[list(selected)]
        )
        axes[0].bar(
            x + (offset - 0.5) * width,
            take.median_diversity_sq_over_family_sq,
            width,
            color=colors[dataset],
            label=dataset_labels[dataset],
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Median $D^2/R^2$")
    axes[0].set_title("A  Certificate tightness is family-specific")
    axes[0].legend(frameon=False, fontsize=8)

    stress_ids = (
        "B2_duplicate_one_flat_scGPT_seed3407",
        "B2_duplicate_one_governed_scGPT_seed3407",
        "B3_overweight_scgpt_flat",
        "B3_overweight_scgpt_governed",
    )
    stress_labels = ("one copy\nflat", "one copy\ngoverned", "3 copies\nflat", "3 copies\ngoverned")
    for offset, dataset in enumerate(DATASETS):
        take = (
            summary.loc[
                summary.dataset.eq(dataset)
                & summary.geometry.eq("absolute_rmse")
                & summary.target_family_id.isin(stress_ids)
            ]
            .set_index("target_family_id")
            .loc[list(stress_ids)]
        )
        axes[1].bar(
            x + (offset - 0.5) * width,
            100.0 * take.relative_diversity_change_vs_a0,
            width,
            color=colors[dataset],
        )
    axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(x, stress_labels)
    axes[1].set_ylabel("Mean diversity change vs A0 (%)")
    axes[1].set_title("B  Lineage weighting cancels duplication")

    for dataset in DATASETS:
        values = []
        centroid_changes = []
        for attack_lambda in (1, 2, 4):
            take = tasks.loc[
                tasks.dataset.eq(dataset)
                & tasks.geometry.eq("absolute_rmse")
                & tasks.target_family_id.eq(
                    f"C4_symmetric_attack_lambda{attack_lambda}"
                )
            ]
            a0 = tasks.loc[
                tasks.dataset.eq(dataset)
                & tasks.geometry.eq("absolute_rmse")
                & tasks.target_family_id.eq("A0_primary_balanced_3x3")
            ]
            values.append(float(take.diversity_lower_bound.mean()))
            centroid_changes.append(
                float(
                    np.max(
                        np.abs(
                            take.centroid_error.to_numpy(float)
                            - a0.centroid_error.to_numpy(float)
                        )
                    )
                )
            )
        axes[2].plot(
            (1, 2, 4),
            values,
            marker="o",
            color=colors[dataset],
            label=dataset_labels[dataset],
        )
        if max(centroid_changes) > INVARIANT_TOL:
            raise AuditFailure("figure audit found changed C4 centroid")
    axes[2].set_xlabel("Symmetric attack λ")
    axes[2].set_ylabel("Mean diversity")
    axes[2].set_xticks((1, 2, 4))
    axes[2].set_title("C  Diversity can be inflated at fixed centroid")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "E194_family_governance_stress.png", dpi=300)
    fig.savefig(FIGURES / "E194_family_governance_stress.pdf")
    plt.close(fig)


def write_reports(
    status: dict[str, Any],
    summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    invariants: pd.DataFrame,
) -> None:
    raw = summary.loc[summary.geometry.eq("absolute_rmse")].copy()
    key_ids = (
        "A0_primary_balanced_3x3",
        "A1_scgpt_seed_only",
        "A2_gears_seed_only",
        "A4_architecture_centroids",
        "B3_overweight_scgpt_flat",
        "B3_overweight_scgpt_governed",
        "C2_add_source_portfolio",
        "C4_symmetric_attack_lambda4",
    )
    key = raw.loc[raw.target_family_id.isin(key_ids), [
        "dataset",
        "target_family_id",
        "mean_family_rms_error",
        "mean_diversity_lower_bound",
        "median_diversity_sq_over_family_sq",
        "diversity_error_spearman",
        "diversity_a0_family_error_spearman",
        "diversity_a0_centroid_error_spearman",
        "oracle_normalized_utility_at_20pct",
        "a0_family_oracle_normalized_utility_at_20pct",
        "relative_diversity_change_vs_a0",
    ]]
    boot_view = bootstrap[[
        "dataset",
        "target_family_id",
        "outcome",
        "metric",
        "point",
        "ci95_lower",
        "ci95_upper",
    ]]
    report = f"""# E194 注册家族构成与治理压力测试报告

状态：**{status["status"]}**

证据标签：`POSTTRUTH_FAMILY_GOVERNANCE_STRESS`。E194 复用已打开的 E190/E192
真值，不能算新的独立确认。

## 运行范围

- 数据：E190 K562 {status["datasets"]["E190_K562"]["n_tasks"]} 个任务；
  E192 RPE1 {status["datasets"]["E192_RPE1"]["n_tasks"]} 个任务；
- 几何：absolute RMSE、cosine、Pearson；
- 主 family：预真值冻结的 3 个 scGPT + 3 个 GEARS；
- 共评估 {status["n_scenario_summaries"]} 个
  `dataset×geometry×scenario`，逐任务记录
  {status["n_task_rows"]} 行。

## 确认性实现检查

- family lower-bound violations：{status["family_lower_violations"]}；
- worst-member lower-bound violations：{status["worst_lower_violations"]}；
- 最大平方恒等式残差：{status["max_identity_residual"]:.3e}；
- governance / C4 不变量检查：
  {status["invariant_passed"]}/{status["invariant_total"]} 通过。

这些检查证明代码对每个声明 family 正确实现了证书；它们不证明任意 family 都有
同样好的经验排序。

`diversity_error_spearman` 以自身 family RMS 为结果，其中含有确定性的平方结构
耦合。表中同时给出固定 A0 family RMS 与 A0 centroid error，跨 family 解释以
固定结果列为准。

## absolute RMSE 关键场景

{markdown_table(key)}

## 基因整簇 bootstrap

{markdown_table(boot_view)}

## 结果边界

1. A0、A1、A2、A4 是不同预测对象。某个子 family 的高相关不能替代 A0；
2. governed duplicate 场景恢复 A0，说明 lineage/架构权重合同能阻止复制成员增加
   话语权；
3. flat duplication 与 leave-one-out 量化 family 构成敏感性，必须如实保留；
4. absolute RMSE 的 C4 在 A0 质心误差不变时放大 diversity，直接否定“成员
   越多、分歧越大，证据越强”的写法；方向几何不创建离开球面的伪预测；
5. zero/source/synthetic 成员改变了 target family，只能作为负控或 portfolio
   分析，不能进入主证书。

完整组合范围见 `tables/E194_GROUP_RANGE_SUMMARY.csv`，逐任务证据见压缩 CSV。
"""
    interpretation = f"""# E194 解释与方法修正

## 可以保留的结论

对任何事先冻结、权重非负且和为 1 的有限 family，平方损失恒等式与两个确定性
下界成立。E194 在两个 target、三种几何和全部压力场景中得到
{status["family_lower_violations"]} 个 family 下界违例、
{status["worst_lower_violations"]} 个 worst 下界违例。

## 必须收紧的结论

证书的对象是“声明 family 的平均误差”，并非抽象的模型风险。成员增删、架构比例
和权重都会改变目标。E194 的 C4 负控还表明：加入关于原质心对称的合成成员，可以
保持质心预测不变，同时任意推高 family diversity。因此，论文不能把 diversity
大小本身解释为跨 family 可比较的证据强度。

## 固定治理规则

- 主 family 只保留 E190/E192 预真值锁定的真实输出；
- scGPT 与 GEARS 各占 1/2；架构内三个唯一 seed lineage 等权；
- 相同预测哈希或 lineage 的副本拆分原权重，不新增权重；
- 不按目标真值、相关性或 top-k 收益挑成员；
- zero、source、架构质心与合成攻击成员只用于诊断；
- 新模型只能形成带版本号的新 target family，重新冻结并重新评估，不能悄悄并入
  现有 A0。

## 对投稿主张的影响

SafeConf 的可辩护贡献应写成：

> 面向满足同一冻结输出合同的预注册加权预测器 family，仅依赖预测输出提供
> post-hoc、pre-truth 可计算的确定性误差下界，并配套 fail-closed 的成员治理
> 与外部转移审计。

不能写成“任意模型集合都能靠分歧可靠识别错误”，也不能写成“首次为单细胞扰动
预测提供不确定性或 conformal 保证”。E193/E194 支持的是几何稳健的数学证书与
治理边界；E192 已显示经验路由会随 target 和 metric 失效。
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "E194_REPORT.md").write_text(report, encoding="utf-8")
    (REPORTS / "E194_INTERPRETATION.md").write_text(
        interpretation, encoding="utf-8"
    )


def main() -> None:
    for path in (TABLES, REPORTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)

    all_tasks: list[pd.DataFrame] = []
    all_hashes: list[dict[str, Any]] = []
    all_members: list[dict[str, Any]] = []
    all_construction_invariants: list[dict[str, Any]] = []
    dataset_status: dict[str, Any] = {}
    for dataset, config in DATASETS.items():
        query, raw_predictions, raw_truth, raw_source, _, hashes = load_dataset(
            dataset, config
        )
        all_hashes.extend(hashes)
        base_hashes = {
            key: array_sha256(raw_predictions[index])
            for index, key in enumerate(MODEL_KEYS)
        }
        if len(set(base_hashes.values())) != len(MODEL_KEYS):
            raise AuditFailure(f"{dataset}: duplicate base prediction arrays")
        for key, digest in base_hashes.items():
            all_hashes.append(
                {
                    "dataset": dataset,
                    "namespace": "prediction_array_content",
                    "path": key,
                    "bytes": int(raw_predictions[MODEL_KEYS.index(key)].nbytes),
                    "sha256": digest,
                }
            )
        dataset_status[dataset] = {
            "n_tasks": len(query),
            "n_gene_clusters": int(query.gene.nunique()),
            "target": config["target"],
        }
        for geometry in GEOMETRIES:
            prediction_flat, prediction_valid, _ = embed_rows(
                raw_predictions.reshape(-1, N_GENES), geometry
            )
            predictions = prediction_flat.reshape(raw_predictions.shape)
            prediction_valid = prediction_valid.reshape(raw_predictions.shape[:2])
            target, target_valid, _ = embed_rows(raw_truth, geometry)
            source, source_valid, _ = embed_rows(raw_source, geometry)
            valid = target_valid & prediction_valid.all(axis=0) & source_valid
            if not valid.all():
                invalid = query.loc[~valid, "task_id"].astype(str).tolist()
                raise AuditFailure(
                    f"{dataset} {geometry}: invalid frozen inputs {invalid[:5]}"
                )
            scenarios = build_scenarios(geometry, predictions, source)
            expected = 55 if geometry == "absolute_rmse" else 50
            if len(scenarios) != expected:
                raise AuditFailure(
                    f"{dataset} {geometry}: expected {expected} scenarios, "
                    f"found {len(scenarios)}"
                )
            all_construction_invariants.extend(
                construction_invariants(dataset, geometry, scenarios)
            )
            for item in scenarios:
                all_members.extend(
                    family_member_rows(
                        dataset, geometry, item, base_hashes
                    )
                )
                all_tasks.append(
                    family_metrics(dataset, geometry, query, target, item)
                )

    tasks = attach_a0_targets(pd.concat(all_tasks, ignore_index=True))
    if tasks.duplicated(
        ["dataset", "geometry", "target_family_id", "task_id"]
    ).any():
        raise AuditFailure("duplicate dataset-geometry-family-task record")
    members = pd.DataFrame(all_members)
    summaries: list[dict[str, Any]] = []
    for (dataset, geometry), block in tasks.groupby(
        ["dataset", "geometry"], sort=False, observed=True
    ):
        a0 = block.loc[
            block.target_family_id.eq("A0_primary_balanced_3x3")
        ].reset_index(drop=True)
        for _, family in block.groupby(
            "target_family_id", sort=False, observed=True
        ):
            summaries.append(
                summarize_scenario(family.reset_index(drop=True), a0)
            )
    summary = pd.DataFrame(summaries)
    group_summary = group_ranges(summary)
    bootstrap = bootstrap_key_scenarios(tasks)
    invariants = invariant_audit(
        tasks, members, all_construction_invariants
    )

    family_violations = int(tasks.family_lower_violation.sum())
    worst_violations = int(tasks.worst_lower_violation.sum())
    max_residual = float(tasks.family_identity_residual.max())
    invariant_passed = int(invariants["pass"].sum())
    status_name = (
        "PASS"
        if (
            family_violations == 0
            and worst_violations == 0
            and max_residual <= IDENTITY_TOL
            and invariant_passed == len(invariants)
        )
        else "FAIL"
    )
    status = {
        "experiment": "E194",
        "stage": "POSTTRUTH_FAMILY_GOVERNANCE_STRESS",
        "status": status_name,
        "frozen_analysis": "ANALYSIS_FREEZE.md",
        "datasets": dataset_status,
        "geometries": list(GEOMETRIES),
        "primary_family": "A0_primary_balanced_3x3",
        "n_task_rows": len(tasks),
        "n_scenario_summaries": len(summary),
        "n_family_member_rows": len(members),
        "family_lower_violations": family_violations,
        "worst_lower_violations": worst_violations,
        "max_identity_residual": max_residual,
        "invariant_passed": invariant_passed,
        "invariant_total": len(invariants),
        "thresholds": {
            "lower_bound": LOWER_TOL,
            "identity": IDENTITY_TOL,
            "invariant": INVARIANT_TOL,
        },
        "bootstrap": {
            "unit": "gene_cluster",
            "spearman_replicates": BOOTSTRAP_SPEARMAN,
            "utility_replicates": BOOTSTRAP_UTILITY,
            "utility_budget": UTILITY_BUDGET,
        },
        "claim_boundary": (
            "The deterministic certificate is family-conditional. "
            "Family construction and weighting are part of the estimand."
        ),
    }

    tasks.to_csv(
        TABLES / "E194_SCENARIO_TASK_METRICS.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    summary.to_csv(TABLES / "E194_SCENARIO_SUMMARY.csv", index=False)
    group_summary.to_csv(TABLES / "E194_GROUP_RANGE_SUMMARY.csv", index=False)
    bootstrap.to_csv(TABLES / "E194_BOOTSTRAP_SUMMARY.csv", index=False)
    invariants.to_csv(TABLES / "E194_INVARIANT_AUDIT.csv", index=False)
    members.to_csv(TABLES / "E194_FAMILY_MEMBER_AUDIT.csv", index=False)
    pd.DataFrame(all_hashes).drop_duplicates().sort_values(
        ["dataset", "namespace", "path"]
    ).to_csv(TABLES / "E194_INPUT_HASHES.csv", index=False)
    (OUT / "E194_STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    make_figure(summary, tasks)
    write_reports(status, summary, group_summary, bootstrap, invariants)
    if status_name != "PASS":
        raise AuditFailure(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
