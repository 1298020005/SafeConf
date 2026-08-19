#!/usr/bin/env python3
"""E8b association between frozen SafeConf task risk and benchmark errors.

The CLI separates identifier/coverage audits from association computation.
The ``dry-audit`` command never computes a correlation, bootstrap interval, or
permutation result. This makes the preregistered pause before the primary
analysis enforceable in code rather than relying on operator discipline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from safetrans_confidence.scoring.protocol_v0_2 import (
    PRIMARY_SCORE_NAME,
    build_protocol_v0_2_primary_scores,
    zscore_by_ref,
)

REFERENCE_PREDICTORS = ("V0StrongBaseline", "ContextSimBaseline")
FROZEN_COMPONENTS = (
    "context_similarity_max",
    "perturbation_support_count",
    "model_disagreement_rmse",
)
SCIPLEX3_DATASETS = ("sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7")
SCIPLEX3_CONTEXTS = {
    "sciplex3_A549": "A549",
    "sciplex3_K562": "K562",
    "sciplex3_MCF7": "MCF7",
}
SAFECONF_SCIPLEX3_CONTEXTS = {
    "alveolar basal epithelial cells": "A549",
    "lymphoblasts": "K562",
    "mammary epithelial cells": "MCF7",
}

# Explicit proposals are retained for Claude's audit only. Because there are
# more than ten manual rows, preregistration excludes all of them from the
# eventual sciplex3 sensitivity analysis.
SCIPLEX3_MANUAL_PROPOSALS = {
    "AZ": "AZ 960",
    "Alendronate": "Alendronate sodium trihydrate",
    "Alvespimycin": "Alvespimycin (17-DMAG) HCl",
    "Epothilone": "Epothilone A",
    "Fasudil": "Fasudil (HA-1077) HCl",
    "Flavopiridol": "Flavopiridol HCl",
    "GSK": "GSK J1",
    "NVPBSK805": "NVP-BSK805 2HCl",
    "Obatoclax": "Obatoclax Mesylate (GX15-070)",
    "Quisinostat": "Quisinostat (JNJ-26481585) 2HCl",
    "Rucaparib": "Rucaparib (AG-014699,PF-01367338) phosphate",
    "SRT1720": "SRT1720 HCl",
    "Tofacitinib": "Tofacitinib (CP-690550) Citrate",
    "Triamcinolone": "Triamcinolone Acetonide",
    "Tubastatin": "Tubastatin A HCl",
}


def _normalize_name(value: str) -> str:
    """Normalize formatting that scPerturBench removes from drug names."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def _strip_parenthetical_aliases(value: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(value)).strip()


def _extract_sciplex3_drug(perturbation: str) -> tuple[str, str, str]:
    text = str(perturbation)
    match = re.match(r"^(A549|K562|MCF7)_(.+)_([^_]+)$", text)
    if not match:
        raise ValueError(f"Unexpected sciplex3 perturbation identifier: {text}")
    return match.group(1), match.group(2), match.group(3)


def build_sciplex3_alias_table(
    safeconf_perturbations: Iterable[str],
    benchmark_perturbations: Iterable[str],
) -> pd.DataFrame:
    """Create an explicit benchmark-drug to SafeConf-drug mapping audit."""
    safe_names = sorted({str(value).strip() for value in safeconf_perturbations if pd.notna(value)})
    benchmark_drugs = sorted(
        {_extract_sciplex3_drug(value)[1] for value in benchmark_perturbations if pd.notna(value)}
    )

    exact_lookup: dict[str, list[str]] = {}
    alias_lookup: dict[str, list[str]] = {}
    for safe_name in safe_names:
        exact_lookup.setdefault(_normalize_name(safe_name), []).append(safe_name)
        stripped = _strip_parenthetical_aliases(safe_name)
        alias_lookup.setdefault(_normalize_name(stripped), []).append(safe_name)

    rows: list[dict] = []
    for benchmark_drug in benchmark_drugs:
        normalized = _normalize_name(benchmark_drug)
        exact = exact_lookup.get(normalized, [])
        alias = alias_lookup.get(normalized, [])
        safe_name = ""
        match_type = "unmatched"
        notes = "no deterministic match"
        if len(exact) == 1:
            safe_name = exact[0]
            match_type = "exact"
            notes = "case/spacing/punctuation normalization only"
        elif len(alias) == 1:
            safe_name = alias[0]
            match_type = "alias"
            notes = "matched after removing SafeConf parenthetical aliases"
        else:
            proposal = SCIPLEX3_MANUAL_PROPOSALS.get(benchmark_drug, "")
            if proposal in safe_names:
                safe_name = proposal
                match_type = "manual"
                notes = "manual proposal; excluded because manual count exceeds preregistered limit"
        rows.append(
            {
                "safeconf_perturbation": safe_name,
                "scperturbench_drug": benchmark_drug,
                "match_type": match_type,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def load_benchmark_errors(
    csv_path: Path,
    dataset: str,
    metric: str,
    deg: int,
) -> pd.DataFrame:
    """Load task-level benchmark errors without aggregating available seeds."""
    usecols = [
        "performance",
        "metric",
        "DataSet",
        "method",
        "perturb",
        "DEG",
        "Nstimulated",
        "seed",
    ]
    frame = pd.read_csv(csv_path, usecols=usecols)
    if dataset == "sciplex3":
        frame = frame[frame["DataSet"].isin(SCIPLEX3_DATASETS)].copy()
        parsed = frame["perturb"].map(_extract_sciplex3_drug)
        frame["context"] = parsed.map(lambda value: value[0])
        frame["perturbation"] = parsed.map(lambda value: value[1])
        frame["dose"] = parsed.map(lambda value: value[2])
    else:
        frame = frame[frame["DataSet"].astype(str).eq(dataset)].copy()
        frame["perturbation"] = frame["perturb"].astype(str)
    frame = frame[
        frame["metric"].astype(str).eq(metric)
        & pd.to_numeric(frame["DEG"], errors="coerce").eq(int(deg))
    ].copy()
    frame = frame.rename(columns={"performance": "error"})
    return frame.drop(columns=["perturb"])


def aggregate_benchmark_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    """Median error across the benchmark seeds available for each task."""
    group_cols = ["method", "perturbation"]
    for optional in ("context", "dose"):
        if optional in frame.columns:
            group_cols.append(optional)
    aggregated = (
        frame.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            error=("error", "median"),
            n_available_seeds=("seed", "nunique"),
            Nstimulated=("Nstimulated", "median"),
        )
    )
    return aggregated


def compute_safeconf_risk_by_context(
    feature_matrix: pd.DataFrame,
    dataset: str,
    formula: str | None = None,
) -> pd.DataFrame:
    """Compute frozen risk through the preregistered context-level aggregation.

    ``formula`` is accepted only as an audit assertion; the scorer itself
    selects the frozen family from dataset ontology.
    """
    base = feature_matrix[
        feature_matrix["dataset_name"].astype(str).eq(dataset)
        & feature_matrix["predictor_name"].astype(str).isin(REFERENCE_PREDICTORS)
    ].copy()
    if base.empty:
        raise ValueError(f"No SafeConf rows found for dataset {dataset}")
    required = {
        "record_id",
        "dataset_name",
        "dataset_family",
        "fold_id",
        "split",
        "context",
        "perturbation",
        "predictor_name",
        *FROZEN_COMPONENTS,
    }
    missing = sorted(required - set(base.columns))
    if missing:
        raise ValueError(f"Missing SafeConf columns: {missing}")
    if formula is not None:
        expected = str(base["dataset_family"].iloc[0])
        if str(formula) != expected:
            raise ValueError(f"Requested formula {formula!r}, dataset ontology requires {expected!r}")

    scores, _ = build_protocol_v0_2_primary_scores(base)
    scores = scores[
        scores["score_name"].astype(str).eq(PRIMARY_SCORE_NAME)
        & scores["split"].astype(str).isin(["train", "val"])
    ].copy()
    if scores.empty:
        raise ValueError("No train/val frozen scores available after scoring")

    per_fold = (
        scores.groupby(["fold_id", "context", "perturbation"], as_index=False)["score_value"]
        .median()
    )
    per_context = (
        per_fold.groupby(["context", "perturbation"], as_index=False)["score_value"]
        .median()
    )
    per_context["risk"] = -pd.to_numeric(per_context["score_value"], errors="coerce")
    return per_context.drop(columns=["score_value"]).sort_values(
        ["context", "perturbation"]
    ).reset_index(drop=True)


def compute_safeconf_risk_per_perturbation(
    feature_matrix: pd.DataFrame,
    dataset: str,
    formula: str | None = None,
) -> pd.Series:
    """Compute frozen risk and aggregate across contexts to perturbations."""
    per_context = compute_safeconf_risk_by_context(feature_matrix, dataset, formula)
    risk = per_context.groupby("perturbation")["risk"].median()
    risk.name = "risk"
    return risk.sort_index()


def _aggregate_component_risk(
    base: pd.DataFrame,
    feature: str,
    sign: float,
) -> pd.Series:
    rows: list[pd.DataFrame] = []
    for (_, fold, predictor), group in base.groupby(
        ["dataset_name", "fold_id", "predictor_name"],
        sort=False,
    ):
        train = group[group["split"].astype(str).eq("train")]
        if train.empty:
            raise ValueError(f"Missing train reference rows for fold={fold}, predictor={predictor}")
        values = pd.to_numeric(group[feature], errors="coerce")
        refs = pd.to_numeric(train[feature], errors="coerce")
        if feature == "perturbation_support_count":
            values = np.log1p(values)
            refs = np.log1p(refs)
        scored = group[["fold_id", "split", "context", "perturbation", "predictor_name"]].copy()
        scored["risk"] = sign * zscore_by_ref(values, refs)
        rows.append(scored)
    scores = pd.concat(rows, ignore_index=True)
    scores = scores[scores["split"].astype(str).isin(["train", "val"])]
    per_fold = (
        scores.groupby(["fold_id", "context", "perturbation"], as_index=False)["risk"].median()
    )
    per_context = (
        per_fold.groupby(["context", "perturbation"], as_index=False)["risk"].median()
    )
    result = per_context.groupby("perturbation")["risk"].median().sort_index()
    return result


def compute_nuisance_baselines(
    feature_matrix: pd.DataFrame,
    dataset: str,
    nstimulated: pd.Series | None = None,
) -> dict[str, pd.Series]:
    base = feature_matrix[
        feature_matrix["dataset_name"].astype(str).eq(dataset)
        & feature_matrix["predictor_name"].astype(str).isin(REFERENCE_PREDICTORS)
    ].copy()
    controls = {
        "context_similarity_only": _aggregate_component_risk(
            base, "context_similarity_max", -1.0
        ),
        "support_only": _aggregate_component_risk(
            base, "perturbation_support_count", -1.0
        ),
        "disagreement_only": _aggregate_component_risk(
            base, "model_disagreement_rmse", 1.0
        ),
    }
    if nstimulated is not None:
        sample_size = pd.to_numeric(nstimulated, errors="coerce")
        controls["sample_size"] = -np.log1p(sample_size)
        controls["sample_size"].name = "risk"
    return controls


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return float("nan"), float("nan")
    if np.unique(x[finite]).size < 2 or np.unique(y[finite]).size < 2:
        return float("nan"), float("nan")
    result = spearmanr(x[finite], y[finite])
    return float(result.statistic), float(result.pvalue)


def _per_method_association(risk: pd.Series, errors: pd.DataFrame) -> pd.DataFrame:
    risk_frame = risk.rename("risk").rename_axis("perturbation").reset_index()
    joined = errors.merge(risk_frame, on="perturbation", how="inner")
    rows = []
    for method, group in joined.groupby("method"):
        rho, p_value = _spearman(
            pd.to_numeric(group["risk"], errors="coerce").to_numpy(),
            pd.to_numeric(group["error"], errors="coerce").to_numpy(),
        )
        rows.append(
            {
                "method_name": method,
                "n_perturbations": int(group["perturbation"].nunique()),
                "spearman_rho": rho,
                "p_raw": p_value,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["bh_q"] = _bh_adjust(result["p_raw"])
    return result.sort_values("method_name").reset_index(drop=True)


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return pd.Series(adjusted, index=p_values.index)
    order = finite_idx[np.argsort(values[finite_idx])]
    ranked = values[order] * finite_idx.size / np.arange(1, finite_idx.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def _bootstrap_median_spearman(
    risk: pd.Series,
    errors: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    common = sorted(set(risk.index.astype(str)) & set(errors["perturbation"].astype(str)))
    if len(common) < 3:
        return np.full(n_bootstrap, np.nan)
    risk_lookup = risk.to_dict()
    method_lookup = {
        method: group.set_index("perturbation")["error"].to_dict()
        for method, group in errors.groupby("method")
    }
    rng = np.random.default_rng(seed)
    values = np.full(n_bootstrap, np.nan, dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.choice(common, size=len(common), replace=True)
        rhos = []
        sampled_risk = np.asarray([risk_lookup[name] for name in sampled], dtype=float)
        for lookup in method_lookup.values():
            available = np.asarray([name in lookup for name in sampled])
            if int(available.sum()) < 3:
                continue
            sampled_error = np.asarray(
                [lookup.get(name, np.nan) for name in sampled], dtype=float
            )
            rho, _ = _spearman(sampled_risk, sampled_error)
            if np.isfinite(rho):
                rhos.append(rho)
        if rhos:
            values[index] = float(np.median(rhos))
    return values


def run_association(
    risk: pd.Series,
    errors: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict:
    per_method = _per_method_association(risk, errors)
    observed = float(per_method["spearman_rho"].median()) if not per_method.empty else np.nan
    boot = _bootstrap_median_spearman(risk, errors, n_bootstrap, seed)
    finite = boot[np.isfinite(boot)]
    ci_low, ci_high = (
        np.quantile(finite, [0.025, 0.975]) if finite.size else (np.nan, np.nan)
    )
    return {
        "per_method": per_method,
        "median_spearman": observed,
        "median_spearman_ci_low": float(ci_low),
        "median_spearman_ci_high": float(ci_high),
        "bootstrap_values": boot,
    }


def run_permutation_null(
    risk: pd.Series,
    errors: pd.DataFrame,
    n_perm: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.full(n_perm, np.nan, dtype=float)
    raw = risk.to_numpy(copy=True)
    for index in range(n_perm):
        shuffled = pd.Series(rng.permutation(raw), index=risk.index, name="risk")
        per_method = _per_method_association(shuffled, errors)
        if not per_method.empty:
            values[index] = float(per_method["spearman_rho"].median())
    return values


def _association_summary_row(
    label: str,
    result: dict,
    errors: pd.DataFrame,
) -> dict:
    per_method = result["per_method"]
    finite = pd.to_numeric(per_method["spearman_rho"], errors="coerce").dropna()
    return {
        "control_type": label,
        "n_methods": int(per_method["method_name"].nunique()),
        "n_perturbations": int(errors["perturbation"].nunique()),
        "median_spearman": float(result["median_spearman"]),
        "ci_low": float(result["median_spearman_ci_low"]),
        "ci_high": float(result["median_spearman_ci_high"]),
        "pct_methods_positive": float(finite.gt(0).mean()) if len(finite) else np.nan,
        "status": "ok" if len(finite) else "undefined_constant_or_insufficient",
    }


def _primary_gate(median_spearman: float, ci_low: float, pct_positive: float) -> str:
    if not np.isfinite(median_spearman) or median_spearman <= 0:
        return "FAIL"
    if np.isfinite(ci_low) and ci_low > 0 and pct_positive >= 0.60:
        return "PASS"
    return "PARTIAL"


def _partial_spearman(
    risk: np.ndarray,
    error: np.ndarray,
    nuisance: np.ndarray,
) -> float:
    frame = pd.DataFrame(
        {"risk": risk, "error": error, "nuisance": nuisance}
    ).apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 5 or frame["nuisance"].nunique() < 2:
        return float("nan")
    ranked = frame.rank(method="average")
    design = np.column_stack([np.ones(len(ranked)), ranked["nuisance"].to_numpy()])
    risk_residual = ranked["risk"].to_numpy() - design @ np.linalg.lstsq(
        design,
        ranked["risk"].to_numpy(),
        rcond=None,
    )[0]
    error_residual = ranked["error"].to_numpy() - design @ np.linalg.lstsq(
        design,
        ranked["error"].to_numpy(),
        rcond=None,
    )[0]
    return float(np.corrcoef(risk_residual, error_residual)[0, 1])


def _per_method_partial_association(
    risk: pd.Series,
    errors: pd.DataFrame,
    nuisance: pd.Series,
) -> pd.DataFrame:
    covariates = pd.concat(
        [risk.rename("risk"), nuisance.rename("nuisance")],
        axis=1,
    ).rename_axis("perturbation").reset_index()
    joined = errors.merge(covariates, on="perturbation", how="inner")
    rows = []
    for method, group in joined.groupby("method"):
        partial = _partial_spearman(
            group["risk"].to_numpy(),
            group["error"].to_numpy(),
            group["nuisance"].to_numpy(),
        )
        rows.append(
            {
                "method_name": method,
                "n_perturbations": int(group["perturbation"].nunique()),
                "partial_spearman_control_log_nstimulated": partial,
            }
        )
    return pd.DataFrame(rows).sort_values("method_name").reset_index(drop=True)


def _bootstrap_median_partial(
    risk: pd.Series,
    errors: pd.DataFrame,
    nuisance: pd.Series,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    common = sorted(
        set(risk.index.astype(str))
        & set(nuisance.index.astype(str))
        & set(errors["perturbation"].astype(str))
    )
    risk_lookup = risk.to_dict()
    nuisance_lookup = nuisance.to_dict()
    method_lookup = {
        method: group.set_index("perturbation")["error"].to_dict()
        for method, group in errors.groupby("method")
    }
    rng = np.random.default_rng(seed)
    values = np.full(n_bootstrap, np.nan, dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.choice(common, size=len(common), replace=True)
        sampled_risk = np.asarray([risk_lookup[name] for name in sampled], dtype=float)
        sampled_nuisance = np.asarray(
            [nuisance_lookup[name] for name in sampled],
            dtype=float,
        )
        partials = []
        for lookup in method_lookup.values():
            sampled_error = np.asarray(
                [lookup.get(name, np.nan) for name in sampled],
                dtype=float,
            )
            partial = _partial_spearman(
                sampled_risk,
                sampled_error,
                sampled_nuisance,
            )
            if np.isfinite(partial):
                partials.append(partial)
        if partials:
            values[index] = float(np.median(partials))
    return values


def _prepare_frangieh_errors(
    genetic_csv: Path,
    metric: str,
    deg: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_benchmark_errors(
        genetic_csv,
        dataset="Frangieh",
        metric=metric,
        deg=deg,
    )
    return raw, aggregate_benchmark_seeds(raw)


def run_frangieh_primary(
    feature_matrix: pd.DataFrame,
    genetic_csv: Path,
    out_dir: Path,
    n_bootstrap: int = 1000,
    n_permutations: int = 200,
    seed: int = 5201,
) -> dict:
    """Run the preregistered Frangieh primary analysis and controls."""
    raw, errors = _prepare_frangieh_errors(genetic_csv, metric="mse", deg=5000)
    risk = compute_safeconf_risk_per_perturbation(
        feature_matrix,
        dataset="Frangieh",
        formula="gene_main",
    )
    primary = run_association(risk, errors, n_bootstrap=n_bootstrap, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    primary["per_method"].to_csv(out_dir / "E8b_FRANGIEH_PER_METHOD.csv", index=False)
    pd.DataFrame(
        {"bootstrap_index": np.arange(n_bootstrap), "median_spearman": primary["bootstrap_values"]}
    ).to_csv(out_dir / "E8b_FRANGIEH_BOOTSTRAP.csv", index=False)

    seed_coverage = (
        raw.groupby("perturbation", as_index=False)["seed"]
        .nunique()
        .rename(columns={"seed": "n_seeds"})
        .sort_values("perturbation")
    )
    seed_coverage.to_csv(out_dir / "E8b_FRANGIEH_SEED_COVERAGE.csv", index=False)

    nstimulated = raw.groupby("perturbation")["Nstimulated"].median().sort_index()
    controls = compute_nuisance_baselines(
        feature_matrix,
        dataset="Frangieh",
        nstimulated=nstimulated,
    )
    control_rows = [_association_summary_row("frozen_v0_2_risk", primary, errors)]
    per_feature_rows = []
    for name, control_risk in controls.items():
        result = run_association(
            control_risk,
            errors,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        row = _association_summary_row(name, result, errors)
        if name == "sample_size":
            control_rows.append(row)
        else:
            per_feature_rows.append({"feature_name": name, **{k: v for k, v in row.items() if k != "control_type"}})

    permutation = run_permutation_null(
        risk,
        errors,
        n_perm=n_permutations,
        seed=seed,
    )
    finite_null = permutation[np.isfinite(permutation)]
    observed = float(primary["median_spearman"])
    empirical_p = (
        float((1 + np.sum(finite_null >= observed)) / (len(finite_null) + 1))
        if len(finite_null)
        else np.nan
    )
    control_rows.append(
        {
            "control_type": "shuffled_risk_null",
            "n_methods": int(primary["per_method"]["method_name"].nunique()),
            "n_perturbations": int(errors["perturbation"].nunique()),
            "median_spearman": float(np.nanmedian(finite_null)),
            "ci_low": float(np.nanquantile(finite_null, 0.025)),
            "ci_high": float(np.nanquantile(finite_null, 0.975)),
            "pct_methods_positive": np.nan,
            "observed_empirical_p_one_sided": empirical_p,
            "status": "ok",
        }
    )
    pd.DataFrame(control_rows).to_csv(out_dir / "E8b_FRANGIEH_CONTROLS.csv", index=False)
    pd.DataFrame(per_feature_rows).to_csv(
        out_dir / "E8b_FRANGIEH_PERFEATURE.csv",
        index=False,
    )
    pd.DataFrame(
        {"permutation_index": np.arange(n_permutations), "median_spearman": permutation}
    ).to_csv(out_dir / "E8b_FRANGIEH_SHUFFLED_NULL.csv", index=False)

    # Post hoc nuisance diagnostic. It does not alter the preregistered gate.
    sample_size_risk = controls["sample_size"]
    partial_per_method = _per_method_partial_association(
        risk,
        errors,
        sample_size_risk,
    )
    partial_per_method.to_csv(
        out_dir / "E8b_FRANGIEH_POSTHOC_SAMPLE_SIZE_PER_METHOD.csv",
        index=False,
    )
    partial_values = pd.to_numeric(
        partial_per_method["partial_spearman_control_log_nstimulated"],
        errors="coerce",
    ).dropna()
    partial_bootstrap = _bootstrap_median_partial(
        risk,
        errors,
        sample_size_risk,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    finite_partial_bootstrap = partial_bootstrap[np.isfinite(partial_bootstrap)]
    risk_sample_rho, risk_sample_p = _spearman(
        risk.reindex(sample_size_risk.index).to_numpy(),
        sample_size_risk.to_numpy(),
    )
    posthoc_sample_size = {
        "analysis_status": "post_hoc_diagnostic_not_part_of_preregistered_gate",
        "n_methods": int(len(partial_values)),
        "median_partial_spearman_control_log_nstimulated": float(partial_values.median()),
        "partial_ci_low": float(np.quantile(finite_partial_bootstrap, 0.025)),
        "partial_ci_high": float(np.quantile(finite_partial_bootstrap, 0.975)),
        "n_methods_partial_positive": int(partial_values.gt(0).sum()),
        "pct_methods_partial_positive": float(partial_values.gt(0).mean()),
        "risk_vs_sample_size_risk_spearman": risk_sample_rho,
        "risk_vs_sample_size_risk_p_value": risk_sample_p,
    }
    _write_json(
        out_dir / "E8b_FRANGIEH_POSTHOC_SAMPLE_SIZE_SUMMARY.json",
        posthoc_sample_size,
    )

    per_method = primary["per_method"]
    finite_rho = pd.to_numeric(per_method["spearman_rho"], errors="coerce").dropna()
    pct_positive = float(finite_rho.gt(0).mean())
    gate = _primary_gate(
        float(primary["median_spearman"]),
        float(primary["median_spearman_ci_low"]),
        pct_positive,
    )
    summary = {
        "dataset": "Frangieh",
        "metric": "mse",
        "DEG": 5000,
        "n_methods": int(per_method["method_name"].nunique()),
        "n_perturbations": int(errors["perturbation"].nunique()),
        "median_spearman": float(primary["median_spearman"]),
        "median_spearman_ci_low": float(primary["median_spearman_ci_low"]),
        "median_spearman_ci_high": float(primary["median_spearman_ci_high"]),
        "n_methods_positive": int(finite_rho.gt(0).sum()),
        "pct_methods_positive": pct_positive,
        "bootstrap_n": int(n_bootstrap),
        "permutation_n": int(n_permutations),
        "shuffled_null_median": float(np.nanmedian(finite_null)),
        "shuffled_null_ci_low": float(np.nanquantile(finite_null, 0.025)),
        "shuffled_null_ci_high": float(np.nanquantile(finite_null, 0.975)),
        "shuffled_null_empirical_p_one_sided": empirical_p,
        "sample_size_baseline_median_spearman": float(
            pd.DataFrame(control_rows)
            .set_index("control_type")
            .loc["sample_size", "median_spearman"]
        ),
        "posthoc_sample_size_adjusted_median_partial_spearman": posthoc_sample_size[
            "median_partial_spearman_control_log_nstimulated"
        ],
        "posthoc_sample_size_adjusted_ci_low": posthoc_sample_size["partial_ci_low"],
        "posthoc_sample_size_adjusted_ci_high": posthoc_sample_size["partial_ci_high"],
        "gate": gate,
        "claim_boundary": "external benchmark method-error association on shared biological datasets",
    }
    _write_json(out_dir / "E8b_FRANGIEH_SUMMARY.json", summary)
    return summary


def run_frangieh_sensitivity(
    feature_matrix: pd.DataFrame,
    genetic_csv: Path,
    out_dir: Path,
) -> pd.DataFrame:
    risk = compute_safeconf_risk_per_perturbation(
        feature_matrix,
        dataset="Frangieh",
        formula="gene_main",
    )
    specifications = [
        ("mse", 20),
        ("mse", 50),
        ("mse", 100),
        ("pearson_distance", 5000),
    ]
    rows = []
    for metric, deg in specifications:
        _, errors = _prepare_frangieh_errors(genetic_csv, metric=metric, deg=deg)
        per_method = _per_method_association(risk, errors)
        rhos = pd.to_numeric(per_method["spearman_rho"], errors="coerce").dropna()
        rows.append(
            {
                "dataset": "Frangieh",
                "metric": metric,
                "DEG": int(deg),
                "n_methods": int(per_method["method_name"].nunique()),
                "n_perturbations": int(errors["perturbation"].nunique()),
                "median_spearman": float(rhos.median()),
                "pct_methods_positive": float(rhos.gt(0).mean()),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "E8b_SENSITIVITY_DEG.csv", index=False)
    return result


def _prepare_sciplex3_errors(
    chemical_csv: Path,
    alias_table: pd.DataFrame,
) -> pd.DataFrame:
    raw = load_benchmark_errors(
        chemical_csv,
        dataset="sciplex3",
        metric="mse",
        deg=5000,
    )
    seeded = aggregate_benchmark_seeds(raw)
    # Four doses are repeated measurements of one context-drug benchmark task.
    across_dose = (
        seeded.groupby(["method", "context", "perturbation"], as_index=False)
        .agg(
            error=("error", "median"),
            n_doses=("dose", "nunique"),
            median_available_seeds=("n_available_seeds", "median"),
        )
    )
    usable = alias_table[
        alias_table["match_type"].astype(str).isin(["exact", "alias"])
    ][["safeconf_perturbation", "scperturbench_drug"]].copy()
    merged = across_dose.merge(
        usable,
        left_on="perturbation",
        right_on="scperturbench_drug",
        how="inner",
        validate="many_to_one",
    )
    return merged.drop(columns=["perturbation"]).rename(
        columns={"safeconf_perturbation": "perturbation"}
    )


def run_sciplex3_sensitivity(
    feature_matrix: pd.DataFrame,
    chemical_csv: Path,
    alias_csv: Path,
    out_dir: Path,
) -> pd.DataFrame:
    alias_table = pd.read_csv(alias_csv)
    errors = _prepare_sciplex3_errors(chemical_csv, alias_table)
    risk_table = compute_safeconf_risk_by_context(
        feature_matrix,
        dataset="SrivatsanTrapnell2020_sciplex3",
        formula="chem_robust",
    )
    risk_table["context"] = risk_table["context"].map(SAFECONF_SCIPLEX3_CONTEXTS)
    if risk_table["context"].isna().any():
        raise ValueError("Unknown SafeConf sciplex3 context encountered")

    rows = []
    for context in ("A549", "K562", "MCF7"):
        context_risk = (
            risk_table[risk_table["context"].eq(context)]
            .set_index("perturbation")["risk"]
            .sort_index()
        )
        context_errors = errors[errors["context"].eq(context)].copy()
        per_method = _per_method_association(context_risk, context_errors)
        for _, row in per_method.iterrows():
            rows.append(
                {
                    "analysis": context,
                    "method_name": row["method_name"],
                    "n_tasks": int(row["n_perturbations"]),
                    "spearman_rho": row["spearman_rho"],
                    "p_raw": row["p_raw"],
                    "bh_q": row["bh_q"],
                }
            )

    pooled_risk = risk_table.copy()
    pooled_risk["task_id"] = (
        pooled_risk["context"].astype(str) + "::" + pooled_risk["perturbation"].astype(str)
    )
    pooled_errors = errors.copy()
    pooled_errors["task_id"] = (
        pooled_errors["context"].astype(str) + "::" + pooled_errors["perturbation"].astype(str)
    )
    pooled_risk_series = pooled_risk.set_index("task_id")["risk"]
    pooled_errors = pooled_errors.drop(columns=["perturbation"]).rename(
        columns={"task_id": "perturbation"}
    )
    pooled_per_method = _per_method_association(pooled_risk_series, pooled_errors)
    for _, row in pooled_per_method.iterrows():
        rows.append(
            {
                "analysis": "pooled",
                "method_name": row["method_name"],
                "n_tasks": int(row["n_perturbations"]),
                "spearman_rho": row["spearman_rho"],
                "p_raw": row["p_raw"],
                "bh_q": row["bh_q"],
            }
        )

    result = pd.DataFrame(rows).sort_values(["analysis", "method_name"]).reset_index(drop=True)
    result.to_csv(out_dir / "E8b_SCIPLEX3_PER_METHOD.csv", index=False)
    return result


def _write_final_report(
    out_dir: Path,
    summary: dict,
    frangieh_sensitivity: pd.DataFrame,
    sciplex3: pd.DataFrame,
) -> None:
    sciplex_summary = (
        sciplex3.groupby("analysis")["spearman_rho"]
        .agg(
            n_methods="count",
            median_spearman="median",
            pct_methods_positive=lambda values: float(pd.Series(values).gt(0).mean()),
        )
        .reset_index()
    )
    text = [
        "# E8b scPerturBench aggregate-error association",
        "",
        "## Claim boundary",
        "",
        "This is an external benchmark method-error association on shared biological",
        "datasets. It is not vector-level SafeConf validation of the benchmark methods.",
        "",
        "## Frangieh primary",
        "",
        f"- Gate: **{summary['gate']}**",
        f"- Methods: {summary['n_methods']}",
        f"- Perturbations: {summary['n_perturbations']}",
        (
            "- Across-method median Spearman: "
            f"{summary['median_spearman']:.4f} "
            f"(95% perturbation-bootstrap CI "
            f"{summary['median_spearman_ci_low']:.4f} to "
            f"{summary['median_spearman_ci_high']:.4f})"
        ),
        (
            "- Positive methods: "
            f"{summary['n_methods_positive']}/{summary['n_methods']} "
            f"({100 * summary['pct_methods_positive']:.1f}%)"
        ),
        (
            "- Shuffled-risk null median: "
            f"{summary['shuffled_null_median']:.4f}; "
            f"empirical one-sided p={summary['shuffled_null_empirical_p_one_sided']:.4g}"
        ),
        (
            "- Sample-size baseline median Spearman: "
            f"{summary['sample_size_baseline_median_spearman']:.4f}"
        ),
        "",
        "## Post hoc sample-size diagnostic",
        "",
        "The sample-size baseline is stronger than the frozen score, so the primary",
        "association must not be presented as free of cell-count confounding.",
        (
            "- Median partial Spearman after controlling log(Nstimulated): "
            f"{summary['posthoc_sample_size_adjusted_median_partial_spearman']:.4f} "
            f"(post hoc 95% perturbation-bootstrap CI "
            f"{summary['posthoc_sample_size_adjusted_ci_low']:.4f} to "
            f"{summary['posthoc_sample_size_adjusted_ci_high']:.4f})"
        ),
        "",
        "## Frangieh sensitivity",
        "",
        "```text",
        frangieh_sensitivity.to_string(index=False),
        "```",
        "",
        "The association weakens for small DEG panels and reverses for",
        "pearson_distance at DEG=5000. The E8b conclusion is therefore metric-",
        "and gene-panel-specific, not universal across benchmark metrics.",
        "",
        "The context-similarity-only component is undefined after perturbation-level",
        "aggregation because it is constant across the 74 aligned Frangieh tasks.",
        "",
        "## sciplex3 sensitivity",
        "",
        "- Uses only 60 exact/parenthetical-alias drug mappings.",
        "- All 15 manual proposals are excluded.",
        "- Benchmark errors are aggregated across available seeds and then four doses.",
        "",
        "```text",
        sciplex_summary.to_string(index=False),
        "```",
        "",
        "## Reproducibility",
        "",
        "- Frozen protocol v0.2 was not modified.",
        "- Primary bootstrap B=1000; shuffled-risk permutations=200; seed=5201.",
        "- Raw scPerturBench aggregate CSV files remain outside Git.",
    ]
    (out_dir / "E8b_FINAL_REPORT.md").write_text(
        "\n".join(text) + "\n",
        encoding="utf-8",
    )


def command_run_authorized(args: argparse.Namespace) -> None:
    features = _read_feature_matrix(args.feature_matrix)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_frangieh_primary(
        features,
        args.genetic_csv,
        out_dir,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    frangieh_sensitivity = run_frangieh_sensitivity(
        features,
        args.genetic_csv,
        out_dir,
    )
    sciplex3 = run_sciplex3_sensitivity(
        features,
        args.chemical_csv,
        args.alias_csv,
        out_dir,
    )
    _write_final_report(out_dir, summary, frangieh_sensitivity, sciplex3)
    status = {
        "status": "ok",
        "primary_gate": summary["gate"],
        "primary_dataset": "Frangieh",
        "primary_metric": "mse",
        "primary_DEG": 5000,
        "n_bootstrap": int(args.n_bootstrap),
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
        "sciplex3_usable_aliases": 60,
        "frozen_protocol_modified": False,
    }
    _write_json(out_dir / "RUN_STATUS.json", status)
    print(json.dumps({"summary": summary, "status": status}, ensure_ascii=False, indent=2))


def build_frangieh_dry_audit(
    feature_matrix: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> dict:
    """Return coverage/provenance facts without computing any association."""
    risk = compute_safeconf_risk_per_perturbation(
        feature_matrix,
        dataset="Frangieh",
        formula="gene_main",
    )
    benchmark_perts = set(benchmark["perturbation"].dropna().astype(str))
    safeconf_perts = set(risk.index.astype(str))
    joined = sorted(benchmark_perts & safeconf_perts)
    seed_coverage = (
        benchmark.groupby("perturbation")["seed"].nunique().value_counts().sort_index()
    )
    nstim_counts = benchmark.groupby("perturbation")["Nstimulated"].nunique()
    return {
        "dataset": "Frangieh",
        "benchmark_perturbations": int(len(benchmark_perts)),
        "safeconf_scored_perturbations": int(len(safeconf_perts)),
        "joined_perturbations": int(len(joined)),
        "join_complete": bool(benchmark_perts == set(joined)),
        "joined_score_non_null": int(risk.reindex(joined).notna().sum()),
        "score_coverage_complete": bool(risk.reindex(joined).notna().all()),
        "seed_coverage_distribution": {
            str(int(seed_count)): int(n_perturbations)
            for seed_count, n_perturbations in seed_coverage.items()
        },
        "nstimulated_source": "scPerturBench aggregate CSV column Nstimulated",
        "nstimulated_is_benchmark_metadata": True,
        "nstimulated_missing_rows": int(benchmark["Nstimulated"].isna().sum()),
        "nstimulated_unique_per_perturbation": bool(nstim_counts.eq(1).all()),
        "association_computed": False,
    }


def _read_feature_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def command_alias(args: argparse.Namespace) -> None:
    features = pd.read_csv(
        args.feature_matrix,
        usecols=["dataset_name", "perturbation"],
    )
    safe_names = features.loc[
        features["dataset_name"].astype(str).eq("SrivatsanTrapnell2020_sciplex3"),
        "perturbation",
    ]
    chemical = pd.read_csv(args.chemical_csv, usecols=["DataSet", "perturb"])
    benchmark_names = chemical.loc[
        chemical["DataSet"].astype(str).isin(SCIPLEX3_DATASETS),
        "perturb",
    ]
    table = build_sciplex3_alias_table(safe_names, benchmark_names)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    counts = table["match_type"].value_counts().to_dict()
    summary = {
        "total_benchmark_drugs": int(len(table)),
        "exact": int(counts.get("exact", 0)),
        "alias": int(counts.get("alias", 0)),
        "manual": int(counts.get("manual", 0)),
        "unmatched": int(counts.get("unmatched", 0)),
        "usable_exact_plus_alias": int(
            table["match_type"].astype(str).isin(["exact", "alias"]).sum()
        ),
        "manual_rows_approved": False,
        "manual_exclusion_reason": (
            "manual count exceeds preregistered limit of 10"
            if counts.get("manual", 0) > 10
            else "awaiting Claude review"
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_dry_audit(args: argparse.Namespace) -> None:
    features = _read_feature_matrix(args.feature_matrix)
    benchmark = load_benchmark_errors(
        args.genetic_csv,
        dataset="Frangieh",
        metric="mse",
        deg=5000,
    )
    audit = build_frangieh_dry_audit(features, benchmark)
    _write_json(args.out, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    alias = subparsers.add_parser("alias-audit", help="Build sciplex3 alias audit table")
    alias.add_argument("--feature-matrix", type=Path, required=True)
    alias.add_argument("--chemical-csv", type=Path, required=True)
    alias.add_argument("--out", type=Path, required=True)
    alias.set_defaults(func=command_alias)

    dry = subparsers.add_parser(
        "dry-audit",
        help="Check Frangieh identifier/score/seed coverage without association",
    )
    dry.add_argument("--feature-matrix", type=Path, required=True)
    dry.add_argument("--genetic-csv", type=Path, required=True)
    dry.add_argument("--out", type=Path, required=True)
    dry.set_defaults(func=command_dry_audit)

    run = subparsers.add_parser(
        "run-authorized",
        help="Run Claude-authorized E8b primary and sensitivity analyses",
    )
    run.add_argument("--feature-matrix", type=Path, required=True)
    run.add_argument("--genetic-csv", type=Path, required=True)
    run.add_argument("--chemical-csv", type=Path, required=True)
    run.add_argument("--alias-csv", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--n-bootstrap", type=int, default=1000)
    run.add_argument("--n-permutations", type=int, default=200)
    run.add_argument("--seed", type=int, default=5201)
    run.set_defaults(func=command_run_authorized)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
