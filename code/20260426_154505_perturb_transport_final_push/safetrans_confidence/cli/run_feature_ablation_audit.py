#!/usr/bin/env python3
"""Feature ablation audit for frozen SafeConf protocol v0.2.

This script does not train perturbation predictors. It recomputes simple
leave-one-feature-out and single-feature variants from existing
PredictionRecord/CONFIDENCE_FEATURES tables, then evaluates whether each
variant still tracks true prediction error after controlling effect magnitude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.cli.run_formal_main_audit import (
    _assign_normalized_rmse,
    _bootstrap_ci,
    _load_effect_magnitudes,
    _metric_row,
)
from safetrans_confidence.data.records import load_merged_records
from safetrans_confidence.scoring.protocol_v0_2 import zscore_by_ref


VARIANT_META = {
    "protocol_full_confidence": ("full", "full frozen v0.2 formula"),
    "loo_no_context_confidence": ("leave_one_out", "remove context_similarity"),
    "loo_no_support_confidence": ("leave_one_out", "remove support_count"),
    "loo_no_disagreement_confidence": ("leave_one_out", "remove model_disagreement"),
    "single_context_similarity_confidence": ("single_feature", "context_similarity only"),
    "single_support_count_confidence": ("single_feature", "support_count only"),
    "single_negative_disagreement_confidence": ("single_feature", "negative model_disagreement only"),
}


def _load_base_with_magnitude(run_dir: Path) -> pd.DataFrame:
    records = pd.read_csv(run_dir / "tables" / "PREDICTION_RECORDS.csv")
    base = load_merged_records(run_dir)
    magnitudes = _load_effect_magnitudes(records, run_dir / "input" / "true_effects.npz")
    keep = [
        "record_id",
        "task_id",
        "task_key",
        "dataset_name",
        "fold_id",
        "split",
        "context",
        "perturbation",
        "predictor_name",
        "true_effect_key",
    ]
    out = base.merge(
        records[keep].drop_duplicates("record_id"),
        on=["record_id", "task_id", "task_key", "dataset_name", "fold_id", "split", "context", "perturbation", "predictor_name"],
        how="left",
        suffixes=("", "_record"),
    )
    out = out.merge(magnitudes, on="record_id", how="left")
    out["source_run_dir"] = str(run_dir)
    return _assign_normalized_rmse(out)


def _score_variants(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    formula_rows: list[dict] = []

    def add_scores(score_name: str, values: pd.Series, variant_kind: str, variant_description: str) -> None:
        for idx, value in values.items():
            row = base.loc[idx]
            rows.append(
                {
                    "record_id": row["record_id"],
                    "dataset_name": row["dataset_name"],
                    "dataset_family": row["dataset_family"],
                    "fold_id": int(row["fold_id"]),
                    "split": row["split"],
                    "context": row["context"],
                    "perturbation": row["perturbation"],
                    "predictor_name": row["predictor_name"],
                    "score_name": score_name,
                    "score_type": "confidence",
                    "score_value": float(value) if pd.notna(value) else np.nan,
                    "true_error_rmse": float(row["true_error_rmse"]),
                    "true_effect_l2_norm": row["true_effect_l2_norm"],
                    "effect_scale_rmse": row["effect_scale_rmse"],
                    "normalized_rmse": row["normalized_rmse"],
                    "variant_kind": variant_kind,
                    "variant_description": variant_description,
                }
            )

    for (dataset, family, fold, predictor), idx_obj in base.groupby(
        ["dataset_name", "dataset_family", "fold_id", "predictor_name"], dropna=False
    ).groups.items():
        idx = list(idx_obj)
        sub = base.loc[idx]
        train = sub[sub["split"].eq("train")]
        if train.empty:
            train = sub[sub["split"].isin(["train", "val"])]

        z_ctx = zscore_by_ref(sub["context_similarity_max"], train["context_similarity_max"])
        z_support = zscore_by_ref(
            np.log1p(sub["perturbation_support_count"].astype(float)),
            np.log1p(train["perturbation_support_count"].astype(float)),
        )
        z_dis = zscore_by_ref(sub["model_disagreement_rmse"], train["model_disagreement_rmse"])

        if str(family) == "chem_robust":
            variants = {
                "protocol_full_confidence": z_support - z_dis,
                # context was not part of the chem_robust frozen formula; this
                # row is retained for shape consistency and marked downstream.
                "loo_no_context_confidence": z_support - z_dis,
                "loo_no_support_confidence": -z_dis,
                "loo_no_disagreement_confidence": z_support,
                "single_context_similarity_confidence": z_ctx,
                "single_support_count_confidence": z_support,
                "single_negative_disagreement_confidence": -z_dis,
            }
            formula = "chem_robust: log_support - model_disagreement"
        else:
            variants = {
                "protocol_full_confidence": z_ctx + z_support - z_dis,
                "loo_no_context_confidence": z_support - z_dis,
                "loo_no_support_confidence": z_ctx - z_dis,
                "loo_no_disagreement_confidence": z_ctx + z_support,
                "single_context_similarity_confidence": z_ctx,
                "single_support_count_confidence": z_support,
                "single_negative_disagreement_confidence": -z_dis,
            }
            formula = "gene_main: context_similarity + log_support - model_disagreement"

        for name, values in variants.items():
            kind, desc = VARIANT_META[name]
            add_scores(name, values, kind, desc)
        formula_rows.append(
            {
                "dataset_name": dataset,
                "dataset_family": family,
                "fold_id": int(fold),
                "predictor_name": predictor,
                "base_formula": formula,
            }
        )

    scores = pd.DataFrame(rows)
    scores["risk_axis"] = -pd.to_numeric(scores["score_value"], errors="coerce")
    return scores, pd.DataFrame(formula_rows)


def _summary(scores: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    test = scores[scores["split"].eq("test")].dropna(
        subset=["score_value", "true_error_rmse", "true_effect_l2_norm"]
    )
    for (dataset, family, score_name), group in test.groupby(
        ["dataset_name", "dataset_family", "score_name"], dropna=False
    ):
        row = {
            "dataset_name": dataset,
            "dataset_family": family,
            "score_name": score_name,
            "variant_kind": str(group["variant_kind"].iloc[0]),
            "variant_description": str(group["variant_description"].iloc[0]),
            **_metric_row(group),
        }
        a_lo, a_hi = _bootstrap_ci(group, "aligned_rho", rng, n_bootstrap)
        p_lo, p_hi = _bootstrap_ci(group, "partial_rho_control_magnitude", rng, n_bootstrap)
        row.update(
            {
                "aligned_rho_ci_low": a_lo,
                "aligned_rho_ci_high": a_hi,
                "partial_rho_ci_low": p_lo,
                "partial_rho_ci_high": p_hi,
                "n_bootstrap": int(n_bootstrap),
            }
        )
        if str(family) == "chem_robust" and score_name == "loo_no_context_confidence":
            row["interpretation_note"] = "not_applicable_same_as_full_for_chem_robust"
        else:
            row["interpretation_note"] = ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset_name", "variant_kind", "score_name"])


def _per_predictor(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    test = scores[scores["split"].eq("test")].dropna(
        subset=["score_value", "true_error_rmse", "true_effect_l2_norm"]
    )
    for (dataset, family, predictor, score_name), group in test.groupby(
        ["dataset_name", "dataset_family", "predictor_name", "score_name"], dropna=False
    ):
        rows.append(
            {
                "dataset_name": dataset,
                "dataset_family": family,
                "predictor_name": predictor,
                "score_name": score_name,
                "variant_kind": str(group["variant_kind"].iloc[0]),
                **_metric_row(group),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_name", "predictor_name", "score_name"])


def _delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    full = summary[summary["score_name"].eq("protocol_full_confidence")][
        ["dataset_name", "partial_rho_control_magnitude", "aligned_rho", "risk_coverage80_improve_pct"]
    ].rename(
        columns={
            "partial_rho_control_magnitude": "full_partial_rho",
            "aligned_rho": "full_aligned_rho",
            "risk_coverage80_improve_pct": "full_rc80_improve_pct",
        }
    )
    out = summary.merge(full, on="dataset_name", how="left")
    out["delta_partial_vs_full"] = out["partial_rho_control_magnitude"] - out["full_partial_rho"]
    out["delta_aligned_vs_full"] = out["aligned_rho"] - out["full_aligned_rho"]
    out["delta_rc80_vs_full"] = out["risk_coverage80_improve_pct"] - out["full_rc80_improve_pct"]
    return out.sort_values(["dataset_name", "variant_kind", "score_name"])


def run_feature_ablation(run_dirs: list[Path], out_dir: Path, n_bootstrap: int, seed: int) -> dict:
    for sub in ["tables", "reports"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    frames = []
    status_rows = []
    for run_dir in run_dirs:
        try:
            base = _load_base_with_magnitude(run_dir)
            frames.append(base)
            status_rows.append(
                {
                    "run_dir": str(run_dir),
                    "dataset_names": sorted(base["dataset_name"].dropna().astype(str).unique().tolist()),
                    "n_records": int(len(base)),
                    "status": "ok",
                }
            )
        except Exception as exc:  # pragma: no cover - operational guard.
            status_rows.append({"run_dir": str(run_dir), "status": "failed", "error": repr(exc)})
    pd.DataFrame(status_rows).to_csv(out_dir / "tables" / "FEATURE_ABLATION_INPUT_STATUS.csv", index=False)
    if not frames:
        raise RuntimeError("No usable run directories for feature ablation.")
    base = pd.concat(frames, ignore_index=True)
    scores, formulas = _score_variants(base)
    summary = _summary(scores, n_bootstrap=n_bootstrap, seed=seed)
    per_predictor = _per_predictor(scores)
    delta = _delta_table(summary)

    scores.to_csv(out_dir / "tables" / "FEATURE_ABLATION_SCORES.csv", index=False)
    formulas.to_csv(out_dir / "tables" / "FEATURE_ABLATION_FORMULAS.csv", index=False)
    summary.to_csv(out_dir / "tables" / "FEATURE_ABLATION_SUMMARY.csv", index=False)
    per_predictor.to_csv(out_dir / "tables" / "FEATURE_ABLATION_PER_PREDICTOR.csv", index=False)
    delta.to_csv(out_dir / "tables" / "FEATURE_ABLATION_DELTA.csv", index=False)

    full = summary[summary["score_name"].eq("protocol_full_confidence")].copy()
    loo = delta[delta["variant_kind"].eq("leave_one_out")].copy()
    lines = [
        "# SafeConf feature ablation audit",
        "",
        "This audit recomputes frozen protocol variants from existing records. It does not train perturbation predictors.",
        "",
        "## Full protocol summary",
        "",
        "```",
        full[["dataset_name", "dataset_family", "n", "aligned_rho", "partial_rho_control_magnitude", "risk_coverage80_improve_pct"]].to_string(index=False),
        "```",
        "",
        "## Leave-one-feature-out delta vs full",
        "",
        "Negative delta means removing the feature hurts the metric.",
        "",
        "```",
        loo[["dataset_name", "score_name", "partial_rho_control_magnitude", "delta_partial_vs_full", "aligned_rho", "delta_aligned_vs_full"]].to_string(index=False),
        "```",
    ]
    (out_dir / "reports" / "FEATURE_ABLATION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status = {
        "out_dir": str(out_dir),
        "n_input_runs": int(len(run_dirs)),
        "n_usable_runs": int(sum(r.get("status") == "ok" for r in status_rows)),
        "n_scores": int(len(scores)),
        "n_bootstrap": int(n_bootstrap),
    }
    (out_dir / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SafeConf feature ablation audit.")
    parser.add_argument("--run-dir", type=Path, action="append", dest="run_dirs", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=5201)
    args = parser.parse_args()
    print(json.dumps(run_feature_ablation(args.run_dirs, args.out_dir, args.bootstrap, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

