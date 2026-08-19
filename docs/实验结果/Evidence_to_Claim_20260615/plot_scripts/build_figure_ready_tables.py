#!/usr/bin/env python3
"""Build figure-ready tables for SafeConf Phase 4c.

This script does not run new experiments. It only consolidates frozen result
tables into small plotting tables with source-file provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "agents").is_dir() and (parent / "docs").is_dir():
            return parent
    raise RuntimeError("Could not locate repo root")


ROOT = find_repo_root()
RESULTS = ROOT / "docs" / "实验结果"
OUT = RESULTS / "Evidence_to_Claim_20260615"
TABLES = OUT / "figure_ready_tables"
SOURCE_COMMIT = "e8a3594"


SOURCES = {
    "formal_main": RESULTS
    / "Formal_main_20260604/corrected_v3_drop_blank_1000_20260609/tables/FORMAL_MAIN_TABLE.csv",
    "a1_variance": RESULTS
    / "Task_risk_audit_20260611/tables/A1_task_vs_predictor_variance_summary.csv",
    "a1_paired": RESULTS
    / "Task_risk_audit_20260611/tables/A1_paired_predictor_error_table.csv",
    "e1_deltas": RESULTS
    / "E1_E4_preregistered_20260614/E1_group_ablation/E1_GROUP_ABLATION_DELTAS.csv",
    "e1_lopo_gate": RESULTS
    / "E1_E4_preregistered_20260614/E1_group_ablation/E1_GROUP_GATE.csv",
    "e1_lodo_gate": RESULTS
    / "E1_E4_preregistered_20260614/E1_group_ablation/E1_LODO_LOPO_GATE.csv",
    "e2": RESULTS
    / "E1_E4_preregistered_20260614/E2_magnitude_residual/E2_MAGNITUDE_RESIDUAL_SUMMARY.csv",
    "e3_empirical": RESULTS
    / "E1_E4_preregistered_20260614/E3_negative_controls/E3_EMPIRICAL_PVALUES.csv",
    "e3_missingness": RESULTS
    / "E1_E4_preregistered_20260614/E3_negative_controls/E3_MISSINGNESS_PAIRED_DELTA.csv",
    "e4_seed_summary": RESULTS
    / "E1_E4_preregistered_20260614/E4_model_stability/E4_SEED_SUMMARY.csv",
    "e4_seed_results": RESULTS
    / "E1_E4_preregistered_20260614/E4_model_stability/E4_SEED_RESULTS.csv",
    "e4_config_summary": RESULTS
    / "E1_E4_preregistered_20260614/E4_model_stability/E4_CONFIG_SUMMARY.csv",
    "e8b_summary": RESULTS
    / "E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_SUMMARY.json",
    "e8b_posthoc": RESULTS
    / "E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_POSTHOC_SAMPLE_SIZE_SUMMARY.json",
    "e8b_per_method": RESULTS
    / "E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_PER_METHOD.csv",
    "e8b_posthoc_per_method": RESULTS
    / "E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_POSTHOC_SAMPLE_SIZE_PER_METHOD.csv",
    "e8b_controls": RESULTS
    / "E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_CONTROLS.csv",
    "e8b_perfeature": RESULTS
    / "E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_PERFEATURE.csv",
    "e8b_shuffled": RESULTS
    / "E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_FRANGIEH_SHUFFLED_NULL.csv",
    "e8b_sensitivity": RESULTS
    / "E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_SENSITIVITY_DEG.csv",
    "e8b_sciplex3": RESULTS
    / "E1_E4_preregistered_20260614/E8b_scperturbench_association/E8b_SCIPLEX3_PER_METHOD.csv",
    "b1_bad_retrieval": RESULTS
    / "Task_risk_audit_20260611/tables/B1_bad_prediction_retrieval.csv",
    "risk_coverage": RESULTS
    / "Formal_main_20260604/paper_figures/tables/PAPER_RISK_COVERAGE_CURVES.csv",
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def add_source(df: pd.DataFrame, source_key: str) -> pd.DataFrame:
    out = df.copy()
    out["source_file"] = rel(SOURCES[source_key])
    out["source_commit"] = SOURCE_COMMIT
    return out


def write(df: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / name, index=False)


def build_fig1_tables() -> None:
    variance = pd.read_csv(SOURCES["a1_variance"])
    variance = add_source(variance, "a1_variance")
    write(variance, "FIG1_A1_VARIANCE_DECOMPOSITION.csv")

    paired = pd.read_csv(SOURCES["a1_paired"])
    paired = add_source(paired, "a1_paired")
    write(paired, "FIG1_A1_ERROR_SCATTER.csv")


def build_fig2_table() -> None:
    df = pd.read_csv(SOURCES["formal_main"])
    cols = [
        "dataset_name",
        "dataset_family",
        "n",
        "aligned_rho",
        "partial_rho_control_magnitude",
        "partial_rho_ci_low",
        "partial_rho_ci_high",
        "magnitude_only_rho",
        "risk_coverage80_improve_pct",
    ]
    out = df[cols].copy()
    out = out.rename(
        columns={
            "partial_rho_control_magnitude": "partial_rho",
            "risk_coverage80_improve_pct": "rc80_improve_pct",
        }
    )
    out["is_failure_boundary"] = out["dataset_name"].eq("McFarlandTsherniak2020")
    out["display_label"] = out["dataset_name"].replace(
        {
            "LaraAstiasoHuntly2023_exvivo": "Lara ex vivo",
            "LaraAstiasoHuntly2023_invivo": "Lara in vivo",
            "SrivatsanTrapnell2020_sciplex3": "Srivatsan sciplex3",
            "McFarlandTsherniak2020": "McFarland",
            "CuiHacohen2023": "Cui",
            "SantinhaPlatt2023": "Santinha",
        }
    )
    out = add_source(out, "formal_main")
    write(out, "FIG2_FORMAL_MAIN_FOREST.csv")
    write(out, "TABLE_DATASET_SUMMARY.csv")


def build_fig3_tables() -> None:
    e2 = pd.read_csv(SOURCES["e2"])
    e2 = e2[e2["calibration_method"].eq("isotonic")].copy()
    e2 = e2.rename(
        columns={
            "aurc_improvement_magnitude_minus_combined_point": "aurc_diff_magnitude_minus_combined",
            "aurc_improvement_magnitude_minus_combined_ci_low": "aurc_diff_ci_low",
            "aurc_improvement_magnitude_minus_combined_ci_high": "aurc_diff_ci_high",
            "residual_partial_rho_point": "residual_partial_rho",
            "residual_partial_rho_ci_low": "residual_partial_ci_low",
            "residual_partial_rho_ci_high": "residual_partial_ci_high",
        }
    )
    e2["aurc_diff_definition"] = "magnitude_only_aurc_minus_combined_predicted_error_aurc"
    e2 = add_source(e2, "e2")
    write(e2, "FIG3_E2_MAGNITUDE_RESIDUAL.csv")

    formal = pd.read_csv(SOURCES["formal_main"])[
        [
            "dataset_name",
            "dataset_family",
            "partial_rho_control_magnitude",
            "partial_rho_ci_low",
            "partial_rho_ci_high",
        ]
    ].copy()
    formal = formal.rename(
        columns={
            "partial_rho_control_magnitude": "frozen_partial_rho",
            "partial_rho_ci_low": "frozen_ci_low",
            "partial_rho_ci_high": "frozen_ci_high",
        }
    )
    learned = pd.read_csv(SOURCES["e4_seed_summary"]).rename(
        columns={
            "median": "learned_partial_rho",
            "minimum": "learned_min_seed",
            "maximum": "learned_max_seed",
            "q1": "learned_q1_seed",
            "q3": "learned_q3_seed",
        }
    )
    panel = formal.merge(learned, on="dataset_name", how="left")
    panel["learned_ci_low"] = pd.NA
    panel["learned_ci_high"] = pd.NA
    panel["learned_interval_low"] = panel["learned_min_seed"]
    panel["learned_interval_high"] = panel["learned_max_seed"]
    panel["learned_interval_type"] = "seed_min_max_not_bootstrap_ci"
    panel["is_mcfarland"] = panel["dataset_name"].eq("McFarlandTsherniak2020")
    panel["source_file"] = rel(SOURCES["formal_main"]) + ";" + rel(SOURCES["e4_seed_summary"])
    panel["source_commit"] = SOURCE_COMMIT
    write(panel, "FIG3_LOPO_LEARNED_PANEL.csv")


def build_fig4_tables() -> None:
    with open(SOURCES["e8b_summary"], encoding="utf-8") as fh:
        summary = json.load(fh)
    with open(SOURCES["e8b_posthoc"], encoding="utf-8") as fh:
        posthoc = json.load(fh)

    per_method = pd.read_csv(SOURCES["e8b_per_method"])
    per_method = per_method.sort_values("spearman_rho", ascending=False).reset_index(drop=True)
    per_method["rank_by_raw_rho"] = per_method.index + 1
    for key in [
        "median_spearman",
        "median_spearman_ci_low",
        "median_spearman_ci_high",
        "shuffled_null_ci_low",
        "shuffled_null_ci_high",
        "sample_size_baseline_median_spearman",
    ]:
        per_method[key] = summary[key]
    per_method = add_source(per_method, "e8b_per_method")
    write(per_method, "FIG4_E8B_EXTERNAL_BENCHMARK.csv")

    partial = pd.read_csv(SOURCES["e8b_posthoc_per_method"])
    joined = per_method.merge(partial, on=["method_name", "n_perturbations"], how="left")
    joined = joined.rename(
        columns={
            "spearman_rho": "raw_spearman_rho",
            "partial_spearman_control_log_nstimulated": "partial_spearman_control_log_nstimulated",
        }
    )
    joined["posthoc_partial_median"] = posthoc[
        "median_partial_spearman_control_log_nstimulated"
    ]
    joined["posthoc_partial_ci_low"] = posthoc["partial_ci_low"]
    joined["posthoc_partial_ci_high"] = posthoc["partial_ci_high"]
    joined["risk_vs_sample_size_risk_spearman"] = posthoc[
        "risk_vs_sample_size_risk_spearman"
    ]
    joined["sample_size_baseline_median_spearman"] = summary[
        "sample_size_baseline_median_spearman"
    ]
    joined["analysis_note"] = "sample_size_adjustment_is_post_hoc_not_preregistered_gate"
    joined["source_file"] = (
        rel(SOURCES["e8b_per_method"])
        + ";"
        + rel(SOURCES["e8b_posthoc_per_method"])
        + ";"
        + rel(SOURCES["e8b_summary"])
        + ";"
        + rel(SOURCES["e8b_posthoc"])
    )
    joined["source_commit"] = SOURCE_COMMIT
    write(joined, "FIG4_E8B_PARTIAL_PER_METHOD.csv")

    controls = add_source(pd.read_csv(SOURCES["e8b_controls"]), "e8b_controls")
    write(controls, "FIG4_E8B_CONTROLS.csv")

    shuffled = add_source(pd.read_csv(SOURCES["e8b_shuffled"]), "e8b_shuffled")
    write(shuffled, "FIG4_E8B_SHUFFLED_NULL.csv")


def build_fig5_tables() -> None:
    b1 = pd.read_csv(SOURCES["b1_bad_retrieval"])
    strategy_labels = {
        "random": "Random",
        "predicted_magnitude": "Magnitude-only",
        "protocol_v0_2_family_confidence": "Frozen v0.2",
        "safeconf_lodo_risk": "LODO risk",
        "safeconf_perdataset_risk": "Per-dataset risk",
        "oracle_magnitude_diagnostic": "Oracle",
    }
    strategy_roles = {
        "random": "baseline",
        "predicted_magnitude": "magnitude_competitor",
        "protocol_v0_2_family_confidence": "frozen_safeconf",
        "safeconf_lodo_risk": "cross_dataset_transfer",
        "safeconf_perdataset_risk": "within_dataset_upper_reference",
        "oracle_magnitude_diagnostic": "non_deployable_oracle_reference",
    }
    b1 = b1[b1["score_name"].isin(strategy_labels)].copy()
    b1["strategy_label"] = b1["score_name"].map(strategy_labels)
    b1["strategy_role"] = b1["score_name"].map(strategy_roles)
    b1["is_macro_summary"] = b1["dataset_name"].eq("__macro_mean__")
    b1["top_percent"] = (b1["top_fraction"] * 100).round().astype(int)
    b1 = add_source(b1, "b1_bad_retrieval")
    write(b1, "FIG5_COST_EFFECTIVENESS.csv")

    macro = b1[b1["is_macro_summary"] & b1["top_fraction"].eq(0.10)].copy()
    macro["macro_definition"] = "macro_mean_across_7_real_datasets"
    write(macro, "FIG5_COST_EFFECTIVENESS_MACRO_TOP10.csv")

    heatmap = b1[
        (~b1["is_macro_summary"])
        & b1["score_name"].isin(
            [
                "random",
                "predicted_magnitude",
                "protocol_v0_2_family_confidence",
                "safeconf_lodo_risk",
                "safeconf_perdataset_risk",
            ]
        )
    ].copy()
    write(heatmap, "FIG5_COST_EFFECTIVENESS_HEATMAP.csv")

    rc = pd.read_csv(SOURCES["risk_coverage"])
    rc = add_source(rc, "risk_coverage")
    write(rc, "SFIG_RISK_COVERAGE.csv")


def build_supplement_tables() -> None:
    e1 = pd.read_csv(SOURCES["e1_deltas"])
    e1 = add_source(e1, "e1_deltas")
    write(e1, "SFIG_E1_GROUP_ABLATION_HEATMAP.csv")

    e3 = pd.read_csv(SOURCES["e3_empirical"])
    e3 = add_source(e3, "e3_empirical")
    write(e3, "SFIG_E3_NEGATIVE_CONTROLS.csv")

    missing = pd.read_csv(SOURCES["e3_missingness"])
    missing = add_source(missing, "e3_missingness")
    write(missing, "SFIG_E3_MISSINGNESS_ONLY_DELTA.csv")

    perfeature = add_source(pd.read_csv(SOURCES["e8b_perfeature"]), "e8b_perfeature")
    sensitivity = add_source(pd.read_csv(SOURCES["e8b_sensitivity"]), "e8b_sensitivity")
    sciplex3 = add_source(pd.read_csv(SOURCES["e8b_sciplex3"]), "e8b_sciplex3")
    write(perfeature, "SFIG_E8B_PERFEATURE.csv")
    write(sensitivity, "SFIG_E8B_SENSITIVITY.csv")
    write(sciplex3, "SFIG_E8B_SCIPLEX3_SENSITIVITY.csv")


def main() -> None:
    build_fig1_tables()
    build_fig2_table()
    build_fig3_tables()
    build_fig4_tables()
    build_fig5_tables()
    build_supplement_tables()

    manifest = pd.DataFrame(
        [{"source_key": key, "source_file": rel(path), "source_commit": SOURCE_COMMIT}
         for key, path in SOURCES.items()]
    )
    write(manifest, "SOURCE_FILES_USED.csv")
    print(f"Wrote figure-ready tables to {TABLES}")


if __name__ == "__main__":
    main()
