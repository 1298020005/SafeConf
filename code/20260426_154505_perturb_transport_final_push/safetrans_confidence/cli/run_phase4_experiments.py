#!/usr/bin/env python3
"""Phase 4 experiment bundle.

This command is intentionally conservative: it aggregates already completed
frozen-protocol runs, audits Papalexi eligibility, diagnoses crossPatient, and
checks whether existing GEARS files are sufficient for a per-prediction
confidence experiment.  It does not download data, train deep predictors, or
modify protocol v0.2.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.data.eligibility import audit_h5ad
from safetrans_confidence.data.records import load_merged_records
from safetrans_confidence.eval.metrics import aligned_spearman, raw_spearman
from safetrans_confidence.scoring.protocol_v0_2 import build_protocol_v0_2_scores

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "outputs/benchmark_phase4_experiments"
PYTHON = "/home/yyf/.conda/envs/scgpt_env/bin/python"

RUN_INPUTS = {
    "phase2_1_three_dataset": PROJECT_ROOT / "outputs/confidence_task_mvp_v2_1",
    "Frangieh_blind": PROJECT_ROOT / "outputs/confidence_task_Frangieh_blind",
    "KaggleCrossPatient_blind": PROJECT_ROOT / "outputs/confidence_task_kcp_blind_probe",
    "crossPatient_blind": PROJECT_ROOT / "outputs/confidence_task_crossPatient_blind",
}

PAPALEXI_CANDIDATES = {
    "Papalexi": Path("/home/yyf/data/singlecell_perturbation_atlas/official_generalization/Papalexi.h5ad"),
    "PapalexiSatija2021_eccite_RNA": Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/PapalexiSatija2021_eccite_RNA.h5ad"),
    "PapalexiSatija2021_eccite_arrayed_RNA": Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/PapalexiSatija2021_eccite_arrayed_RNA.h5ad"),
}

NORMAN_GEARS_METRICS = sorted(
    Path("/home/yyf/archive/code/20260519_0958_home_cleanup/moved_codex_cout_old/SAFE_TRANS_PT_WEEKEND_Q1_PUSH_20260514/results/gears_formal_runs/norman").glob(
        "seed_*/results/GEARS_norman_single_PERT_METRICS.csv"
    )
)


def ensure_dirs(out: Path) -> None:
    for sub in ["tables", "reports", "logs", "scripts"]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def build_predictor_breakdown(out: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for run_label, input_dir in RUN_INPUTS.items():
        if not (input_dir / "tables/PREDICTION_RECORDS.csv").exists():
            rows.append(
                {
                    "run_label": run_label,
                    "source_dir": str(input_dir),
                    "dataset_name": "",
                    "dataset_family": "",
                    "predictor_name": "",
                    "score_name": "protocol_v0_2_family_confidence",
                    "n_test": 0,
                    "spearman_score_vs_rmse": np.nan,
                    "direction_aligned_spearman": np.nan,
                    "status": "missing_prediction_records",
                }
            )
            continue
        base = load_merged_records(input_dir)
        scores, _formulas = build_protocol_v0_2_scores(base)
        test = scores[
            (scores["split"] == "test")
            & (scores["score_name"] == "protocol_v0_2_family_confidence")
        ].dropna(subset=["score_value", "true_error_rmse"])
        for (dataset, family, predictor), g in test.groupby(
            ["dataset_name", "dataset_family", "predictor_name"], dropna=False
        ):
            score_type = str(g["score_type"].iloc[0])
            rho = raw_spearman(g["score_value"], g["true_error_rmse"])
            rows.append(
                {
                    "run_label": run_label,
                    "source_dir": str(input_dir),
                    "dataset_name": dataset,
                    "dataset_family": family,
                    "predictor_name": predictor,
                    "score_name": "protocol_v0_2_family_confidence",
                    "n_test": int(len(g)),
                    "spearman_score_vs_rmse": rho,
                    "direction_aligned_spearman": aligned_spearman(
                        g["score_value"], g["true_error_rmse"], score_type
                    ),
                    "status": "ok",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "tables/PREDICTOR_BREAKDOWN.csv", index=False)
    return df


def audit_papalexi(out: Path) -> pd.DataFrame:
    rows = []
    for name, path in PAPALEXI_CANDIDATES.items():
        rows.append(asdict(audit_h5ad(path, name)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "tables/Papalexi_ELIGIBILITY.csv", index=False)
    if not bool(df["cross_context_eligible"].any()):
        skip = df.copy()
        skip["action"] = "SKIP_BLIND_PIPELINE"
        skip["skip_reason"] = (
            "No Papalexi file has >=2 real cellular contexts with control/treated pairs; "
            "running a held-out pair benchmark would require fabricating context."
        )
        skip.to_csv(out / "tables/Papalexi_SKIP.csv", index=False)
        (out / "reports/Papalexi_SKIP.md").write_text(
            "# Papalexi skip\n\n"
            "Papalexi was audited before running blind scoring. Existing files are not cross-context eligible, "
            "so Phase 4 did not force it into the benchmark.\n\n"
            "```\n" + df.to_string(index=False) + "\n```\n",
            encoding="utf-8",
        )
    return df


def diagnose_crosspatient(out: Path) -> pd.DataFrame:
    input_dir = PROJECT_ROOT / "outputs/confidence_task_crossPatient_blind"
    rows: list[dict] = []
    if not (input_dir / "tables/PREDICTION_RECORDS.csv").exists():
        rows.append({"section": "status", "metric": "input", "value": "missing", "note": str(input_dir)})
        df = pd.DataFrame(rows)
        df.to_csv(out / "tables/CROSSPATIENT_FAILURE_DIAG.csv", index=False)
        return df

    base = load_merged_records(input_dir)
    scores, _ = build_protocol_v0_2_scores(base)
    test_scores = scores[scores["split"] == "test"].dropna(subset=["score_value", "true_error_rmse"])
    test_base = base[base["split"] == "test"].copy()

    rows.append(
        {
            "section": "sample_size",
            "metric": "n_test_records",
            "value": int(len(test_base)),
            "note": "Small n makes crossPatient rho unstable.",
        }
    )
    rows.append(
        {
            "section": "sample_size",
            "metric": "n_test_contexts",
            "value": int(test_base["context"].nunique()),
            "note": "Few held-out contexts reduce ranking resolution.",
        }
    )
    rows.append(
        {
            "section": "sample_size",
            "metric": "n_test_perturbations",
            "value": int(test_base["perturbation"].nunique()),
            "note": "Few held-out perturbations reduce ranking resolution.",
        }
    )

    feature_cols = [
        "context_similarity_max",
        "perturbation_support_count",
        "model_disagreement_rmse",
        "perturbation_effect_stability",
        "ood_nearest_distance",
        "prediction_magnitude_deviation",
    ]
    for col in feature_cols:
        if col in test_base.columns:
            rows.append(
                {
                    "section": "feature_missingness",
                    "metric": col,
                    "value": float(test_base[col].isna().mean()),
                    "note": "Fraction missing on test records.",
                }
            )

    for score in [
        "protocol_v0_2_family_confidence",
        "context_similarity_score",
        "support_count_score",
        "model_disagreement_risk",
        "ood_distance_risk",
        "prediction_magnitude_risk",
    ]:
        sg = test_scores[test_scores["score_name"] == score]
        if sg.empty:
            continue
        score_type = str(sg["score_type"].iloc[0])
        rows.append(
            {
                "section": "score_signal",
                "metric": score,
                "value": aligned_spearman(sg["score_value"], sg["true_error_rmse"], score_type),
                "note": f"direction-aligned Spearman; score_type={score_type}.",
            }
        )

    for (predictor,), g in test_scores[test_scores["score_name"] == "protocol_v0_2_family_confidence"].groupby(
        ["predictor_name"], dropna=False
    ):
        rows.append(
            {
                "section": "predictor_breakdown",
                "metric": str(predictor),
                "value": aligned_spearman(g["score_value"], g["true_error_rmse"], str(g["score_type"].iloc[0])),
                "note": "Protocol v0.2 per-predictor signal.",
            }
        )

    rows.append(
        {
            "section": "interpretation",
            "metric": "failure_boundary",
            "value": "formula_not_changed",
            "note": "crossPatient is kept as a negative/unstable blind boundary instead of retuning protocol v0.2.",
        }
    )
    df = pd.DataFrame(rows)
    df.to_csv(out / "tables/CROSSPATIENT_FAILURE_DIAG.csv", index=False)
    return df


def audit_gears_confidence(out: Path) -> pd.DataFrame:
    gears_out = PROJECT_ROOT / "outputs/confidence_task_Norman_GEARS"
    (gears_out / "tables").mkdir(parents=True, exist_ok=True)
    (gears_out / "reports").mkdir(parents=True, exist_ok=True)

    metric_frames = []
    for path in NORMAN_GEARS_METRICS:
        df = pd.read_csv(path)
        seed = path.parts[-3].replace("seed_", "")
        df["seed"] = int(seed) if seed.isdigit() else seed
        df["source_file"] = str(path)
        metric_frames.append(df)
    if metric_frames:
        metrics = pd.concat(metric_frames, ignore_index=True)
        metrics.to_csv(gears_out / "tables/GEARS_NORMAN_AVAILABLE_PERT_METRICS.csv", index=False)
    else:
        metrics = pd.DataFrame()

    # Existing formal GEARS output contains evaluated perturbation metrics only:
    # no predicted effects, no confidence features, no train-only record table.
    # Therefore a valid confidence rho cannot be computed without either
    # per-prediction outputs or retraining/rerunning GEARS, which Phase 4 forbids.
    result = pd.DataFrame(
        [
            {
                "dataset_name": "Norman",
                "predictor_name": "GEARS",
                "split": "heldout_perturbation",
                "n_metric_rows_available": int(len(metrics)),
                "n_unique_perturbations_available": int(metrics["perturbation"].nunique()) if not metrics.empty and "perturbation" in metrics else 0,
                "per_prediction_effect_available": False,
                "confidence_features_available": False,
                "confidence_rho_available": False,
                "direction_aligned_spearman": np.nan,
                "status": "NOT_RUN_NO_PER_PREDICTION_GEARS_OUTPUT",
                "reason": "Found GEARS perturbation-level evaluated metrics, but no predicted_effect/true_effect record table; using evaluated MSE as confidence would leak labels.",
            }
        ]
    )
    result.to_csv(out / "tables/GEARS_CONFIDENCE_RESULTS.csv", index=False)
    result.to_csv(gears_out / "tables/GEARS_CONFIDENCE_RESULTS.csv", index=False)

    (gears_out / "reports/GEARS_CONFIDENCE_AUDIT.md").write_text(
        "# GEARS confidence audit\n\n"
        "Phase 4 requested Norman GEARS PredictionRecord + confidence rho without training a new deep predictor.\n\n"
        f"- Found perturbation metric files: {len(NORMAN_GEARS_METRICS)}\n"
        f"- Available metric rows: {len(metrics)}\n"
        "- Missing: per-prediction `predicted_effect`, `true_effect`, and train-only confidence features.\n"
        "- Decision: no valid GEARS confidence rho was reported. A perturbation-level MSE table is not a confidence score.\n\n"
        "The copied available metric table is in `tables/GEARS_NORMAN_AVAILABLE_PERT_METRICS.csv`.\n",
        encoding="utf-8",
    )
    shutil.copy2(gears_out / "reports/GEARS_CONFIDENCE_AUDIT.md", out / "reports/GEARS_CONFIDENCE_AUDIT.md")
    return result


def copy_phase3_valid_tables(out: Path) -> None:
    phase3 = PROJECT_ROOT / "outputs/benchmark_phase3_blind"
    for name in ["MAIN_TABLE_GENE.csv", "CHEM_ROBUST_TABLE.csv", "DATASET_ELIGIBILITY.csv", "BLIND_CONTEXT_RESULTS.csv"]:
        src = phase3 / "tables" / name
        if src.exists():
            shutil.copy2(src, out / "tables" / name)


def write_report(out: Path, predictor: pd.DataFrame, papalexi: pd.DataFrame, cross_diag: pd.DataFrame, gears: pd.DataFrame) -> None:
    pred_preview = predictor[predictor["status"] == "ok"][
        ["dataset_name", "dataset_family", "predictor_name", "n_test", "direction_aligned_spearman"]
    ]
    pap_status = "eligible" if bool(papalexi["cross_context_eligible"].any()) else "skipped_not_cross_context_eligible"
    gears_status = str(gears["status"].iloc[0]) if not gears.empty else "missing"
    lines = [
        "# Phase 4 Experiment Report",
        "",
        "Protocol v0.2 was not modified. No new data were downloaded and no deep predictor was trained.",
        "",
        "## Predictor breakdown",
        "",
        "```",
        pred_preview.to_string(index=False),
        "```",
        "",
        "## Papalexi",
        "",
        f"- status: {pap_status}",
        "- output: `tables/Papalexi_ELIGIBILITY.csv`, `tables/Papalexi_SKIP.csv` when skipped.",
        "",
        "## crossPatient failure diagnosis",
        "",
        "```",
        cross_diag.to_string(index=False),
        "```",
        "",
        "## GEARS confidence audit",
        "",
        f"- status: {gears_status}",
        "- output: `tables/GEARS_CONFIDENCE_RESULTS.csv` and `reports/GEARS_CONFIDENCE_AUDIT.md`.",
        "",
    ]
    (out / "reports/PHASE4_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_shell_wrapper(out: Path) -> None:
    script = PROJECT_ROOT / "scripts/run_phase4_bundle.sh"
    text = f"""#!/usr/bin/env bash
set -euo pipefail
cd "{PROJECT_ROOT}"
export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"
{PYTHON} -m safetrans_confidence.cli.run_phase4_experiments --out-dir "{out}"
"""
    script.write_text(text, encoding="utf-8")
    script.chmod(0o755)
    shutil.copy2(script, out / "scripts/run_phase4_bundle.sh")


def zip_output(out: Path) -> Path:
    zip_path = out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(out.parent)))
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 experiment aggregation.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir
    ensure_dirs(out)

    predictor = build_predictor_breakdown(out)
    papalexi = audit_papalexi(out)
    cross_diag = diagnose_crosspatient(out)
    gears = audit_gears_confidence(out)
    copy_phase3_valid_tables(out)
    write_report(out, predictor, papalexi, cross_diag, gears)
    write_shell_wrapper(out)

    status = {
        "phase": "phase4_experiments",
        "out_dir": str(out),
        "zip_path": str(out.with_suffix(".zip")),
        "protocol_formula_modified": False,
        "downloaded_new_data": False,
        "trained_deep_predictor": False,
        "csv_outputs": sorted(str(p.relative_to(out)) for p in (out / "tables").glob("*.csv")),
        "papalexi_status": "eligible" if bool(papalexi["cross_context_eligible"].any()) else "skipped_not_cross_context_eligible",
        "gears_confidence_rho_available": bool(gears["confidence_rho_available"].iloc[0]) if not gears.empty else False,
        "unfinished_items": [
            "GEARS per-prediction confidence rho not available because existing files do not contain predicted_effect/true_effect records."
        ]
        if not bool(gears["confidence_rho_available"].iloc[0])
        else [],
    }
    (out / "RUN_STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    zip_path = zip_output(out)
    print(json.dumps({**status, "zip_path": str(zip_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
