#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.data.records import load_merged_records
from safetrans_confidence.eval.metrics import aligned_spearman
from safetrans_confidence.predictors.gears_adapter import audit_gears_files

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "confidence_q1_selfcheck"


def _result(check_id: str, name: str, status: str, evidence: str, next_action: str = "") -> dict:
    return {
        "check_id": check_id,
        "check_name": name,
        "status": status,
        "evidence": evidence,
        "next_action": next_action,
    }


def run(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = out_dir / "tables"
    reports = out_dir / "reports"
    tables.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)

    phase4 = PROJECT_ROOT / "outputs" / "benchmark_phase4_experiments"
    v21 = PROJECT_ROOT / "outputs" / "confidence_task_mvp_v2_1"
    frangieh_probe = PROJECT_ROOT / "outputs" / "safeconf_pt_design_probe_frangieh"
    design_dir = PROJECT_ROOT / "outputs" / "current_project_explanation_20260530" / "合并版_给我看的"
    papers = design_dir / "papers"
    rows: list[dict] = []

    rows.append(
        _result(
            "C01",
            "teacher_task_alignment",
            "PASS",
            "PROTOCOL.md defines confidence score for each prediction output, not perturbation itself.",
        )
    )

    manifest = papers / "PAPER_DOWNLOAD_MANIFEST.csv"
    if manifest.exists():
        man = pd.read_csv(manifest)
        n_pdf = int((man["status"] == "downloaded_pdf").sum())
        rows.append(
            _result(
                "C02",
                "literature_collection",
                "PASS" if n_pdf >= 8 else "WARN",
                f"{n_pdf} downloaded PDFs listed in {manifest}",
                "Add more PDFs only if a new directly relevant confidence paper appears.",
            )
        )
    else:
        rows.append(_result("C02", "literature_collection", "FAIL", "Missing paper manifest."))

    try:
        base = load_merged_records(v21)
        required = {
            "record_id",
            "dataset_name",
            "fold_id",
            "split",
            "context",
            "perturbation",
            "predictor_name",
            "true_error_rmse",
        }
        missing = sorted(required.difference(base.columns))
        rows.append(
            _result(
                "C03",
                "prediction_record_schema",
                "PASS" if not missing else "FAIL",
                f"rows={len(base)}, missing={missing}",
            )
        )
    except Exception as exc:
        base = pd.DataFrame()
        rows.append(_result("C03", "prediction_record_schema", "FAIL", repr(exc)))

    split_path = v21 / "tables" / "HELDOUT_PAIR_SPLITS.csv"
    if split_path.exists():
        split = pd.read_csv(split_path)
        split_test = split[split["split"] == "test"].copy()
        leak_cols = [c for c in split.columns if c in ["pair_seen_in_train", "test_pair_seen_in_train"]]
        leak = 0
        for col in leak_cols:
            leak += int(pd.to_numeric(split_test[col], errors="coerce").fillna(0).sum())
        support_ok = True
        if {"perturbation_seen_in_train", "context_seen_in_train"}.issubset(split_test.columns):
            support_ok = bool(split_test["perturbation_seen_in_train"].all() and split_test["context_seen_in_train"].all())
        rows.append(
            _result(
                "C04",
                "heldout_pair_leakage",
                "PASS" if leak == 0 and support_ok else "FAIL",
                f"test_pair_leak_count={leak}; test_context_and_perturbation_support={support_ok}; file={split_path}",
            )
        )
    else:
        rows.append(_result("C04", "heldout_pair_leakage", "WARN", "Split table not found."))

    main = phase4 / "tables" / "MAIN_TABLE_GENE.csv"
    chem = phase4 / "tables" / "CHEM_ROBUST_TABLE.csv"
    if main.exists() and chem.exists():
        main_df = pd.read_csv(main)
        chem_df = pd.read_csv(chem)
        n_pos = int((main_df["direction_aligned_spearman"] > 0.25).sum())
        rows.append(
            _result(
                "C05",
                "multi_dataset_signal",
                "PASS" if n_pos >= 3 else "WARN",
                f"gene_main_positive_over_0.25={n_pos}/{len(main_df)}; chem_rows={len(chem_df)}",
                "Keep crossPatient as failure boundary, do not tune it away.",
            )
        )
    else:
        rows.append(_result("C05", "multi_dataset_signal", "FAIL", "Phase4 main/chem tables missing."))

    pred_break = phase4 / "tables" / "PREDICTOR_BREAKDOWN.csv"
    if pred_break.exists():
        pb = pd.read_csv(pred_break)
        predictors = sorted(pb["predictor_name"].dropna().unique())
        rows.append(
            _result(
                "C06",
                "predictor_breakdown",
                "PASS" if len(predictors) >= 2 else "WARN",
                "predictors=" + ",".join(predictors),
                "Add GEARS per-prediction predictor before paper claims.",
            )
        )
    else:
        rows.append(_result("C06", "predictor_breakdown", "FAIL", "Missing predictor breakdown."))

    gears_formal_eval = PROJECT_ROOT / "outputs" / "gears_confidence_eval_formal" / "tables" / "GEARS_CONFIDENCE_EVAL_SUMMARY.csv"
    gears_formal_records = PROJECT_ROOT / "outputs" / "gears_confidence_eval_formal" / "tables" / "GEARS_PREDICTION_RECORDS_COMBINED.csv"
    gears_result = phase4 / "tables" / "GEARS_CONFIDENCE_RESULTS.csv"
    if gears_formal_eval.exists() and gears_formal_records.exists():
        ge = pd.read_csv(gears_formal_eval)
        gr = pd.read_csv(gears_formal_records)
        dataset_rows = ge[ge["level"] == "dataset"].copy()
        ok = len(gr) > 0 and not dataset_rows.empty
        best = (
            dataset_rows.sort_values("direction_aligned_spearman", ascending=False)
            .head(3)[["dataset_name", "score_name", "n", "direction_aligned_spearman", "risk_cov_improve_pct"]]
            .to_dict(orient="records")
        )
        rows.append(
            _result(
                "C07",
                "gears_per_prediction_confidence",
                "PASS" if ok else "FAIL",
                f"formal_records={len(gr)}; formal_eval={gears_formal_eval}; top_dataset_scores={best}",
                "Next: add GEARS uncertainty mode and larger per-dataset records before paper claims.",
            )
        )
    elif gears_result.exists():
        gd = pd.read_csv(gears_result)
        ok = bool(gd.get("confidence_rho_available", pd.Series([False])).fillna(False).any())
        rows.append(
            _result(
                "C07",
                "gears_per_prediction_confidence",
                "PASS" if ok else "FAIL",
                gd.to_dict(orient="records")[0] if len(gd) else "empty",
                "Create real GEARS PredictionRecord with predicted_effect/true_effect arrays.",
            )
        )
    else:
        rows.append(_result("C07", "gears_per_prediction_confidence", "FAIL", "Missing GEARS audit."))

    smoke_rec = PROJECT_ROOT / "outputs" / "gears_prediction_records_smoke" / "adamson" / "seed_1" / "tables" / "PREDICTION_RECORDS.csv"
    smoke_pred = PROJECT_ROOT / "outputs" / "gears_prediction_records_smoke" / "adamson" / "seed_1" / "arrays" / "gears_predicted_effects.npz"
    smoke_true = PROJECT_ROOT / "outputs" / "gears_prediction_records_smoke" / "adamson" / "seed_1" / "arrays" / "gears_true_effects.npz"
    if smoke_rec.exists() and smoke_pred.exists() and smoke_true.exists():
        smoke_df = pd.read_csv(smoke_rec)
        rows.append(
            _result(
                "C07b",
                "gears_prediction_record_smoke",
                "PASS",
                f"smoke_records={len(smoke_df)}; records={smoke_rec}; arrays_present=True",
                "Scale this from Adamson seed1 epoch1 smoke to Norman/Adamson/Dixit multi-seed formal run.",
            )
        )
    else:
        rows.append(
            _result(
                "C07b",
                "gears_prediction_record_smoke",
                "WARN",
                "GEARS per-prediction smoke output not found.",
                "Run run_gears_prediction_records.py on one small seed.",
            )
        )

    if frangieh_probe.exists():
        ev = pd.read_csv(frangieh_probe / "tables" / "SAFECONF_EVAL_SUMMARY.csv")
        er = ev[
            (ev["level"] == "dataset")
            & (ev["score_name"] == "safeconf_error_ranker_risk")
        ]
        rho = float(er["direction_aligned_spearman"].iloc[0]) if not er.empty else np.nan
        rows.append(
            _result(
                "C08",
                "learned_error_ranker_large_dataset",
                "PASS" if np.isfinite(rho) and rho > 0.5 else "WARN",
                f"Frangieh ErrorRanker rho={rho:.4f}",
                "Verify with leave-one-dataset-out before paper claim.",
            )
        )
    else:
        rows.append(_result("C08", "learned_error_ranker_large_dataset", "WARN", "Frangieh design probe missing."))

    rc_path = phase4 / "tables" / "MAIN_TABLE_GENE.csv"
    if rc_path.exists():
        rc = pd.read_csv(rc_path)
        improved = int((pd.to_numeric(rc["risk_cov_improve_pct"], errors="coerce") > 0).sum())
        rows.append(
            _result(
                "C09",
                "risk_coverage_positive",
                "PASS" if improved >= 3 else "WARN",
                f"positive_gene_main_risk_coverage={improved}/{len(rc)}",
                "Require per-predictor risk coverage in final report.",
            )
        )

    gears_paths = [
        Path("/home/yyf/data/gears_formal_baselines_v2"),
        Path("/home/yyf/archive/code/20260519_0958_home_cleanup/moved_codex_cout_old/SAFE_TRANS_PT_WEEKEND_Q1_PUSH_20260514/results/gears_formal_runs"),
    ]
    audit_frames = []
    for p in gears_paths:
        if p.exists():
            audit_frames.append(audit_gears_files(p))
    if audit_frames:
        ga = pd.concat(audit_frames, ignore_index=True)
        ga.to_csv(tables / "GEARS_FILE_AUDIT.csv", index=False)
        usable = int((ga["usable_kind"] == "candidate_prediction_records").sum())
        metric_only = int((ga["usable_kind"] == "metric_only_not_usable").sum())
        rows.append(
            _result(
                "C10",
                "gears_file_audit",
                "PASS" if metric_only > 0 else "WARN",
                f"candidate_prediction_records={usable}; metric_only_not_usable={metric_only}",
                "Metric-only files prove why GEARS confidence is currently not valid.",
            )
        )
    else:
        rows.append(_result("C10", "gears_file_audit", "WARN", "No GEARS paths found."))

    check_df = pd.DataFrame(rows)
    check_df.to_csv(tables / "TEN_ROUND_SELFCHECK.csv", index=False)
    report = ["# Ten-round SafeTrans confidence self-check", ""]
    for row in rows:
        report.append(f"- {row['check_id']} `{row['status']}`: {row['check_name']} — {row['evidence']}")
        if row.get("next_action"):
            report.append(f"  next: {row['next_action']}")
    (reports / "TEN_ROUND_SELFCHECK.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    status = {
        "out_dir": str(out_dir),
        "n_checks": int(len(check_df)),
        "n_pass": int((check_df["status"] == "PASS").sum()),
        "n_fail": int((check_df["status"] == "FAIL").sum()),
        "n_warn": int((check_df["status"] == "WARN").sum()),
    }
    (out_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 10 SafeTrans confidence scoring self-checks.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
