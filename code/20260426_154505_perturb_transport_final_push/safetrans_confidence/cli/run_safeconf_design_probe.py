#!/usr/bin/env python3
"""Run a small SafeConf-PT design probe on existing PredictionRecords."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

from safetrans_confidence.data.records import load_merged_records
from safetrans_confidence.eval.metrics import evaluate_scores
from safetrans_confidence.eval.reliability import build_reliability_tables
from safetrans_confidence.features.schema import (
    build_feature_missingness,
    build_feature_provenance_table,
    build_feature_schema_table,
)
from safetrans_confidence.scoring.conformal_calibrator import build_unsafe_flags
from safetrans_confidence.scoring.error_ranker import build_error_ranker_scores
from safetrans_confidence.scoring.protocol_v0_2 import build_protocol_v0_2_scores

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "confidence_task_mvp_v2_1"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "safeconf_pt_design_probe"


def _mkdirs(out: Path) -> None:
    for sub in ["tables", "reports", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def _zip(out: Path) -> Path:
    zip_path = out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in out.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(out.parent)))
    return zip_path


def run(input_dir: Path, out_dir: Path) -> dict:
    _mkdirs(out_dir)
    base = load_merged_records(input_dir)
    feature_schema = build_feature_schema_table(base)
    feature_provenance = build_feature_provenance_table(base)
    feature_missingness = build_feature_missingness(base)
    protocol_scores, protocol_formulas = build_protocol_v0_2_scores(base)
    ranker_scores, ranker_status = build_error_ranker_scores(base)
    all_scores = pd.concat([protocol_scores, ranker_scores], ignore_index=True)
    eval_df, risk_cov = evaluate_scores(all_scores)
    unsafe = build_unsafe_flags(all_scores, target_coverage=0.8)
    reliability_tables = build_reliability_tables(all_scores)

    base.to_csv(out_dir / "tables" / "RECORD_FEATURE_TABLE.csv", index=False)
    feature_schema.to_csv(out_dir / "tables" / "FEATURE_SCHEMA.csv", index=False)
    feature_provenance.to_csv(out_dir / "tables" / "FEATURE_PROVENANCE.csv", index=False)
    feature_missingness.to_csv(out_dir / "tables" / "FEATURE_MISSINGNESS.csv", index=False)
    all_scores.to_csv(out_dir / "tables" / "SAFECONF_SCORES.csv", index=False)
    eval_df.to_csv(out_dir / "tables" / "SAFECONF_EVAL_SUMMARY.csv", index=False)
    risk_cov.to_csv(out_dir / "tables" / "SAFECONF_RISK_COVERAGE.csv", index=False)
    unsafe.to_csv(out_dir / "tables" / "SAFECONF_UNSAFE_FLAGS.csv", index=False)
    for table_name, table in reliability_tables.items():
        table.to_csv(out_dir / "tables" / f"{table_name}.csv", index=False)
    protocol_formulas.to_csv(out_dir / "tables" / "PROTOCOL_FORMULAS.csv", index=False)
    ranker_status.to_csv(out_dir / "tables" / "ERROR_RANKER_STATUS.csv", index=False)

    main = eval_df[
        (eval_df["level"] == "dataset")
        & (eval_df["score_name"].isin(["protocol_v0_2_family_confidence", "safeconf_error_ranker_risk"]))
    ][
        [
            "dataset_family",
            "dataset_name",
            "score_name",
            "score_type",
            "n",
            "direction_aligned_spearman",
            "risk_cov_improve_pct",
        ]
    ].sort_values(["dataset_name", "score_name"])
    report = [
        "# SafeConf-PT design probe",
        "",
        "This probe uses existing PredictionRecords only. It does not train a perturbation predictor.",
        "",
        "## Main table",
        "",
        "```",
        main.to_string(index=False),
        "```",
        "",
        "## What this means",
        "",
        "- `protocol_v0_2_family_confidence` is the existing interpretable rule score.",
        "- `safeconf_error_ranker_risk` is a new shallow error-ranking risk score trained fold-locally on train+val rows only.",
        "- Extra reliability tables are exported: high/low RMSE, calibration buckets, and failure detection.",
        "- This is a code-level probe, not a paper conclusion.",
    ]
    (out_dir / "reports" / "SAFECONF_DESIGN_PROBE_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    status = {
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "n_records": int(len(base)),
        "n_scores": int(len(all_scores)),
        "extra_reliability_tables": sorted(reliability_tables),
        "zip_path": str(_zip(out_dir)),
    }
    (out_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SafeConf-PT design probe.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.input_dir, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
