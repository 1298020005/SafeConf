#!/usr/bin/env python3
"""Run protocol v0.2 rescoring benchmark from Phase 2.1 outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from safetrans_confidence.data.records import load_merged_records
from safetrans_confidence.eval.metrics import evaluate_scores
from safetrans_confidence.scoring.protocol_v0_2 import build_protocol_v0_2_scores

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V21 = PROJECT_ROOT / "outputs" / "confidence_task_mvp_v2_1"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "benchmark_protocol_v0_2_pkg"


def mkdirs(out: Path) -> None:
    for name in ["tables", "figures", "reports", "logs"]:
        (out / name).mkdir(parents=True, exist_ok=True)


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_report(out: Path, eval_df: pd.DataFrame, n_records: int, n_test: int) -> None:
    ds = eval_df[
        (eval_df["level"] == "dataset")
        & (eval_df["score_name"] == "protocol_v0_2_family_confidence")
    ][["dataset_family", "dataset_name", "direction_aligned_spearman", "risk_cov_improve_pct", "n"]]
    gene = ds[ds["dataset_family"] == "gene_main"]["direction_aligned_spearman"].median()
    chem = ds[ds["dataset_family"] == "chem_robust"]["direction_aligned_spearman"].median()
    text = [
        "# SafeTrans confidence benchmark (safetrans_confidence package)",
        "",
        f"- n_records: {n_records}",
        f"- n_test_records: {n_test}",
        f"- gene_main median protocol rho: {gene:.4f}",
        f"- chem_robust protocol rho (KCC): {chem:.4f}",
        "",
        "## Per-dataset",
        "",
        "```\n" + ds.to_string(index=False) + "\n```",
        "",
    ]
    (out / "RUN_REPORT.md").write_text("\n".join(text), encoding="utf-8")


def make_zip(out: Path) -> Path:
    zip_path = out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(out.parent)))
    return zip_path


def run(input_dir: Path, out_dir: Path, config: dict) -> dict:
    mkdirs(out_dir)
    base = load_merged_records(input_dir)
    scores, formulas = build_protocol_v0_2_scores(base)
    eval_df, cov_df = evaluate_scores(scores)

    scores.to_csv(out_dir / "tables" / "CONFIDENCE_SCORES.csv", index=False)
    eval_df.to_csv(out_dir / "tables" / "CONFIDENCE_EVAL_SUMMARY.csv", index=False)
    cov_df.to_csv(out_dir / "tables" / "RISK_COVERAGE.csv", index=False)
    base.to_csv(out_dir / "tables" / "RECORD_FEATURE_TABLE.csv", index=False)
    formulas.to_csv(out_dir / "tables" / "PROTOCOL_V0_2_FORMULAS.csv", index=False)

    # Copy resolved config
    (out_dir / "config_resolved.yaml").write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")

    n_test = int((base["split"] == "test").sum())
    write_report(out_dir, eval_df, len(base), n_test)
    zip_path = make_zip(out_dir)
    status = {
        "package": "safetrans_confidence",
        "out_dir": str(out_dir),
        "zip_path": str(zip_path),
        "input_dir": str(input_dir),
        "n_records": int(len(base)),
        "n_test_records": n_test,
    }
    (out_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run safetrans_confidence protocol v0.2 benchmark.")
    parser.add_argument("--config", type=Path, default=None, help="Optional yaml config.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_V21)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_dir = Path(cfg.get("input_dir", args.input_dir))
    out_dir = Path(cfg.get("out_dir", args.out_dir))
    status = run(input_dir, out_dir, cfg)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
