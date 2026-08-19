#!/usr/bin/env python3
"""Audit which datasets are eligible for cross-context held-out pair benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dataclasses import asdict

from safetrans_confidence.data.eligibility import audit_from_scan, audit_h5ad

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATLAS = Path("/home/yyf/data/singlecell_perturbation_atlas")
DEFAULT_OUT = PROJECT_ROOT / "outputs/benchmark_phase3_blind"

CANDIDATES = [
    "Haber",
    "Parekh",
    "kangCrossCell",
    "kangCrossPatient",
    "KaggleCrossCell",
    "KaggleCrossPatient",
    "TCDD",
    "sciplex3",
    "crossPatient",
    "Frangieh",
    "ShifrutMarson2018",
    "AissaBenevolenskaya2021",
    "LaraAstiasoHuntly2023",
    "Norman",
    "Adamson",
]


def write_report(df: pd.DataFrame, out: Path) -> None:
    lines = [
        "# Dataset eligibility audit",
        "",
        "Rule: cross-context eligible requires >=2 contexts, control labels, >=8 (context,pert) pairs, context!=perturbation.",
        "",
        "```",
        df.to_string(index=False),
        "```",
        "",
        "## Eligible for blind cross-context",
        "",
    ]
    ok = df[df["cross_context_eligible"]]
    if ok.empty:
        lines.append("- (none)")
    else:
        for _, r in ok.iterrows():
            lines.append(f"- **{r['dataset_name']}** ({r['dataset_family']})")
    lines.extend(["", "## Not eligible (honest exclusion)", ""])
    bad = df[~df["cross_context_eligible"]]
    for _, r in bad.iterrows():
        lines.append(f"- {r['dataset_name']}: {r['reason_if_not']}")
    (out / "reports" / "dataset_adapter_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gears-norman", type=Path, default=Path("/home/yyf/data/gears_formal_baselines_v2/norman_local_atlas/perturb_processed.h5ad"))
    parser.add_argument("--gears-adamson", type=Path, default=Path("/home/yyf/data/gears_formal_baselines_v2/adamson_local_atlas/perturb_processed.h5ad"))
    args = parser.parse_args()

    out = args.out_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    df = audit_from_scan(args.atlas_root, CANDIDATES)
    extra = []
    for label, path in [("Norman_GEARS", args.gears_norman), ("Adamson_GEARS", args.gears_adamson)]:
        row = audit_h5ad(path, label)
        extra.append(row)
    df = pd.concat([df, pd.DataFrame([asdict(r) for r in extra])], ignore_index=True)

    csv_path = out / "tables" / "DATASET_ELIGIBILITY.csv"
    df.to_csv(csv_path, index=False)
    write_report(df, out)
    status = {
        "n_candidates": len(df),
        "n_eligible": int(df["cross_context_eligible"].sum()),
        "eligible": df[df["cross_context_eligible"]]["dataset_name"].tolist(),
    }
    (out / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
