#!/usr/bin/env python3
"""Phase 3: score blind pipeline outputs + assemble main tables (frozen protocol v0.2)."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from safetrans_confidence.cli.run_benchmark import run as run_benchmark_scoring
from safetrans_confidence.data.eligibility import audit_from_scan
from safetrans_confidence.gears.supplement import write_gears_supplement

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "outputs/benchmark_phase3_blind"
ATLAS = Path("/home/yyf/data/singlecell_perturbation_atlas")

# Completed or to-be-scored blind run directories (input must have v2_1-style tables/)
BLIND_INPUTS = {
    "KaggleCrossPatient": PROJECT_ROOT / "outputs/confidence_task_kcp_blind_probe",
    "crossPatient": PROJECT_ROOT / "outputs/confidence_task_crossPatient_blind",
    "Frangieh": PROJECT_ROOT / "outputs/confidence_task_Frangieh_blind",
}


def score_blind_dataset(name: str, input_dir: Path, out_root: Path) -> pd.DataFrame:
    pkg_out = out_root / f"scored_{name}"
    status = run_benchmark_scoring(input_dir, pkg_out, {"dataset": name})
    eval_df = pd.read_csv(pkg_out / "tables" / "CONFIDENCE_EVAL_SUMMARY.csv")
    sub = eval_df[
        (eval_df["level"] == "dataset")
        & (eval_df["score_name"] == "protocol_v0_2_family_confidence")
    ].copy()
    sub["blind_source"] = str(input_dir)
    return sub


def build_main_tables(out: Path, blind_rows: list[pd.DataFrame], baseline_pkg: Path, eligibility: Path) -> dict[str, pd.DataFrame]:
    base_eval = pd.read_csv(baseline_pkg / "tables" / "CONFIDENCE_EVAL_SUMMARY.csv")
    base_sub = base_eval[
        (base_eval["level"] == "dataset")
        & (base_eval["score_name"] == "protocol_v0_2_family_confidence")
    ][["dataset_family", "dataset_name", "direction_aligned_spearman", "n", "risk_cov_improve_pct"]]
    base_sub = base_sub.copy()
    base_sub["blind_source"] = ""

    blind = pd.concat(blind_rows, ignore_index=True) if blind_rows else pd.DataFrame()
    if not blind.empty:
        blind = blind[
            [
                "dataset_family",
                "dataset_name",
                "direction_aligned_spearman",
                "n",
                "risk_cov_improve_pct",
                "blind_source",
            ]
        ]

    elig = pd.read_csv(eligibility)
    gene_ok = set(elig[(elig["cross_context_eligible"]) & (elig["dataset_family"] == "gene_main")]["dataset_name"])

    gene = base_sub[base_sub["dataset_family"] == "gene_main"].copy()
    if not blind.empty:
        gene_blind = blind[blind["dataset_name"].isin(gene_ok)]
        gene = pd.concat([gene, gene_blind[gene_blind["dataset_family"] == "gene_main"]], ignore_index=True)

    chem = base_sub[base_sub["dataset_family"] == "chem_robust"].copy()
    if not blind.empty:
        chem_blind = blind[blind["dataset_family"] == "chem_robust"]
        chem = pd.concat([chem, chem_blind], ignore_index=True).drop_duplicates(subset=["dataset_name"])

    stale = out / "tables" / "MAIN_GENE_TABLE.csv"
    if stale.exists():
        stale.unlink()
    gene.to_csv(out / "tables" / "MAIN_TABLE_GENE.csv", index=False)
    chem.to_csv(out / "tables" / "CHEM_ROBUST_TABLE.csv", index=False)
    return {"gene": gene, "chem": chem, "blind": blind, "eligibility": elig, "baseline": base_sub}


def collect_risk_coverage(out: Path, baseline_pkg: Path) -> pd.DataFrame:
    frames = []
    base_path = baseline_pkg / "tables" / "RISK_COVERAGE.csv"
    if base_path.exists():
        df = pd.read_csv(base_path)
        df["source"] = "baseline_protocol_v0_2_pkg"
        frames.append(df)
    for p in sorted(out.glob("scored_*/tables/RISK_COVERAGE.csv")):
        df = pd.read_csv(p)
        df["source"] = p.parents[1].name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    cov = pd.concat(frames, ignore_index=True)
    cov.to_csv(out / "tables" / "PHASE3_RISK_COVERAGE.csv", index=False)
    return cov


def write_phase3_figures(out: Path, tables: dict[str, pd.DataFrame], risk_cov: pd.DataFrame) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([tables["gene"], tables["chem"]], ignore_index=True)
    if not combined.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        colors = combined["dataset_family"].map({"gene_main": "#4C78A8", "chem_robust": "#F58518"}).fillna("#777777")
        ax.bar(combined["dataset_name"], combined["direction_aligned_spearman"], color=colors)
        ax.axhline(0.25, color="#555555", linestyle="--", linewidth=1, label="blind evidence threshold")
        ax.set_ylabel("aligned Spearman")
        ax.set_title("Phase 3 confidence score vs true error")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "phase3_per_dataset_spearman.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        fam = combined.groupby("dataset_family", as_index=False)["direction_aligned_spearman"].median()
        ax.bar(fam["dataset_family"], fam["direction_aligned_spearman"], color=["#F58518" if x == "chem_robust" else "#4C78A8" for x in fam["dataset_family"]])
        ax.set_ylabel("median aligned Spearman")
        ax.set_title("Gene main vs chemical robustness")
        fig.tight_layout()
        fig.savefig(fig_dir / "phase3_gene_vs_chem_summary.png", dpi=220)
        plt.close(fig)

    if not risk_cov.empty:
        score_col = "score_name"
        keep_score = "protocol_v0_2_family_confidence"
        rc = risk_cov[risk_cov.get(score_col, "").eq(keep_score)].copy()
        if {"dataset_name", "coverage", "mean_rmse"}.issubset(rc.columns) and not rc.empty:
            fig, ax = plt.subplots(figsize=(8.5, 5.0))
            for dataset, g in rc.groupby("dataset_name"):
                gg = g.sort_values("coverage")
                ax.plot(gg["coverage"], gg["mean_rmse"], marker="o", label=str(dataset))
            ax.set_xlabel("coverage")
            ax.set_ylabel("mean true RMSE")
            ax.set_title("Phase 3 risk-coverage")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / "phase3_risk_coverage.png", dpi=220)
            plt.close(fig)


def write_phase3_report(out: Path, tables: dict[str, pd.DataFrame], gears: pd.DataFrame) -> None:
    report = [
        "# Phase 3 Blind Report",
        "",
        "Protocol v0.2 was kept frozen. These tables do not use pooled all-dataset Spearman as a headline.",
        "",
        "## Gene main table",
        "",
        "```",
        tables["gene"].to_string(index=False),
        "```",
        "",
        "## Chemical robustness table",
        "",
        "```",
        tables["chem"].to_string(index=False),
        "```",
        "",
        "## Blind context results",
        "",
        "```",
        (tables["blind"].to_string(index=False) if not tables["blind"].empty else "No blind scored datasets found."),
        "```",
        "",
        "## Dataset eligibility",
        "",
        "```",
        tables["eligibility"].to_string(index=False),
        "```",
        "",
        "## Interpretation rules",
        "",
        "- KaggleCrossPatient is chemical robustness, not gene_main.",
        "- crossPatient and Frangieh are blind gene_main only if their rows appear in MAIN_TABLE_GENE.csv.",
        "- Norman and Adamson official h5ad files are excluded from cross-context main tables because they lack a true context column.",
        "- GEARS is reported as a formal baseline supplement only; current files are not per-prediction confidence records.",
        "",
        "## GEARS supplement preview",
        "",
        "```",
        gears.to_string(index=False),
        "```",
        "",
    ]
    (out / "reports" / "PHASE3_BLIND_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--baseline-pkg", type=Path, default=PROJECT_ROOT / "outputs/benchmark_protocol_v0_2_pkg")
    args = parser.parse_args()
    out = args.out_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    elig_path = out / "tables" / "DATASET_ELIGIBILITY.csv"
    if not elig_path.exists():
        df = audit_from_scan(ATLAS, ["Haber", "Parekh", "KaggleCrossCell", "KaggleCrossPatient", "crossPatient", "Frangieh", "Norman", "Adamson"])
        gears_rows = []
        from safetrans_confidence.data.eligibility import audit_h5ad
        for ds, path in {
            "Norman_GEARS": Path("/home/yyf/data/gears_formal_baselines_v2/norman_local_atlas/perturb_processed.h5ad"),
            "Adamson_GEARS": Path("/home/yyf/data/gears_formal_baselines_v2/adamson_local_atlas/perturb_processed.h5ad"),
        }.items():
            gears_rows.append(audit_h5ad(path, ds).__dict__)
        df = pd.concat([df, pd.DataFrame(gears_rows)], ignore_index=True)
        df.to_csv(elig_path, index=False)
        (out / "reports" / "dataset_adapter_audit.md").write_text(
            "# Dataset eligibility audit\n\n"
            "Rule: cross-context eligible requires >=2 contexts, control labels, >=8 (context,pert) pairs, context!=perturbation.\n\n"
            "```\n" + df.to_string(index=False) + "\n```\n",
            encoding="utf-8",
        )

    blind_results = []
    for name, inp in BLIND_INPUTS.items():
        if (inp / "tables" / "PREDICTION_RECORDS.csv").exists():
            blind_results.append(score_blind_dataset(name, inp, out))

    tables = build_main_tables(out, blind_results, args.baseline_pkg, elig_path)

    if blind_results:
        pd.concat(blind_results, ignore_index=True).to_csv(out / "tables" / "BLIND_CONTEXT_RESULTS.csv", index=False)
    else:
        pd.DataFrame().to_csv(out / "tables" / "BLIND_CONTEXT_RESULTS.csv", index=False)

    risk_cov = collect_risk_coverage(out, args.baseline_pkg)
    gears = write_gears_supplement(out)
    write_phase3_figures(out, tables, risk_cov)
    write_phase3_report(out, tables, gears)

    status = {
        "out_dir": str(out),
        "zip": str(out.with_suffix(".zip")),
        "blind_scored": [name for name, inp in BLIND_INPUTS.items() if (inp / "tables" / "PREDICTION_RECORDS.csv").exists()],
        "required_outputs": {
            "MAIN_TABLE_GENE.csv": (out / "tables" / "MAIN_TABLE_GENE.csv").exists(),
            "CHEM_ROBUST_TABLE.csv": (out / "tables" / "CHEM_ROBUST_TABLE.csv").exists(),
            "BLIND_CONTEXT_RESULTS.csv": (out / "tables" / "BLIND_CONTEXT_RESULTS.csv").exists(),
            "GEARS_SUPPLEMENT_TABLE.csv": (out / "tables" / "GEARS_SUPPLEMENT_TABLE.csv").exists(),
            "PHASE3_BLIND_REPORT.md": (out / "reports" / "PHASE3_BLIND_REPORT.md").exists(),
        },
    }
    (out / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    zip_path = out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(out.parent)))

    print(f"Phase3 assembly done: {out}")


if __name__ == "__main__":
    main()
