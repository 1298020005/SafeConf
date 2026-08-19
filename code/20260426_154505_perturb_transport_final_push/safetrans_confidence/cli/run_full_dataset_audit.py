#!/usr/bin/env python3
"""Full local h5ad eligibility audit for SafeConf dataset planning.

This audit is intentionally data-only: it does not train predictors and does not
download anything. It checks whether local h5ad files can support
cross-context perturbation confidence scoring.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATLAS_ROOT = Path("/home/yyf/data/singlecell_perturbation_atlas")
DEFAULT_OUT = PROJECT_ROOT / "outputs/safeconf_phaseB_full_dataset_audit"


PHASE_A_PATHS = [
    "extra_official/cellular_context_generalization/Haber.h5ad",
    "extra_official/cellular_context_generalization/Parekh.h5ad",
    "extra_official/cellular_context_generalization/kangCrossCell.h5ad",
    "extra_official/cellular_context_generalization/kangCrossPatient.h5ad",
    "extra_official/cellular_context_generalization/KaggleCrossCell.h5ad",
    "extra_official/cellular_context_generalization/KaggleCrossPatient.h5ad",
    "extra_official/cellular_context_generalization/TCDD.h5ad",
    "extra_official/cellular_context_generalization/sciplex3.h5ad",
    "extra_official/cellular_context_generalization/crossPatient.h5ad",
    "official_generalization/Frangieh.h5ad",
    "official_scperturb/ShifrutMarson2018.h5ad",
    "official_scperturb/AissaBenevolenskaya2021.h5ad",
    "official_scperturb/LaraAstiasoHuntly2023_exvivo.h5ad",
    "official_generalization/Norman.h5ad",
    "official_generalization/Adamson.h5ad",
]


CLAUDE_PHASE_B_PATHS = [
    "official_scperturb/SrivatsanTrapnell2020_sciplex3.h5ad",
    "official_scperturb/SrivatsanTrapnell2020_sciplex4.h5ad",
    "official_scperturb/SrivatsanTrapnell2020_sciplex2.h5ad",
    "official_scperturb/ReplogleWeissman2022_rpe1.h5ad",
    "official_scperturb/NormanWeissman2019_filtered.h5ad",
    "official_scperturb/ZhaoSims2021.h5ad",
    "official_scperturb/LaraAstiasoHuntly2023_exvivo.h5ad",
    "official_scperturb/LaraAstiasoHuntly2023_invivo.h5ad",
    "official_scperturb/GasperiniShendure2019_atscale.h5ad",
    "official_scperturb/XuCao2023.h5ad",
    "official_scperturb/SantinhaPlatt2023.h5ad",
    "official_scperturb/AissaBenevolenskaya2021.h5ad",
    "official_scperturb/LotfollahiTheis2023.h5ad",
    "official_scperturb/TianKampmann2019_iPSC.h5ad",
    "official_scperturb/TianKampmann2019_day7neuron.h5ad",
    "official_scperturb/TianKampmann2021_CRISPRi.h5ad",
    "official_scperturb/TianKampmann2021_CRISPRa.h5ad",
    "official_scperturb/LiangWang2023.h5ad",
    "official_scperturb/DixitRegev2016_K562_TFs_7_days.h5ad",
    "official_scperturb/DixitRegev2016_K562_TFs_13_days.h5ad",
    "official_scperturb/DixitRegev2016_K562_TFs_High_MOI.h5ad",
    "official_scperturb/WesselsSatija2023.h5ad",
    "official_scperturb/SchiebingerLander2019_GSE106340.h5ad",
    "official_scperturb/PapalexiSatija2021_eccite_RNA.h5ad",
    "official_scperturb/SchraivogelSteinmetz2020_TAP_SCREEN__chromosome_8_screen.h5ad",
    "official_scperturb/SchraivogelSteinmetz2020_TAP_SCREEN__chromosome_11_screen.h5ad",
    "official_scperturb/SunshineHein2023.h5ad",
    "official_scperturb/GasperiniShendure2019_highMOI.h5ad",
    "official_scperturb/GasperiniShendure2019_lowMOI.h5ad",
    "official_scperturb/DatlingerBock2017.h5ad",
    "official_scperturb/AdamsonWeissman2016_GSM2406677_10X005.h5ad",
    "official_generalization/sciplex3_comb.h5ad",
]


CONTROL_TOKENS = {
    "control",
    "ctrl",
    "non-targeting",
    "non_targeting",
    "non targeting",
    "nt",
    "ntc",
    "vehicle",
    "dmso",
    "untreated",
    "mock",
    "negative_control",
    "negctrl",
    "scramble",
    "scrambled",
}

PERTURBATION_CANDIDATES = [
    "perturbation",
    "condition2",
    "target",
    "target_gene",
    "gene",
    "gene_name",
    "gene_short_name",
    "sgRNA",
    "sgRNA_ID",
    "guide",
    "guide_id",
    "gRNA",
    "GenePair",
    "treatment",
    "drug",
    "drug_name",
    "compound",
    "condition",
]

CONTEXT_CANDIDATES = [
    "cell_type",
    "celltype",
    "cell_types_broad",
    "celltype_broad",
    "cell_label",
    "cell_line",
    "cellline",
    "condition1",
    "patient",
    "patient_id",
    "donor",
    "donor_id",
    "sample",
    "sample_id",
    "tissue",
    "organ",
    "time",
    "timepoint",
    "day",
    "condition",
    "batch",
    "replicate",
]

HIGH_CONTEXT_HINTS = [
    "cell_type",
    "celltype",
    "cell_types_broad",
    "celltype_broad",
    "cell_label",
    "cell_line",
    "cellline",
    "condition1",
    "patient",
    "donor",
    "sample",
    "tissue",
    "organ",
]
MEDIUM_CONTEXT_HINTS = ["time", "timepoint", "day", "condition"]
RISKY_CONTEXT_HINTS = ["batch", "replicate", "well", "plate", "library"]
FORBIDDEN_CONTEXT_HINTS = ["gene", "target", "perturb", "guide", "sgrna", "grna"]
MEASUREMENT_COL_HINTS = [
    "n_gene",
    "ngene",
    "ncounts",
    "total_count",
    "pct_count",
    "percent_",
    "umi",
    "barcode",
    "read_count",
    "size_factor",
    "mito",
    "ribo",
]


@dataclass
class FullAuditRow:
    dataset_name: str
    rel_path: str
    source_dir: str
    audit_source: str
    dataset_line: str
    file_exists: bool
    size_gb: float
    n_cells: int
    n_genes: int
    context_col_used: str
    context_quality: str
    perturbation_col_used: str
    control_label_examples: str
    has_control: bool
    n_contexts: int
    n_perturbations_total: int
    n_perturbations_noncontrol: int
    n_total_pairs_min_cells: int
    n_eligible_task_pairs: int
    max_pairs_per_context: int
    max_pairs_per_perturbation: int
    recommended_role: str
    reason: str
    obs_columns_preview: str


def _norm(value: object) -> str:
    return str(value).strip().lower()


def _has_control_value(values: pd.Series) -> tuple[bool, list[str]]:
    unique_values = pd.Series(values.dropna().astype(str).unique())
    hits = []
    for raw in unique_values:
        low = _norm(raw)
        if low in CONTROL_TOKENS or any(tok in low for tok in ["control", "non-target", "dmso", "vehicle", "untreated"]):
            hits.append(str(raw))
    return bool(hits), hits[:5]


def _is_measurement_col(col: str) -> bool:
    low = col.lower()
    return any(h in low for h in MEASUREMENT_COL_HINTS)


def _candidate_columns(obs: pd.DataFrame, candidates: list[str]) -> list[str]:
    exact = []
    lower_to_real = {str(col).lower(): str(col) for col in obs.columns}
    for cand in candidates:
        if cand.lower() in lower_to_real:
            real = lower_to_real[cand.lower()]
            if not _is_measurement_col(real):
                exact.append(real)
    fuzzy = []
    fuzzy_allowed = {
        "perturbation",
        "condition2",
        "target_gene",
        "sgrna_id",
        "guide_id",
        "genepair",
        "treatment",
        "drug_name",
        "compound",
        "cell_type",
        "cell_types_broad",
        "celltype_broad",
        "cell_line",
        "patient_id",
        "donor_id",
        "sample_id",
        "timepoint",
    }
    for cand in candidates:
        low = cand.lower()
        if low not in fuzzy_allowed:
            continue
        for col in obs.columns:
            col_s = str(col)
            if _is_measurement_col(col_s):
                continue
            if low in col_s.lower() and col_s not in exact:
                fuzzy.append(col_s)
    out = []
    for col in exact + fuzzy:
        if col not in out:
            out.append(col)
    return out


def _pick_perturbation_col(obs: pd.DataFrame) -> tuple[str, bool, list[str]]:
    best_col = ""
    best_has_control = False
    best_hits: list[str] = []
    best_score = -1
    for col in _candidate_columns(obs, PERTURBATION_CANDIDATES):
        try:
            n_unique = int(obs[col].nunique(dropna=True))
        except Exception:
            continue
        if n_unique < 2:
            continue
        has_ctrl, hits = _has_control_value(obs[col])
        score = (100 if has_ctrl else 0) + min(n_unique, 5000)
        low = col.lower()
        if any(x in low for x in ["perturb", "condition2", "target", "gene", "sgrna", "guide", "drug", "treatment"]):
            score += 50
        if score > best_score:
            best_col, best_has_control, best_hits, best_score = col, has_ctrl, hits, score
    return best_col, best_has_control, best_hits


def _context_quality(col: str) -> str:
    low = col.lower()
    if any(h in low for h in FORBIDDEN_CONTEXT_HINTS):
        return "forbidden_perturbation_like"
    if any(h in low for h in RISKY_CONTEXT_HINTS):
        return "risky_technical_proxy"
    if any(h in low for h in HIGH_CONTEXT_HINTS):
        return "high_biological_or_sample"
    if any(h in low for h in MEDIUM_CONTEXT_HINTS):
        return "medium_time_or_condition_proxy"
    return "unknown"


def _pick_context_col(obs: pd.DataFrame, perturbation_col: str) -> tuple[str, str]:
    best_col = ""
    best_quality = ""
    best_score = -1
    for col in _candidate_columns(obs, CONTEXT_CANDIDATES):
        if col == perturbation_col:
            continue
        quality = _context_quality(col)
        if quality == "forbidden_perturbation_like":
            continue
        try:
            n_unique = int(obs[col].nunique(dropna=True))
        except Exception:
            continue
        if n_unique < 2:
            continue
        if n_unique > max(5000, int(len(obs) * 0.5)):
            continue
        quality_score = {
            "high_biological_or_sample": 300,
            "medium_time_or_condition_proxy": 150,
            "risky_technical_proxy": 25,
            "unknown": 10,
        }.get(quality, 0)
        low = col.lower()
        if any(x in low for x in ["celltype", "cell_type", "cell_types", "cell_line", "cellline"]):
            priority_bonus = 120
        elif any(x in low for x in ["patient", "donor"]):
            priority_bonus = 80
        elif any(x in low for x in ["sample"]):
            priority_bonus = 40
        else:
            priority_bonus = 0
        score = quality_score + priority_bonus + min(n_unique, 200)
        if score > best_score:
            best_col, best_quality, best_score = col, quality, score
    return best_col, best_quality


def _dataset_line(name: str, rel_path: str, obs: pd.DataFrame | None = None) -> str:
    low = f"{name} {rel_path}".lower()
    if obs is not None:
        if "perturbation_type" in obs.columns:
            vals = set(obs["perturbation_type"].dropna().astype(str).str.lower().unique().tolist())
            if vals and vals.issubset({"drug", "compound", "chemical"}):
                return "chemical"
            if "drug" in vals and "crispr" not in vals and "genetic" not in vals:
                return "chemical"
        if any(col in obs.columns for col in ["sm_name", "SMILES", "dose_uM", "chembl-ID", "dose_value"]):
            return "chemical"
    if any(x in low for x in ["sciplex", "tcdd", "drug", "chemical", "santinha", "xu", "lotfollahi", "liang", "sunshine", "kagglecross", "mcfarland"]):
        return "chemical"
    if any(x in low for x in ["gasperini", "schraivogel", "enhancer"]):
        return "enhancer_or_regulatory"
    if any(x in low for x in ["comb", "combinatorial", "genepair"]):
        return "combinatorial"
    return "genetic"


def _recommend_role(row: dict, min_pairs: int, big_pair_threshold: int) -> tuple[str, str]:
    reasons = []
    if not row["file_exists"]:
        return "not_eligible", "file_missing"
    if not row["context_col_used"]:
        reasons.append("no_context_col")
    if not row["perturbation_col_used"]:
        reasons.append("no_perturbation_col")
    if row["context_col_used"] and row["context_col_used"] == row["perturbation_col_used"]:
        reasons.append("context_equals_perturbation")
    if row["context_quality"] in {"risky_technical_proxy", "unknown"}:
        reasons.append(f"context_quality={row['context_quality']}")
    if not row["has_control"]:
        reasons.append("no_control_label")
    if row["n_contexts"] < 2:
        reasons.append(f"n_contexts={row['n_contexts']}<2")
    if row["n_perturbations_noncontrol"] < 2:
        reasons.append(f"n_noncontrol_perturbations={row['n_perturbations_noncontrol']}<2")
    if row["n_eligible_task_pairs"] < min_pairs:
        reasons.append(f"eligible_task_pairs={row['n_eligible_task_pairs']}<{min_pairs}")
    if reasons:
        if row["n_eligible_task_pairs"] >= min_pairs and row["has_control"]:
            return "small_or_proxy_supplement", ";".join(reasons)
        return "not_eligible", ";".join(reasons)
    if row["n_eligible_task_pairs"] >= big_pair_threshold:
        return "big_main_candidate", ""
    if row["n_eligible_task_pairs"] >= 50:
        return "main_candidate", ""
    return "small_support_candidate", "eligible_but_pair_count_small"


def audit_one(path: Path, rel_path: str, audit_source: str, min_cells: int, min_pairs: int, big_pair_threshold: int) -> FullAuditRow:
    name = path.stem
    source_dir = "/".join(Path(rel_path).parts[:-1])
    base = {
        "dataset_name": name,
        "rel_path": rel_path,
        "source_dir": source_dir,
        "audit_source": audit_source,
        "dataset_line": _dataset_line(name, rel_path),
        "file_exists": path.exists(),
        "size_gb": round(path.stat().st_size / (1024**3), 3) if path.exists() else 0.0,
        "n_cells": 0,
        "n_genes": 0,
        "context_col_used": "",
        "context_quality": "",
        "perturbation_col_used": "",
        "control_label_examples": "",
        "has_control": False,
        "n_contexts": 0,
        "n_perturbations_total": 0,
        "n_perturbations_noncontrol": 0,
        "n_total_pairs_min_cells": 0,
        "n_eligible_task_pairs": 0,
        "max_pairs_per_context": 0,
        "max_pairs_per_perturbation": 0,
        "recommended_role": "not_eligible",
        "reason": "",
        "obs_columns_preview": "",
    }
    if not path.exists():
        base["recommended_role"], base["reason"] = _recommend_role(base, min_pairs, big_pair_threshold)
        return FullAuditRow(**base)

    adata = sc.read_h5ad(path, backed="r")
    try:
        obs = adata.obs.copy()
        base["n_cells"] = int(adata.n_obs)
        base["n_genes"] = int(adata.n_vars)
        base["dataset_line"] = _dataset_line(name, rel_path, obs)
        base["obs_columns_preview"] = ", ".join(map(str, list(obs.columns)[:40]))
        pert_col, has_ctrl, ctrl_hits = _pick_perturbation_col(obs)
        ctx_col, ctx_quality = _pick_context_col(obs, pert_col)
        base["perturbation_col_used"] = pert_col
        base["context_col_used"] = ctx_col
        base["context_quality"] = ctx_quality
        base["has_control"] = bool(has_ctrl)
        base["control_label_examples"] = "|".join(ctrl_hits)
        if ctx_col:
            base["n_contexts"] = int(obs[ctx_col].nunique(dropna=True))
        if pert_col:
            pert_values = obs[pert_col].astype(str)
            is_control = pert_values.map(lambda x: _norm(x) in CONTROL_TOKENS or any(tok in _norm(x) for tok in ["control", "non-target", "dmso", "vehicle", "untreated"]))
            base["n_perturbations_total"] = int(pert_values.nunique(dropna=True))
            base["n_perturbations_noncontrol"] = int(pert_values[~is_control].nunique(dropna=True))
        if ctx_col and pert_col:
            frame = pd.DataFrame(
                {
                    "context": obs[ctx_col].astype(str).values,
                    "perturbation": obs[pert_col].astype(str).values,
                }
            )
            control_mask = frame["perturbation"].map(
                lambda x: _norm(x) in CONTROL_TOKENS or any(tok in _norm(x) for tok in ["control", "non-target", "dmso", "vehicle", "untreated"])
            )
            control_contexts = set(
                frame[control_mask].groupby("context", observed=True).size().loc[lambda s: s >= min_cells].index
            )
            pair_counts = frame.groupby(["context", "perturbation"], observed=True).size()
            eligible_total = pair_counts[pair_counts >= min_cells].reset_index(name="n")
            base["n_total_pairs_min_cells"] = int(len(eligible_total))
            eligible_task = eligible_total[
                (~eligible_total["perturbation"].map(lambda x: _norm(x) in CONTROL_TOKENS or any(tok in _norm(x) for tok in ["control", "non-target", "dmso", "vehicle", "untreated"])))
                & (eligible_total["context"].isin(control_contexts))
            ]
            base["n_eligible_task_pairs"] = int(len(eligible_task))
            if not eligible_task.empty:
                base["max_pairs_per_context"] = int(eligible_task.groupby("context", observed=True).size().max())
                base["max_pairs_per_perturbation"] = int(eligible_task.groupby("perturbation", observed=True).size().max())
        base["recommended_role"], base["reason"] = _recommend_role(base, min_pairs, big_pair_threshold)
        return FullAuditRow(**base)
    finally:
        try:
            adata.file.close()
        except Exception:
            pass


def collect_paths(atlas_root: Path, include_all_h5ad: bool) -> list[tuple[str, str]]:
    phase_a = {p: "phaseA_known" for p in PHASE_A_PATHS}
    phase_b = {p: "claude_phaseB_candidate" for p in CLAUDE_PHASE_B_PATHS}
    merged: dict[str, str] = {}
    for rel, source in list(phase_a.items()) + list(phase_b.items()):
        merged[rel] = source if rel not in merged else f"{merged[rel]}+{source}"
    if include_all_h5ad:
        for path in sorted(atlas_root.rglob("*.h5ad")):
            rel = str(path.relative_to(atlas_root))
            if rel not in merged:
                merged[rel] = "local_extra_h5ad"
    return sorted(merged.items())


def write_report(df: pd.DataFrame, out: Path, min_pairs: int, big_pair_threshold: int) -> None:
    def md_table(table: pd.DataFrame) -> str:
        if table.empty:
            return "- None."
        clean = table.copy()
        for col in clean.columns:
            clean[col] = clean[col].map(lambda x: "" if pd.isna(x) else str(x))
        header = "| " + " | ".join(clean.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in clean.to_numpy(dtype=str)]
        return "\n".join([header, sep] + body)

    role_counts = df["recommended_role"].value_counts(dropna=False).to_dict()
    eligible = df[df["recommended_role"].isin(["big_main_candidate", "main_candidate", "small_support_candidate"])]
    big = df[df["recommended_role"] == "big_main_candidate"]
    lines = [
        "# SafeConf Phase B full dataset audit",
        "",
        "This is a server-side audit of local h5ad files. No data were downloaded and no model was trained.",
        "",
        "## What the counts mean",
        "",
        "- `n_cells`: number of single cells in the h5ad file.",
        "- `n_eligible_task_pairs`: usable non-control `(context, perturbation)` tasks with enough cells and a control group in the same context.",
        "- For this project, task pairs matter more than raw cell count, because one prediction record is an aggregated perturbation-effect task.",
        "",
        "## Thresholds",
        "",
        f"- Minimum usable dataset: `n_eligible_task_pairs >= {min_pairs}` and `n_contexts >= 2`.",
        f"- Big evidence dataset: `n_eligible_task_pairs >= {big_pair_threshold}`.",
        "- `batch` / `replicate` / unknown context columns are flagged as proxy or risky, not automatically main-table evidence.",
        "",
        "## Summary",
        "",
        f"- Audited h5ad files: {len(df)}",
        f"- Usable candidates: {len(eligible)}",
        f"- Big candidates: {len(big)}",
        f"- Role counts: `{json.dumps(role_counts, ensure_ascii=False)}`",
        "",
        "## Big candidates",
        "",
    ]
    if big.empty:
        lines.append("- None.")
    else:
        show_cols = [
            "dataset_name",
            "dataset_line",
            "source_dir",
            "size_gb",
            "n_cells",
            "n_contexts",
            "n_perturbations_noncontrol",
            "n_eligible_task_pairs",
            "context_col_used",
            "context_quality",
        ]
        lines.append(md_table(big[show_cols].sort_values("n_eligible_task_pairs", ascending=False)))
    lines.extend(["", "## All usable candidates", ""])
    if eligible.empty:
        lines.append("- None.")
    else:
        show_cols = [
            "dataset_name",
            "recommended_role",
            "dataset_line",
            "size_gb",
            "n_cells",
            "n_contexts",
            "n_perturbations_noncontrol",
            "n_eligible_task_pairs",
            "context_col_used",
            "context_quality",
            "reason",
        ]
        lines.append(md_table(eligible[show_cols].sort_values("n_eligible_task_pairs", ascending=False)))
    lines.extend(["", "## Not eligible or risky", ""])
    bad = df[~df.index.isin(eligible.index)]
    if bad.empty:
        lines.append("- None.")
    else:
        show_cols = [
            "dataset_name",
            "recommended_role",
            "dataset_line",
            "n_cells",
            "n_contexts",
            "n_perturbations_noncontrol",
            "n_eligible_task_pairs",
            "context_col_used",
            "perturbation_col_used",
            "reason",
        ]
        lines.append(md_table(bad[show_cols].sort_values(["recommended_role", "n_cells"], ascending=[True, False])))
    (out / "reports" / "FULL_DATASET_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full SafeConf local h5ad eligibility audit.")
    parser.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-cells", type=int, default=6)
    parser.add_argument("--min-pairs", type=int, default=8)
    parser.add_argument("--big-pair-threshold", type=int, default=100)
    parser.add_argument("--all-h5ad", action="store_true", help="Audit every local h5ad under atlas-root, not just Phase A + Claude candidates.")
    args = parser.parse_args()

    out = args.out_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    path_rows = collect_paths(args.atlas_root, include_all_h5ad=args.all_h5ad)
    candidate_df = pd.DataFrame(path_rows, columns=["rel_path", "audit_source"])
    candidate_df.to_csv(out / "tables" / "FULL_AUDIT_CANDIDATES.csv", index=False)

    rows = []
    for rel_path, audit_source in path_rows:
        path = args.atlas_root / rel_path
        print(f"[audit] {rel_path}", flush=True)
        rows.append(asdict(audit_one(path, rel_path, audit_source, args.min_cells, args.min_pairs, args.big_pair_threshold)))

    df = pd.DataFrame(rows)
    df.to_csv(out / "tables" / "FULL_DATASET_ELIGIBILITY.csv", index=False)
    df[df["recommended_role"].isin(["big_main_candidate", "main_candidate", "small_support_candidate"])].to_csv(
        out / "tables" / "ELIGIBLE_MAIN_CANDIDATES.csv", index=False
    )
    df[df["recommended_role"].isin(["small_or_proxy_supplement", "small_support_candidate"])].to_csv(
        out / "tables" / "BORDERLINE_DATASETS.csv", index=False
    )
    write_report(df, out, args.min_pairs, args.big_pair_threshold)

    status = {
        "atlas_root": str(args.atlas_root),
        "out_dir": str(out),
        "audited_h5ad": int(len(df)),
        "usable_candidates": int(df["recommended_role"].isin(["big_main_candidate", "main_candidate", "small_support_candidate"]).sum()),
        "big_candidates": int((df["recommended_role"] == "big_main_candidate").sum()),
        "role_counts": {str(k): int(v) for k, v in df["recommended_role"].value_counts(dropna=False).to_dict().items()},
        "big_dataset_names": df[df["recommended_role"] == "big_main_candidate"]["dataset_name"].tolist(),
    }
    (out / "RUN_STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
