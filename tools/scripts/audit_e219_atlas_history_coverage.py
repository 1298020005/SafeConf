#!/usr/bin/env python3
"""Audit whether the downloaded atlas can expand SafeConf history coverage.

Only AnnData observation labels and the frozen E201 gene list are read.  The
script never reads expression matrices, predictions, target effects or errors.
It deliberately separates raw-file availability from usable, harmonized gene
history coverage.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import anndata as ad
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ATLAS_MANIFEST = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/manifests/dataset_manifest.json"
)
E201_TASKS = (
    ROOT
    / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
    / "tables/E201_PRETRUTH_TASK_BASE.csv"
)
OUT = ROOT / "docs/实验结果/E219_database_expansion_contract_20260920"

PERTURBATION_COLUMNS = (
    "perturbation",
    "condition",
    "gene",
    "target_gene",
    "target",
    "guide_target",
)
CONTEXT_COLUMNS = (
    "cell_type",
    "celltype",
    "cell_line",
    "cellline",
    "replicate",
    "batch",
    "tissue",
    "patient",
    "donor",
    "media",
)
CONTROL_LABELS = {
    "control",
    "ctrl",
    "ntc",
    "non-targeting",
    "non_targeting",
    "nontargeting",
    "vehicle",
    "dmso",
}
GENE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9.-]{0,30}$")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def first_present(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {column.casefold(): column for column in columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def normalize_gene_label(value: object) -> str | None:
    text = str(value).strip()
    if not text or text.casefold() in CONTROL_LABELS:
        return None
    # Common single-gene task suffixes in the existing atlas.
    if text.endswith("+ctrl"):
        text = text[: -len("+ctrl")]
    if text.endswith("_1+1"):
        text = text[: -len("_1+1")]
    # Combination or compound encodings are not silently converted to a
    # single-gene history record.
    if any(separator in text for separator in ("+", ",", ";", "|", "&")):
        return None
    if not GENE_PATTERN.fullmatch(text):
        return None
    return text.upper()


def main() -> None:
    records = json.loads(ATLAS_MANIFEST.read_text(encoding="utf-8"))
    e201 = pd.read_csv(E201_TASKS)
    e201_genes = sorted(e201.loc[e201.analysis_stratum.eq("primary_ge30"), "gene"].astype(str).str.upper().unique())
    e201_set = set(e201_genes)
    schema_rows: list[dict] = []
    history_rows: list[dict] = []

    for record in records:
        path = Path(record["local_path"])
        base = {
            "study_family": str(record["study_family"]),
            "dataset_family": str(record["dataset_family"]),
            "perturbation_type_manifest": str(record["perturbation_type"]),
            "local_path": str(path),
        }
        if not path.is_file():
            schema_rows.append({**base, "read_status": "MISSING"})
            continue
        try:
            data = ad.read_h5ad(path, backed="r")
            columns = list(map(str, data.obs.columns))
            perturbation_column = first_present(columns, PERTURBATION_COLUMNS)
            context_column = first_present(columns, CONTEXT_COLUMNS)
            n_cells, n_genes = map(int, data.shape)
            labels = (
                data.obs[perturbation_column].astype(str).dropna().unique().tolist()
                if perturbation_column
                else []
            )
            contexts = (
                data.obs[context_column].astype(str).dropna().nunique()
                if context_column
                else 1
            )
            data.file.close()
            normalized = sorted(
                {
                    gene
                    for label in labels
                    if (gene := normalize_gene_label(label)) is not None
                }
            )
            overlap = sorted(set(normalized) & e201_set)
            status = (
                "EXPLICIT_PERTURBATION_COLUMN"
                if perturbation_column == "perturbation"
                else "MAPPING_REVIEW_NEEDED"
                if perturbation_column
                else "NO_RECOGNIZED_PERTURBATION_COLUMN"
            )
            schema_rows.append(
                {
                    **base,
                    "read_status": "PASS",
                    "n_cells": n_cells,
                    "n_genes": n_genes,
                    "perturbation_column": perturbation_column or "",
                    "context_column": context_column or "",
                    "n_context_labels": int(contexts),
                    "n_raw_perturbation_labels": len(labels),
                    "n_single_gene_like_labels": len(normalized),
                    "n_e201_gene_overlap": len(overlap),
                    "integration_status": status,
                    "expression_values_read": False,
                }
            )
            # Only manifest-confirmed single-gene perturbation records count
            # toward gene-history coverage.  Chemical names, regulatory
            # elements and combinations may happen to resemble gene symbols.
            if str(record["perturbation_type"]) == "genetic_single":
                for gene in overlap:
                    history_rows.append(
                        {
                            "gene": gene,
                            "study_family": str(record["study_family"]),
                            "dataset_family": str(record["dataset_family"]),
                            "perturbation_column": perturbation_column,
                            "integration_status": status,
                        }
                    )
        except Exception as exc:  # Preserve per-file failures instead of hiding them.
            schema_rows.append(
                {
                    **base,
                    "read_status": "FAIL",
                    "read_error": f"{type(exc).__name__}: {exc}",
                    "expression_values_read": False,
                }
            )

    schema = pd.DataFrame(schema_rows).sort_values(
        ["read_status", "study_family", "dataset_family"]
    )
    history = pd.DataFrame(history_rows)
    if history.empty:
        coverage = pd.DataFrame(
            {"gene": e201_genes, "n_study_families": 0, "n_dataset_records": 0}
        )
    else:
        coverage = (
            history.groupby("gene", as_index=False)
            .agg(
                n_study_families=("study_family", "nunique"),
                n_dataset_records=("study_family", "size"),
                study_families=("study_family", lambda x: "|".join(sorted(set(x)))),
            )
        )
        coverage = pd.DataFrame({"gene": e201_genes}).merge(
            coverage, on="gene", how="left"
        )
        coverage[["n_study_families", "n_dataset_records"]] = coverage[
            ["n_study_families", "n_dataset_records"]
        ].fillna(0).astype(int)
        coverage["study_families"] = coverage["study_families"].fillna("")

    readable = schema.read_status.eq("PASS")
    explicit = schema.integration_status.eq("EXPLICIT_PERTURBATION_COLUMN") if "integration_status" in schema else pd.Series(False, index=schema.index)
    covered = coverage.n_study_families.gt(0)
    multi = coverage.n_study_families.ge(2)
    summary = {
        "experiment": "E219_atlas_history_coverage_audit",
        "status": "PASS" if readable.all() else "PASS_WITH_FILE_LEVEL_FAILURES",
        "n_manifest_records": len(schema),
        "n_readable_records": int(readable.sum()),
        "n_records_with_explicit_perturbation_column": int(explicit.sum()),
        "n_e201_primary_genes": len(e201_genes),
        "n_e201_genes_with_any_atlas_history": int(covered.sum()),
        "n_e201_genes_with_two_or_more_studies": int(multi.sum()),
        "fraction_with_any_atlas_history": float(covered.mean()),
        "fraction_with_two_or_more_studies": float(multi.mean()),
        "expression_values_read": False,
        "truth_prediction_or_error_used": False,
        "coverage_scope": "manifest-confirmed genetic_single records only",
        "coverage_is_label_presence_not_harmonized_effect": True,
        "interpretation": (
            "Gene labels are broadly covered, but label presence is not yet a usable historical "
            "effect database. Expression normalization, context harmonization and source-only "
            "effect estimation remain separate gates."
        ),
    }
    atomic_text(OUT / "tables/E219_ATLAS_SCHEMA_AUDIT.csv", schema.to_csv(index=False))
    atomic_text(OUT / "tables/E219_ATLAS_E201_GENE_HISTORY.csv", coverage.to_csv(index=False))
    atomic_text(
        OUT / "ATLAS_HISTORY_COVERAGE_STATUS.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
