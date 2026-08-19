#!/usr/bin/env python3
"""Create a row-addressable GSE225807 source without expression analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/home/yyf/data/safeconf_e182_gse225807/raw")
SOURCE_ROOT = Path("/home/yyf/data/safeconf_e182_gse225807/source")
MATRIX = RAW / "GSM7056649_matrix.mtx.gz"
FEATURES = RAW / "GSM7056649_features.tsv.gz"
BARCODES = RAW / "GSM7056649_barcodes.tsv.gz"
GUIDES = RAW / "GSM7056650_bc_to_sgrna_mapping_3_06.csv.gz"
OUTPUT = SOURCE_ROOT / "GSE225807_RBP_CRISPRI.h5ad"
ATTESTATION = SOURCE_ROOT / "F0_SOURCE_REFORMAT.json"


class IntegrityError(RuntimeError):
    """A downloaded input or structural transformation is invalid."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def unique_symbols(symbols: list[str]) -> tuple[list[str], int]:
    seen: dict[str, int] = {}
    result: list[str] = []
    duplicates = 0
    for symbol in symbols:
        occurrence = seen.get(symbol, 0)
        result.append(symbol if occurrence == 0 else f"{symbol}-{occurrence}")
        seen[symbol] = occurrence + 1
        duplicates += occurrence > 0
    return result, duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-matrix-sha256", required=True)
    args = parser.parse_args()

    inputs = (MATRIX, FEATURES, BARCODES, GUIDES)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing GSE225807 files: {missing}")
    if OUTPUT.exists() or ATTESTATION.exists():
        raise IntegrityError("E182 F0 source conversion is append-only")
    observed_matrix_sha = sha256_file(MATRIX)
    if observed_matrix_sha != args.expected_matrix_sha256:
        raise IntegrityError("GSE225807 matrix hash differs from the registered download")

    features = pd.read_csv(
        FEATURES,
        sep="\t",
        header=None,
        names=["ensembl_id", "gene_symbol", "feature_type"],
        dtype=str,
    )
    barcodes = pd.read_csv(
        BARCODES, sep="\t", header=None, names=["source_barcode"], dtype=str
    )
    guides = pd.read_csv(GUIDES, sep="\t", index_col=0, dtype=str)
    if list(guides.columns) != ["barcode", "sgRNA", "RBP"]:
        raise IntegrityError(f"unexpected GSE225807 guide columns: {list(guides.columns)}")
    if guides["barcode"].duplicated().any():
        raise IntegrityError("one GSE225807 barcode has multiple registered assignments")

    matrix = mmread(MATRIX).tocsr()
    if matrix.shape != (len(features), len(barcodes)):
        raise IntegrityError(
            f"matrix shape {matrix.shape} != features/barcodes "
            f"{(len(features), len(barcodes))}"
        )
    cell_by_gene = matrix.transpose().tocsr()
    del matrix

    guide_lookup = guides.set_index("barcode")
    bare_barcodes = barcodes["source_barcode"].str.removesuffix("-1")
    joined = guide_lookup.reindex(bare_barcodes)
    obs = pd.DataFrame(index=barcodes["source_barcode"].astype(str))
    obs.index.name = "source_barcode"
    obs["assignment_barcode"] = bare_barcodes.to_numpy()
    obs["guide_id"] = joined["sgRNA"].fillna("unassigned").to_numpy()
    obs["perturbation"] = joined["RBP"].fillna("unassigned").to_numpy()
    obs.loc[obs["perturbation"].eq("negative"), "perturbation"] = "control"
    obs["assignment_status"] = "assigned"
    obs.loc[obs["guide_id"].eq("unassigned"), "assignment_status"] = "unassigned"
    obs["source_row_index"] = range(len(obs))

    symbols, duplicate_occurrences = unique_symbols(
        features["gene_symbol"].astype(str).tolist()
    )
    var = features.copy()
    var.index = pd.Index(symbols, name="gene_symbol_unique")
    var["original_gene_symbol"] = features["gene_symbol"].astype(str).to_numpy()
    var["source_column_index"] = range(len(var))

    adata = ad.AnnData(X=cell_by_gene, obs=obs, var=var)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(OUTPUT, compression="gzip")

    check = ad.read_h5ad(OUTPUT, backed="r")
    try:
        if check.shape != (len(barcodes), len(features)):
            raise IntegrityError("E182 H5AD shape changed after round trip")
        if not check.obs_names.equals(obs.index) or not check.var_names.equals(var.index):
            raise IntegrityError("E182 H5AD axes changed after round trip")
    finally:
        check.file.close()

    record = {
        "schema": "safeconf_e182_f0_source_reformat_v1",
        "status": "PASS",
        "study": "GSE225807",
        "source_title": (
            "A Unified Framework for Systematic Identification of "
            "Post-Transcriptional Regulatory Modules [perturb_seq]"
        ),
        "raw_files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in inputs
        },
        "output": {
            "path": str(OUTPUT),
            "bytes": OUTPUT.stat().st_size,
            "sha256": sha256_file(OUTPUT),
            "shape": [int(adata.n_obs), int(adata.n_vars)],
        },
        "n_assigned_cells": int(obs["assignment_status"].eq("assigned").sum()),
        "n_unassigned_cells": int(obs["assignment_status"].eq("unassigned").sum()),
        "n_negative_control_cells": int(obs["perturbation"].eq("control").sum()),
        "n_duplicate_gene_symbol_occurrences_made_unique": duplicate_occurrences,
        "expression_values_loaded_only_for_lossless_axis_reformat": True,
        "expression_values_aggregated_or_summarized": False,
        "perturbation_effects_computed": False,
        "targets_selected_or_split": False,
        "model_predictions_or_errors_computed": False,
        "formal_truth_access_boundary_begins_after_f0_reformat": True,
        "converter": str(Path(__file__).resolve()),
        "converter_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_text(
        ATTESTATION,
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
