#!/usr/bin/env python3
"""Build the E237 expression asset only after its metadata contract is frozen."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import anndata as ad
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/实验结果/E237_nadig_disjoint_gene_confirmation_20260923"
DATA_OUT = Path("/home/yyf/data/safeconf_e237_nadig")
ASSET = DATA_OUT / "Nadig_E237_disjoint.h5ad"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    status = json.loads((CONTRACT / "CONTRACT_STATUS.json").read_text())
    if status["status"] != "FROZEN_BEFORE_E237_EXPRESSION_OR_PREDICTION":
        raise RuntimeError("E237 contract not frozen")
    if sha256(CONTRACT / "manifests/E237_TASK_MANIFEST.csv") != status["manifest_sha256"]:
        raise RuntimeError("E237 task manifest changed")
    if ASSET.exists():
        raise RuntimeError(f"refusing to overwrite existing asset: {ASSET}")
    selected = set(pd.read_csv(CONTRACT / "tables/E237_SELECTED_PERTURBATIONS.csv").perturbation.astype(str))
    if len(selected) != status["n_selected_disjoint_genes"]:
        raise RuntimeError("selected gene count changed")
    sources = {name: Path(record["path"]) for name, record in status["source_files"].items()}
    for name, path in sources.items():
        if sha256(path) != status["source_files"][name]["sha256"]:
            raise RuntimeError(f"source file changed: {name}")
    loaded = {name: ad.read_h5ad(path) for name, path in sources.items()}
    common_genes = sorted(set.intersection(*(set(data.var_names.astype(str)) for data in loaded.values())))
    pieces, audit = [], []
    for name, data in loaded.items():
        labels = data.obs.perturbation.astype(str)
        mask = labels.isin(selected | {"control"}).to_numpy()
        piece = data[mask, common_genes].copy()
        piece.obs["context"] = name
        piece.obs["source_cell_line"] = name
        if not sp.issparse(piece.X):
            piece.X = sp.csr_matrix(piece.X)
        counts = piece.obs.perturbation.astype(str).value_counts()
        if set(selected) - set(counts.index) or min(counts[gene] for gene in selected) < 50:
            raise RuntimeError(f"selected gene coverage failed in {name}")
        audit.append({"context": name, "n_cells": int(piece.n_obs), "n_genes": int(piece.n_vars),
                      "n_control_cells": int(counts.get("control", 0)),
                      "n_selected_genes": len(selected),
                      "min_cells_per_gene": int(min(counts[gene] for gene in selected))})
        pieces.append(piece)
    combined = ad.concat(pieces, axis=0, join="inner", merge="same", index_unique="::")
    combined.var_names = common_genes
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(ASSET, compression="gzip")
    pd.DataFrame(audit).to_csv(CONTRACT / "E237_ASSET_AUDIT.csv", index=False)
    record = {"status": "COMPLETE", "created_at": datetime.now().isoformat(timespec="seconds"),
              "asset": str(ASSET), "asset_sha256": sha256(ASSET),
              "shape": [int(combined.n_obs), int(combined.n_vars)],
              "n_selected_genes": len(selected),
              "selection_or_split_changed_after_expression_read": False,
              "target_error_computed": False}
    (CONTRACT / "E237_ASSET_STATUS.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
