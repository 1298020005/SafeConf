#!/usr/bin/env python3
"""E137: build the contract-filtered two-cell-line Nadig h5ad asset."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import anndata as ad
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = ROOT / "docs/实验结果/E136_nadig_two_cellline_contract_20260714"
SELECTION = CONTRACT_ROOT / "tables/E136_SELECTED_PERTURBATIONS.csv"
OUT = ROOT / "docs/实验结果/E137_nadig_combined_asset_20260714"
DATA_OUT = Path("/home/yyf/data/safeconf_e137_nadig")
COMBINED = DATA_OUT / "Nadig_two_cellline_E136_selected.h5ad"
SOURCES = {
    "HepG2": Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/NadigOConner2024_hepg2.h5ad"),
    "Jurkat": Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/NadigOConner2024_jurkat.h5ad"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((CONTRACT_ROOT / "RUN_STATUS.json").read_text())
    for context, path in SOURCES.items():
        if sha256(path) != contract["sources"][context]["sha256"]:
            raise RuntimeError(f"source hash changed: {context}")
    selected = set(pd.read_csv(SELECTION).perturbation.astype(str))
    loaded = {context: ad.read_h5ad(path) for context, path in SOURCES.items()}
    common_genes = sorted(set.intersection(*(set(data.var_names.astype(str)) for data in loaded.values())))
    pieces, audit = [], []
    for context, data in loaded.items():
        perturbations = data.obs.perturbation.astype(str)
        mask = perturbations.isin(selected | {"control"}).to_numpy()
        piece = data[mask, common_genes].copy()
        piece.obs["context"] = context
        piece.obs["source_cell_line"] = context
        if not sp.issparse(piece.X):
            piece.X = sp.csr_matrix(piece.X)
        pieces.append(piece)
        counts = piece.obs.perturbation.astype(str).value_counts()
        audit.append({
            "context": context,
            "n_cells": int(piece.n_obs),
            "n_genes": int(piece.n_vars),
            "n_control_cells": int(counts.get("control", 0)),
            "n_selected_perturbations_present": int(len(set(counts.index) & selected)),
            "min_cells_per_selected_perturbation": int(min(counts[value] for value in selected)),
        })
    combined = ad.concat(pieces, axis=0, join="inner", merge="same", index_unique="::")
    combined.var_names = common_genes
    combined.write_h5ad(COMBINED, compression="gzip")
    pd.DataFrame(audit).to_csv(OUT / "E137_ASSET_AUDIT.csv", index=False)
    status = {
        "experiment": "E137_nadig_combined_asset",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "source_contract": str(CONTRACT_ROOT.relative_to(ROOT)),
        "combined_asset": str(COMBINED),
        "combined_sha256": sha256(COMBINED),
        "shape": [int(combined.n_obs), int(combined.n_vars)],
        "contexts": sorted(combined.obs.context.astype(str).unique().tolist()),
        "selected_perturbations": len(selected),
        "expression_values_used_for_contract_selection_or_split": False,
        "test_effect_or_error_computed": False,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    (OUT / "E137_REPORT.md").write_text(
        "# E137｜Nadig 双细胞系组合资产\n\n"
        f"按 E136 已冻结的 96 个扰动和 control 过滤两个源文件，取共同 {combined.n_vars} 个基因，合并得到 {combined.n_obs} 个细胞。"
        "本步骤在合同冻结后读取表达矩阵，只构建模型输入资产，没有重新挑任务，也没有计算测试效应或误差。\n"
    )
    (OUT / "README_先看这个.md").write_text("# E137 先看这个\n\n先读 `E137_REPORT.md`。大体积 h5ad 位于状态文件记录的数据目录，不进入 Git。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(pd.DataFrame(audit).to_string(index=False))


if __name__ == "__main__":
    main()
