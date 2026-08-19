#!/usr/bin/env python3
"""E67: Norman replication of E65's formal GEARS--scGPT adapter contract.

This is intentionally a thin, explicit adapter around E65's tested training
implementation.  It changes the dataset, frozen GEARS audit, output paths and
provenance labels; it does not reuse Adamson cells, tasks, predictions or
errors.  The shared-contract rewrite is kept here so every E67 record says
``Norman``/``E66`` rather than inheriting E65 labels.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "scripts"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_e65_scgpt_formal_fixed_panel as core


E66 = ROOT / "docs" / "实验结果" / "E66_norman_gears_fixed_panel_formal_20260711"
OUT = ROOT / "docs" / "实验结果" / "E67_norman_scgpt_formal_fixed_panel_20260711"
DATA_ROOT = Path("/home/yyf/data/scgpt_formal_norman_fixed_panel_20260711")
SOURCE = Path("/home/yyf/data/gears_formal_baselines_v2/norman_local_atlas/perturb_processed.h5ad")


def configure() -> None:
    core.GEARS_AUDIT = E66
    core.OUT = OUT
    core.TABLES, core.ARRAYS, core.REPORTS, core.FIGURES, core.RAW = (
        OUT / "tables", OUT / "arrays", OUT / "reports", OUT / "figures", OUT / "raw_scgpt"
    )
    core.SOURCE_H5AD = SOURCE
    core.DATA_ROOT = DATA_ROOT
    core.PROCESSED_DIR = DATA_ROOT / "norman_e67_fixed512"
    core.PROCESSED_H5AD = core.PROCESSED_DIR / "perturb_processed.h5ad"
    core.MANIFEST = E66 / "tables" / "E60_FIXED_TEST_PERTURBATIONS.csv"
    core.PANEL_SEED = 20260767
    core.TRAIN_SEED = 20260767


def prepare_pertdata_norman(genes: list[str], force_rebuild: bool = False):
    """E65 processing logic, with Norman-specific data provenance."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if force_rebuild and core.PROCESSED_DIR.exists():
        raise RuntimeError("Refusing to delete existing E67 processed data automatically.")
    pert_data = core.PertData(str(DATA_ROOT))
    if core.PROCESSED_H5AD.exists():
        pert_data.load(data_path=str(core.PROCESSED_DIR))
        return pert_data
    source = core.sc.read_h5ad(SOURCE)
    try:
        source = source[:, genes].copy()
        source.var["gene_name"] = source.var_names.astype(str)
        if "cell_type" not in source.obs:
            source.obs["cell_type"] = "Norman"
        pert_data.new_data_process(core.PROCESSED_DIR.name, adata=source)
    finally:
        del source
    return pert_data


def _rewrite_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    return (
        value.replace("E65", "E67")
        .replace("E60", "E66")
        .replace("Adamson", "Norman")
        .replace("adamson", "norman")
    )


def write_records_norman(*args, **kwargs):
    """Run E65's vector construction, then make every contract field Norman-specific."""
    records, tasks, _issues = ORIGINAL_WRITE_RECORDS(*args, **kwargs)
    records = records.copy()
    for column in [
        "record_id", "task_key", "dataset_name", "dataset_group", "context",
        "predictor_name", "gene_panel_id", "normalization_id",
        "predicted_effect_key", "true_effect_key",
    ]:
        records[column] = records[column].map(_rewrite_value)
    records.to_csv(core.TABLES / "PREDICTION_RECORDS.csv", index=False)

    def rewrite_store(path: Path) -> None:
        with np.load(path) as arrays:
            payload = {_rewrite_value(key): np.asarray(arrays[key], dtype=np.float32) for key in arrays.files}
        np.savez_compressed(path, **payload)

    rewrite_store(core.ARRAYS / "predicted_effects.npz")
    rewrite_store(core.ARRAYS / "true_effects.npz")
    issues = core.validate_prediction_record_artifacts(core.OUT, records=records, strict=True)
    pd.DataFrame({"strict_issue": issues}).to_csv(core.TABLES / "E65_STRICT_CONTRACT_ISSUES.csv", index=False)
    return records, tasks, issues


def repair_text_provenance() -> None:
    """Correct the inherited human-readable labels after a completed E67 run."""
    for path in [
        OUT / "README_先看这个.md",
        OUT / "reports" / "E65_REPORT.md",
        OUT / "PREPARE_STATUS.json",
        OUT / "PREPARE_RUN_STATUS.json",
        OUT / "PREFLIGHT_STATUS.json",
        OUT / "RUN_STATUS.json",
    ]:
        if not path.exists():
            continue
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data["experiment"] = "E67_norman_scGPT_formal_fixed_panel"
                data["dataset"] = "Norman"
                data["gears_audit"] = "E66_norman_gears_fixed_panel_formal_20260711"
                data["adapter_origin"] = "E65 implementation reused; no Adamson cell/task/prediction/error reused"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            path.write_text(_rewrite_value(path.read_text(encoding="utf-8")), encoding="utf-8")
    # Keep E65-named core artifacts for resumability, and add E67 aliases used
    # by the human-readable report.  The files are byte-identical copies.
    aliases = {
        OUT / "reports" / "E65_REPORT.md": OUT / "reports" / "E67_REPORT.md",
        OUT / "tables" / "E65_FIXED_SPLIT.csv": OUT / "tables" / "E67_FIXED_SPLIT.csv",
        OUT / "tables" / "E65_GENE_PANEL.csv": OUT / "tables" / "E67_GENE_PANEL.csv",
        OUT / "tables" / "E65_TASK_RISK_TABLE.csv": OUT / "tables" / "E67_TASK_RISK_TABLE.csv",
        OUT / "tables" / "E65_RISK_ERROR_SUMMARY.csv": OUT / "tables" / "E67_RISK_ERROR_SUMMARY.csv",
        OUT / "tables" / "E65_STRICT_CONTRACT_ISSUES.csv": OUT / "tables" / "E67_STRICT_CONTRACT_ISSUES.csv",
        OUT / "tables" / "E65_TRAINING_HISTORY.csv": OUT / "tables" / "E67_TRAINING_HISTORY.csv",
    }
    for source, target in aliases.items():
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


configure()
ORIGINAL_WRITE_RECORDS = core.write_records
core.prepare_pertdata = prepare_pertdata_norman
core.write_records = write_records_norman


def main() -> None:
    core.main()
    repair_text_provenance()


if __name__ == "__main__":
    main()
