#!/usr/bin/env python3
"""Run E65's audited scGPT protocol on pre-registered repeated panels.

The adapter keeps one implementation for panel replications.  Each config
specifies an independent source AnnData, a completed GEARS audit, frozen test
manifest, seeds and output namespace.  Core E65 artifact names are retained
for resumability and mirrored to experiment-specific aliases after validation.
"""

from __future__ import annotations

import argparse
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


CONFIGS = {
    "adamson_panel2": {
        "dataset": "Adamson", "dataset_lower": "adamson", "experiment_code": "E76a", "gears_code": "E75a",
        "gears_audit": "E75a_adamson_gears_panel2_20260711", "out": "E76a_adamson_scgpt_panel2_20260711",
        "source": "/home/yyf/data/gears_formal_baselines_v2/adamson_local_atlas/perturb_processed.h5ad",
        "data_root": "/home/yyf/data/scgpt_formal_adamson_panel2_20260711", "processed": "adamson_e76a_fixed512",
        "panel_seed": 202607761, "train_seed": 202607761,
    },
    "norman_panel2": {
        "dataset": "Norman", "dataset_lower": "norman", "experiment_code": "E76b", "gears_code": "E75b",
        "gears_audit": "E75b_norman_gears_panel2_20260711", "out": "E76b_norman_scgpt_panel2_20260711",
        "source": "/home/yyf/data/gears_formal_baselines_v2/norman_local_atlas/perturb_processed.h5ad",
        "data_root": "/home/yyf/data/scgpt_formal_norman_panel2_20260711", "processed": "norman_e76b_fixed512",
        "panel_seed": 202607762, "train_seed": 202607762,
    },
    "frangieh_panel2": {
        "dataset": "Frangieh", "dataset_lower": "frangieh", "experiment_code": "E76c", "gears_code": "E75c",
        "gears_audit": "E75c_frangieh_gears_panel2_20260711", "out": "E76c_frangieh_scgpt_panel2_20260711",
        "source": "/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/perturb_processed.h5ad",
        "data_root": "/home/yyf/data/scgpt_formal_frangieh_panel2_20260711", "processed": "frangieh_e76c_fixed512",
        "panel_seed": 202607763, "train_seed": 202607763,
    },
    "adamson_panel3": {
        "dataset": "Adamson", "dataset_lower": "adamson", "experiment_code": "E79a", "gears_code": "E78a",
        "gears_audit": "E78a_adamson_gears_panel3_20260711", "out": "E79a_adamson_scgpt_panel3_20260711",
        "source": "/home/yyf/data/gears_formal_baselines_v2/adamson_local_atlas/perturb_processed.h5ad",
        "data_root": "/home/yyf/data/scgpt_formal_adamson_panel3_20260711", "processed": "adamson_e79a_fixed512",
        "panel_seed": 202607791, "train_seed": 202607791,
    },
    "norman_panel3": {
        "dataset": "Norman", "dataset_lower": "norman", "experiment_code": "E79b", "gears_code": "E78b",
        "gears_audit": "E78b_norman_gears_panel3_20260711", "out": "E79b_norman_scgpt_panel3_20260711",
        "source": "/home/yyf/data/gears_formal_baselines_v2/norman_local_atlas/perturb_processed.h5ad",
        "data_root": "/home/yyf/data/scgpt_formal_norman_panel3_20260711", "processed": "norman_e79b_fixed512",
        "panel_seed": 202607792, "train_seed": 202607792,
    },
    "frangieh_panel3": {
        "dataset": "Frangieh", "dataset_lower": "frangieh", "experiment_code": "E79c", "gears_code": "E78c",
        "gears_audit": "E78c_frangieh_gears_panel3_20260711", "out": "E79c_frangieh_scgpt_panel3_20260711",
        "source": "/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/perturb_processed.h5ad",
        "data_root": "/home/yyf/data/scgpt_formal_frangieh_panel3_20260711", "processed": "frangieh_e79c_fixed512",
        "panel_seed": 202607793, "train_seed": 202607793,
    },
}


def configure(config: dict) -> None:
    gears_audit = ROOT / "docs" / "实验结果" / config["gears_audit"]
    out = ROOT / "docs" / "实验结果" / config["out"]
    data_root = Path(config["data_root"])
    core.GEARS_AUDIT = gears_audit
    core.OUT = out
    core.TABLES, core.ARRAYS, core.REPORTS, core.FIGURES, core.RAW = (
        out / "tables", out / "arrays", out / "reports", out / "figures", out / "raw_scgpt"
    )
    core.SOURCE_H5AD = Path(config["source"])
    core.DATA_ROOT = data_root
    core.PROCESSED_DIR = data_root / config["processed"]
    core.PROCESSED_H5AD = core.PROCESSED_DIR / "perturb_processed.h5ad"
    core.MANIFEST = gears_audit / "tables" / "E60_FIXED_TEST_PERTURBATIONS.csv"
    core.PANEL_SEED = int(config["panel_seed"])
    core.TRAIN_SEED = int(config["train_seed"])


def prepare_pertdata_generic(config: dict, genes: list[str], force_rebuild: bool = False):
    core.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if force_rebuild and core.PROCESSED_DIR.exists():
        raise RuntimeError(f"Refusing to delete existing {config['experiment_code']} processed data automatically.")
    pert_data = core.PertData(str(core.DATA_ROOT))
    if core.PROCESSED_H5AD.exists():
        pert_data.load(data_path=str(core.PROCESSED_DIR))
        return pert_data
    source = core.sc.read_h5ad(core.SOURCE_H5AD)
    try:
        source = source[:, genes].copy()
        source.var["gene_name"] = source.var_names.astype(str)
        if "cell_type" not in source.obs:
            source.obs["cell_type"] = config["dataset"]
        pert_data.new_data_process(core.PROCESSED_DIR.name, adata=source)
    finally:
        del source
    return pert_data


def replacement(config: dict, value: object) -> object:
    if not isinstance(value, str):
        return value
    return (
        value.replace("E65", config["experiment_code"]).replace("E60", config["gears_code"])
        .replace("e65", config["experiment_code"].lower()).replace("e60", config["gears_code"].lower())
        .replace("Adamson", config["dataset"]).replace("adamson", config["dataset_lower"])
    )


RECORD_COLUMNS = [
    "record_id", "task_key", "dataset_name", "dataset_group", "context",
    "predictor_name", "gene_panel_id", "normalization_id",
    "predicted_effect_key", "true_effect_key",
]


def rewrite_npz(path: Path, config: dict) -> None:
    with np.load(path) as arrays:
        payload = {replacement(config, key): np.asarray(arrays[key], dtype=np.float32) for key in arrays.files}
    np.savez_compressed(path, **payload)


def write_records_generic(config: dict, original, *args, **kwargs):
    records, tasks, _issues = original(*args, **kwargs)
    records = records.copy()
    for column in RECORD_COLUMNS:
        records[column] = records[column].map(lambda value: replacement(config, value))
    records.to_csv(core.TABLES / "PREDICTION_RECORDS.csv", index=False)
    rewrite_npz(core.ARRAYS / "predicted_effects.npz", config)
    rewrite_npz(core.ARRAYS / "true_effects.npz", config)
    issues = core.validate_prediction_record_artifacts(core.OUT, records=records, strict=True)
    pd.DataFrame({"strict_issue": issues}).to_csv(core.TABLES / "E65_STRICT_CONTRACT_ISSUES.csv", index=False)
    return records, tasks, issues


def repair(config: dict) -> None:
    out = core.OUT
    text_paths = [
        out / "README_先看这个.md", out / "reports" / "E65_REPORT.md",
        out / "figures" / "F1_gears_scgpt_disagreement_vs_mean_error.svg",
    ]
    for path in text_paths:
        if path.exists():
            path.write_text(replacement(config, path.read_text(encoding="utf-8")), encoding="utf-8")
    for path in [out / "PREPARE_STATUS.json", out / "PREPARE_RUN_STATUS.json", out / "PREFLIGHT_STATUS.json", out / "RUN_STATUS.json"]:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "panel_selection" in data:
            data["panel_selection"] = replacement(config, data["panel_selection"])
        if "normalization_id" in data:
            data["normalization_id"] = replacement(config, data["normalization_id"])
        data["experiment"] = f"{config['experiment_code']}_{config['dataset_lower']}_scGPT_formal_panel2"
        data["dataset"] = config["dataset"]
        data["gears_audit"] = config["gears_audit"]
        data["adapter_origin"] = "E65 audited implementation reused; no prior-panel task/prediction/error reused"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    records_path = out / "tables" / "PREDICTION_RECORDS.csv"
    if records_path.exists():
        records = pd.read_csv(records_path)
        for column in RECORD_COLUMNS:
            records[column] = records[column].map(lambda value: replacement(config, value))
        records.to_csv(records_path, index=False)
        issues = core.validate_prediction_record_artifacts(out, records=records, strict=True)
        pd.DataFrame({"strict_issue": issues}).to_csv(out / "tables" / "E65_STRICT_CONTRACT_ISSUES.csv", index=False)
    code = config["experiment_code"]
    aliases = {
        out / "reports" / "E65_REPORT.md": out / "reports" / f"{code}_REPORT.md",
        out / "tables" / "E65_FIXED_SPLIT.csv": out / "tables" / f"{code}_FIXED_SPLIT.csv",
        out / "tables" / "E65_GENE_PANEL.csv": out / "tables" / f"{code}_GENE_PANEL.csv",
        out / "tables" / "E65_TASK_RISK_TABLE.csv": out / "tables" / f"{code}_TASK_RISK_TABLE.csv",
        out / "tables" / "E65_RISK_ERROR_SUMMARY.csv": out / "tables" / f"{code}_RISK_ERROR_SUMMARY.csv",
        out / "tables" / "E65_STRICT_CONTRACT_ISSUES.csv": out / "tables" / f"{code}_STRICT_CONTRACT_ISSUES.csv",
        out / "tables" / "E65_TRAINING_HISTORY.csv": out / "tables" / f"{code}_TRAINING_HISTORY.csv",
    }
    for source, target in aliases.items():
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def main() -> None:
    adapter = argparse.ArgumentParser(add_help=False)
    adapter.add_argument("--adapter-config", required=True, choices=sorted(CONFIGS))
    known, remaining = adapter.parse_known_args()
    config = CONFIGS[known.adapter_config]
    sys.argv = [sys.argv[0], *remaining]
    configure(config)
    original_write = core.write_records
    core.prepare_pertdata = lambda genes, force_rebuild=False: prepare_pertdata_generic(config, genes, force_rebuild)
    core.write_records = lambda *args, **kwargs: write_records_generic(config, original_write, *args, **kwargs)
    core.main()
    repair(config)


if __name__ == "__main__":
    main()
