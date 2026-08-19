#!/usr/bin/env python3
"""Run PRESCRIBE's Step1 on an E91 frozen Norman panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRESCRIBE = Path("/home/yyf/archive/external/PRESCRIBE")
RAW = Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/NormanWeissman2019_filtered.h5ad")
E91 = ROOT / "docs/实验结果/E91_prescribe_norman_contract_20260712"
OUT = ROOT / "docs/实验结果/E93_prescribe_preprocess_20260712"
PANELS = {"p1": "Norman_P1", "p2": "Norman_P2"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=sorted(PANELS), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    panel_name = PANELS[args.panel]
    dataset_name = f"norman_{args.panel}"
    split_source = E91 / "manifests" / f"{panel_name}_set2conditions.pkl"
    dataset_dir = PRESCRIBE / "data" / dataset_name
    processed = dataset_dir / "perturb_processed.h5ad"
    if processed.exists() and not args.force:
        print(f"Already processed: {processed}")
        return
    dataset_dir.mkdir(parents=True, exist_ok=True)
    split_target = dataset_dir / "set2conditions_3407.pkl"
    shutil.copy2(split_source, split_target)

    source_path = PRESCRIBE / "Step1_preprocess.py"
    source = source_path.read_text()
    needle = '"data/norman/set2conditions_3407.pkl"'
    replacement = f'"data/{dataset_name}/set2conditions_3407.pkl"'
    if source.count(needle) != 1:
        raise RuntimeError("Upstream Step1 split-path anchor changed")
    source = source.replace(needle, replacement)
    old_argv = sys.argv
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    try:
        os.chdir(PRESCRIBE)
        sys.path.insert(0, str(PRESCRIBE))
        sys.argv = [str(source_path), "--file", str(RAW), "--dataset_name", dataset_name]
        exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
    finally:
        sys.argv = old_argv
        sys.path[:] = old_sys_path
        os.chdir(old_cwd)
    required = [processed, dataset_dir / "perturb_e_distance.h5ad", dataset_dir / "data_pyg/mean.npy", dataset_dir / "data_pyg/cov.npy"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("PRESCRIBE preprocessing outputs missing: " + ", ".join(missing))
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / f"{dataset_name}_STATUS.json"
    status = {
        "experiment": "E93_prescribe_preprocess",
        "panel": panel_name,
        "dataset_name": dataset_name,
        "phase": "preprocessing_complete",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prescribe_commit": "6f7264a205aaff654a9594863c5c10b656f88ebe",
        "upstream_step1_sha256": sha256(source_path),
        "runtime_change": f"split path only: {needle} -> {replacement}",
        "raw_dataset": str(RAW),
        "raw_dataset_sha256": sha256(RAW),
        "split_manifest": str(split_source),
        "split_manifest_sha256": sha256(split_source),
        "processed_h5ad": str(processed),
        "processed_h5ad_sha256": sha256(processed),
        "target_test_expression_used_for_task_selection": False,
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
