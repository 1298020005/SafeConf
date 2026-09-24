#!/usr/bin/env python3
"""Map Feng count columns to train/validation tasks or permitted controls.

No count-gene row is opened. Test-donor targeted cells and ineligible tasks map
to -1; the native aggregator will skip their numeric tokens without parsing.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_e258_raw_header import count_matrix_id
from build_e258_official_dev_view import TRAIN, VALIDATION, TEST, task_contract


RAW_MD5 = "f0d613b1ea8147425501baddba034158"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(metadata: Path, counts: Path, view: Path, output_dir: Path) -> dict:
    meta, task_keys = task_contract(metadata)
    if len(task_keys) != 4292:
        raise ValueError("eligible train/validation task count changed")
    control = meta.Guide_Call.eq("unassigned") | meta.Guide_Call.str.startswith("NonTarget_")
    target = meta.Guide_Call.str.rsplit("_", n=1).str[0]
    donor = meta.Cell_Line.str.split("_").str[0]
    controls = sorted(set(zip(meta.loc[control, "Cell_Line"], meta.loc[control, "Batch"])))
    groups = [{"kind": "control", "line": line, "batch": batch,
               "donor": line.split("_")[0]}
              for line, batch in controls]
    groups.extend({"kind": "target", "line": line, "target": gene,
                   "donor": line.split("_")[0]}
                  for line, gene in task_keys)
    control_ids = {key: i for i, key in enumerate(controls)}
    target_ids = {key: i + len(controls) for i, key in enumerate(task_keys)}
    labels = np.full(len(meta), -1, dtype=np.int32)
    control_rows = np.flatnonzero(control.to_numpy())
    for row in control_rows:
        labels[row] = control_ids[(meta.Cell_Line.iat[row], meta.Batch.iat[row])]
    candidate_rows = np.flatnonzero((~control & donor.isin(TRAIN + VALIDATION)).to_numpy())
    for row in candidate_rows:
        labels[row] = target_ids.get((meta.Cell_Line.iat[row], target.iat[row]), -1)
    if np.any(labels[donor.isin(TEST).to_numpy() & ~control.to_numpy()] >= 0):
        raise ValueError("test targeted cell assigned to an output group")
    if np.any(labels[control.to_numpy()] < 0):
        raise ValueError("permitted control cell omitted")
    counts_per_group = np.bincount(labels[labels >= 0], minlength=len(groups))
    for group, n in zip(groups, counts_per_group):
        group["n_cells"] = int(n)
        if n == 0:
            raise ValueError(f"empty group: {group}")
    batch_counts = (
        meta.loc[labels >= len(controls), ["Cell_Line", "Batch"]]
        .assign(group=labels[labels >= len(controls)])
        .groupby(["group", "Batch"]).size()
    )
    batch_by_group = defaultdict(list)
    for (group_id, batch), n in batch_counts.items():
        line = groups[group_id]["line"]
        batch_by_group[group_id].append({"control_group": control_ids[(line, batch)],
                                         "target_cells": int(n)})
    for group_id in range(len(controls), len(groups)):
        batches = batch_by_group[group_id]
        if sum(b["target_cells"] for b in batches) != groups[group_id]["n_cells"]:
            raise ValueError("task batch mapping incomplete")
        groups[group_id]["matched_control_batches"] = batches
    normalized = [count_matrix_id(value) for value in meta.Cell_ID.astype(str)]
    if len(set(normalized)) != len(meta):
        raise ValueError("duplicate normalized metadata IDs")
    lookup = dict(zip(normalized, labels))
    with gzip.open(counts, "rt") as stream:
        header = stream.readline().rstrip("\r\n").split(",")
    if header[0] != "" or len(header) != 1_161_866:
        raise ValueError("raw count header changed")
    cells = header[1:]
    if len(set(cells)) != len(cells) or len(set(normalized) & set(cells)) != len(meta):
        raise ValueError("count/metadata cell alignment failed")
    raw_labels = np.asarray([lookup.get(cell, -1) for cell in cells], dtype=np.int32)
    if np.count_nonzero(raw_labels >= 0) != np.count_nonzero(labels >= 0):
        raise ValueError("selected cell count changed after raw-column mapping")
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "E258_RAW_DEV_COLUMN_GROUPS.i32"
    manifest_path = output_dir / "E258_RAW_DEV_GROUPS.json"
    axis_path = output_dir / "E258_DEV_SYMBOL_AXIS.txt"
    if mapping_path.exists() or manifest_path.exists() or axis_path.exists():
        raise FileExistsError("raw group artifacts already exist")
    with np.load(view, allow_pickle=False) as engineering:
        symbols = engineering["gene"].astype(str).tolist()
    if len(symbols) != 6520 or len(set(symbols)) != len(symbols):
        raise ValueError("engineering train/validation symbol axis changed")
    axis_path.write_text("\n".join(symbols) + "\n")
    pending = mapping_path.with_suffix(mapping_path.suffix + ".part")
    raw_labels.tofile(pending)
    pending.replace(mapping_path)
    result = {
        "stage": "E258_RAW_TRAIN_VALIDATION_PLUS_CONTROL_GROUP_MAPPING",
        "raw_count_md5_verified_separately": RAW_MD5,
        "metadata_cells": len(meta),
        "raw_columns": len(cells),
        "group_count": len(groups),
        "control_groups": len(controls),
        "target_groups": len(task_keys),
        "selected_control_cells": int(control.sum()),
        "selected_train_validation_target_cells": int(
            np.count_nonzero(labels >= len(controls))),
        "test_target_numeric_values_read": 0,
        "count_gene_rows_read": 0,
        "mapping_sha256": sha256(mapping_path),
        "gene_axis_sha256": sha256(axis_path),
        "gene_axis_origin": "symbols common to official-LFC train/validation engineering view; no test perturbation effect values",
        "n_gene_symbols": len(symbols),
        "groups": groups,
    }
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "groups"},
                     ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/TargetedScreen_Cell-Metadata.tsv.gz"))
    parser.add_argument("--counts", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/TargetedScreen_RNA-UMI-Counts.csv.gz"))
    parser.add_argument("--view", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/E258_OFFICIAL_DEV_VIEW.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/raw_dev"))
    args = parser.parse_args()
    prepare(args.metadata, args.counts, args.view, args.output_dir)
