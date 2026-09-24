#!/usr/bin/env python3
"""Deterministically split only allowed E258 cell groups into balanced halves.

The original -1 mapping for final-test targeted cells remains -1. No expression
values are opened. Half assignment alternates within each original group in
raw-column order, keeping target and batch-control groups separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split(mapping: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    if mapping.dtype != np.int32:
        raise ValueError("mapping must be int32")
    if not np.all((mapping == -1) | ((mapping >= 0) & (mapping < n_groups))):
        raise ValueError("invalid original group indices")
    counts = np.zeros(n_groups, dtype=np.int64)
    half = np.full(len(mapping), -1, dtype=np.int32)
    for i, group in enumerate(mapping):
        if group >= 0:
            half[i] = 2 * group + (counts[group] % 2)
            counts[group] += 1
    if not np.array_equal(half[mapping < 0], mapping[mapping < 0]):
        raise ValueError("forbidden cells became selected")
    return half, np.bincount(half[half >= 0], minlength=2 * n_groups)


def prepare(original: Path, manifest: Path, output: Path) -> dict:
    record = json.loads(manifest.read_text())
    if record["test_target_numeric_values_read"] != 0:
        raise ValueError("original group manifest violates test isolation")
    if sha256(original) != record["mapping_sha256"]:
        raise ValueError("original group map hash mismatch")
    mapping = np.fromfile(original, dtype=np.int32)
    if len(mapping) != record["raw_columns"]:
        raise ValueError("original group map length mismatch")
    half, counts = split(mapping, record["group_count"])
    expected = np.asarray([g["n_cells"] for g in record["groups"]])
    if not np.array_equal(counts.reshape(-1, 2).sum(axis=1), expected):
        raise ValueError("half assignments do not reconstruct full groups")
    if counts.min() < 10:
        raise ValueError("one split-half group has fewer than 10 cells")
    if output.exists() or output.with_suffix(".status.json").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    half.tofile(output)
    status = {
        "stage": "E258_TRAIN_VALIDATION_SPLIT_HALF_QUALITY_AUDIT",
        "original_mapping_sha256": sha256(original),
        "half_mapping_sha256": sha256(output),
        "n_original_groups": record["group_count"],
        "n_half_groups": len(counts),
        "n_raw_columns": len(mapping),
        "n_skipped_columns_unchanged": int((mapping < 0).sum()),
        "n_selected_columns": int((mapping >= 0).sum()),
        "min_half_cells": int(counts.min()),
        "max_half_imbalance": int(np.max(np.abs(counts[0::2] - counts[1::2]))),
        "test_donor_target_numeric_values_loaded": 0,
    }
    output.with_suffix(".status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("/home/yyf/data/feng2025_candidate/raw_dev")
    parser.add_argument("--original", type=Path,
                        default=root / "E258_RAW_DEV_COLUMN_GROUPS.i32")
    parser.add_argument("--manifest", type=Path,
                        default=root / "E258_RAW_DEV_GROUPS.json")
    parser.add_argument("--output", type=Path,
                        default=root / "E258_RAW_DEV_SPLIT_HALF_GROUPS.i32")
    args = parser.parse_args()
    prepare(args.original, args.manifest, args.output)
