#!/usr/bin/env python3
"""Verify the Kaden asset and freeze a label-only perturbation split.

Never indexes X, X/data, layers, or any target-expression values. Output is
runtime-local until a human or a separate supervisor commits it to both remotes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


EXPECTED_BYTES = 5_641_728_720
EXPECTED_MD5 = "bda7f3ea0173f64a0898ab6bb21a4a16"
EXPECTED_CELLS = 850_225
EXPECTED_CONTROL_CELLS = 42_233
EXPECTED_PERTURBATIONS = 1_836
MIN_CELLS = 30
MIN_SPLIT_TASKS = 100


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest_file(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def split_label(label: str) -> str:
    digest = hashlib.sha256(f"kaden25rpe1:{label}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < 0.60:
        return "train"
    if fraction < 0.80:
        return "validation"
    return "test"


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def wait_for_size(path: Path, out: Path, seconds: int, timeout_hours: float) -> None:
    deadline = time.monotonic() + timeout_hours * 3600
    while True:
        observed = path.stat().st_size if path.exists() else 0
        if observed == EXPECTED_BYTES:
            return
        if observed > EXPECTED_BYTES:
            raise RuntimeError(f"asset larger than frozen size: {observed}")
        if seconds == 0 or time.monotonic() >= deadline:
            raise RuntimeError(f"asset incomplete: {observed}/{EXPECTED_BYTES} bytes")
        write_status(
            out,
            {"status": "WAITING_FOR_ASSET", "observed_bytes": observed,
             "expected_bytes": EXPECTED_BYTES, "expression_values_read": 0},
        )
        time.sleep(seconds)


def write_status(out: Path, payload: dict[str, object]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": "E247_kaden_crispra_unseen_tf", "updated_at": timestamp(), **payload}
    temp = out / ".E247_ASSET_LABEL_STATUS.json.tmp"
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(out / "E247_ASSET_LABEL_STATUS.json")


def read_label_counts(path: Path) -> tuple[list[dict[str, object]], int, int, int]:
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]["perturbation"]
        categories = [decode(x) for x in obs["categories"][:]]
        if len(set(categories)) != len(categories):
            raise RuntimeError("duplicate perturbation categories")
        codes = np.asarray(obs["codes"][:], dtype=np.int64)
        if np.any((codes < -1) | (codes >= len(categories))):
            raise RuntimeError("invalid categorical codes")
        if np.any(codes == -1):
            raise RuntimeError("unlabeled cells present; label-only split refused")
        n_cells = len(codes)
        x_shape = tuple(int(x) for x in handle["X"].attrs["shape"])
        n_genes = len(handle["var"]["_index"])
        if x_shape != (n_cells, n_genes):
            raise RuntimeError(f"X metadata/axis mismatch: {x_shape}, {n_cells}, {n_genes}")
        counts = np.bincount(codes[codes >= 0], minlength=len(categories))
    rows = []
    for label, count in zip(categories, counts, strict=True):
        eligible = label != "control" and count >= MIN_CELLS
        rows.append(
            {"perturbation": label, "n_cells": int(count),
             "eligible_ge30": int(eligible), "split": split_label(label) if eligible else "excluded"}
        )
    control_count = next((int(x["n_cells"]) for x in rows if x["perturbation"] == "control"), 0)
    return rows, n_cells, n_genes, control_count


def audit(path: Path, out: Path, wait_seconds: int, timeout_hours: float) -> None:
    wait_for_size(path, out, wait_seconds, timeout_hours)
    md5, sha256 = digest_file(path)
    if md5 != EXPECTED_MD5:
        raise RuntimeError(f"MD5 mismatch: {md5} != {EXPECTED_MD5}")
    rows, n_cells, n_genes, n_control = read_label_counts(path)
    n_perturbations = sum(x["perturbation"] != "control" for x in rows)
    split_counts = {
        split: sum(x["split"] == split for x in rows)
        for split in ("train", "validation", "test")
    }
    failures = []
    if n_cells != EXPECTED_CELLS:
        failures.append(f"cell count {n_cells} != {EXPECTED_CELLS}")
    if n_control != EXPECTED_CONTROL_CELLS:
        failures.append(f"control count {n_control} != {EXPECTED_CONTROL_CELLS}")
    if n_perturbations != EXPECTED_PERTURBATIONS:
        failures.append(f"perturbation labels {n_perturbations} != {EXPECTED_PERTURBATIONS}")
    if any(count < MIN_SPLIT_TASKS for count in split_counts.values()):
        failures.append(f"split below {MIN_SPLIT_TASKS} qualified perturbations: {split_counts}")
    if failures:
        raise RuntimeError("; ".join(failures))

    out.mkdir(parents=True, exist_ok=True)
    table = out / "E247_LABEL_ONLY_SPLIT.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["perturbation", "n_cells", "eligible_ge30", "split"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row["perturbation"])))
    write_status(
        out,
        {"status": "PASS_LABEL_ONLY", "asset": str(path), "bytes": EXPECTED_BYTES,
         "md5": md5, "sha256": sha256, "n_cells": n_cells, "n_genes": n_genes,
         "n_controls": n_control, "n_perturbations": n_perturbations,
         "n_eligible": sum(int(x["eligible_ge30"]) for x in rows),
         "split_counts": split_counts, "label_table_sha256": digest_file(table)[1],
         "expression_values_read": 0, "test_truth_read": False},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    args = parser.parse_args()
    try:
        audit(args.asset, args.out, args.wait_seconds, args.timeout_hours)
    except Exception as exc:
        write_status(args.out, {"status": "FAIL", "reason": str(exc), "expression_values_read": 0})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
