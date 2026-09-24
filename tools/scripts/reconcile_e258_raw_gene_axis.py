#!/usr/bin/env python3
"""Resolve symbol-name availability after the sealed raw-count scan.

Uses gene identifiers only from a completed raw scan that failed its exact-axis
assertion; it never reads the group-count matrix or a test-donor effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconcile(original: Path, raw_names: Path, aggregate_log: Path,
              manifest_path: Path, output: Path) -> dict:
    if "raw aggregation failed: not all predeclared gene symbols found" not in \
       aggregate_log.read_text():
        raise ValueError("raw scanner did not reach exact-axis terminal check")
    manifest = json.loads(manifest_path.read_text())
    if sha256(original) != manifest["gene_axis_sha256"]:
        raise ValueError("original gene axis changed")
    original_symbols = original.read_text().splitlines()
    raw_gene_ids = raw_names.read_text().splitlines()
    if raw_names.stat().st_size == 0 or \
       (raw_names.parent / "E258_RAW_ALLOWED_GROUP_SUMS.u32.part").stat().st_size != \
       len(raw_gene_ids) * manifest["group_count"] * 4:
        raise ValueError("incomplete failed-scan outputs")
    raw_symbols = set()
    for raw_gene in raw_gene_ids:
        parts = raw_gene.split(":")
        if len(parts) < 3:
            raise ValueError(f"invalid selected raw gene: {raw_gene}")
        raw_symbols.add(":".join(parts[1:-1]))
    missing = sorted(set(original_symbols) - raw_symbols)
    expected_missing = ["DDX3Y", "EIF1AY", "NLGN4Y", "RPS4Y1", "USP9Y"]
    if missing != expected_missing or len(original_symbols) != 6520:
        raise ValueError(f"unexpected missing raw symbols: {missing}")
    resolved = [symbol for symbol in original_symbols if symbol in raw_symbols]
    if output.exists():
        raise FileExistsError(output)
    output.write_text("\n".join(resolved) + "\n")
    manifest["raw_common_axis_sha256"] = sha256(output)
    manifest["n_gene_symbols_raw_common"] = len(resolved)
    manifest["raw_absent_gene_symbols"] = missing
    manifest["axis_reconciliation_basis"] = (
        "raw gene ID names from completed scanner, before any validation/test effect computation")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    result = {"stage": "E258_RAW_NAME_ONLY_AXIS_RECONCILIATION",
              "original_symbols": len(original_symbols),
              "raw_common_symbols": len(resolved),
              "absent_symbols": missing,
              "new_axis_sha256": sha256(output),
              "test_perturbation_effect_values_read": 0}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("/home/yyf/data/feng2025_candidate/raw_dev")
    parser.add_argument("--original", type=Path, default=root / "E258_DEV_SYMBOL_AXIS.txt")
    parser.add_argument("--raw-names", type=Path,
                        default=root / "E258_RAW_ALLOWED_GROUP_SUMS.genes.txt.part")
    parser.add_argument("--aggregate-log", type=Path, default=root / "aggregate.log")
    parser.add_argument("--manifest", type=Path, default=root / "E258_RAW_DEV_GROUPS.json")
    parser.add_argument("--output", type=Path,
                        default=root / "E258_DEV_SYMBOL_AXIS_RAW_COMMON.txt")
    args = parser.parse_args()
    reconcile(args.original, args.raw_names, args.aggregate_log, args.manifest, args.output)
