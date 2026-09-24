#!/usr/bin/env python3
"""Build E258 train/validation effects from isolated raw UMI group sums.

This is an independently defined pseudobulk effect, not the authors' adjusted
LFC. Test-donor controls may enter the query context; test perturbation counts
cannot enter because the native aggregator never converts their CSV tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from build_e258_official_dev_view import TRAIN, VALIDATION, TEST


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(manifest_path: Path, axis_path: Path, prefix: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    groups = manifest["groups"]
    if manifest["test_target_numeric_values_read"] != 0 or \
       manifest["raw_common_axis_sha256"] != sha256(axis_path) or \
       len(groups) != 5616:
        raise ValueError("E258 raw isolation/axis contract failed")
    axis = axis_path.read_text().splitlines()
    if len(axis) != manifest["n_gene_symbols_raw_common"] or \
       len(set(axis)) != len(axis):
        raise ValueError("predeclared symbol axis changed")
    raw_genes_path = Path(f"{prefix}.genes.txt")
    matrix_path = Path(f"{prefix}.u32")
    library_path = Path(f"{prefix}.library.u64")
    raw_genes = raw_genes_path.read_text().splitlines()
    if matrix_path.stat().st_size != 4 * len(raw_genes) * len(groups):
        raise ValueError("aggregate matrix shape/size mismatch")
    raw = np.memmap(matrix_path, dtype=np.uint32, mode="r",
                    shape=(len(raw_genes), len(groups)))
    library = np.fromfile(library_path, dtype=np.uint64)
    if len(library) != len(groups) or np.any(library == 0):
        raise ValueError("missing/zero whole-transcriptome group library sizes")
    symbol_index = {symbol: i for i, symbol in enumerate(axis)}
    summed = np.zeros((len(axis), len(groups)), dtype=np.uint64)
    seen = set()
    duplicates = 0
    for row, raw_gene in enumerate(raw_genes):
        parts = raw_gene.split(":")
        if len(parts) < 3:
            raise ValueError(f"unrecognized raw gene name: {raw_gene}")
        symbol = ":".join(parts[1:-1])
        symbol_row = symbol_index.get(symbol)
        if symbol_row is None:
            raise ValueError(f"selected raw symbol outside fixed axis: {symbol}")
        duplicates += symbol in seen
        seen.add(symbol)
        summed[symbol_row] += raw[row]
    if seen != set(axis):
        raise ValueError("raw count/engineering symbol axes differ")
    # Group-level log-normalized pseudobulk expression. Denominators include
    # *all* raw gene rows, not only the selected 6,520-gene output axis.
    expression = np.log1p(10000.0 * summed.astype(np.float64) / library[None, :])
    expression = expression.astype(np.float32)
    control_count = manifest["control_groups"]
    if any(g["kind"] != "control" for g in groups[:control_count]) or \
       any(g["kind"] != "target" for g in groups[control_count:]):
        raise ValueError("unexpected control/target group order")
    lines = sorted({g["line"] for g in groups[:control_count]})
    control_by_line = np.empty((len(lines), len(axis)), dtype=np.float32)
    for i, line in enumerate(lines):
        ids = [j for j, g in enumerate(groups[:control_count]) if g["line"] == line]
        weights = np.asarray([groups[j]["n_cells"] for j in ids], dtype=np.float64)
        control_by_line[i] = np.average(expression[:, ids], axis=1, weights=weights)
    tasks = groups[control_count:]
    effects = np.empty((len(tasks), len(axis)), dtype=np.float32)
    for i, task in enumerate(tasks):
        batch_groups = [entry["control_group"] for entry in
                        task["matched_control_batches"]]
        batch_weights = np.asarray([entry["target_cells"] for entry in
                                    task["matched_control_batches"]], dtype=np.float64)
        matched_control = np.average(expression[:, batch_groups], axis=1,
                                     weights=batch_weights)
        effects[i] = expression[:, control_count + i] - matched_control
    donors = [g["donor"] for g in tasks]
    if not set(donors) <= set(TRAIN + VALIDATION) or set(donors) & set(TEST):
        raise ValueError("test donor target entered raw development view")
    if not np.isfinite(effects).all() or not np.isfinite(control_by_line).all():
        raise ValueError("nonfinite raw-derived expression")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(
        output,
        task_line=np.asarray([g["line"] for g in tasks]),
        task_target=np.asarray([g["target"] for g in tasks]),
        task_donor=np.asarray(donors),
        task_cells=np.asarray([g["n_cells"] for g in tasks], dtype=np.int32),
        line=np.asarray(lines), gene=np.asarray(axis),
        effect=effects, control=control_by_line,
    )
    result = {
        "stage": "E258_RAW_PSEUDOBULK_TRAIN_VALIDATION_DEV_VIEW",
        "effect_definition": "log1p(10000*target_count/all_gene_target_UMI) minus target-batch-weighted matched-control log1p(10000*control_count/all_gene_control_UMI)",
        "gene_axis": "official train/validation-common symbol names intersected with raw gene identifiers; no test effect values",
        "n_train_tasks": sum(d in TRAIN for d in donors),
        "n_validation_tasks": sum(d in VALIDATION for d in donors),
        "n_test_target_tasks": 0,
        "n_symbols": len(axis),
        "raw_rows_for_symbols": len(raw_genes),
        "duplicate_symbol_raw_rows_collapsed": duplicates,
        "test_perturbation_numeric_values_loaded": 0,
        "input_group_matrix_sha256": sha256(matrix_path),
        "input_library_sha256": sha256(library_path),
        "output_sha256": sha256(output),
    }
    status_path = output.with_suffix(".status.json")
    status_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("/home/yyf/data/feng2025_candidate/raw_dev")
    parser.add_argument("--manifest", type=Path, default=root / "E258_RAW_DEV_GROUPS.json")
    parser.add_argument("--axis", type=Path,
                        default=root / "E258_DEV_SYMBOL_AXIS_RAW_COMMON.txt")
    parser.add_argument("--prefix", type=Path,
                        default=root / "E258_RAW_ALLOWED_GROUP_SUMS")
    parser.add_argument("--output", type=Path,
                        default=root / "E258_RAW_DEV_VIEW.npz")
    args = parser.parse_args()
    build(args.manifest, args.axis, args.prefix, args.output)
