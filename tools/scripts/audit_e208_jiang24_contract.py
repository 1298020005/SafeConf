#!/usr/bin/env python3
"""Audit Jiang24 file integrity, split alignment and schema without reading X."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path


EXPECTED_GZIP_BYTES = 15_232_554_616
EXPECTED_SPLIT_SHA256 = (
    "5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d"
)
EXPECTED_SPLIT_ROWS = 1_628_476
EXPECTED_SPLIT_LABELS = {"train", "val", "test"}


class ContractFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def gzip_stream_test(path: Path) -> int:
    uncompressed_bytes = 0
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            uncompressed_bytes += len(block)
    return uncompressed_bytes


def audit_split(path: Path) -> tuple[dict, set[str]]:
    counts: Counter[str] = Counter()
    barcodes: set[str] = set()
    duplicate_count = 0
    malformed = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 2 or not row[0] or row[1] not in EXPECTED_SPLIT_LABELS:
                if len(malformed) < 20:
                    malformed.append({"line": line_number, "row": row})
                continue
            barcode, split = row
            counts[split] += 1
            if barcode in barcodes:
                duplicate_count += 1
            else:
                barcodes.add(barcode)
    rows = sum(counts.values())
    result = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "valid_rows": rows,
        "unique_barcodes": len(barcodes),
        "duplicate_barcodes": duplicate_count,
        "split_counts": dict(sorted(counts.items())),
        "malformed_rows": malformed,
    }
    result["gates"] = {
        "sha256": result["sha256"] == EXPECTED_SPLIT_SHA256,
        "row_count": rows == EXPECTED_SPLIT_ROWS,
        "labels": set(counts) == EXPECTED_SPLIT_LABELS,
        "unique": duplicate_count == 0 and len(barcodes) == rows,
        "well_formed": not malformed,
    }
    return result, barcodes


def candidate_columns(columns: list[str], terms: tuple[str, ...]) -> list[str]:
    return [
        column
        for column in columns
        if any(term in column.lower() for term in terms)
    ]


def preview_counts(values, limit: int = 30) -> dict:
    counts = values.astype(str).value_counts(dropna=False)
    return {
        "n_unique": int(len(counts)),
        "top_values": [
            {"value": str(key), "count": int(value)}
            for key, value in counts.iloc[:limit].items()
        ],
    }


def audit_h5ad(path: Path, split_barcodes: set[str]) -> dict:
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        obs_columns = [str(column) for column in adata.obs.columns]
        var_columns = [str(column) for column in adata.var.columns]
        obs_names = adata.obs_names.astype(str)
        obs_name_set = set(obs_names)
        candidates = {
            "perturbation": candidate_columns(
                obs_columns,
                ("pert", "target", "gene", "guide", "condition"),
            ),
            "cell_context": candidate_columns(
                obs_columns, ("cell_line", "celltype", "cell_type", "context")
            ),
            "treatment_state": candidate_columns(
                obs_columns, ("state", "treatment", "stim", "cytokine")
            ),
            "control": candidate_columns(
                obs_columns, ("control", "ctrl", "non_target", "nontarget")
            ),
        }
        previews = {
            column: preview_counts(adata.obs[column])
            for group in candidates.values()
            for column in group
        }
        split_missing_from_h5ad = split_barcodes - obs_name_set
        h5ad_missing_from_split = obs_name_set - split_barcodes
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "x_type": type(adata.X).__name__,
            "obs_names_unique": bool(adata.obs_names.is_unique),
            "var_names_unique": bool(adata.var_names.is_unique),
            "obs_columns": obs_columns,
            "var_columns": var_columns,
            "obsm_keys": sorted(map(str, adata.obsm.keys())),
            "layers": sorted(map(str, adata.layers.keys())),
            "uns_keys": sorted(map(str, adata.uns.keys())),
            "schema_candidates_not_yet_frozen": candidates,
            "candidate_value_previews": previews,
            "split_alignment": {
                "exact_barcode_set": not split_missing_from_h5ad
                and not h5ad_missing_from_split,
                "split_missing_from_h5ad_count": len(split_missing_from_h5ad),
                "h5ad_missing_from_split_count": len(h5ad_missing_from_split),
                "split_missing_examples": sorted(split_missing_from_h5ad)[:20],
                "h5ad_missing_examples": sorted(h5ad_missing_from_split)[:20],
            },
            "expression_values_read": False,
        }
    finally:
        adata.file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gzip", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--h5ad", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-gzip-stream-test",
        action="store_true",
        help="only for a quick dry run; a formal D0 audit must not use this flag",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gzip_path = args.gzip.resolve()
    split_path = args.split.resolve()
    h5ad_path = args.h5ad.resolve() if args.h5ad else None
    for path in (gzip_path, split_path):
        if not path.is_file():
            raise ContractFailure(f"missing input: {path}")
    if h5ad_path is not None and not h5ad_path.is_file():
        raise ContractFailure(f"missing input: {h5ad_path}")

    split, barcodes = audit_split(split_path)
    gzip_record = {
        "path": str(gzip_path),
        "bytes": gzip_path.stat().st_size,
        "sha256": sha256_file(gzip_path),
        "stream_test_run": not args.skip_gzip_stream_test,
    }
    gzip_record["expected_bytes_match"] = (
        gzip_record["bytes"] == EXPECTED_GZIP_BYTES
    )
    if args.skip_gzip_stream_test:
        gzip_record["stream_test_pass"] = None
        gzip_record["uncompressed_bytes"] = None
    else:
        gzip_record["uncompressed_bytes"] = gzip_stream_test(gzip_path)
        gzip_record["stream_test_pass"] = True

    h5ad = audit_h5ad(h5ad_path, barcodes) if h5ad_path else None
    file_gates = list(split["gates"].values()) + [
        gzip_record["expected_bytes_match"],
        gzip_record["stream_test_pass"] is True,
    ]
    alignment_pass = bool(
        h5ad is not None
        and h5ad["obs_names_unique"]
        and h5ad["var_names_unique"]
        and h5ad["split_alignment"]["exact_barcode_set"]
    )
    payload = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D0_file_and_schema_audit",
        "created_at": now(),
        "status": (
            "SCHEMA_REVIEW_REQUIRED"
            if all(file_gates) and alignment_pass
            else "INCOMPLETE_OR_FAILED"
        ),
        "gzip": gzip_record,
        "split": split,
        "h5ad": h5ad,
        "gates": {
            "file_integrity": all(file_gates),
            "split_h5ad_alignment": alignment_pass,
            "schema_mapping_frozen": False,
            "expression_values_read": False,
            "target_truth_used_for_model_or_score_selection": False,
        },
        "interpretation": (
            "Passing this script does not make D0 PASS. Candidate column names must "
            "be manually mapped and frozen before any formal model run."
        ),
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload["gates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
