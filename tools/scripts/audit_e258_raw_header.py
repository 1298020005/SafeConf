#!/usr/bin/env python3
"""Audit Feng raw-count CELL HEADER only; never reads a gene-count row."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd

from audit_e258_feng_metadata import EXPECTED_MD5, md5


CELL_ID_PATTERN = re.compile(r"^PC-(P[0-9]+)-D[0-9]+_(.+)$")


def count_matrix_id(metadata_id: str) -> str:
    match = CELL_ID_PATTERN.fullmatch(metadata_id)
    if match is None:
        raise ValueError(f"unexpected metadata cell ID: {metadata_id[:50]}")
    return f"{match.group(1)}_{match.group(2)}"


def audit(metadata: Path, counts: Path) -> dict:
    if md5(metadata) != EXPECTED_MD5:
        raise ValueError("metadata checksum changed")
    meta = pd.read_csv(metadata, sep="\t", usecols=["Cell_ID"], compression="gzip")
    normalized = [count_matrix_id(value) for value in meta.Cell_ID.astype(str)]
    if len(set(normalized)) != len(normalized):
        raise ValueError("normalized metadata IDs are not unique")
    with gzip.open(counts, "rt") as stream:
        header = stream.readline().rstrip("\r\n").split(",")
    if header[0] != "":
        raise ValueError("raw matrix header axis changed")
    cells = header[1:]
    if len(set(cells)) != len(cells):
        raise ValueError("duplicate cell IDs in count matrix")
    matched = set(normalized) & set(cells)
    missing = set(normalized) - set(cells)
    extra = set(cells) - set(normalized)
    result = {
        "stage": "E258_RAW_COUNT_HEADER_ONLY_AUDIT",
        "metadata_cells": len(normalized),
        "count_matrix_cell_columns": len(cells),
        "matched_metadata_cells": len(matched),
        "missing_metadata_cells": len(missing),
        "unmapped_count_columns": len(extra),
        "unmapped_count_ids": sorted(extra),
        "count_gene_rows_read": 0,
        "test_target_expression_values_read": 0,
    }
    if len(missing) != 0 or len(extra) != 1:
        raise ValueError(json.dumps(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/TargetedScreen_Cell-Metadata.tsv.gz"))
    parser.add_argument("--counts", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/TargetedScreen_RNA-UMI-Counts.csv.gz"))
    args = parser.parse_args()
    audit(args.metadata, args.counts)
