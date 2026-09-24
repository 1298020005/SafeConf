#!/usr/bin/env python3
"""E258 Feng targeted-screen metadata audit; never opens expression or LFC data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EXPECTED_MD5 = "980a440a8132a20567c4c3610e6a5b20"
SPLIT_SALT = "E258:Feng2025:v1:"


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metadata",
        type=Path,
        nargs="?",
        default=Path("/home/yyf/data/feng2025_candidate/TargetedScreen_Cell-Metadata.tsv.gz"),
    )
    args = parser.parse_args()
    if md5(args.metadata) != EXPECTED_MD5:
        raise SystemExit("official metadata MD5 mismatch")
    frame = pd.read_csv(args.metadata, sep="\t", compression="gzip")
    columns = ["Cell_ID", "Batch", "Guide_Call", "Cell_Line"]
    if list(frame.columns) != columns or frame.Cell_ID.duplicated().any():
        raise SystemExit("metadata identity/column contract failed")
    frame["donor"] = frame.Cell_Line.str.split("_").str[0]
    donors = sorted(
        frame.donor.unique(),
        key=lambda donor: hashlib.sha256((SPLIT_SALT + donor).encode()).hexdigest(),
    )
    split = {**dict.fromkeys(donors[:4], "train"),
             **dict.fromkeys(donors[4:6], "validation"),
             **dict.fromkeys(donors[6:], "test")}
    assigned = frame[frame.Guide_Call.ne("unassigned")].copy()
    assigned["gene"] = assigned.Guide_Call.str.rsplit("_", n=1).str[0]
    controls = assigned[assigned.gene.eq("NonTarget")]
    genes = assigned[assigned.gene.ne("NonTarget")]
    tasks = (
        genes.groupby(["donor", "Cell_Line", "gene"])
        .Guide_Call.agg(cells="size", guides="nunique")
        .reset_index()
    )
    tasks = tasks[(tasks.cells >= 30) & (tasks.guides >= 2)].copy()
    train_support = tasks[tasks.donor.map(split).eq("train")].groupby("gene").donor.nunique()
    tasks["source_donors"] = tasks.gene.map(train_support).fillna(0).astype(int)
    tasks["split"] = tasks.donor.map(split)
    line_rows = []
    for line, block in tasks.groupby("Cell_Line", sort=True):
        line_rows.append({
            "cell_line": line,
            "donor": block.donor.iloc[0],
            "split": block.split.iloc[0],
            "eligible_tasks": len(block),
            "eligible_with_2_train_donors": int((block.source_donors >= 2).sum()),
            "non_target_cells": int((controls.Cell_Line == line).sum()),
        })
    result = {
        "experiment": "E258_feng2025_external_candidate_metadata_only",
        "source_article": "10.6084/m9.figshare.27989294.v2",
        "source_file_id": 51051809,
        "metadata_md5": EXPECTED_MD5,
        "metadata_rows": len(frame),
        "unique_cell_ids": frame.Cell_ID.nunique(),
        "cell_lines": frame.Cell_Line.nunique(),
        "donors": len(donors),
        "target_genes": genes.gene.nunique(),
        "assigned_cells": len(assigned),
        "gene_targeted_cells": len(genes),
        "non_target_control_cells": len(controls),
        "eligible_line_gene_tasks": len(tasks),
        "task_rule": ">=30 assigned cells and >=2 guides per line-gene",
        "source_rule": ">=2 independent training donors satisfying the task rule",
        "split_salt": SPLIT_SALT,
        "hash_order": donors,
        "train_donors": donors[:4],
        "validation_donors": donors[4:6],
        "test_donors": donors[6:],
        "validation_tasks_with_source_rule": int(
            ((tasks.split == "validation") & (tasks.source_donors >= 2)).sum()
        ),
        "test_tasks_with_source_rule": int(
            ((tasks.split == "test") & (tasks.source_donors >= 2)).sum()
        ),
        "per_line": line_rows,
        "perturbed_expression_rows_read": 0,
        "official_lfc_rows_read": 0,
    }
    if (
        len(frame) != 1_161_864
        or len(donors) != 10
        or frame.Cell_Line.nunique() != 19
        or len(tasks) != 7201
        or result["test_tasks_with_source_rule"] != 2895
    ):
        raise SystemExit("metadata count contract changed")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
