#!/usr/bin/env python3
"""Build a TRAIN/VALIDATION-only engineering view of Feng's official LFC table.

This is NOT E258's formal raw-count history. It never parses a test donor's LFC
or wild-type expression number; those rows are mechanically skipped by ID.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_e258_feng_metadata import EXPECTED_MD5, SPLIT_SALT, md5


TRAIN = ("pipw", "kolf", "paab", "fiaj")
VALIDATION = ("eipl", "oikd")
TEST = ("tolg", "jejf", "zapk", "iudw")
LFC_MD5 = "27ae8d0109b9c42f1f972c45987db77e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_contract(metadata: Path):
    if md5(metadata) != EXPECTED_MD5:
        raise ValueError("metadata MD5 mismatch")
    meta = pd.read_csv(metadata, sep="\t", compression="gzip")
    meta["donor"] = meta.Cell_Line.str.split("_").str[0]
    donors = sorted(meta.donor.unique(),
                    key=lambda d: hashlib.sha256((SPLIT_SALT + d).encode()).hexdigest())
    if tuple(donors[:4]) != TRAIN or tuple(donors[4:6]) != VALIDATION or tuple(donors[6:]) != TEST:
        raise ValueError("donor split contract changed")
    targeted = meta.loc[~meta.Guide_Call.isin(["unassigned"])].copy()
    targeted["target"] = targeted.Guide_Call.str.rsplit("_", n=1).str[0]
    targeted = targeted.loc[targeted.target.ne("NonTarget")]
    tasks = targeted.groupby(["Cell_Line", "target"]).Guide_Call.agg(cells="size", guides="nunique")
    tasks = tasks.loc[(tasks.cells >= 30) & (tasks.guides >= 2)].reset_index()
    tasks["donor"] = tasks.Cell_Line.str.split("_").str[0]
    source_donors = tasks.loc[tasks.donor.isin(TRAIN)].groupby("target").donor.nunique()
    tasks["n_source_donors"] = tasks.target.map(source_donors).fillna(0).astype(int)
    tasks = tasks.loc[tasks.n_source_donors >= 2].copy()
    dev = tasks.loc[tasks.donor.isin(TRAIN + VALIDATION)].copy()
    keys = sorted(zip(dev.Cell_Line, dev.target))
    if len(tasks.loc[tasks.donor.isin(VALIDATION)]) != 1530 or \
       len(tasks.loc[tasks.donor.isin(TEST)]) != 2895:
        raise ValueError("eligible task contract changed")
    return meta, keys


def build(metadata: Path, lfc: Path, output: Path, *, verify_hash: bool = True,
          min_common_genes: int = 3000) -> dict:
    meta, keys = task_contract(metadata)
    if verify_hash and md5(lfc) != LFC_MD5:
        raise ValueError("official LFC MD5 mismatch")
    allowed = {line for line in meta.Cell_Line.unique()
               if line.split("_")[0] in TRAIN + VALIDATION}
    key_to_row = {key: i for i, key in enumerate(keys)}
    lines = sorted(allowed)
    line_to_row = {line: i for i, line in enumerate(lines)}
    gene_set: set[str] = set()
    skipped_test = 0
    with gzip.open(lfc, "rt") as stream:
        if stream.readline().rstrip("\n") != (
            "Target\tExpressed_Gene_Symbol\tExpressed_Gene_Ens_ID\t"
            "Cell_Line\twt_expr\tlfc\tpval_adj"
        ):
            raise ValueError("official LFC schema changed")
        for line in stream:
            prefix = line.split("\t", 4)
            if len(prefix) < 5:
                raise ValueError("malformed LFC row")
            cell_line = prefix[3]
            if cell_line not in allowed:
                skipped_test += 1
                continue
            if (cell_line, prefix[0]) in key_to_row and cell_line.split("_")[0] in TRAIN:
                gene_set.add(prefix[1])
    genes = sorted(gene_set)
    gene_to_col = {gene: i for i, gene in enumerate(genes)}
    effect = np.full((len(keys), len(genes)), np.nan, np.float32)
    control = np.full((len(lines), len(genes)), np.nan, np.float32)
    filled = np.zeros(len(keys), np.int32)
    with gzip.open(lfc, "rt") as stream:
        next(stream)
        for line in stream:
            prefix = line.split("\t", 4)
            cell_line = prefix[3]
            if cell_line not in allowed:
                # Do not parse either numerical field for a test-donor row.
                continue
            target, gene = prefix[0], prefix[1]
            task_row = key_to_row.get((cell_line, target))
            gene_col = gene_to_col.get(gene)
            if task_row is None or gene_col is None:
                continue
            parts = line.rstrip("\n").split("\t")
            value = np.float32(parts[5])
            wt_value = np.float32(parts[4])
            if not np.isfinite(value) or not np.isfinite(wt_value):
                raise ValueError("non-finite train/validation value")
            if np.isfinite(effect[task_row, gene_col]):
                raise ValueError("duplicate task-gene official effect")
            effect[task_row, gene_col] = value
            control[line_to_row[cell_line], gene_col] = wt_value
            filled[task_row] += 1
    available = filled > 0
    missing_task_keys = [list(keys[i]) for i in np.flatnonzero(~available)]
    if len(missing_task_keys) > 10:
        raise ValueError("too many task rows absent from official summary")
    complete = np.isfinite(effect[available]).all(axis=0) & np.isfinite(control).all(axis=0)
    print(json.dumps({
        "stage": "E258_OFFICIAL_DEV_AXIS_AUDIT",
        "candidate_genes": len(genes),
        "common_genes": int(complete.sum()),
        "incomplete_tasks": int((filled < len(genes)).sum()),
        "missing_task_keys": missing_task_keys,
        "minimum_genes_per_task": int(filled.min()),
        "maximum_genes_per_task": int(filled.max()),
        "control_missing_entries": int((~np.isfinite(control)).sum()),
        "test_numeric_values_parsed": 0,
    }, ensure_ascii=False), flush=True)
    if int(complete.sum()) < min_common_genes:
        raise ValueError(f"fewer than {min_common_genes} common train/validation genes")
    effect = effect[available][:, complete]
    control = control[:, complete]
    keys = [key for key, keep in zip(keys, available) if keep]
    genes = [gene for gene, keep in zip(genes, complete) if keep]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(
        output, task_line=np.asarray([k[0] for k in keys]),
        task_target=np.asarray([k[1] for k in keys]),
        task_donor=np.asarray([k[0].split("_")[0] for k in keys]),
        line=np.asarray(lines), gene=np.asarray(genes),
        effect=effect, control=control,
    )
    result = {
        "stage": "E258_OFFICIAL_LFC_ENGINEERING_VIEW_ONLY",
        "source_lfc_sha256": sha256(lfc),
        "output_sha256": sha256(output),
        "n_train_tasks": sum(k[0].split("_")[0] in TRAIN for k in keys),
        "n_validation_tasks": sum(k[0].split("_")[0] in VALIDATION for k in keys),
        "n_genes": len(genes),
        "test_rows_mechanically_skipped": skipped_test,
        "test_numeric_values_parsed": 0,
        "formal_raw_count_history_ready": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/TargetedScreen_Cell-Metadata.tsv.gz"))
    parser.add_argument("--lfc", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/TargetedScreen_LFC_byGene-perLine.tsv.gz"))
    parser.add_argument("--output", type=Path, default=Path(
        "/home/yyf/data/feng2025_candidate/E258_OFFICIAL_DEV_VIEW.npz"))
    args = parser.parse_args()
    build(args.metadata, args.lfc, args.output)
