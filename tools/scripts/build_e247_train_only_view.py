#!/usr/bin/env python3
"""Materialize a Kaden training-only H5AD without indexing held-out expression.

The label split is frozen elsewhere. This script reads only control and train
perturbation X intervals; validation/test perturbation values are not indexed.
The source H5 uses CSR X and is sorted enough for contiguous-run copying.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


SOURCE_SHA256 = "837f05b08c5227bb91e19929896251f6d263ad8d3eb4b3f60f8fffc887c42012"
SPLIT_SHA256 = "3b6992bad649e4c7a12932ba016eb99fb61830209d29a3ffc7389b55c2ea09af"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def approved_labels(
    split_csv: Path, expected_counts: dict[str, int]
) -> tuple[set[str], set[str], dict[str, int]]:
    counts = {split: 0 for split in ("train", "validation", "test")}
    selected = {"control"}
    seen: set[str] = set()
    with split_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["perturbation"]
            if label in seen:
                raise ValueError(f"duplicate label in split: {label}")
            seen.add(label)
            split = row["split"]
            if split in counts:
                counts[split] += 1
            if split == "train":
                selected.add(label)
    if counts != expected_counts:
        raise ValueError(f"split count mismatch: {counts}")
    if "control" not in seen or len(seen) != sum(expected_counts.values()) + 1:
        raise ValueError(f"split label inventory mismatch: {len(seen)}")
    return selected, seen, counts


def copy_attrs(source: h5py.Group | h5py.Dataset, target: h5py.Group | h5py.Dataset) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def runs_from_indices(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    cuts = np.flatnonzero(np.diff(indices) != 1) + 1
    starts = np.r_[0, cuts]
    stops = np.r_[cuts, len(indices)]
    return [(int(indices[a]), int(indices[b - 1]) + 1) for a, b in zip(starts, stops, strict=True)]


def build_view(
    source: Path, split_csv: Path, output: Path, *, source_sha: str,
    split_sha: str, expected_counts: dict[str, int] | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if sha256_file(source) != source_sha:
        raise ValueError("source sha256 does not match frozen asset")
    if sha256_file(split_csv) != split_sha:
        raise ValueError("split sha256 does not match frozen labels")
    if expected_counts is None:
        expected_counts = {"train": 1093, "validation": 376, "test": 367}
    selected_labels, all_labels, split_counts = approved_labels(split_csv, expected_counts)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite partial view: {temporary}")

    try:
        with h5py.File(source, "r") as src, h5py.File(temporary, "w") as dst:
            source_labels = [decode(x) for x in src["obs/perturbation/categories"][:]]
            source_codes = np.asarray(src["obs/perturbation/codes"][:], dtype=np.int32)
            if len(source_labels) != len(all_labels) or np.any(source_codes < 0):
                raise ValueError("source label inventory changed")
            if set(source_labels) != all_labels:
                raise ValueError("source labels do not match frozen split")
            keep_codes = [i for i, label in enumerate(source_labels) if label in selected_labels]
            keep = np.isin(source_codes, keep_codes)
            selected_rows = np.flatnonzero(keep)
            runs = runs_from_indices(selected_rows)
            if not runs:
                raise ValueError("empty training view")

            source_ptr = np.asarray(src["X/indptr"][:], dtype=np.int64)
            shape = tuple(map(int, src["X"].attrs["shape"]))
            if shape != (len(source_codes), len(src["var/_index"])):
                raise ValueError("source axes do not match X")
            row_lengths = source_ptr[selected_rows + 1] - source_ptr[selected_rows]
            nnz = int(row_lengths.sum())
            if np.any(row_lengths < 0) or nnz <= 0:
                raise ValueError("invalid CSR pointers")

            copy_attrs(src, dst)
            for name in ("var", "layers", "obsm", "obsp", "uns", "varm", "varp"):
                src.copy(name, dst)
            obs = dst.create_group("obs")
            copy_attrs(src["obs"], obs)
            index_key = decode(src["obs"].attrs["_index"])
            categorical = obs.create_group("perturbation")
            copy_attrs(src["obs/perturbation"], categorical)
            # Retain only actually available categories; no test label appears
            # inside the physical train view, even as an unused category.
            retained = [label for label in source_labels if label in selected_labels]
            category_ids = [i for i, label in enumerate(source_labels) if label in selected_labels]
            code_map = np.full(len(source_labels), -1, dtype=np.int16)
            code_map[category_ids] = np.arange(len(retained), dtype=np.int16)
            string_dtype = h5py.string_dtype(encoding="utf-8")
            categories = categorical.create_dataset(
                "categories", data=np.asarray(retained, dtype=object), dtype=string_dtype
            )
            copy_attrs(src["obs/perturbation/categories"], categories)
            codes = categorical.create_dataset(
                "codes", shape=(len(selected_rows),), dtype="i2", compression="gzip", compression_opts=4
            )
            copy_attrs(src["obs/perturbation/codes"], codes)
            barcodes = obs.create_dataset(
                index_key, shape=(len(selected_rows),), dtype=string_dtype,
                compression="gzip", compression_opts=4,
            )
            copy_attrs(src["obs"][index_key], barcodes)

            x = dst.create_group("X")
            copy_attrs(src["X"], x)
            x.attrs["shape"] = np.asarray((len(selected_rows), shape[1]), dtype=np.int64)
            chunks = min(1_000_000, max(1, nnz))
            out_data = x.create_dataset(
                "data", shape=(nnz,), dtype=src["X/data"].dtype,
                chunks=(chunks,), compression="gzip", compression_opts=4,
            )
            out_indices = x.create_dataset(
                "indices", shape=(nnz,), dtype=src["X/indices"].dtype,
                chunks=(chunks,), compression="gzip", compression_opts=4,
            )
            out_ptr = x.create_dataset("indptr", shape=(len(selected_rows) + 1,), dtype="i8")
            out_ptr[0] = 0

            row_cursor = 0
            nnz_cursor = 0
            for start, stop in runs:
                lo, hi = int(source_ptr[start]), int(source_ptr[stop])
                nrows, ndata = stop - start, hi - lo
                out_data[nnz_cursor:nnz_cursor + ndata] = src["X/data"][lo:hi]
                out_indices[nnz_cursor:nnz_cursor + ndata] = src["X/indices"][lo:hi]
                out_ptr[row_cursor + 1:row_cursor + 1 + nrows] = (
                    source_ptr[start + 1:stop + 1] - lo + nnz_cursor
                )
                codes[row_cursor:row_cursor + nrows] = code_map[source_codes[start:stop]]
                barcodes[row_cursor:row_cursor + nrows] = src["obs"][index_key][start:stop]
                row_cursor += nrows
                nnz_cursor += ndata
            if row_cursor != len(selected_rows) or nnz_cursor != nnz:
                raise ValueError("copy counts mismatch")
            if np.any(codes[:] < 0) or int(out_ptr[-1]) != nnz:
                raise ValueError("train view contains forbidden label or invalid CSR")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "experiment": "E247_kaden_crispra_unseen_tf",
        "status": "PASS_TRAIN_ONLY_VIEW",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_sha256": source_sha,
        "split_sha256": split_sha,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "n_train_labels": split_counts["train"],
        "n_validation_labels_excluded": split_counts["validation"],
        "n_test_labels_excluded": split_counts["test"],
        "n_rows": len(selected_rows),
        "n_controls": int(np.sum(source_codes[selected_rows] == source_labels.index("control"))),
        "n_genes": shape[1],
        "nnz": nnz,
        "source_contiguous_runs_indexed": len(runs),
        "heldout_expression_rows_indexed": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    status = build_view(
        args.source, args.split, args.output,
        source_sha=SOURCE_SHA256, split_sha=SPLIT_SHA256,
    )
    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.status.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
