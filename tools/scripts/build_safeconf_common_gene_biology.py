#!/usr/bin/env python3
"""October common-gene contract: train/validation-only real biology and quality.

The script scans the official processed CSR matrix in bounded row chunks and
never aggregates test expression. Quality is defined by independent guide,
plate, and deterministic split-half effect reproducibility; raw support counts
remain separate fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.safeconf_continual import PublicMemoryItem, PublicMemoryStore
H5AD = Path("/home/yyf/data/perturbench_mcfaline23_official/mcfaline23_gxe_processed.h5ad")
SPLIT = Path("/home/yyf/data/perturbench_mcfaline23_official/splits/mcfaline23_gxe_splits/full_covariate_split.csv")
OUT = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/mcfaline_quality"
STORE = Path("/home/yyf/data/safeconf_dual_memory_20260929/public_mcfaline_trainval")
SEED = 20260929
N_GENES = 512


class QualityFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=H5AD)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--store-root", type=Path, default=STORE)
    parser.add_argument("--n-genes", type=int, default=N_GENES)
    parser.add_argument("--gene-ids", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=1000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def categorical(group: h5py.Group, name: str) -> np.ndarray:
    node = group[name]
    if isinstance(node, h5py.Group):
        categories = np.asarray(node["categories"]).astype(str)
        codes = np.asarray(node["codes"], dtype=int)
        values = np.empty(len(codes), dtype=object)
        values[codes < 0] = None
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values.astype(str)
    return np.asarray(node).astype(str)


def factorize_keys(frame: pd.DataFrame, columns: list[str], mask: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    codes = np.full(len(frame), -1, dtype=np.int64)
    selected = frame.loc[mask, columns].astype(str)
    multi = pd.MultiIndex.from_frame(selected)
    local_codes, uniques = pd.factorize(multi, sort=True)
    codes[np.flatnonzero(mask)] = local_codes
    key_frame = uniques.to_frame(index=False)
    key_frame.columns = columns
    return codes, key_frame


def accumulate(
    sums: np.ndarray,
    counts: np.ndarray,
    codes: np.ndarray,
    values: np.ndarray,
) -> None:
    valid = codes >= 0
    if not valid.any():
        return
    use_codes = codes[valid]
    use_values = values[valid]
    order = np.argsort(use_codes, kind="stable")
    sorted_codes = use_codes[order]
    sorted_values = use_values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    unique = sorted_codes[starts]
    sums[unique] += np.add.reduceat(sorted_values, starts, axis=0)
    counts[unique] += np.diff(np.r_[starts, len(sorted_codes)])


def allowed_row_spans(mask: np.ndarray, chunk_rows: int):
    """Yield bounded contiguous allowed spans before reading expression.

    HDF5 may decompress shared storage chunks internally; no disallowed row
    values are returned or converted into the biological aggregation arrays.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or chunk_rows < 1:
        raise ValueError("row mask must be one-dimensional and chunk size positive")
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for first, stop in zip(boundaries[::2], boundaries[1::2]):
        for start in range(int(first), int(stop), chunk_rows):
            yield start, min(start + chunk_rows, int(stop))


def pairwise_cosine(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return float("nan")
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-12
    vectors = vectors[valid]
    norms = norms[valid]
    if len(vectors) < 2:
        return float("nan")
    matrix = (vectors @ vectors.T) / (norms[:, None] * norms[None, :])
    upper = matrix[np.triu_indices(len(vectors), k=1)]
    return float(np.median(upper))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def stable_half(cell_id: str) -> int:
    digest = hashlib.sha256(f"SafeConf-quality-v1\0{cell_id}".encode()).digest()
    return digest[0] & 1


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.store_root.exists():
        raise FileExistsError("refusing to overwrite McFaline quality/public memory")
    split = pd.read_csv(args.split, header=None, names=["cell_id", "split"])
    if set(split.split.unique()) != {"train", "val", "test"}:
        raise QualityFailure("official split labels changed")
    split_map = pd.Series(split.split.to_numpy(str), index=split.cell_id.astype(str)).to_dict()
    with h5py.File(args.h5ad, "r") as handle:
        obs_group = handle["obs"]
        cell_ids = categorical(obs_group, "_index")
        roles = np.asarray([split_map.get(cell, "missing") for cell in cell_ids], dtype=object)
        if (roles == "missing").any():
            raise QualityFailure("official cell split failed to align")
        # Test metadata are needed only to enforce exclusion; no test expression
        # is aggregated and no test covariate summary is emitted.
        dev = np.isin(roles, ["train", "val"])
        control = categorical(obs_group, "control") == "1"
        treated = dev & ~control
        control_dev = dev & control
        frame = pd.DataFrame({
            "cell_id": cell_ids,
            "role": roles,
            "perturbation": categorical(obs_group, "perturbation"),
            "context": categorical(obs_group, "cell_type"),
            "treatment": categorical(obs_group, "treatment"),
            "guide": categorical(obs_group, "gRNA_id"),
            "plate": categorical(obs_group, "PCR_plate"),
        })
        frame["half"] = np.fromiter((stable_half(x) for x in frame.cell_id), dtype=np.int8, count=len(frame))
        frame["task_id"] = frame.perturbation + "::" + frame.context + "::" + frame.treatment
        guide_codes, guide_keys = factorize_keys(frame, ["task_id", "guide"], treated)
        plate_codes, plate_keys = factorize_keys(frame, ["task_id", "plate"], treated)
        half_codes, half_keys = factorize_keys(frame, ["task_id", "half"], treated)
        control_codes, control_keys = factorize_keys(frame, ["context", "treatment"], control_dev)
        control_plate_codes, control_plate_keys = factorize_keys(frame, ["context", "treatment", "plate"], control_dev)
        control_half_codes, control_half_keys = factorize_keys(frame, ["context", "treatment", "half"], control_dev)
        genes = np.asarray(handle["var"]["gene_name"]).astype(str)
        # Stable schema-only gene selection: independent of expression and test
        # outcome distributions.
        payload = json.loads(args.gene_ids.read_text())
        registered_genes = payload["gene_ids"] if isinstance(payload, dict) else payload
        mapping = {str(g): i for i, g in enumerate(genes)}
        if len(registered_genes) < 2000 or len(set(registered_genes)) != len(registered_genes):
            raise QualityFailure("common-axis gene minimum or uniqueness failed")
        if not set(registered_genes) <= set(mapping):
            raise QualityFailure("common-axis gene IDs absent from native target gene space")
        selected = np.asarray([mapping[g] for g in registered_genes], dtype=int)
        selected_genes = genes[selected]
        specifications = [
            (guide_codes, guide_keys, "guide"),
            (plate_codes, plate_keys, "plate"),
            (half_codes, half_keys, "half"),
            (control_codes, control_keys, "control"),
            (control_plate_codes, control_plate_keys, "control_plate"),
            (control_half_codes, control_half_keys, "control_half"),
        ]
        aggregates = {
            name: [np.zeros((len(keys), len(selected)), dtype=np.float64), np.zeros(len(keys), dtype=np.int64)]
            for _, keys, name in specifications
        }
        x = handle["X"]
        indptr = x["indptr"]
        n_rows, n_cols = len(frame), len(genes)
        expression_rows_materialized = 0
        for start, end in allowed_row_spans(dev, args.chunk_rows):
            if not dev[start:end].all():
                raise QualityFailure("disallowed expression row entered read span")
            p0, p1 = int(indptr[start]), int(indptr[end])
            block = sparse.csr_matrix(
                (
                    np.asarray(x["data"][p0:p1]),
                    np.asarray(x["indices"][p0:p1]),
                    np.asarray(indptr[start : end + 1], dtype=np.int64) - p0,
                ),
                shape=(end - start, n_cols),
            )[:, selected].toarray()
            expression_rows_materialized += end - start
            for codes, _, name in specifications:
                accumulate(aggregates[name][0], aggregates[name][1], codes[start:end], block)
            if start % 50000 == 0:
                print(f"[McFalineQuality] rows {start}:{end}/{n_rows}", flush=True)
        if expression_rows_materialized != int(dev.sum()):
            raise QualityFailure("allowed expression rows incomplete or duplicated")
        means = {}
        for _, keys, name in specifications:
            sums, counts = aggregates[name]
            if (counts == 0).any():
                raise QualityFailure(f"empty aggregate group: {name}")
            means[name] = sums / counts[:, None]

    control_map = {
        tuple(row): means["control"][i]
        for i, row in enumerate(control_keys[["context", "treatment"]].itertuples(index=False, name=None))
    }
    control_plate_map = {
        tuple(row): means["control_plate"][i]
        for i, row in enumerate(control_plate_keys[["context", "treatment", "plate"]].itertuples(index=False, name=None))
    }
    control_half_map = {
        tuple(map(str, row)): means["control_half"][i]
        for i, row in enumerate(control_half_keys[["context", "treatment", "half"]].itertuples(index=False, name=None))
    }
    task_meta = frame.loc[treated, ["task_id", "perturbation", "context", "treatment", "role"]].drop_duplicates()
    role_map = task_meta.groupby("task_id").role.apply(lambda s: ";".join(sorted(s.unique()))).to_dict()
    task_info = task_meta.drop_duplicates("task_id").set_index("task_id")
    guide_effects: dict[str, list[np.ndarray]] = {}
    for i, row in enumerate(guide_keys.itertuples(index=False)):
        context, treatment = task_info.loc[row.task_id, ["context", "treatment"]]
        guide_effects.setdefault(row.task_id, []).append(means["guide"][i] - control_map[(context, treatment)])
    plate_effects: dict[str, list[np.ndarray]] = {}
    for i, row in enumerate(plate_keys.itertuples(index=False)):
        context, treatment = task_info.loc[row.task_id, ["context", "treatment"]]
        key = (context, treatment, row.plate)
        if key in control_plate_map:
            plate_effects.setdefault(row.task_id, []).append(means["plate"][i] - control_plate_map[key])
    half_effects: dict[str, dict[str, np.ndarray]] = {}
    for i, row in enumerate(half_keys.itertuples(index=False)):
        context, treatment = task_info.loc[row.task_id, ["context", "treatment"]]
        key = tuple(map(str, (context, treatment, row.half)))
        if key in control_half_map:
            half_effects.setdefault(row.task_id, {})[str(row.half)] = means["half"][i] - control_half_map[key]
    treated_frame = frame.loc[treated]
    support = treated_frame.groupby("task_id").agg(
        n_cells=("cell_id", "size"), n_guides=("guide", "nunique"),
        n_plates=("plate", "nunique"), n_roles=("role", "nunique"),
    )
    records = []
    public_items: list[PublicMemoryItem] = []
    public_effects: list[np.ndarray] = []
    public_controls: list[np.ndarray] = []
    for task_id, row in task_info.iterrows():
        guides = np.asarray(guide_effects.get(task_id, []))
        plates = np.asarray(plate_effects.get(task_id, []))
        halves = half_effects.get(task_id, {})
        record = {
            "task_id": task_id,
            "perturbation": row.perturbation,
            "context": row.context,
            "treatment": row.treatment,
            "split_roles": role_map[task_id],
            "n_cells": int(support.loc[task_id, "n_cells"]),
            "n_guides": int(support.loc[task_id, "n_guides"]),
            "n_plates": int(support.loc[task_id, "n_plates"]),
            "guide_reproducibility": pairwise_cosine(guides),
            "plate_reproducibility": pairwise_cosine(plates),
            "split_half_stability": cosine(halves["0"], halves["1"]) if set(halves) == {"0", "1"} else float("nan"),
            "batch_agreement": pairwise_cosine(plates),
            "quality_gene_count": len(selected_genes),
            "quality_gene_schema_sha256": hashlib.sha256("\n".join(selected_genes).encode()).hexdigest(),
            "test_expression_opened": False,
        }
        records.append(record)
        if len(guides) == 0:
            raise QualityFailure(f"task has no guide effect: {task_id}")
        context_control = control_map[(row.context, row.treatment)]
        public_items.append(PublicMemoryItem(
            experiment_id=f"McFaline23::{task_id}",
            study_id="McFalineFigueroa23_PerturBench",
            context=str(row.context),
            perturbation_type="genetic_single_gene",
            perturbation_target=str(row.perturbation),
            condition=str(row.treatment),
            effect_vector_row=len(public_items),
            effect_contract_id="McFaline23_common2840gene_log1p_delta_v1",
            control_source="trainval_context_treatment_control",
            gene_space_id="E201_McFaline_common2840gene_v1",
            n_cells=int(record["n_cells"]),
            n_guides=int(record["n_guides"]),
            n_plates=int(record["n_plates"]),
            n_batches=int(record["n_plates"]),
            guide_reproducibility=float(record["guide_reproducibility"]),
            plate_reproducibility=float(record["plate_reproducibility"]),
            split_half_stability=float(record["split_half_stability"]),
            batch_agreement=float(record["batch_agreement"]),
            provenance=f"{args.h5ad}::{record['split_roles']}",
            eligibility=True,
            timestamp="2026-09-29T00:00:00+00:00",
        ))
        public_effects.append(np.mean(guides, axis=0).astype(np.float32))
        public_controls.append(context_control.astype(np.float32))
    result = pd.DataFrame(records)
    if result.empty or result.task_id.duplicated().any():
        raise QualityFailure("quality task output failed")
    args.output.mkdir(parents=True)
    table_path = args.output / "MCFALINE_TRAINVAL_QUALITY.csv.gz"
    result.to_csv(table_path, index=False, compression={"method": "gzip", "mtime": 0})
    public_manifest = PublicMemoryStore(args.store_root).create(
        public_items,
        np.stack(public_effects),
        np.stack(public_controls),
        selected_genes.tolist(),
        {
            "h5ad_sha256": sha256_file(args.h5ad),
            "split_sha256": sha256_file(args.split),
            "allowed_roles": ["train", "val"],
            "test_expression_opened": False,
            "test_expression_rows_materialized": 0,
            "expression_access_policy": "trainval_contiguous_spans_before_CSR_read_v2",
            "quality_definition": "held-out guide/plate/split-half reproducibility",
        },
    )
    audit = {
        "status": "PASS",
        "schema": "SafeConf-McFaline-Quality-v1",
        "h5ad_sha256": sha256_file(args.h5ad),
        "split_sha256": sha256_file(args.split),
        "n_trainval_cells": int(dev.sum()),
        "n_test_cells_aggregated": 0,
        "n_test_expression_rows_materialized": 0,
        "n_allowed_expression_rows_materialized": expression_rows_materialized,
        "n_quality_tasks": len(result),
        "n_quality_genes": len(selected_genes),
        "guide_quality_coverage": float(result.guide_reproducibility.notna().mean()),
        "plate_quality_coverage": float(result.plate_reproducibility.notna().mean()),
        "split_half_quality_coverage": float(result.split_half_stability.notna().mean()),
        "support_is_not_quality": True,
        "test_truth_opened": False,
        "result_sha256": sha256_file(table_path),
        "public_memory_manifest": public_manifest,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "QUALITY_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
