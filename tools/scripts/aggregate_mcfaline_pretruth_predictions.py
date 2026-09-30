#!/usr/bin/env python3
"""Aggregate McFaline test predictions without reading test treated expression."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_number(path: Path) -> int:
    match = re.search(r"chunk_(\d+)", path.name)
    if not match:
        raise RuntimeError(f"unexpected prediction filename: {path}")
    return int(match.group(1))


def aggregate_prediction_states(
    prediction_dir: Path, gene_ids: list[str], n_tasks: int
) -> np.ndarray:
    files = sorted(prediction_dir.glob("prediction_rank*_chunk_*.h5ad"), key=prediction_number)
    if len(files) != n_tasks:
        raise RuntimeError(f"expected {n_tasks} prediction chunks, got {len(files)}")
    states = np.full((n_tasks, len(gene_ids)), np.nan, dtype=np.float32)
    seen: set[int] = set()
    selected = None
    for number, path in enumerate(files, 1):
        item = ad.read_h5ad(path, backed="r")
        if selected is None:
            axis = pd.Series(np.arange(item.n_vars), index=item.var_names.astype(str))
            if not set(gene_ids) <= set(axis.index):
                raise RuntimeError("prediction gene axis does not cover registered genes")
            selected = axis.loc[gene_ids].to_numpy(int)
        obs = item.obs.reset_index(drop=True)
        matrix = np.asarray(item.X, dtype=np.float32)[:, selected]
        for condition_index, positions in obs.groupby("_condition_idx").groups.items():
            index = int(condition_index)
            if index in seen or index < 0 or index >= n_tasks:
                raise RuntimeError(f"duplicate/out-of-range condition index {index}")
            seen.add(index)
            states[index] = matrix[np.asarray(list(positions), dtype=int)].mean(axis=0)
        item.file.close()
        del matrix
        if number % 50 == 0:
            print(f"[PretruthPredictions] files {number}/{len(files)}", flush=True)
    if len(seen) != n_tasks or not np.isfinite(states).all():
        raise RuntimeError("prediction aggregation is incomplete")
    return states


def aggregate_test_controls_only(
    h5ad: Path, split_path: Path, gene_ids: list[str]
) -> dict[tuple[str, str], np.ndarray]:
    split = pd.read_csv(split_path, header=None, names=["cell_id", "split"])
    split_map = pd.Series(split["split"].to_numpy(str), index=split.cell_id.astype(str)).to_dict()
    with h5py.File(h5ad, "r") as handle:
        obs = handle["obs"]
        cell_ids = categorical(obs, "_index")
        roles = np.asarray([split_map.get(value, "missing") for value in cell_ids], dtype=object)
        if (roles == "missing").any():
            raise RuntimeError("split failed to align")
        control = categorical(obs, "control") == "1"
        contexts = categorical(obs, "cell_type")
        treatments = categorical(obs, "treatment")
        selected_rows = np.flatnonzero((roles == "test") & control)
        all_genes = np.asarray(handle["var"]["gene_name"]).astype(str)
        axis = pd.Series(np.arange(len(all_genes)), index=all_genes)
        if not set(gene_ids) <= set(axis.index):
            raise RuntimeError("H5AD gene axis does not cover registered genes")
        selected_columns = axis.loc[gene_ids].to_numpy(int)
        output_index = np.full(len(all_genes), -1, dtype=int)
        output_index[selected_columns] = np.arange(len(selected_columns))
        x = handle["X"]
        indptr = np.asarray(x["indptr"])
        sums: dict[tuple[str, str], np.ndarray] = {}
        counts: dict[tuple[str, str], int] = {}
        # Access exactly the registered test control rows. No treated row's X
        # slice is read, even transiently.
        for number, row in enumerate(selected_rows, 1):
            p0, p1 = int(indptr[row]), int(indptr[row + 1])
            columns = np.asarray(x["indices"][p0:p1], dtype=int)
            values = np.asarray(x["data"][p0:p1], dtype=np.float32)
            mapped = output_index[columns]
            keep = mapped >= 0
            vector = np.zeros(len(gene_ids), dtype=np.float64)
            vector[mapped[keep]] = values[keep]
            key = (str(contexts[row]), str(treatments[row]))
            sums.setdefault(key, np.zeros(len(gene_ids), dtype=np.float64))
            sums[key] += vector
            counts[key] = counts.get(key, 0) + 1
            if number % 2500 == 0:
                print(f"[PretruthControls] rows {number}/{len(selected_rows)}", flush=True)
    return {key: value / counts[key] for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--public-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    tasks = pd.read_csv(args.task_manifest)
    gene_payload = json.loads((args.public_store / "gene_ids.json").read_text())
    gene_ids = gene_payload["gene_ids"] if isinstance(gene_payload, dict) else gene_payload
    predicted_state = aggregate_prediction_states(args.prediction_dir, gene_ids, len(tasks))
    test_controls = aggregate_test_controls_only(args.h5ad, args.split, gene_ids)
    control_matrix = np.stack(
        [test_controls[(str(row.cell_type), str(row.treatment))] for row in tasks.itertuples()]
    )
    raw_effect = predicted_state - control_matrix

    memory = pd.read_parquet(args.public_store / "public_memory.parquet")
    effects = np.load(args.public_store / "effect_vectors.npy", mmap_mode="r")
    train_mask = memory.provenance.astype(str).str.contains(r"::train(?:;|$)", regex=True)
    state_means: dict[tuple[str, str], np.ndarray] = {}
    for key, group in memory.loc[train_mask].groupby(["context", "condition"]):
        rows = group.effect_vector_row.to_numpy(int)
        state_means[(str(key[0]), str(key[1]))] = np.asarray(effects[rows], dtype=float).mean(axis=0)
    missing = {
        (str(row.cell_type), str(row.treatment))
        for row in tasks.itertuples()
        if (str(row.cell_type), str(row.treatment)) not in state_means
    }
    if missing:
        raise RuntimeError(f"missing train-state baselines: {sorted(missing)}")
    state_mean = np.stack(
        [state_means[(str(row.cell_type), str(row.treatment))] for row in tasks.itertuples()]
    )
    calibrated = args.alpha * raw_effect + (1.0 - args.alpha) * state_mean

    np.save(args.output_dir / "TEST_PREDICTED_STATES.npy", predicted_state.astype(np.float32))
    np.save(args.output_dir / "TEST_CONTROL_STATES.npy", control_matrix.astype(np.float32))
    np.save(args.output_dir / "TEST_RAW_EFFECTS.npy", raw_effect.astype(np.float32))
    np.save(args.output_dir / "TEST_STATE_MEAN_EFFECTS.npy", state_mean.astype(np.float32))
    np.save(args.output_dir / "TEST_CALIBRATED_EFFECTS.npy", calibrated.astype(np.float32))
    tasks.to_csv(args.output_dir / "TEST_TASKS.csv", index=False)
    (args.output_dir / "GENE_IDS.json").write_text(json.dumps(gene_ids, indent=2) + "\n")

    status = {
        "schema": "SafeConf-McFaline-Pretruth-Predictions-v1",
        "status": "COMPLETE",
        "n_tasks": int(len(tasks)),
        "n_perturbation_clusters": int(tasks.condition.nunique()),
        "n_strata": int(tasks[["cell_type", "treatment"]].drop_duplicates().shape[0]),
        "n_genes": int(len(gene_ids)),
        "decoder_validation_alpha": float(args.alpha),
        "effect_contract": "predicted treated state minus observed test control; convex validation calibration",
        "test_control_expression_accessed": True,
        "test_treated_expression_accessed": False,
        "task_manifest_sha256": sha256(args.task_manifest),
        "split_sha256": sha256(args.split),
        "calibrated_effects_sha256": sha256(args.output_dir / "TEST_CALIBRATED_EFFECTS.npy"),
    }
    (args.output_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
