#!/usr/bin/env python3
"""Generate E208 counterfactual centroids without reading test perturbation truth.

The official PerturBench prediction loader repeatedly scans the 93.5 GB Jiang24
H5 file.  E208 freezes a control-only sparse cache before prediction, then this
script feeds exactly those controls through an unchanged trained model.  Only
task-level centroids are written; no test perturbed expression is opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class PredictionFailure(RuntimeError):
    """A frozen-input, model, or output contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def load_cache(cache_path: Path) -> tuple[csr_matrix, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as packed:
        required = {"data", "indices", "indptr", "shape", "gene_names"}
        if not required.issubset(packed.files):
            raise PredictionFailure(f"control cache fields missing: {required-set(packed.files)}")
        shape = tuple(map(int, packed["shape"]))
        matrix = csr_matrix(
            (packed["data"], packed["indices"], packed["indptr"]), shape=shape
        )
        genes = packed["gene_names"].astype(str)
    if matrix.shape != (6799, 15473) or len(genes) != 15473:
        raise PredictionFailure(f"unexpected cache shape: {matrix.shape}, genes={len(genes)}")
    if matrix.dtype != np.float32 or not np.isfinite(matrix.data).all():
        raise PredictionFailure("control cache must contain finite float32 values")
    return matrix, genes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    parser.add_argument("--architecture", choices=("latent", "linear"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--manifest-status", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=0,
        help="Explicit nonformal mode: run only the first N tasks.",
    )
    args = parser.parse_args()

    source_root = args.perturbench_repo.expanduser().absolute() / "src"
    if not source_root.is_dir():
        raise PredictionFailure(f"PerturBench source missing: {source_root}")
    sys.path.insert(0, str(source_root))

    import torch
    from perturbench.data.datasets.population import Counterfactual
    from perturbench.data.transforms.pipelines import LinearModelPipelineControls
    from perturbench.data.types import Batch, FrozenDictKeyMap
    from perturbench.modelcore.models.latent_additive import LatentAdditive
    from perturbench.modelcore.models.linear_additive import LinearAdditive

    cache_dir = args.cache_dir.expanduser().absolute()
    cache_path = cache_dir / "E208_TARGET_CONTROL_CACHE.npz"
    rows_path = cache_dir / "E208_TARGET_CONTROL_ROWS.csv"
    cache_status_path = cache_dir / "E208_CONTROL_CACHE_STATUS.json"
    checkpoint = args.checkpoint.expanduser().absolute()
    tasks_path = args.tasks.expanduser().absolute()
    output = args.output_dir.expanduser().absolute()
    for path in (
        cache_path,
        rows_path,
        cache_status_path,
        checkpoint,
        tasks_path,
        args.manifest_status,
    ):
        if not path.is_file():
            raise PredictionFailure(f"required input missing: {path}")

    cache_status = json.loads(cache_status_path.read_text(encoding="utf-8"))
    manifest_status = json.loads(args.manifest_status.read_text(encoding="utf-8"))
    if cache_status.get("status") != "PASS":
        raise PredictionFailure("control cache did not pass its frozen gate")
    if cache_status.get("test_perturbed_expression_rows_read") != 0:
        raise PredictionFailure("control cache reports test perturbation truth access")
    if sha256_file(cache_path) != cache_status["cache"]["sha256"]:
        raise PredictionFailure("control cache checksum changed")
    if sha256_file(rows_path) != cache_status["row_manifest"]["sha256"]:
        raise PredictionFailure("control row manifest checksum changed")
    if manifest_status.get("status") != "PASS" or manifest_status.get("n_tasks") != 224:
        raise PredictionFailure("prediction manifest did not pass its frozen gate")
    if sha256_file(tasks_path) != manifest_status["manifest_sha256"]:
        raise PredictionFailure("prediction manifest checksum changed")

    controls, gene_names = load_cache(cache_path)
    rows = pd.read_csv(rows_path)
    expected_rows = {"cache_row", "cell_type", "treatment"}
    if not expected_rows.issubset(rows.columns) or len(rows) != controls.shape[0]:
        raise PredictionFailure("control row manifest no longer matches sparse cache")
    if not np.array_equal(rows.cache_row.to_numpy(int), np.arange(len(rows))):
        raise PredictionFailure("control cache rows are not in canonical order")

    tasks = pd.read_csv(tasks_path)
    expected_tasks = {"condition", "cell_type", "treatment", "task_id"}
    if not expected_tasks.issubset(tasks.columns) or len(tasks) != 224:
        raise PredictionFailure("formal task manifest is not 224 unique tasks")
    if tasks.task_id.nunique() != len(tasks):
        raise PredictionFailure("task identifiers are not unique")
    formal = args.smoke_limit <= 0
    if not formal:
        if args.smoke_limit > len(tasks):
            raise PredictionFailure("smoke limit exceeds task count")
        tasks = tasks.iloc[: args.smoke_limit].copy()

    model_class = LatentAdditive if args.architecture == "latent" else LinearAdditive
    model = model_class.load_from_checkpoint(
        checkpoint, map_location="cpu", weights_only=False
    )
    context = model.training_record["train_context"]
    checkpoint_genes = np.asarray(model.hparams["gene_names"], dtype=str)
    if len(checkpoint_genes) != len(gene_names) or not np.array_equal(
        checkpoint_genes, gene_names
    ):
        raise PredictionFailure("checkpoint and cache gene axes differ")
    perturbation_uniques = set(map(str, context["perturbation_uniques"]))
    missing_perturbations = sorted(set(tasks.condition.astype(str)) - perturbation_uniques)
    if missing_perturbations:
        raise PredictionFailure(
            f"tasks contain perturbations absent from training encoder: {missing_perturbations}"
        )
    covariate_uniques = {
        str(key): None if values is None else set(map(str, values))
        for key, values in context["covariate_uniques"].items()
    }
    for key in ("cell_type", "treatment"):
        absent = sorted(set(tasks[key].astype(str)) - covariate_uniques[key])
        if absent:
            raise PredictionFailure(f"unknown {key} values in tasks: {absent}")

    index_pairs = []
    for (cell_type, treatment), block in rows.groupby(
        ["cell_type", "treatment"], sort=True
    ):
        index_pairs.append(
            (
                {"cell_type": str(cell_type), "treatment": str(treatment)},
                block.cache_row.to_numpy(np.int64),
            )
        )
    control_indexes = FrozenDictKeyMap(index_pairs)
    dataset = Counterfactual(
        perturbations=[[str(value)] for value in tasks.condition],
        covariates={
            "cell_type": tasks.cell_type.astype(str).tolist(),
            "treatment": tasks.treatment.astype(str).tolist(),
        },
        control_expression=controls,
        control_indexes=control_indexes,
        gene_names=gene_names.tolist(),
        transforms=None,
        info={
            "covariate_uniques": covariate_uniques,
            "perturbation_key": str(context["perturbation_key"]),
            "covariate_keys": list(map(str, context["covariate_keys"])),
            "perturbation_combination_delimiter": str(
                context["perturbation_combination_delimiter"]
            ),
            "perturbation_control_value": str(context["perturbation_control_value"]),
        },
    )
    dataset.transform = LinearModelPipelineControls(
        perturbation_uniques=perturbation_uniques,
        covariate_uniques=covariate_uniques,
        use_perturbation_embedding=bool(context["use_perturbation_embedding"]),
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PredictionFailure("CUDA device requested but CUDA is unavailable")
    model = model.to(device).eval()
    predictions = np.empty((len(tasks), len(gene_names)), dtype=np.float32)
    control_centroids = np.empty_like(predictions)
    selected_control_counts = np.empty(len(tasks), dtype=np.int32)

    def move_batch(batch: Batch) -> Batch:
        return batch._replace(
            gene_expression=batch.gene_expression.to(device),
            perturbations=batch.perturbations.to(device),
            covariates={key: value.to(device) for key, value in batch.covariates.items()},
        )

    with torch.inference_mode():
        for task_index in range(len(tasks)):
            raw_batch, _ = dataset.__getitems__([task_index])
            selected_control_counts[task_index] = len(raw_batch.gene_expression)
            batch = move_batch(raw_batch)
            prediction = model.predict(batch)
            if not isinstance(prediction, torch.Tensor):
                prediction = prediction.mean
            if prediction.ndim != 2 or prediction.shape[1] != len(gene_names):
                raise PredictionFailure(
                    f"unexpected prediction shape for task {task_index}: {prediction.shape}"
                )
            predictions[task_index] = (
                prediction.float().mean(dim=0).detach().cpu().numpy()
            )
            key = {
                "cell_type": str(tasks.iloc[task_index].cell_type),
                "treatment": str(tasks.iloc[task_index].treatment),
            }
            control_centroids[task_index] = np.asarray(
                controls[control_indexes[key]].mean(axis=0), dtype=np.float32
            ).reshape(-1)

    if not np.isfinite(predictions).all() or not np.isfinite(control_centroids).all():
        raise PredictionFailure("non-finite prediction or control centroid")
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "E208_PREDICTION_CENTROIDS.npy"
    control_path = output / "E208_CONTROL_CENTROIDS.npy"
    task_output_path = output / "E208_PREDICTION_TASKS.csv"
    temporary_tasks = task_output_path.with_name(f".{task_output_path.name}.tmp")
    tasks.to_csv(temporary_tasks, index=False)
    os.replace(temporary_tasks, task_output_path)
    atomic_npy(prediction_path, predictions)
    atomic_npy(control_path, control_centroids)
    status = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_PRETRUTH_PREDICTION_CENTROIDS",
        "status": "PASS" if formal else "SMOKE_PASS",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "architecture": args.architecture,
        "seed": args.seed,
        "device": str(device),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "cache_sha256": cache_status["cache"]["sha256"],
        "control_rows_sha256": cache_status["row_manifest"]["sha256"],
        "input_manifest_sha256": manifest_status["manifest_sha256"],
        "n_tasks": len(tasks),
        "n_genes": len(gene_names),
        "n_contexts": len(tasks[["cell_type", "treatment"]].drop_duplicates()),
        "selected_control_count_min": int(selected_control_counts.min()),
        "selected_control_count_max": int(selected_control_counts.max()),
        "prediction_centroids": {
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
        },
        "control_centroids": {
            "path": str(control_path),
            "sha256": sha256_file(control_path),
        },
        "task_manifest": {
            "path": str(task_output_path),
            "sha256": sha256_file(task_output_path),
        },
        "finite_prediction_values": True,
        "finite_control_values": True,
        "test_control_expression_rows_available": controls.shape[0],
        "test_perturbed_expression_rows_read": 0,
        "target_truth_access": "NOT_AUTHORIZED",
    }
    atomic_json(output / "E208_PREDICTION_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
