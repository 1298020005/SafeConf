#!/usr/bin/env python3
"""Predict the frozen E208 validation tasks for the preregistered competence gate."""

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


class ValidationPredictionFailure(RuntimeError):
    """A validation input, model, or output contract failed."""


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    parser.add_argument("--architecture", choices=("latent", "linear"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-limit", type=int, default=0)
    args = parser.parse_args()

    source_root = args.perturbench_repo.expanduser().absolute() / "src"
    sys.path.insert(0, str(source_root))
    import torch
    from perturbench.data.datasets.population import Counterfactual
    from perturbench.data.transforms.pipelines import LinearModelPipelineControls
    from perturbench.data.types import Batch, FrozenDictKeyMap
    from perturbench.modelcore.models.latent_additive import LatentAdditive
    from perturbench.modelcore.models.linear_additive import LinearAdditive

    package = args.validation_cache_dir.expanduser().absolute()
    cache_path = package / "E208_VALIDATION_CONTROL_CACHE.npz"
    rows_path = package / "E208_VALIDATION_CONTROL_ROWS.csv"
    tasks_path = package / "E208_VALIDATION_TASKS.csv"
    status_path = package / "E208_VALIDATION_CACHE_STATUS.json"
    checkpoint = args.checkpoint.expanduser().absolute()
    for path in (cache_path, rows_path, tasks_path, status_path, checkpoint):
        if not path.is_file():
            raise ValidationPredictionFailure(f"required input missing: {path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "PASS" or status.get("n_tasks") != 216:
        raise ValidationPredictionFailure("validation cache did not pass its frozen gate")
    if status.get("test_perturbed_expression_rows_read") != 0:
        raise ValidationPredictionFailure("validation package reports test truth access")
    for path, key in (
        (cache_path, "control_cache"),
        (rows_path, "control_rows"),
        (tasks_path, "task_manifest"),
    ):
        if sha256_file(path) != status[key]["sha256"]:
            raise ValidationPredictionFailure(f"validation {key} checksum changed")

    with np.load(cache_path, allow_pickle=False) as packed:
        shape = tuple(map(int, packed["shape"]))
        controls = csr_matrix(
            (packed["data"], packed["indices"], packed["indptr"]), shape=shape
        )
        gene_names = packed["gene_names"].astype(str)
    if controls.shape != (6794, 15473) or controls.dtype != np.float32:
        raise ValidationPredictionFailure(f"unexpected validation cache: {controls.shape}")
    rows = pd.read_csv(rows_path)
    tasks = pd.read_csv(tasks_path)
    if len(rows) != controls.shape[0] or len(tasks) != 216 or tasks.task_id.nunique() != 216:
        raise ValidationPredictionFailure("validation rows or tasks changed")
    formal = args.smoke_limit <= 0
    if not formal:
        if args.smoke_limit > len(tasks):
            raise ValidationPredictionFailure("smoke limit exceeds validation tasks")
        tasks = tasks.iloc[: args.smoke_limit].copy()

    model_class = LatentAdditive if args.architecture == "latent" else LinearAdditive
    model = model_class.load_from_checkpoint(
        checkpoint, map_location="cpu", weights_only=False
    )
    context = model.training_record["train_context"]
    checkpoint_genes = np.asarray(model.hparams["gene_names"], dtype=str)
    if not np.array_equal(checkpoint_genes, gene_names):
        raise ValidationPredictionFailure("checkpoint and validation gene axes differ")
    perturbation_uniques = set(map(str, context["perturbation_uniques"]))
    missing = sorted(set(tasks.condition.astype(str)) - perturbation_uniques)
    if missing:
        raise ValidationPredictionFailure(f"validation perturbations absent from encoder: {missing}")
    covariate_uniques = {
        str(key): None if values is None else set(map(str, values))
        for key, values in context["covariate_uniques"].items()
    }

    control_indexes = FrozenDictKeyMap(
        [
            (
                {"cell_type": str(cell), "treatment": str(treatment)},
                block.cache_row.to_numpy(np.int64),
            )
            for (cell, treatment), block in rows.groupby(
                ["cell_type", "treatment"], sort=True
            )
        ]
    )
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
        raise ValidationPredictionFailure("CUDA requested but unavailable")
    model = model.to(device).eval()
    predictions = np.empty((len(tasks), len(gene_names)), dtype=np.float32)
    control_centroids = np.empty_like(predictions)

    def move_batch(batch: Batch) -> Batch:
        return batch._replace(
            gene_expression=batch.gene_expression.to(device),
            perturbations=batch.perturbations.to(device),
            covariates={key: value.to(device) for key, value in batch.covariates.items()},
        )

    with torch.inference_mode():
        for task_index in range(len(tasks)):
            raw_batch, _ = dataset.__getitems__([task_index])
            prediction = model.predict(move_batch(raw_batch))
            if not isinstance(prediction, torch.Tensor):
                prediction = prediction.mean
            predictions[task_index] = prediction.float().mean(dim=0).cpu().numpy()
            key = {
                "cell_type": str(tasks.iloc[task_index].cell_type),
                "treatment": str(tasks.iloc[task_index].treatment),
            }
            control_centroids[task_index] = np.asarray(
                controls[control_indexes[key]].mean(axis=0), dtype=np.float32
            ).reshape(-1)
    if not np.isfinite(predictions).all() or not np.isfinite(control_centroids).all():
        raise ValidationPredictionFailure("non-finite validation centroids")

    output = args.output_dir.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "E208_VALIDATION_PREDICTION_CENTROIDS.npy"
    control_path = output / "E208_VALIDATION_CONTROL_CENTROIDS.npy"
    task_output = output / "E208_VALIDATION_TASKS.csv"
    atomic_npy(prediction_path, predictions)
    atomic_npy(control_path, control_centroids)
    temporary_tasks = task_output.with_name(f".{task_output.name}.tmp")
    tasks.to_csv(temporary_tasks, index=False)
    os.replace(temporary_tasks, task_output)
    result = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_VALIDATION_COMPETENCE_PREDICTION",
        "status": "PASS" if formal else "SMOKE_PASS",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "architecture": args.architecture,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation_package_sha256": sha256_file(status_path),
        "n_tasks": len(tasks),
        "n_genes": len(gene_names),
        "prediction_centroids_sha256": sha256_file(prediction_path),
        "control_centroids_sha256": sha256_file(control_path),
        "task_manifest_sha256": sha256_file(task_output),
        "validation_expression_used": True,
        "test_perturbed_expression_rows_read": 0,
        "target_truth_access": "NOT_AUTHORIZED",
    }
    atomic_json(output / "E208_VALIDATION_PREDICTION_STATUS.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
