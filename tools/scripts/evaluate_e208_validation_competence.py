#!/usr/bin/env python3
"""Evaluate the frozen E208 validation competence gate before test prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


class CompetenceFailure(RuntimeError):
    """The registered validation competence calculation cannot be completed."""


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


def atomic_csv(path: Path, value: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    value.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_prediction(directory: Path, architecture: str, seed: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    status_path = directory / "E208_VALIDATION_PREDICTION_STATUS.json"
    prediction_path = directory / "E208_VALIDATION_PREDICTION_CENTROIDS.npy"
    control_path = directory / "E208_VALIDATION_CONTROL_CENTROIDS.npy"
    tasks_path = directory / "E208_VALIDATION_TASKS.csv"
    for path in (status_path, prediction_path, control_path, tasks_path):
        if not path.is_file():
            raise CompetenceFailure(f"validation prediction artifact missing: {path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("status") != "PASS"
        or status.get("architecture") != architecture
        or int(status.get("seed", -1)) != seed
        or status.get("n_tasks") != 216
        or status.get("test_perturbed_expression_rows_read") != 0
    ):
        raise CompetenceFailure(f"invalid prediction status: {directory}")
    if sha256_file(prediction_path) != status["prediction_centroids_sha256"]:
        raise CompetenceFailure(f"prediction checksum changed: {directory}")
    if sha256_file(control_path) != status["control_centroids_sha256"]:
        raise CompetenceFailure(f"control checksum changed: {directory}")
    if sha256_file(tasks_path) != status["task_manifest_sha256"]:
        raise CompetenceFailure(f"task checksum changed: {directory}")
    predictions = np.load(prediction_path, allow_pickle=False)
    controls = np.load(control_path, allow_pickle=False)
    tasks = pd.read_csv(tasks_path)
    if predictions.shape != (216, 15473) or controls.shape != predictions.shape:
        raise CompetenceFailure(f"unexpected validation prediction shape: {directory}")
    return predictions, controls, tasks, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-cache-dir", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, action="append", required=True)
    parser.add_argument("--linear-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.latent_dir) != 4:
        raise CompetenceFailure("exactly four LatentAdditive prediction directories required")

    package = args.validation_cache_dir.expanduser().absolute()
    package_status_path = package / "E208_VALIDATION_CACHE_STATUS.json"
    truth_path = package / "E208_VALIDATION_TRUTH_CENTROIDS.npy"
    tasks_path = package / "E208_VALIDATION_TASKS.csv"
    package_status = json.loads(package_status_path.read_text(encoding="utf-8"))
    if (
        package_status.get("status") != "PASS"
        or package_status.get("n_tasks") != 216
        or package_status.get("test_perturbed_expression_rows_read") != 0
    ):
        raise CompetenceFailure("validation cache status failed")
    if sha256_file(truth_path) != package_status["truth_centroids"]["sha256"]:
        raise CompetenceFailure("validation truth checksum changed")
    if sha256_file(tasks_path) != package_status["task_manifest"]["sha256"]:
        raise CompetenceFailure("validation task checksum changed")
    truth = np.load(truth_path, allow_pickle=False)
    frozen_tasks = pd.read_csv(tasks_path)
    if truth.shape != (216, 15473):
        raise CompetenceFailure("validation truth shape changed")

    latent_sum = np.zeros_like(truth, dtype=np.float64)
    common_controls = None
    common_tasks = None
    input_records = []
    for seed, directory in enumerate(args.latent_dir, start=1):
        prediction, controls, tasks, status = load_prediction(
            directory.expanduser().absolute(), "latent", seed
        )
        latent_sum += prediction.astype(np.float64)
        if common_controls is None:
            common_controls = controls
            common_tasks = tasks
        elif not np.array_equal(controls, common_controls) or not tasks.equals(common_tasks):
            raise CompetenceFailure("latent models did not use identical controls/tasks")
        input_records.append(status)
    linear, linear_controls, linear_tasks, linear_status = load_prediction(
        args.linear_dir.expanduser().absolute(), "linear", 1
    )
    input_records.append(linear_status)
    if (
        not np.array_equal(linear_controls, common_controls)
        or not linear_tasks.equals(common_tasks)
        or not common_tasks.equals(frozen_tasks)
    ):
        raise CompetenceFailure("model families did not use the frozen shared task/control inputs")

    latent = latent_sum / 4.0
    truth64 = truth.astype(np.float64)
    control64 = common_controls.astype(np.float64)
    linear64 = linear.astype(np.float64)
    no_change_mse = np.mean((control64 - truth64) ** 2, axis=1)
    latent_mse = np.mean((latent - truth64) ** 2, axis=1)
    linear_mse = np.mean((linear64 - truth64) ** 2, axis=1)
    if not all(np.isfinite(values).all() for values in (no_change_mse, latent_mse, linear_mse)):
        raise CompetenceFailure("non-finite validation MSE")

    task_results = frozen_tasks.copy()
    task_results["no_change_mse"] = no_change_mse
    task_results["latent_architecture_centroid_mse"] = latent_mse
    task_results["linear_mse"] = linear_mse
    task_results["latent_relative_improvement"] = 1.0 - latent_mse / no_change_mse
    task_results["linear_relative_improvement"] = 1.0 - linear_mse / no_change_mse
    by_context = (
        task_results.groupby(["cell_type", "treatment"], sort=True)
        .agg(
            n_tasks=("task_id", "size"),
            no_change_mse=("no_change_mse", "mean"),
            latent_mse=("latent_architecture_centroid_mse", "mean"),
            linear_mse=("linear_mse", "mean"),
        )
        .reset_index()
    )
    by_context["latent_relative_improvement"] = 1.0 - by_context.latent_mse / by_context.no_change_mse
    by_context["linear_relative_improvement"] = 1.0 - by_context.linear_mse / by_context.no_change_mse
    means = {
        "no_change_mse": float(no_change_mse.mean()),
        "latent_architecture_centroid_mse": float(latent_mse.mean()),
        "linear_mse": float(linear_mse.mean()),
    }
    latent_pass = means["latent_architecture_centroid_mse"] < means["no_change_mse"]
    linear_pass = means["linear_mse"] < means["no_change_mse"]
    gate_status = "PASS" if latent_pass and linear_pass else "FAMILY_COMPETENCE_BLOCKED"

    output = args.output_dir.expanduser().absolute()
    task_result_path = output / "E208_VALIDATION_COMPETENCE_TASKS.csv"
    context_result_path = output / "E208_VALIDATION_COMPETENCE_CONTEXTS.csv"
    atomic_csv(task_result_path, task_results)
    atomic_csv(context_result_path, by_context)
    result = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_VALIDATION_COMPETENCE_GATE",
        "status": gate_status,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "decision_rule": "both architecture mean task MSE values strictly below no-change mean task MSE",
        "n_tasks": len(task_results),
        "n_contexts": len(by_context),
        "mean_task_mse": means,
        "relative_improvement_vs_no_change": {
            "latent_architecture_centroid": float(1.0 - means["latent_architecture_centroid_mse"] / means["no_change_mse"]),
            "linear": float(1.0 - means["linear_mse"] / means["no_change_mse"]),
        },
        "gates": {
            "latent_strictly_better_than_no_change": bool(latent_pass),
            "linear_strictly_better_than_no_change": bool(linear_pass),
        },
        "inputs": [
            {
                "architecture": item["architecture"],
                "seed": item["seed"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "prediction_centroids_sha256": item["prediction_centroids_sha256"],
            }
            for item in input_records
        ],
        "task_results": {
            "path": str(task_result_path),
            "sha256": sha256_file(task_result_path),
        },
        "context_results": {
            "path": str(context_result_path),
            "sha256": sha256_file(context_result_path),
        },
        "validation_expression_used": True,
        "test_perturbed_expression_rows_read": 0,
        "target_truth_access": "NOT_AUTHORIZED",
    }
    atomic_json(output / "E208_VALIDATION_COMPETENCE_STATUS.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
