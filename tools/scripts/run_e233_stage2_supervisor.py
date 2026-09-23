#!/usr/bin/env python3
"""Automatically complete E233 selected-variant seeds 2-4 after stage-1 PASS."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from run_e233_stage1_supervisor import best_checkpoint, evaluate_variant, status


TERMINAL = {"STAGE1_PASS", "STAGE1_COMPETENCE_BLOCKED", "BLOCKED_TRAINING_FAILED",
            "BLOCKED_VALIDATION_FAILED"}
VARIANTS = {"matched_control_softplus", "matched_control_linear"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def run_training(args: argparse.Namespace, variant: str, seed: int, gpu: int) -> tuple[int, dict]:
    run = args.root / variant / f"seed_{seed}"
    if run.exists():
        raise RuntimeError(f"refuse to overwrite stage-2 run: {run}")
    run.mkdir(parents=True)
    command = [
        str(args.python), str(args.job_script), "--stage", "stage2",
        "--perturbench-repo", str(args.perturbench_repo), "--python", str(args.python),
        "--data-dir", str(args.data_dir), "--run-dir", str(run),
        "--variant", variant, "--seed", str(seed),
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = (run / "SUPERVISOR.log").open("w")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    return process.pid, {"process": process, "log": log, "run": run, "seed": seed, "gpu": gpu}


def wait_jobs(jobs: list[dict], output: Path, state: dict, poll: int) -> bool:
    while any(job["process"].poll() is None for job in jobs):
        state["training_states"] = {
            str(job["seed"]): (status(job["run"] / "E233_RUN_STATUS.json") or {}).get("status", "STARTING")
            for job in jobs
        }
        atomic_json(output, state)
        time.sleep(poll)
    codes = [job["process"].wait() for job in jobs]
    for job in jobs:
        job["log"].close()
    state["training_exit_codes"].update({str(job["seed"]): code for job, code in zip(jobs, codes)})
    atomic_json(output, state)
    return all(code == 0 for code in codes)


def predict_validation(args: argparse.Namespace, variant: str, seed: int) -> Path:
    run = args.root / variant / f"seed_{seed}"
    record = status(run / "E233_RUN_STATUS.json")
    if record is None or record.get("status") != "COMPLETE" or record.get("seed") != seed:
        raise RuntimeError(f"invalid stage-2 training record for seed {seed}")
    checkpoint = best_checkpoint(record)
    output = args.output / "validation" / f"seed_{seed}"
    if output.exists():
        raise RuntimeError(f"refuse to overwrite validation: {output}")
    output.mkdir(parents=True)
    command = [str(args.python), str(args.prediction_script),
               "--perturbench-repo", str(args.perturbench_repo),
               "--architecture", "linear", "--seed", str(seed),
               "--checkpoint", str(checkpoint),
               "--validation-cache-dir", str(args.validation_cache),
               "--output-dir", str(output), "--device", "cuda:0"]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    with (output / "SUPERVISOR.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False)
    if result.returncode:
        raise RuntimeError(f"seed {seed} validation prediction exited {result.returncode}")
    return output


def evaluate_family(args: argparse.Namespace, variant: str, outputs: list[Path]) -> dict:
    first = args.stage1_validation / "validation" / variant
    paths = [first, *outputs]
    predictions = []
    task_ref = None
    control_ref = None
    input_hashes = []
    for seed, path in enumerate(paths, start=1):
        tasks_path = path / "E208_VALIDATION_TASKS.csv"
        control_path = path / "E208_VALIDATION_CONTROL_CENTROIDS.npy"
        prediction_path = path / "E208_VALIDATION_PREDICTION_CENTROIDS.npy"
        tasks = pd.read_csv(tasks_path)
        controls = np.load(control_path).astype(np.float64)
        predictions.append(np.load(prediction_path).astype(np.float64))
        if task_ref is None:
            task_ref, control_ref = tasks, controls
        elif not tasks.equals(task_ref) or not np.array_equal(controls, control_ref):
            raise RuntimeError(f"seed {seed} validation task/control alignment changed")
        input_hashes.append({"seed": seed, "tasks": sha256(tasks_path), "controls": sha256(control_path),
                             "prediction": sha256(prediction_path)})
    family = np.mean(predictions, axis=0)
    if not np.isfinite(family).all() or len(family) != 216:
        raise RuntimeError("non-finite or incomplete stage-2 family")
    family_output = args.output / "family_validation"
    family_output.mkdir(parents=True, exist_ok=True)
    np.save(family_output / "E208_VALIDATION_PREDICTION_CENTROIDS.npy", family)
    np.save(family_output / "E208_VALIDATION_CONTROL_CENTROIDS.npy", control_ref)
    task_ref.to_csv(family_output / "E208_VALIDATION_TASKS.csv", index=False)
    _, contexts, summary = evaluate_variant(family_output, args.truth)
    contexts.to_csv(args.output / "E233_FAMILY_VALIDATION_CONTEXTS.csv", index=False)
    summary["seed_prediction_hashes"] = input_hashes
    summary["family_prediction_sha256"] = sha256(family_output / "E208_VALIDATION_PREDICTION_CENTROIDS.npy")
    summary["four_seed_mean"] = True
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "stage1-status", "stage1-validation", "python", "job-script",
                 "prediction-script", "perturbench-repo", "data-dir", "validation-cache",
                 "truth", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("poll-seconds must be >= 5")
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "E233_STAGE2_SUPERVISOR_STATUS.json"
    if status_path.exists():
        raise RuntimeError("stage-2 supervisor already exists; refusing duplicate launch")
    state = {"status": "WAITING_FOR_STAGE1", "test_perturbed_expression_rows_read": 0,
             "training_exit_codes": {}, "seed_plan": [2, 3, 4]}
    atomic_json(status_path, state)
    try:
        while True:
            first = status(args.stage1_status)
            if first and first.get("status") in TERMINAL:
                break
            time.sleep(args.poll_seconds)
        state["stage1_status_sha256"] = sha256(args.stage1_status)
        if first["status"] != "STAGE1_PASS":
            state.update(status="BLOCKED_STAGE1", stage1_result=first["status"])
            return
        variant = first.get("selected_variant")
        if variant not in VARIANTS or first.get("summaries", {}).get(variant, {}).get("competence_gate") != "PASS":
            raise RuntimeError("stage-1 selection record is not a passing registered variant")
        state.update(status="TRAINING", selected_variant=variant)
        atomic_json(status_path, state)
        first_pair = [run_training(args, variant, 2, 0)[1], run_training(args, variant, 3, 1)[1]]
        if not wait_jobs(first_pair, status_path, state, args.poll_seconds):
            state["status"] = "BLOCKED_TRAINING_FAILED"
            return
        final = [run_training(args, variant, 4, 0)[1]]
        if not wait_jobs(final, status_path, state, args.poll_seconds):
            state["status"] = "BLOCKED_TRAINING_FAILED"
            return
        state["status"] = "VALIDATING_FAMILY"
        atomic_json(status_path, state)
        outputs = [predict_validation(args, variant, seed) for seed in (2, 3, 4)]
        summary = evaluate_family(args, variant, outputs)
        state.update(status="STAGE2_FAMILY_PASS" if summary["competence_gate"] == "PASS"
                     else "STAGE2_FAMILY_COMPETENCE_BLOCKED", family_validation=summary)
    except BaseException as error:
        state.update(status="FAILED", reason=repr(error))
        raise
    finally:
        atomic_json(status_path, state)


if __name__ == "__main__":
    main()
