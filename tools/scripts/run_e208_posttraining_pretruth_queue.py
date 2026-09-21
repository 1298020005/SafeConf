#!/usr/bin/env python3
"""Automatically run E208 validation gating and truth-isolated test prediction."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


JOBS = (("latent", 1), ("linear", 1), ("latent", 2), ("latent", 3), ("latent", 4))


class PostTrainingFailure(RuntimeError):
    """Training, competence, or prediction queue contract failed."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def output_dir(root: Path, stage: str, architecture: str, seed: int) -> Path:
    return root / stage / architecture / f"seed_{seed}"


def completed_prediction(root: Path, stage: str, architecture: str, seed: int) -> bool:
    name = (
        "E208_VALIDATION_PREDICTION_STATUS.json"
        if stage == "validation"
        else "E208_PREDICTION_STATUS.json"
    )
    path = output_dir(root, stage, architecture, seed) / name
    if not path.is_file():
        return False
    try:
        state = read_json(path)
    except (OSError, ValueError, TypeError):
        return False
    return (
        state.get("status") == "PASS"
        and state.get("architecture") == architecture
        and int(state.get("seed", -1)) == seed
        and int(state.get("n_tasks", -1)) == (216 if stage == "validation" else 224)
        and int(state.get("test_perturbed_expression_rows_read", -1)) == 0
    )


def checkpoints(training_root: Path) -> dict[tuple[str, int], Path] | None:
    queue_status = training_root / "E208_FORMAL_QUEUE_STATUS.json"
    if not queue_status.is_file():
        return None
    queue = read_json(queue_status)
    if queue.get("status") != "COMPLETE" or int(queue.get("completed", -1)) != 5:
        return None
    result = {}
    for architecture, seed in JOBS:
        state_path = training_root / architecture / f"seed_{seed}" / "E208_RUN_STATUS.json"
        if not state_path.is_file():
            raise PostTrainingFailure(f"completed queue lacks status: {state_path}")
        state = read_json(state_path)
        checkpoint = Path(str(state.get("best_checkpoint", "")))
        if (
            state.get("status") != "COMPLETE"
            or state.get("architecture") != architecture
            or int(state.get("seed", -1)) != seed
            or int(state.get("test_perturbed_expression_rows_read", -1)) != 0
            or not checkpoint.is_file()
            or checkpoint.stat().st_size <= 0
        ):
            raise PostTrainingFailure(f"invalid completed training job: {architecture}/{seed}")
        result[(architecture, seed)] = checkpoint
    return result


def launch_prediction(
    *,
    args: argparse.Namespace,
    stage: str,
    architecture: str,
    seed: int,
    checkpoint: Path,
    device: str,
    log_path: Path,
) -> subprocess.Popen:
    if stage == "validation":
        script = args.repo / "tools/scripts/run_e208_validation_prediction_job.py"
        command = [
            str(args.python), str(script),
            "--perturbench-repo", str(args.perturbench_repo),
            "--architecture", architecture,
            "--seed", str(seed),
            "--checkpoint", str(checkpoint),
            "--validation-cache-dir", str(args.validation_cache_dir),
            "--output-dir", str(output_dir(args.output_root, stage, architecture, seed)),
            "--device", "cuda:0",
        ]
    else:
        script = args.repo / "tools/scripts/run_e208_pretruth_prediction_job.py"
        command = [
            str(args.python), str(script),
            "--perturbench-repo", str(args.perturbench_repo),
            "--architecture", architecture,
            "--seed", str(seed),
            "--checkpoint", str(checkpoint),
            "--cache-dir", str(args.test_control_cache_dir),
            "--tasks", str(args.tasks),
            "--manifest-status", str(args.manifest_status),
            "--output-dir", str(output_dir(args.output_root, stage, architecture, seed)),
            "--device", "cuda:0",
        ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = device
    environment["HDF5_USE_FILE_LOCKING"] = "FALSE"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    try:
        log.write(f"\n===== {now()} START GPU {device}: {command} =====\n")
        log.flush()
        return subprocess.Popen(
            command,
            cwd=args.repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log.close()


def run_stage(
    args: argparse.Namespace,
    stage: str,
    registered_checkpoints: dict[tuple[str, int], Path],
    status: dict,
    status_path: Path,
) -> None:
    waiting = [job for job in JOBS if not completed_prediction(args.output_root, stage, *job)]
    active: dict[str, tuple[subprocess.Popen, tuple[str, int], Path]] = {}
    attempts = {job: 0 for job in JOBS}
    while waiting or active:
        for device, (process, job, log_path) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            del active[device]
            if returncode != 0 or not completed_prediction(args.output_root, stage, *job):
                if attempts[job] < 2:
                    waiting.insert(0, job)
                else:
                    raise PostTrainingFailure(
                        f"{stage} prediction failed twice for {job}; see {log_path}"
                    )
        for device in args.devices:
            if device in active or not waiting:
                continue
            architecture, seed = waiting.pop(0)
            attempts[(architecture, seed)] += 1
            log_path = (
                args.output_root
                / "_logs"
                / f"{stage}_{architecture}_seed{seed}_attempt{attempts[(architecture, seed)]}.log"
            )
            process = launch_prediction(
                args=args,
                stage=stage,
                architecture=architecture,
                seed=seed,
                checkpoint=registered_checkpoints[(architecture, seed)],
                device=device,
                log_path=log_path,
            )
            active[device] = (process, (architecture, seed), log_path)
        status.update(
            {
                "status": f"RUNNING_{stage.upper()}_PREDICTION",
                "updated_at": now(),
                f"{stage}_completed": sum(completed_prediction(args.output_root, stage, *job) for job in JOBS),
                f"{stage}_waiting": len(waiting),
                f"{stage}_active": [
                    {"device": device, "architecture": job[0], "seed": job[1], "pid": process.pid}
                    for device, (process, job, _) in active.items()
                ],
            }
        )
        atomic_json(status_path, status)
        if waiting or active:
            time.sleep(args.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--validation-cache-dir", type=Path, required=True)
    parser.add_argument("--test-control-cache-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--manifest-status", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cuda-devices", default="0,1")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    args.repo = args.repo.expanduser().absolute()
    args.perturbench_repo = args.perturbench_repo.expanduser().absolute()
    args.python = args.python.expanduser().absolute()
    args.training_root = args.training_root.expanduser().absolute()
    args.validation_cache_dir = args.validation_cache_dir.expanduser().absolute()
    args.test_control_cache_dir = args.test_control_cache_dir.expanduser().absolute()
    args.tasks = args.tasks.expanduser().absolute()
    args.manifest_status = args.manifest_status.expanduser().absolute()
    args.output_root = args.output_root.expanduser().absolute()
    args.devices = [value.strip() for value in args.cuda_devices.split(",") if value.strip()]
    if not args.devices or not 10 <= args.poll_seconds <= 600:
        raise PostTrainingFailure("invalid CUDA device list or poll interval")
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "E208_POSTTRAINING_PRETRUTH_QUEUE_STATUS.json"
    status = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_POSTTRAINING_PRETRUTH_QUEUE",
        "status": "WAITING_FOR_TRAINING_AND_VALIDATION_CACHE",
        "started_at": now(),
        "pid": os.getpid(),
        "jobs": [{"architecture": a, "seed": s} for a, s in JOBS],
        "cuda_devices": args.devices,
        "test_perturbed_expression_rows_read": 0,
        "target_truth_access": "NOT_AUTHORIZED",
    }
    atomic_json(status_path, status)
    while True:
        registered_checkpoints = checkpoints(args.training_root)
        cache_status_path = args.validation_cache_dir / "E208_VALIDATION_CACHE_STATUS.json"
        cache_ready = False
        if cache_status_path.is_file():
            try:
                cache_state = read_json(cache_status_path)
                cache_ready = (
                    cache_state.get("status") == "PASS"
                    and cache_state.get("n_tasks") == 216
                    and cache_state.get("test_perturbed_expression_rows_read") == 0
                )
            except (OSError, ValueError, TypeError):
                cache_ready = False
        status.update(
            {
                "updated_at": now(),
                "training_ready": registered_checkpoints is not None,
                "validation_cache_ready": cache_ready,
            }
        )
        atomic_json(status_path, status)
        if registered_checkpoints is not None and cache_ready:
            break
        time.sleep(args.poll_seconds)

    run_stage(args, "validation", registered_checkpoints, status, status_path)
    competence_dir = args.output_root / "validation_competence"
    command = [
        str(args.python),
        str(args.repo / "tools/scripts/evaluate_e208_validation_competence.py"),
        "--validation-cache-dir", str(args.validation_cache_dir),
    ]
    for seed in (1, 2, 3, 4):
        command.extend(
            ["--latent-dir", str(output_dir(args.output_root, "validation", "latent", seed))]
        )
    command.extend(
        [
            "--linear-dir", str(output_dir(args.output_root, "validation", "linear", 1)),
            "--output-dir", str(competence_dir),
        ]
    )
    completed = subprocess.run(command, cwd=args.repo, check=False)
    if completed.returncode:
        raise PostTrainingFailure("validation competence evaluation failed")
    competence = read_json(competence_dir / "E208_VALIDATION_COMPETENCE_STATUS.json")
    status["competence_status"] = competence.get("status")
    if competence.get("status") != "PASS":
        status.update(
            {
                "status": "FAMILY_COMPETENCE_BLOCKED",
                "finished_at": now(),
                "test_prediction_started": False,
            }
        )
        atomic_json(status_path, status)
        return

    run_stage(args, "test_pretruth", registered_checkpoints, status, status_path)
    status.update(
        {
            "status": "PRETRUTH_PREDICTIONS_COMPLETE_AWAITING_RISK_SEAL",
            "finished_at": now(),
            "validation_completed": 5,
            "test_pretruth_completed": 5,
            "test_perturbed_expression_rows_read": 0,
            "target_truth_access": "NOT_AUTHORIZED",
        }
    )
    atomic_json(status_path, status)


if __name__ == "__main__":
    main()
