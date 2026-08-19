#!/usr/bin/env python3
"""Run the frozen E201 blind-training jobs sequentially on one GPU."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
SEEDS = (1, 2, 3, 4)
JOBS = tuple((target, seed) for seed in SEEDS for target in TARGETS)


class QueueFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safeconf-repo", type=Path, required=True)
    parser.add_argument("--txpert-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--cuda-device", default="1")
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_runtime(python: Path, txpert_repo: Path, environment: dict) -> dict:
    """Check the exact training interpreter before a formal run directory exists."""
    code = """
import json
import sys

import hydra
import lightning
import omegaconf
import pandas
import torch
import gspp

print(json.dumps({
    "python_executable": sys.executable,
    "python_prefix": sys.prefix,
    "python_base_prefix": sys.base_prefix,
    "python_version": sys.version.split()[0],
    "hydra_version": hydra.__version__,
    "lightning_version": lightning.__version__,
    "omegaconf_version": omegaconf.__version__,
    "pandas_version": pandas.__version__,
    "torch_version": torch.__version__,
}, sort_keys=True))
"""
    result = subprocess.run(
        [str(python), "-c", code],
        cwd=txpert_repo,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise QueueFailure(
            f"training runtime preflight failed with exit {result.returncode}: {detail}"
        )
    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise QueueFailure(
            f"training runtime preflight returned invalid JSON: {result.stdout!r}"
        ) from exc
    if Path(payload.get("python_executable", "")).absolute() != python:
        raise QueueFailure(
            "training runtime preflight used an unexpected Python executable"
        )
    expected_prefix = python.parent.parent
    if Path(payload.get("python_prefix", "")).resolve() != expected_prefix.resolve():
        raise QueueFailure(
            "training runtime preflight escaped the requested virtual environment"
        )
    return payload


def matching_training_pids(adapter: Path, run_dir: Path) -> list[int]:
    adapter_text = str(adapter.resolve())
    run_text = str(run_dir.resolve())
    matches = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            UnicodeDecodeError,
        ):
            continue
        if adapter_text in command and run_text in command:
            matches.append(int(proc_dir.name))
    return sorted(matches)


def gpu_compute_pids(device: str) -> list[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            device,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    pids = []
    for line in output.splitlines():
        value = line.strip()
        if value and value != "[N/A]":
            pids.append(int(value))
    return sorted(set(pids))


def validate_complete(status: dict, target: str, seed: int) -> None:
    required = {
        "status": status.get("status") == "COMPLETE",
        "kind": status.get("kind") == "formal",
        "target": status.get("target") == target,
        "seed": int(status.get("seed", -1)) == seed,
        "epochs": int(status.get("current_epoch", -1)) == 80,
        "target_access": int(status.get("target_perturbed_cells_accessed", -1)) == 0,
        "test_not_constructed": status.get("target_test_dataset_constructed") is False,
    }
    failed = sorted(name for name, passed in required.items() if not passed)
    if failed:
        raise QueueFailure(
            f"completed status failed gates for {target}/seed_{seed}: {failed}"
        )


def wait_for_existing_job(
    status_path: Path,
    adapter: Path,
    run_dir: Path,
    target: str,
    seed: int,
    poll_seconds: int,
) -> bool:
    if not status_path.is_file():
        if run_dir.exists():
            raise QueueFailure(f"run directory has no status: {run_dir}")
        return False
    while True:
        status = read_json(status_path)
        state = status.get("status")
        if state == "COMPLETE":
            validate_complete(status, target, seed)
            print(f"{now()} SKIP complete {target}/seed_{seed}", flush=True)
            return True
        if state == "FAILED":
            raise QueueFailure(f"existing job failed: {target}/seed_{seed}")
        if state != "RUNNING":
            raise QueueFailure(f"unexpected job state {state}: {target}/seed_{seed}")
        pids = matching_training_pids(adapter, run_dir)
        if not pids:
            time.sleep(5)
            status = read_json(status_path)
            if status.get("status") == "COMPLETE":
                validate_complete(status, target, seed)
                return True
            raise QueueFailure(
                f"RUNNING status has no live training process: {target}/seed_{seed}"
            )
        print(
            f"{now()} WAIT existing {target}/seed_{seed} pid={','.join(map(str, pids))}",
            flush=True,
        )
        time.sleep(poll_seconds)


def main() -> None:
    args = parse_args()
    if not 10 <= args.poll_seconds <= 300:
        raise QueueFailure("poll-seconds must be between 10 and 300")
    safeconf_repo = args.safeconf_repo.resolve()
    txpert_repo = args.txpert_repo.resolve()
    # Do not resolve this symlink: a virtual environment depends on invoking its
    # own bin/python path rather than the base interpreter it points to.
    python = args.python.expanduser().absolute()
    runs_root = args.runs_root.resolve()
    adapter = safeconf_repo / "tools/scripts/txpert_blind_training_adapter.py"
    for path in (safeconf_repo, txpert_repo, python, adapter):
        if not path.exists():
            raise QueueFailure(f"missing required path: {path}")

    runs_root.mkdir(parents=True, exist_ok=True)
    log_root = runs_root / "_queue_logs"
    log_root.mkdir(exist_ok=True)
    lock_path = runs_root / "E201_QUEUE.lock"
    state_path = runs_root / "E201_QUEUE_STATUS.json"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise QueueFailure(
            "another E201 training queue already holds the lock"
        ) from exc

    queue_status = {
        "experiment": "E201_txpert_multitarget_retraining",
        "status": "RUNNING",
        "started_at": now(),
        "pid": os.getpid(),
        "cuda_device": args.cuda_device,
        "jobs": [{"target": target, "seed": seed} for target, seed in JOBS],
        "target_truth_release": "NOT_AUTHORIZED",
        "command": sys.argv,
    }
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    queue_status["runtime_preflight"] = validate_runtime(
        python, txpert_repo, environment
    )
    write_json(state_path, queue_status)
    try:
        for ordinal, (target, seed) in enumerate(JOBS, start=1):
            run_dir = runs_root / target / f"seed_{seed}"
            status_path = run_dir / "E201_RUN_STATUS.json"
            queue_status.update(
                {
                    "current_job": ordinal,
                    "current_target": target,
                    "current_seed": seed,
                    "updated_at": now(),
                }
            )
            write_json(state_path, queue_status)
            if wait_for_existing_job(
                status_path,
                adapter,
                run_dir,
                target,
                seed,
                args.poll_seconds,
            ):
                continue
            foreign_pids = gpu_compute_pids(args.cuda_device)
            if foreign_pids:
                raise QueueFailure(
                    f"GPU {args.cuda_device} is occupied before launch: {foreign_pids}"
                )
            queue_status["runtime_preflight"] = validate_runtime(
                python, txpert_repo, environment
            )
            queue_status["runtime_preflight_at"] = now()
            write_json(state_path, queue_status)
            log_path = log_root / f"{ordinal:02d}_{target}_seed_{seed}.log"
            command = [
                str(python),
                str(adapter),
                "--txpert-repo",
                str(txpert_repo),
                "--task-type",
                f"E201_blind_{target}",
                "--target",
                target,
                "--seed",
                str(seed),
                "--run-dir",
                str(run_dir),
                "--kind",
                "formal",
                "--batch-size",
                "64",
            ]
            print(f"{now()} START {target}/seed_{seed}", flush=True)
            with log_path.open("x", encoding="utf-8") as log_handle:
                result = subprocess.run(
                    command,
                    cwd=txpert_repo,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode != 0:
                raise QueueFailure(
                    f"training exited {result.returncode}: {target}/seed_{seed}"
                )
            validate_complete(read_json(status_path), target, seed)
            print(f"{now()} COMPLETE {target}/seed_{seed}", flush=True)
        queue_status.update(
            {
                "status": "COMPLETE_BLIND_TRAINING_ONLY",
                "finished_at": now(),
                "current_job": len(JOBS),
                "target_truth_release": "NOT_PERFORMED",
            }
        )
        write_json(state_path, queue_status)
    except Exception as exc:
        queue_status.update(
            {
                "status": "FAILED_STOPPED",
                "finished_at": now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "target_truth_release": "NOT_PERFORMED",
            }
        )
        write_json(state_path, queue_status)
        raise


if __name__ == "__main__":
    main()
