#!/usr/bin/env python3
"""Wait for the E208 smoke gate, then run the five frozen formal jobs.

The queue uses one process per available GPU, is checkpoint-restartable, and
refuses to share a GPU with another substantial compute process.  It only
trains; prediction and test-truth evaluation remain separately authorized.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence


JOBS = (
    ("latent", 1),
    ("linear", 1),
    ("latent", 2),
    ("latent", 3),
    ("latent", 4),
)
REGISTERED_BATCH_SIZES = {2000, 1000, 500, 250}


class QueueFailure(RuntimeError):
    """The smoke gate, resource contract, or formal job contract failed."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--smoke-supervisor-status", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--cuda-devices", default="0,1")
    parser.add_argument("--min-free-mb", type=int, default=22_000)
    parser.add_argument("--foreign-proc-mb", type=int, default=1024)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args(argv)


def smoke_gate(path: Path) -> tuple[str, str, int | None]:
    """Return WAIT/GO/FAIL, reason, and the smoke-selected batch size."""
    if not path.is_file():
        return "WAIT", "smoke supervisor status is missing", None
    try:
        state = read_json(path)
    except (OSError, ValueError, TypeError) as exc:
        return "WAIT", f"smoke status is being updated: {exc}", None
    status = str(state.get("status", ""))
    if status == "SMOKE_PASS_FORMAL_QUEUE_NOT_STARTED":
        batch_size = int(state.get("selected_batch_size", -1))
        if batch_size not in REGISTERED_BATCH_SIZES:
            return "FAIL", f"unregistered selected batch size: {batch_size}", None
        if state.get("test_truth_access") != "NOT_AUTHORIZED":
            return "FAIL", "smoke truth-isolation state changed", None
        return "GO", f"smoke passed at batch size {batch_size}", batch_size
    if status.startswith("BLOCKED"):
        return "FAIL", f"smoke supervisor blocked: {status}", None
    if status in {"", "WAITING_FOR_E205", "RUNNING_SMOKE"}:
        return "WAIT", f"smoke supervisor state: {status or 'not initialized'}", None
    return "FAIL", f"unexpected smoke supervisor state: {status}", None


def run_dir_for(root: Path, architecture: str, seed: int) -> Path:
    return root / architecture / f"seed_{seed}"


def validate_complete(run_dir: Path, architecture: str, seed: int, batch_size: int) -> dict:
    path = run_dir / "E208_RUN_STATUS.json"
    if not path.is_file():
        raise QueueFailure(f"missing run status: {path}")
    status = read_json(path)
    last = Path(str(status.get("last_checkpoint", "")))
    best = Path(str(status.get("best_checkpoint", "")))
    checks = {
        "complete": status.get("status") == "COMPLETE",
        "experiment": status.get("experiment")
        == "E208_jiang24_external_confirmation",
        "stage": status.get("stage") == "D1_FORMAL_TRAINING",
        "architecture": status.get("architecture") == architecture,
        "seed": int(status.get("seed", -1)) == seed,
        "batch_size": int(status.get("batch_size", -1)) == batch_size,
        "truth_not_authorized": status.get("test_truth_access") == "NOT_AUTHORIZED",
        "zero_truth_rows": int(status.get("test_perturbed_expression_rows_read", -1))
        == 0,
        "last_checkpoint": last.is_file() and last.stat().st_size > 0,
        "best_checkpoint": best.is_file() and best.stat().st_size > 0,
        "config_hash": len(str(status.get("resolved_config_sha256", ""))) == 64,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise QueueFailure(
            f"formal gates failed for {architecture}/seed_{seed}: {failed}"
        )
    return status


def job_state(run_dir: Path) -> str:
    path = run_dir / "E208_RUN_STATUS.json"
    if not path.is_file():
        return "orphan" if run_dir.exists() else "pending"
    try:
        status = read_json(path).get("status")
    except (OSError, ValueError, TypeError):
        return "failed"
    if status == "COMPLETE":
        return "complete"
    if status == "RUNNING":
        return "running"
    return "failed"


def count_valid_completed(root: Path, batch_size: int) -> int:
    completed = 0
    for architecture, seed in JOBS:
        try:
            validate_complete(
                run_dir_for(root, architecture, seed),
                architecture,
                seed,
                batch_size,
            )
        except (QueueFailure, OSError, ValueError, TypeError):
            continue
        completed += 1
    return completed


def resume_checkpoint_for(run_dir: Path, architecture: str, seed: int) -> Path | None:
    status_path = run_dir / "E208_RUN_STATUS.json"
    checkpoint = run_dir / "checkpoints/last.ckpt"
    if not status_path.is_file() or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        return None
    try:
        status = read_json(status_path)
    except (OSError, ValueError, TypeError):
        return None
    if (
        status.get("status") not in {"RUNNING", "FAILED"}
        or status.get("architecture") != architecture
        or int(status.get("seed", -1)) != seed
    ):
        return None
    return checkpoint.resolve()


def attempts_from_logs(log_root: Path, architecture: str, seed: int) -> int:
    pattern = re.compile(
        rf"^{re.escape(architecture)}_seed{seed}_attempt(\d+)\.log$"
    )
    attempts = []
    for path in log_root.glob(f"{architecture}_seed{seed}_attempt*.log"):
        match = pattern.match(path.name)
        if match:
            attempts.append(int(match.group(1)))
    return max(attempts, default=0)


def archive_nonresumable_run(
    run_dir: Path, root: Path, architecture: str, seed: int, attempt: int
) -> Path:
    archive = root / "_failed_attempts"
    archive.mkdir(exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    destination = archive / f"{architecture}_seed{seed}_attempt{attempt}_{stamp}"
    os.replace(run_dir, destination)
    return destination


def nvidia_free_mb(device: str) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            device,
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def foreign_gpu_pids(device: str, own_pids: set[int], min_mb: int) -> list[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            device,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    foreign = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        pid, used_mb = int(fields[0]), int(fields[1])
        if pid not in own_pids and used_mb >= min_mb:
            foreign.append(pid)
    return sorted(set(foreign))


def job_pids(run_dir: Path, runner: Path) -> set[int]:
    runner_text = str(runner.resolve())
    run_text = str(run_dir.resolve())
    matches = set()
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if runner_text in command and run_text in command:
            matches.add(int(proc_dir.name))
    return matches


def start_job(
    *,
    runner: Path,
    python: Path,
    perturbench_repo: Path,
    data_dir: Path,
    run_dir: Path,
    architecture: str,
    seed: int,
    batch_size: int,
    device: str,
    log_path: Path,
) -> subprocess.Popen:
    resume = resume_checkpoint_for(run_dir, architecture, seed)
    if run_dir.exists() and resume is None and job_state(run_dir) != "pending":
        raise QueueFailure(f"refusing non-resumable existing run: {run_dir}")
    command = [
        str(python),
        str(runner),
        "--perturbench-repo",
        str(perturbench_repo),
        "--python",
        str(python),
        "--data-dir",
        str(data_dir),
        "--run-dir",
        str(run_dir),
        "--architecture",
        architecture,
        "--seed",
        str(seed),
        "--batch-size",
        str(batch_size),
    ]
    if resume is not None:
        command.extend(["--resume-checkpoint", str(resume)])
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = device
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    try:
        log_handle.write(f"\n===== {now()} START GPU {device}: {command} =====\n")
        log_handle.flush()
        return subprocess.Popen(
            command,
            cwd=perturbench_repo,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()


def run(args: argparse.Namespace) -> dict:
    if not 10 <= args.poll_seconds <= 1800:
        raise QueueFailure("poll-seconds must be between 10 and 1800")
    if not 1 <= args.max_attempts <= 3:
        raise QueueFailure("max-attempts must be between 1 and 3")
    devices = [item.strip() for item in args.cuda_devices.split(",") if item.strip()]
    if not devices:
        raise QueueFailure("no CUDA devices configured")
    repo = args.repo.resolve()
    perturbench_repo = args.perturbench_repo.resolve()
    # Preserve the venv entry-point symlink instead of resolving to its base
    # interpreter.
    python = args.python.expanduser().absolute()
    data_dir = args.data_dir.resolve()
    root = args.runs_root.resolve()
    runner = repo / "tools/scripts/run_e208_formal_training_job.py"
    for path in (repo, perturbench_repo, python, data_dir, runner):
        if not path.exists():
            raise QueueFailure(f"missing required path: {path}")

    root.mkdir(parents=True, exist_ok=True)
    log_root = root / "_queue_logs"
    log_root.mkdir(exist_ok=True)
    status_path = root / "E208_FORMAL_QUEUE_STATUS.json"
    lock_handle = (root / "E208_FORMAL_QUEUE.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise QueueFailure("another E208 formal queue holds the lock") from exc

    status: dict[str, object] = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_FORMAL_TRAINING_4_LATENT_PLUS_1_LINEAR",
        "status": "WAITING_FOR_SMOKE",
        "started_at": now(),
        "pid": os.getpid(),
        "jobs": [
            {"architecture": architecture, "seed": seed}
            for architecture, seed in JOBS
        ],
        "cuda_devices": devices,
        "test_truth_access": "NOT_AUTHORIZED",
        "test_perturbed_expression_rows_read": 0,
    }
    atomic_json(status_path, status)
    while True:
        gate, reason, batch_size = smoke_gate(args.smoke_supervisor_status.resolve())
        status.update({"updated_at": now(), "smoke_gate_reason": reason})
        atomic_json(status_path, status)
        if gate == "GO":
            break
        if gate == "FAIL":
            status.update({"status": "BLOCKED_SMOKE", "finished_at": now()})
            atomic_json(status_path, status)
            raise QueueFailure(reason)
        time.sleep(args.poll_seconds)
    assert batch_size is not None
    status.update({"status": "RUNNING", "selected_batch_size": batch_size})
    atomic_json(status_path, status)

    attempts = {job: attempts_from_logs(log_root, *job) for job in JOBS}
    active: dict[str, dict] = {}
    failures: list[dict] = []
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        while True:
            for device, entry in list(active.items()):
                process = entry["process"]
                if process.poll() is None:
                    continue
                architecture, seed = entry["job"]
                run_dir = run_dir_for(root, architecture, seed)
                try:
                    validate_complete(run_dir, architecture, seed, batch_size)
                except (QueueFailure, OSError, ValueError, TypeError):
                    if attempts[(architecture, seed)] >= args.max_attempts:
                        failures.append(
                            {"architecture": architecture, "seed": seed}
                        )
                del active[device]
            if stop_requested:
                break

            remaining = []
            detached = []
            for architecture, seed in JOBS:
                run_dir = run_dir_for(root, architecture, seed)
                try:
                    validate_complete(run_dir, architecture, seed, batch_size)
                    continue
                except (QueueFailure, OSError, ValueError, TypeError):
                    pass
                if any(
                    entry["job"] == (architecture, seed)
                    for entry in active.values()
                ):
                    continue
                live = job_pids(run_dir, runner)
                if live:
                    detached.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "pids": sorted(live),
                        }
                    )
                    continue
                record = {"architecture": architecture, "seed": seed}
                if record in failures:
                    continue
                if attempts[(architecture, seed)] >= args.max_attempts:
                    failures.append(record)
                    continue
                state = job_state(run_dir)
                if state in {"failed", "running", "orphan"} and resume_checkpoint_for(
                    run_dir, architecture, seed
                ) is None:
                    if attempts[(architecture, seed)] >= args.max_attempts:
                        failures.append(
                            {**record, "reason": "existing run is not checkpoint-resumable"}
                        )
                        continue
                    archive_nonresumable_run(
                        run_dir,
                        root,
                        architecture,
                        seed,
                        max(attempts[(architecture, seed)], 1),
                    )
                remaining.append((architecture, seed))

            if not remaining and not active and not detached:
                break
            own_pids = {
                entry["process"].pid
                for entry in active.values()
                if entry["process"].poll() is None
            }
            for device in devices:
                if device in active or not remaining:
                    continue
                if nvidia_free_mb(device) < args.min_free_mb:
                    continue
                if foreign_gpu_pids(device, own_pids, args.foreign_proc_mb):
                    continue
                architecture, seed = remaining.pop(0)
                attempts[(architecture, seed)] += 1
                attempt = attempts[(architecture, seed)]
                run_dir = run_dir_for(root, architecture, seed)
                log_path = (
                    log_root
                    / f"{architecture}_seed{seed}_attempt{attempt}.log"
                )
                process = start_job(
                    runner=runner,
                    python=python,
                    perturbench_repo=perturbench_repo,
                    data_dir=data_dir,
                    run_dir=run_dir,
                    architecture=architecture,
                    seed=seed,
                    batch_size=batch_size,
                    device=device,
                    log_path=log_path,
                )
                active[device] = {
                    "job": (architecture, seed),
                    "process": process,
                    "log_path": str(log_path),
                }
                own_pids.add(process.pid)

            status.update(
                {
                    "updated_at": now(),
                    "completed": count_valid_completed(root, batch_size),
                    "waiting": len(remaining),
                    "active": [
                        {
                            "device": device,
                            "architecture": entry["job"][0],
                            "seed": entry["job"][1],
                            "pid": entry["process"].pid,
                            "log_path": entry["log_path"],
                        }
                        for device, entry in active.items()
                    ],
                    "detached": detached,
                    "attempts": {
                        f"{architecture}/seed_{seed}": value
                        for (architecture, seed), value in attempts.items()
                        if value
                    },
                    "permanent_failures": failures,
                }
            )
            atomic_json(status_path, status)
            time.sleep(args.poll_seconds)
    finally:
        completed = count_valid_completed(root, batch_size)
        if stop_requested:
            final_status = "INTERRUPTED"
        elif failures or completed != len(JOBS):
            final_status = "FAILED"
        else:
            final_status = "COMPLETE"
        status.update(
            {
                "status": final_status,
                "finished_at": now(),
                "completed": completed,
                "waiting": max(len(JOBS) - completed - len(active), 0),
                "active": [
                    {
                        "device": device,
                        "architecture": entry["job"][0],
                        "seed": entry["job"][1],
                        "pid": entry["process"].pid,
                    }
                    for device, entry in active.items()
                    if entry["process"].poll() is None
                ],
                "permanent_failures": failures,
                "test_truth_access": "NOT_AUTHORIZED",
                "test_perturbed_expression_rows_read": 0,
            }
        )
        atomic_json(status_path, status)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
