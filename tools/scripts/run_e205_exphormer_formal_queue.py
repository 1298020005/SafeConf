#!/usr/bin/env python3
"""Run 16 E205 Exphormer whole-context training jobs on shared GPUs.

The queue refuses to start until the one-epoch profile passes every contract
gate. It is restart-safe, preserves failed attempts and never preempts foreign
GPU processes.
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
    parser.add_argument("--profile-run-dir", type=Path, required=True)
    parser.add_argument("--cuda-devices", default="0,1")
    parser.add_argument("--min-free-mb", type=int, default=20480)
    parser.add_argument("--foreign-proc-mb", type=int, default=1024)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_dir_for(root: Path, target: str, seed: int) -> Path:
    return root / target / f"seed_{seed}"


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
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        pid, used_mb = int(parts[0]), int(parts[1])
        if pid not in own_pids and used_mb >= min_mb:
            foreign.append(pid)
    return sorted(set(foreign))


def training_pids_for(run_dir: Path, adapter: Path) -> set[int]:
    adapter_text = str(adapter.resolve())
    run_text = str(run_dir.resolve())
    matches = set()
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if adapter_text in command and run_text in command:
            matches.add(int(proc_dir.name))
    return matches


def validate_profile(profile_run_dir: Path) -> dict:
    path = profile_run_dir / "E205_RUN_STATUS.json"
    if not path.is_file():
        raise QueueFailure(f"profile status missing: {path}")
    status = read_json(path)
    checkpoint = Path(str(status.get("last_model_path", "")))
    required = {
        "complete": status.get("status") == "COMPLETE",
        "profile": status.get("kind") == "profile",
        "target": status.get("target") == "RPE1",
        "seed": int(status.get("seed", -1)) == 1,
        "batch_size": int(status.get("batch_size", -1)) == 64,
        "family": status.get("model_family") == "TxPert-Exphormer",
        "architecture_only": status.get("architecture_change_only") is True,
        "whole_context_base": status.get("base_config") == "config-x-cell-gat",
        "exphormer_config": status.get("architecture_config") == "config-exphormer",
        "epoch": int(status.get("current_epoch", -1)) == 1,
        "zero_target_access": int(status.get("target_perturbed_cells_accessed", -1)) == 0,
        "test_not_constructed": status.get("target_test_dataset_constructed") is False,
        "checkpoint": checkpoint.is_file() and checkpoint.stat().st_size > 0,
        "memory_recorded": int(status.get("cuda_peak_memory_reserved_bytes", 0)) > 0,
        "time_recorded": float(status.get("fit_wall_seconds", 0.0)) > 0,
    }
    failed = sorted(name for name, passed in required.items() if not passed)
    if failed:
        raise QueueFailure(f"profile failed expansion gates: {failed}")
    return status


def validate_complete(run_dir: Path, target: str, seed: int) -> dict:
    path = run_dir / "E205_RUN_STATUS.json"
    if not path.is_file():
        raise QueueFailure(f"run status missing: {path}")
    status = read_json(path)
    checkpoint = Path(str(status.get("last_model_path", "")))
    model_config = status.get("resolved_model_config", {})
    pert_model = model_config.get("pert_model", {})
    required = {
        "complete": status.get("status") == "COMPLETE",
        "formal": status.get("kind") == "formal",
        "target": status.get("target") == target,
        "seed": int(status.get("seed", -1)) == seed,
        "family": status.get("model_family") == "TxPert-Exphormer",
        "architecture_only": status.get("architecture_change_only") is True,
        "exphormer": pert_model.get("model_type") == "exphormer",
        "whole_context_base": status.get("base_config") == "config-x-cell-gat",
        "epoch": int(status.get("current_epoch", -1)) == 80,
        "zero_target_access": int(status.get("target_perturbed_cells_accessed", -1)) == 0,
        "test_not_constructed": status.get("target_test_dataset_constructed") is False,
        "checkpoint": checkpoint.is_file() and checkpoint.stat().st_size > 0,
    }
    failed = sorted(name for name, passed in required.items() if not passed)
    if failed:
        raise QueueFailure(f"formal gates failed for {target}/seed_{seed}: {failed}")
    return status


def job_state(run_dir: Path) -> str:
    path = run_dir / "E205_RUN_STATUS.json"
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


def resume_checkpoint_for(run_dir: Path, target: str, seed: int) -> Path | None:
    status_path = run_dir / "E205_RUN_STATUS.json"
    checkpoint = run_dir / "checkpoints/last.ckpt"
    if not status_path.is_file() or not checkpoint.is_file():
        return None
    try:
        status = read_json(status_path)
    except (OSError, ValueError, TypeError):
        return None
    if (
        status.get("status") not in {"RUNNING", "FAILED"}
        or status.get("kind") != "formal"
        or status.get("target") != target
        or int(status.get("seed", -1)) != seed
        or status.get("model_family") != "TxPert-Exphormer"
        or checkpoint.stat().st_size <= 0
    ):
        return None
    return checkpoint.resolve()


def attempts_from_logs(log_root: Path, target: str, seed: int) -> int:
    pattern = re.compile(rf"^{re.escape(target)}_seed{seed}_attempt(\d+)\.log$")
    attempts = []
    for path in log_root.glob(f"{target}_seed{seed}_attempt*.log"):
        match = pattern.match(path.name)
        if match:
            attempts.append(int(match.group(1)))
    return max(attempts, default=0)


def archive_run(run_dir: Path, root: Path, target: str, seed: int, attempt: int) -> Path:
    archive = root / "_failed_attempts"
    archive.mkdir(exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    destination = archive / f"{target}_seed{seed}_attempt{attempt}_{stamp}"
    os.replace(run_dir, destination)
    return destination


def start_job(
    adapter: Path,
    python: Path,
    txpert_repo: Path,
    run_dir: Path,
    target: str,
    seed: int,
    device: str,
    batch_size: int,
    log_path: Path,
) -> subprocess.Popen:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = resume_checkpoint_for(run_dir, target, seed)
    if run_dir.exists() and resume_checkpoint is None:
        raise QueueFailure(f"refusing existing run directory: {run_dir}")
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
        str(batch_size),
    ]
    if resume_checkpoint is not None:
        command.extend(["--resume-checkpoint", str(resume_checkpoint)])
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = device
    log_handle = log_path.open("a", encoding="utf-8")
    try:
        log_handle.write(f"\n===== {now()} START {command} GPU {device} =====\n")
        log_handle.flush()
        return subprocess.Popen(
            command,
            cwd=txpert_repo,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()


def main() -> None:
    args = parse_args()
    if not 30 <= args.poll_seconds <= 1800:
        raise QueueFailure("poll-seconds must be between 30 and 1800")
    if not 1 <= args.max_attempts <= 3:
        raise QueueFailure("max-attempts must be between 1 and 3")
    devices = [item.strip() for item in args.cuda_devices.split(",") if item.strip()]
    if not devices:
        raise QueueFailure("no CUDA devices configured")

    safeconf_repo = args.safeconf_repo.resolve()
    txpert_repo = args.txpert_repo.resolve()
    python = args.python.expanduser().absolute()
    root = args.runs_root.resolve()
    profile_run_dir = args.profile_run_dir.resolve()
    adapter = safeconf_repo / "tools/scripts/txpert_blind_training_exphormer_adapter.py"
    for path in (safeconf_repo, txpert_repo, python, adapter):
        if not path.exists():
            raise QueueFailure(f"missing required path: {path}")
    profile_status_path = profile_run_dir / "E205_RUN_STATUS.json"
    while not profile_status_path.is_file():
        time.sleep(args.poll_seconds)
    while True:
        profile_state = read_json(profile_status_path).get("status")
        if profile_state == "COMPLETE":
            profile = validate_profile(profile_run_dir)
            break
        if profile_state == "FAILED":
            raise QueueFailure("E205 profile failed; formal expansion is blocked")
        time.sleep(args.poll_seconds)

    root.mkdir(parents=True, exist_ok=True)
    log_root = root / "_queue_logs"
    log_root.mkdir(exist_ok=True)
    state_path = root / "E205_FORMAL_QUEUE_STATUS.json"
    supervisor_log_path = root / "E205_FORMAL_QUEUE.log"
    lock_handle = (root / "E205_FORMAL_QUEUE.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise QueueFailure("another E205 formal queue holds the lock") from exc

    supervisor_log = supervisor_log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{now()} {message}"
        print(line, flush=True)
        supervisor_log.write(line + "\n")
        supervisor_log.flush()

    stop_requested = False

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        log(f"received signal {signum}; preserve active children")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    attempts = {job: attempts_from_logs(log_root, *job) for job in JOBS}
    active: dict[str, dict] = {}
    permanent_failures: list[dict] = []
    archives: list[dict] = []
    queue_status = {
        "experiment": "E205_cross_family_exphormer",
        "stage": "formal_training_4_targets_x_4_seeds",
        "status": "RUNNING",
        "started_at": now(),
        "pid": os.getpid(),
        "cuda_devices": devices,
        "min_free_mb": args.min_free_mb,
        "foreign_proc_mb": args.foreign_proc_mb,
        "profile_gate": {
            "path": str(profile_run_dir / "E205_RUN_STATUS.json"),
            "status": "PASS",
            "fit_wall_seconds": profile.get("fit_wall_seconds"),
            "cuda_peak_memory_reserved_bytes": profile.get(
                "cuda_peak_memory_reserved_bytes"
            ),
        },
        "jobs": [{"target": target, "seed": seed} for target, seed in JOBS],
        "target_truth_access": "NOT_AUTHORIZED",
    }
    write_json(state_path, queue_status)
    log(f"formal queue opened after profile PASS: {len(JOBS)} jobs")

    try:
        while True:
            for device, entry in list(active.items()):
                process = entry["process"]
                if process.poll() is None:
                    continue
                target, seed = entry["job"]
                run_dir = run_dir_for(root, target, seed)
                try:
                    validate_complete(run_dir, target, seed)
                except QueueFailure as exc:
                    log(f"FAILED {target}/seed_{seed} exit={process.returncode}: {exc}")
                    if attempts[(target, seed)] >= args.max_attempts:
                        permanent_failures.append({"target": target, "seed": seed})
                else:
                    log(f"COMPLETE {target}/seed_{seed} GPU {device}")
                del active[device]

            if stop_requested:
                break

            remaining = []
            detached = []
            for target, seed in JOBS:
                run_dir = run_dir_for(root, target, seed)
                state = job_state(run_dir)
                if state == "complete":
                    try:
                        validate_complete(run_dir, target, seed)
                    except QueueFailure:
                        state = "failed"
                    else:
                        continue
                if any(entry["job"] == (target, seed) for entry in active.values()):
                    continue
                live = training_pids_for(run_dir, adapter)
                if live:
                    detached.append(
                        {"target": target, "seed": seed, "pids": sorted(live)}
                    )
                    continue
                record = {"target": target, "seed": seed}
                if record in permanent_failures:
                    continue
                used = attempts[(target, seed)]
                resume_checkpoint = resume_checkpoint_for(run_dir, target, seed)
                if state in {"failed", "running", "orphan"} and resume_checkpoint is None:
                    archived = archive_run(run_dir, root, target, seed, max(used, 1))
                    archives.append({**record, "path": str(archived)})
                    log(f"ARCHIVE {target}/seed_{seed} to {archived}")
                elif resume_checkpoint is not None:
                    log(
                        f"RESUME-READY {target}/seed_{seed} from {resume_checkpoint}"
                    )
                if used >= args.max_attempts:
                    permanent_failures.append(record)
                    continue
                remaining.append((target, seed))

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
                free_mb = nvidia_free_mb(device)
                foreign = foreign_gpu_pids(device, own_pids, args.foreign_proc_mb)
                if free_mb < args.min_free_mb or foreign:
                    log(
                        f"GPU {device} busy free={free_mb}MB foreign={foreign}; "
                        f"{len(remaining)} waiting"
                    )
                    continue
                target, seed = remaining.pop(0)
                attempts[(target, seed)] += 1
                attempt = attempts[(target, seed)]
                run_dir = run_dir_for(root, target, seed)
                log_path = log_root / f"{target}_seed{seed}_attempt{attempt}.log"
                process = start_job(
                    adapter,
                    python,
                    txpert_repo,
                    run_dir,
                    target,
                    seed,
                    device,
                    args.batch_size,
                    log_path,
                )
                active[device] = {
                    "job": (target, seed),
                    "process": process,
                    "log_path": str(log_path),
                }
                own_pids.add(process.pid)
                log(f"START {target}/seed_{seed} attempt={attempt} GPU {device}")

            completed = 0
            for target, seed in JOBS:
                run_dir = run_dir_for(root, target, seed)
                if job_state(run_dir) == "complete":
                    try:
                        validate_complete(run_dir, target, seed)
                    except QueueFailure:
                        pass
                    else:
                        completed += 1
            queue_status.update(
                {
                    "updated_at": now(),
                    "status": "RUNNING",
                    "completed": completed,
                    "waiting": len(remaining),
                    "active": [
                        {
                            "device": device,
                            "target": entry["job"][0],
                            "seed": entry["job"][1],
                            "pid": entry["process"].pid,
                            "log_path": entry["log_path"],
                        }
                        for device, entry in active.items()
                    ],
                    "detached": detached,
                    "attempts": {
                        f"{target}/seed_{seed}": count
                        for (target, seed), count in attempts.items()
                        if count
                    },
                    "permanent_failures": permanent_failures,
                    "archives": archives,
                }
            )
            write_json(state_path, queue_status)
            time.sleep(args.poll_seconds)
    finally:
        running = [
            {
                "device": device,
                "target": entry["job"][0],
                "seed": entry["job"][1],
                "pid": entry["process"].pid,
            }
            for device, entry in active.items()
            if entry["process"].poll() is None
        ]
        if stop_requested:
            final_status = "INTERRUPTED"
        elif permanent_failures:
            final_status = "FAILED"
        else:
            final_status = "COMPLETE"
        queue_status.update(
            {
                "status": final_status,
                "finished_at": now(),
                "running_children_preserved": running,
                "permanent_failures": permanent_failures,
                "archives": archives,
            }
        )
        write_json(state_path, queue_status)
        supervisor_log.close()


if __name__ == "__main__":
    main()
