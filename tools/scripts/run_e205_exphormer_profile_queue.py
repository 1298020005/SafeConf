#!/usr/bin/env python3
"""Wait politely for one GPU and run the E205 Exphormer contract profile.

The profile is one complete training epoch on RPE1/seed 1. It checks data
isolation, architecture composition, runtime, memory and checkpoint writing.
It does not construct target outcomes and it is not a scientific result.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


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
    parser.add_argument("--min-free-mb", type=int, default=20480)
    parser.add_argument("--foreign-proc-mb", type=int, default=1024)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--target", default="RPE1")
    parser.add_argument("--seed", type=int, default=1)
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


def foreign_gpu_pids(device: str, min_mb: int) -> list[int]:
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
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            if int(parts[1]) >= min_mb:
                foreign.append(int(parts[0]))
    return sorted(set(foreign))


def matching_pids(adapter: Path, run_dir: Path) -> list[int]:
    adapter_text = str(adapter.resolve())
    run_text = str(run_dir.resolve())
    matches = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if adapter_text in command and run_text in command:
            matches.append(int(proc_dir.name))
    return sorted(matches)


def validate_complete(run_dir: Path, target: str, seed: int) -> dict:
    status_path = run_dir / "E205_RUN_STATUS.json"
    if not status_path.is_file():
        raise QueueFailure("E205 profile status is missing")
    status = read_json(status_path)
    last_path = Path(str(status.get("last_model_path", "")))
    required = {
        "complete": status.get("status") == "COMPLETE",
        "profile": status.get("kind") == "profile",
        "target": status.get("target") == target,
        "seed": int(status.get("seed", -1)) == seed,
        "model_family": status.get("model_family") == "TxPert-Exphormer",
        "architecture_only": status.get("architecture_change_only") is True,
        "one_epoch": int(status.get("current_epoch", -1)) == 1,
        "zero_target_access": int(status.get("target_perturbed_cells_accessed", -1)) == 0,
        "test_not_constructed": status.get("target_test_dataset_constructed") is False,
        "checkpoint": last_path.is_file() and last_path.stat().st_size > 0,
    }
    failed = sorted(name for name, passed in required.items() if not passed)
    if failed:
        raise QueueFailure(f"E205 profile failed gates: {failed}")
    return status


def main() -> None:
    args = parse_args()
    if not 30 <= args.poll_seconds <= 1800:
        raise QueueFailure("poll-seconds must be between 30 and 1800")
    if args.target not in {"K562", "RPE1", "hepg2", "jurkat"}:
        raise QueueFailure("unsupported target")
    if args.seed not in {1, 2, 3, 4}:
        raise QueueFailure("seed must be 1..4")

    safeconf_repo = args.safeconf_repo.resolve()
    txpert_repo = args.txpert_repo.resolve()
    python = args.python.expanduser().absolute()
    runs_root = args.runs_root.resolve()
    adapter = safeconf_repo / "tools/scripts/txpert_blind_training_exphormer_adapter.py"
    for path in (safeconf_repo, txpert_repo, python, adapter):
        if not path.exists():
            raise QueueFailure(f"missing required path: {path}")

    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / args.target / f"seed_{args.seed}"
    state_path = runs_root / "E205_PROFILE_QUEUE_STATUS.json"
    log_path = runs_root / "E205_PROFILE.log"
    lock_handle = (runs_root / "E205_PROFILE_QUEUE.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise QueueFailure("another E205 profile queue holds the lock") from exc

    if (run_dir / "E205_RUN_STATUS.json").is_file():
        try:
            validate_complete(run_dir, args.target, args.seed)
        except QueueFailure:
            live = matching_pids(adapter, run_dir)
            if live:
                raise QueueFailure(f"detached E205 profile is still running: {live}")
            stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
            failed_root = runs_root / "_failed_attempts"
            failed_root.mkdir(exist_ok=True)
            os.replace(run_dir, failed_root / f"{args.target}_seed{args.seed}_{stamp}")
        else:
            write_json(
                state_path,
                {"status": "COMPLETE", "updated_at": now(), "run_dir": str(run_dir)},
            )
            return
    elif run_dir.exists():
        raise QueueFailure(f"profile directory exists without a status: {run_dir}")

    stop_requested = False

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    state = {
        "experiment": "E205_cross_family_exphormer",
        "stage": "one_epoch_contract_profile",
        "status": "WAITING_FOR_GPU",
        "started_at": now(),
        "target": args.target,
        "seed": args.seed,
        "cuda_device": args.cuda_device,
        "min_free_mb": args.min_free_mb,
        "foreign_proc_mb": args.foreign_proc_mb,
        "run_dir": str(run_dir),
    }
    write_json(state_path, state)

    while not stop_requested:
        free_mb = nvidia_free_mb(args.cuda_device)
        foreign = foreign_gpu_pids(args.cuda_device, args.foreign_proc_mb)
        state.update(
            {"updated_at": now(), "free_mb": free_mb, "foreign_pids": foreign}
        )
        write_json(state_path, state)
        if free_mb >= args.min_free_mb and not foreign:
            break
        time.sleep(args.poll_seconds)
    if stop_requested:
        state.update({"status": "INTERRUPTED", "updated_at": now()})
        write_json(state_path, state)
        return

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(adapter),
        "--txpert-repo",
        str(txpert_repo),
        "--task-type",
        f"E201_blind_{args.target}",
        "--target",
        args.target,
        "--seed",
        str(args.seed),
        "--run-dir",
        str(run_dir),
        "--kind",
        "profile",
        "--batch-size",
        str(args.batch_size),
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n===== {now()} START {command} =====\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=txpert_repo,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state.update({"status": "RUNNING", "updated_at": now(), "pid": process.pid})
    write_json(state_path, state)
    return_code = process.wait()
    if return_code != 0:
        state.update(
            {"status": "FAILED", "updated_at": now(), "return_code": return_code}
        )
        write_json(state_path, state)
        raise QueueFailure(f"E205 profile exited with {return_code}; see {log_path}")
    profile = validate_complete(run_dir, args.target, args.seed)
    state.update(
        {
            "status": "COMPLETE",
            "updated_at": now(),
            "return_code": return_code,
            "fit_wall_seconds": profile.get("fit_wall_seconds"),
            "cuda_peak_memory_reserved_bytes": profile.get(
                "cuda_peak_memory_reserved_bytes"
            ),
            "formal_expansion_authorized": True,
        }
    )
    write_json(state_path, state)


if __name__ == "__main__":
    main()
