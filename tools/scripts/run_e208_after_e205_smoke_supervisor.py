#!/usr/bin/env python3
"""Wait for a clean blind E205 completion, then run the E208 H5 smoke gate.

Only CUDA out-of-memory failures advance through the preregistered batch-size
fallbacks.  Any data, configuration, checkpoint, or truth-isolation failure
stops the supervisor.  Passing the smoke does not start formal E208 training.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence


BATCH_FALLBACKS = (2000, 1000, 500, 250)


class SupervisorFailure(RuntimeError):
    """The upstream queue or E208 smoke cannot continue safely."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--e205-queue-status", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-free-mb", type=int, default=22_000)
    return parser.parse_args(argv)


def e205_gate(path: Path) -> tuple[str, str]:
    """Return WAIT, GO, or FAIL plus a human-readable reason."""
    if not path.is_file():
        return "WAIT", "queue status has not been created"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "WAIT", f"queue status is being updated: {exc}"
    if state.get("permanent_failures"):
        return "FAIL", "E205 contains permanent failures"
    if state.get("target_truth_access") != "NOT_AUTHORIZED":
        return "FAIL", "E205 truth access changed before E208 smoke"
    completed = int(state.get("completed", 0))
    status = str(state.get("status", ""))
    if status == "COMPLETE" and completed == 16:
        return "GO", "E205 completed 16/16 in blind state"
    if status in {"RUNNING", "WAITING", ""} and 0 <= completed < 16:
        return "WAIT", f"E205 completed {completed}/16"
    return "FAIL", f"unexpected E205 state: status={status}, completed={completed}"


def is_oom_failure(attempt_dir: Path, combined_log: str = "") -> bool:
    needles = (
        "cuda out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "outofmemoryerror",
    )
    chunks = [combined_log.lower()]
    if attempt_dir.exists():
        for path in attempt_dir.rglob("*.log"):
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="replace").lower())
            except OSError:
                continue
        status = attempt_dir / "E208_H5_SMOKE_STATUS.json"
        if status.is_file():
            chunks.append(status.read_text(encoding="utf-8", errors="replace").lower())
    text = "\n".join(chunks)
    return any(needle in text for needle in needles)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict:
    if args.poll_seconds < 10:
        raise SupervisorFailure("poll interval must be at least 10 seconds")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "E208_AFTER_E205_SUPERVISOR_STATUS.json"
    status: dict[str, object] = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "WAIT_E205_THEN_H5_SMOKE",
        "status": "WAITING_FOR_E205",
        "started_at": now(),
        "e205_queue_status": str(args.e205_queue_status.resolve()),
        "test_truth_access": "NOT_AUTHORIZED",
        "attempts": [],
    }
    atomic_json(status_path, status)

    while True:
        gate, reason = e205_gate(args.e205_queue_status.resolve())
        status["last_e205_check_at"] = now()
        status["last_e205_reason"] = reason
        atomic_json(status_path, status)
        if gate == "GO":
            break
        if gate == "FAIL":
            status["status"] = "BLOCKED_E205"
            atomic_json(status_path, status)
            raise SupervisorFailure(reason)
        time.sleep(args.poll_seconds)

    smoke = args.repo.resolve() / "tools/scripts/run_e208_jiang24_h5_smoke.py"
    if not smoke.is_file():
        raise SupervisorFailure(f"missing smoke runner: {smoke}")
    status["status"] = "RUNNING_SMOKE"
    atomic_json(status_path, status)
    for index, batch_size in enumerate(BATCH_FALLBACKS):
        attempt = root / f"smoke_batch_{batch_size}"
        command = [
            str(args.python.resolve()),
            str(smoke),
            "--perturbench-repo",
            str(args.perturbench_repo.resolve()),
            "--python",
            str(args.python.resolve()),
            "--data-dir",
            str(args.data_dir.resolve()),
            "--output-root",
            str(attempt),
            "--e205-queue-status",
            str(args.e205_queue_status.resolve()),
            "--cuda-device",
            str(args.cuda_device),
            "--batch-size",
            str(batch_size),
            "--min-free-mb",
            str(args.min_free_mb),
        ]
        if index == 0:
            command.append("--rehash-h5")
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        attempt_record = {
            "batch_size": batch_size,
            "returncode": completed.returncode,
            "finished_at": now(),
            "output_tail": completed.stdout[-4000:],
        }
        status["attempts"].append(attempt_record)
        atomic_json(status_path, status)
        if completed.returncode == 0:
            status["status"] = "SMOKE_PASS_FORMAL_QUEUE_NOT_STARTED"
            status["selected_batch_size"] = batch_size
            status["finished_at"] = now()
            atomic_json(status_path, status)
            return status
        if not is_oom_failure(attempt, completed.stdout):
            status["status"] = "BLOCKED_NON_OOM_SMOKE_FAILURE"
            status["finished_at"] = now()
            atomic_json(status_path, status)
            raise SupervisorFailure(
                f"E208 smoke failed for a non-OOM reason at batch {batch_size}"
            )

    status["status"] = "BLOCKED_OOM_ALL_REGISTERED_BATCHES"
    status["finished_at"] = now()
    atomic_json(status_path, status)
    raise SupervisorFailure("all preregistered E208 batch sizes exhausted by OOM")


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
