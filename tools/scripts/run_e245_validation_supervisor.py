#!/usr/bin/env python3
"""After train-only cache completion, evaluate E245 validation and stop before test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "evaluator", "source-cache", "tasks", "target-controls",
                 "target-control-status", "validation-control-cache", "validation-truth",
                 "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-hours", type=float, default=12.0)
    args = parser.parse_args()
    if args.poll_seconds < 10 or args.max_wait_hours <= 0:
        raise RuntimeError("invalid E245 supervisor wait settings")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "E245_VALIDATION_SUPERVISOR_STATUS.json"
    if status_path.exists():
        raise RuntimeError("E245 validation supervisor already exists")
    status = {"status": "WAITING_FOR_TRAIN_SOURCE_CACHE", "started_at": now(),
              "test_perturbed_expression_rows_read": 0}
    atomic_json(status_path, status)
    start = time.monotonic()
    try:
        source_status_path = args.source_cache / "E245_VALIDATION_SOURCE_CACHE_STATUS.json"
        while not source_status_path.exists():
            if time.monotonic() - start > args.max_wait_hours * 3600:
                raise RuntimeError("E245 train-source cache did not finish before timeout")
            status["updated_at"] = now()
            atomic_json(status_path, status)
            time.sleep(args.poll_seconds)
        source = json.loads(source_status_path.read_text())
        if (source.get("status") != "PASS" or source.get("test_perturbed_expression_rows_read") != 0
                or source.get("validation_perturbed_expression_rows_read") != 0):
            raise RuntimeError("E245 train-source cache did not pass")
        status.update(status="RUNNING_VALIDATION", source_cache_completed_at=now())
        atomic_json(status_path, status)
        command = [str(args.python), str(args.evaluator), "--source-cache", str(args.source_cache),
                   "--tasks", str(args.tasks), "--target-controls", str(args.target_controls),
                   "--target-control-status", str(args.target_control_status),
                   "--validation-control-cache", str(args.validation_control_cache),
                   "--validation-truth", str(args.validation_truth),
                   "--output-dir", str(output / "evaluation")]
        with (output / "E245_VALIDATION.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f"E245 validation evaluator exited {completed.returncode}")
        result = json.loads((output / "evaluation/E245_VALIDATION_STATUS.json").read_text())
        if (result.get("status") not in ("VALIDATION_COMPETENCE_PASS", "BLOCKED_UPSTREAM_COMPETENCE")
                or result.get("test_perturbed_expression_rows_read") != 0):
            raise RuntimeError("E245 validation result malformed")
        status.update(status=result["status"], selected_alpha=result.get("selected_alpha"),
                      finished_at=now())
    except BaseException as error:
        status.update(status="FAILED", reason=repr(error), failed_at=now())
        raise
    finally:
        atomic_json(status_path, status)


if __name__ == "__main__":
    main()
