#!/usr/bin/env python3
"""Wait for E208 pretruth predictions, then build the frozen risk seal."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


class SealQueueFailure(RuntimeError):
    """The pretruth inputs or risk-seal builder failed."""


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--validation-cache-dir", type=Path, required=True)
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    for field in (
        "repo",
        "python",
        "prediction_root",
        "validation_cache_dir",
        "source_cache_dir",
        "output_dir",
    ):
        setattr(args, field, getattr(args, field).expanduser().absolute())
    if not 10 <= args.poll_seconds <= 600:
        raise SealQueueFailure("poll interval must be 10--600 seconds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue_status_path = args.output_dir / "E208_RISK_SEAL_QUEUE_STATUS.json"
    state = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_PRETRUTH_RISK_SEAL_QUEUE",
        "status": "WAITING_FOR_PREDICTIONS_AND_SOURCE_CACHE",
        "started_at": now(),
        "pid": os.getpid(),
        "test_perturbed_expression_rows_read": 0,
        "target_truth_access": "NOT_AUTHORIZED",
    }
    atomic_json(queue_status_path, state)
    post_status_path = (
        args.prediction_root / "E208_POSTTRAINING_PRETRUTH_QUEUE_STATUS.json"
    )
    source_status_path = args.source_cache_dir / "E208_SOURCE_EFFECT_CACHE_STATUS.json"
    while True:
        predictions_ready = False
        source_ready = False
        if post_status_path.is_file():
            try:
                post = read_json(post_status_path)
                predictions_ready = (
                    post.get("status")
                    == "PRETRUTH_PREDICTIONS_COMPLETE_AWAITING_RISK_SEAL"
                    and int(post.get("test_pretruth_completed", -1)) == 5
                    and int(post.get("test_perturbed_expression_rows_read", -1)) == 0
                )
            except (OSError, ValueError, TypeError):
                predictions_ready = False
        if source_status_path.is_file():
            try:
                source = read_json(source_status_path)
                source_ready = (
                    source.get("status") == "PASS"
                    and int(source.get("n_source_effects", -1)) == 238
                    and int(source.get("test_perturbed_expression_rows_read", -1)) == 0
                )
            except (OSError, ValueError, TypeError):
                source_ready = False
        state.update(
            {
                "updated_at": now(),
                "predictions_ready": predictions_ready,
                "source_cache_ready": source_ready,
            }
        )
        atomic_json(queue_status_path, state)
        if predictions_ready and source_ready:
            break
        time.sleep(args.poll_seconds)

    state.update({"status": "BUILDING_PRETRUTH_RISK_SEAL", "updated_at": now()})
    atomic_json(queue_status_path, state)
    seal_dir = args.output_dir / "seal"
    command = [
        str(args.python),
        str(args.repo / "tools/scripts/build_e208_pretruth_risk_seal.py"),
        "--prediction-root", str(args.prediction_root),
        "--validation-cache-dir", str(args.validation_cache_dir),
        "--competence-dir", str(args.prediction_root / "validation_competence"),
        "--source-cache-dir", str(args.source_cache_dir),
        "--output-dir", str(seal_dir),
    ]
    log_path = args.output_dir / "RISK_SEAL_BUILD.log"
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=args.repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        state.update(
            {
                "status": "RISK_SEAL_BUILD_FAILED",
                "finished_at": now(),
                "log_path": str(log_path),
            }
        )
        atomic_json(queue_status_path, state)
        raise SealQueueFailure(f"risk seal builder exited {completed.returncode}")
    seal_status = read_json(seal_dir / "E208_PRETRUTH_RISK_SEAL_STATUS.json")
    if (
        seal_status.get("status") != "PASS_AWAITING_REMOTE_SEAL"
        or seal_status.get("n_tasks") != 224
        or seal_status.get("test_perturbed_expression_rows_read") != 0
    ):
        raise SealQueueFailure("risk seal output failed its completion gate")
    state.update(
        {
            "status": "PRETRUTH_RISK_SEAL_READY_FOR_GIT",
            "finished_at": now(),
            "seal_dir": str(seal_dir),
            "risk_table_sha256": seal_status["risk_table"]["sha256"],
            "test_perturbed_expression_rows_read": 0,
            "target_truth_access": "NOT_AUTHORIZED",
            "next_required_gate": "independent audit, GitHub/Gitee push, then separate truth authorization",
        }
    )
    atomic_json(queue_status_path, state)


if __name__ == "__main__":
    main()
