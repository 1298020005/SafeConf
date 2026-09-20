#!/usr/bin/env python3
"""Wait for all E216 models, then generate and hash truth-free predictions."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


EXPECTED_RUNS = (
    ("latent", 1),
    ("latent", 2),
    ("latent", 3),
    ("latent", 4),
    ("linear", 1),
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_status(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def completed_training_runs(runs_root: Path) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    complete = []
    waiting = []
    for architecture, seed in EXPECTED_RUNS:
        status = read_status(
            runs_root / architecture / f"seed_{seed}" / "E216_RUN_STATUS.json"
        )
        if status and status.get("status") == "COMPLETE":
            complete.append((architecture, seed))
        else:
            waiting.append((architecture, seed))
    return complete, waiting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--control-asset", type=Path, required=True)
    parser.add_argument("--control-status", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cuda-device", default="1")
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    runs_root = args.runs_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    queue_status_path = output_root / "E216_PREDICTION_QUEUE_STATUS.json"
    control_status = read_status(args.control_status.resolve())
    if not control_status or control_status.get("status") != "PASS":
        raise RuntimeError("E216 control-only prediction asset has not passed")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    state = {
        "experiment": "E216_jiang24_resource_bounded_confirmation",
        "stage": "D2_TRUTH_FREE_PREDICTION_QUEUE",
        "status": "WAITING_FOR_TRAINING",
        "started_at": now_iso(),
        "code_commit": commit,
        "expected_runs": [f"{arch}/seed_{seed}" for arch, seed in EXPECTED_RUNS],
        "test_truth_access": "NOT_AUTHORIZED",
        "test_perturbed_expression_rows_read": 0,
    }
    atomic_json(queue_status_path, state)

    while True:
        complete, waiting = completed_training_runs(runs_root)
        state.update(
            {
                "last_checked_at": now_iso(),
                "training_complete": [f"{a}/seed_{s}" for a, s in complete],
                "training_waiting": [f"{a}/seed_{s}" for a, s in waiting],
            }
        )
        atomic_json(queue_status_path, state)
        if not waiting:
            break
        time.sleep(max(5, args.poll_seconds))

    state["status"] = "PREDICTING"
    atomic_json(queue_status_path, state)
    prediction_script = repo / "tools" / "scripts" / "run_e216_prediction_job.py"
    completed_predictions = []
    for architecture, seed in EXPECTED_RUNS:
        run_dir = runs_root / architecture / f"seed_{seed}"
        output_dir = output_root / architecture / f"seed_{seed}"
        prediction_status = read_status(output_dir / "E216_PREDICTION_STATUS.json")
        if prediction_status and prediction_status.get("status") == "COMPLETE":
            completed_predictions.append(f"{architecture}/seed_{seed}")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.python.resolve()),
            str(prediction_script),
            "--perturbench-repo", str(args.perturbench_repo.resolve()),
            "--run-dir", str(run_dir),
            "--tasks", str(args.tasks.resolve()),
            "--control-asset", str(args.control_asset.resolve()),
            "--output-dir", str(output_dir),
            "--cuda-device", str(args.cuda_device),
            "--chunk-size", "8",
        ]
        log_path = output_dir / "E216_PREDICTION_CONSOLE.log"
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            state.update(
                {
                    "status": "FAILED",
                    "failed_run": f"{architecture}/seed_{seed}",
                    "finished_at": now_iso(),
                }
            )
            atomic_json(queue_status_path, state)
            return result.returncode
        completed_predictions.append(f"{architecture}/seed_{seed}")
        state["predictions_complete"] = completed_predictions
        atomic_json(queue_status_path, state)

    state.update(
        {
            "status": "PREDICTIONS_COMPLETE",
            "finished_at": now_iso(),
            "predictions_complete": completed_predictions,
        }
    )
    atomic_json(queue_status_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
