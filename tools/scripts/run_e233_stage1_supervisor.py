#!/usr/bin/env python3
"""Wait for E233 stage-1 jobs, validate both, and freeze a winner without test truth."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

VARIANTS = ("matched_control_softplus", "matched_control_linear")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def status(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def best_checkpoint(record: dict) -> Path:
    candidates = [Path(x["path"]) for x in record["checkpoints"] if Path(x["path"]).name != "last.ckpt"]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one best checkpoint, found {candidates}")
    return candidates[0]


def evaluate_variant(output: Path, truth_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    tasks = pd.read_csv(output / "E208_VALIDATION_TASKS.csv")
    predictions = np.load(output / "E208_VALIDATION_PREDICTION_CENTROIDS.npy").astype(float)
    controls = np.load(output / "E208_VALIDATION_CONTROL_CENTROIDS.npy").astype(float)
    truth = np.load(truth_path).astype(float)
    if predictions.shape != truth.shape or controls.shape != truth.shape or len(tasks) != len(truth):
        raise RuntimeError("E233 validation arrays are not aligned")
    frame = tasks[["cell_type", "treatment", "condition", "task_id"]].copy()
    frame["no_change_mse"] = np.mean((controls - truth) ** 2, axis=1)
    frame["model_mse"] = np.mean((predictions - truth) ** 2, axis=1)
    frame["context"] = frame.cell_type.astype(str) + "|" + frame.treatment.astype(str)
    contexts = frame.groupby("context")[["no_change_mse", "model_mse"]].mean().reset_index()
    contexts["not_worse"] = contexts.model_mse <= contexts.no_change_mse
    summary = {
        "n_tasks": int(len(frame)), "n_contexts": int(len(contexts)),
        "no_change_mean_mse": float(frame.no_change_mse.mean()),
        "model_mean_mse": float(frame.model_mse.mean()),
        "relative_improvement": float(1 - frame.model_mse.mean() / frame.no_change_mse.mean()),
        "contexts_not_worse": int(contexts.not_worse.sum()),
    }
    summary["competence_gate"] = "PASS" if (
        summary["model_mean_mse"] < summary["no_change_mean_mse"]
        and summary["contexts_not_worse"] >= 8
    ) else "FAIL"
    return frame, contexts, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", default="seed_1_attempt3")
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--prediction-script", type=Path, required=True)
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    supervisor = {"status": "WAITING_FOR_TRAINING", "test_perturbed_expression_rows_read": 0,
                  "attempt": args.attempt, "variants": list(VARIANTS)}
    atomic_json(args.output / "E233_STAGE1_SUPERVISOR_STATUS.json", supervisor)

    while True:
        records = {v: status(args.root / v / args.attempt / "E233_RUN_STATUS.json") for v in VARIANTS}
        states = {v: None if r is None else r.get("status") for v, r in records.items()}
        supervisor.update(training_states=states)
        atomic_json(args.output / "E233_STAGE1_SUPERVISOR_STATUS.json", supervisor)
        if any(value == "FAILED" for value in states.values()):
            supervisor.update(status="BLOCKED_TRAINING_FAILED")
            atomic_json(args.output / "E233_STAGE1_SUPERVISOR_STATUS.json", supervisor)
            return
        if all(value == "COMPLETE" for value in states.values()):
            break
        time.sleep(args.poll_seconds)

    supervisor["status"] = "VALIDATING"
    atomic_json(args.output / "E233_STAGE1_SUPERVISOR_STATUS.json", supervisor)
    processes = []
    for device, variant in enumerate(VARIANTS):
        output = args.output / "validation" / variant
        checkpoint = best_checkpoint(records[variant])
        command = [
            str(args.python), str(args.prediction_script),
            "--perturbench-repo", str(args.perturbench_repo), "--architecture", "linear",
            "--seed", "1", "--checkpoint", str(checkpoint),
            "--validation-cache-dir", str(args.validation_cache), "--output-dir", str(output),
            "--device", f"cuda:{device}",
        ]
        log = (args.output / f"validation_{variant}.log").open("w")
        processes.append((variant, output, log, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)))
    for variant, _, log, process in processes:
        code = process.wait(); log.close()
        if code:
            supervisor.update(status="BLOCKED_VALIDATION_FAILED", failed_variant=variant, exit_code=code)
            atomic_json(args.output / "E233_STAGE1_SUPERVISOR_STATUS.json", supervisor)
            return

    summaries = {}
    for variant, output, _, _ in processes:
        tasks, contexts, summary = evaluate_variant(output, args.truth)
        tasks.to_csv(args.output / f"E233_{variant}_TASKS.csv", index=False)
        contexts.to_csv(args.output / f"E233_{variant}_CONTEXTS.csv", index=False)
        summaries[variant] = summary
    passing = [v for v in VARIANTS if summaries[v]["competence_gate"] == "PASS"]
    selected = min(passing, key=lambda v: summaries[v]["model_mean_mse"]) if passing else None
    supervisor.update(
        status="STAGE1_PASS" if selected else "STAGE1_COMPETENCE_BLOCKED",
        summaries=summaries, selected_variant=selected,
        selection_rule="lowest validation mean MSE among preregistered variants passing mean and 8/12-context gates",
        validation_truth_used=True, test_perturbed_expression_rows_read=0,
    )
    atomic_json(args.output / "E233_STAGE1_SUPERVISOR_STATUS.json", supervisor)


if __name__ == "__main__":
    main()
