#!/usr/bin/env python3
"""Wait for E233 competence PASS, predict all E235 tasks, and stop at score seal.

This worker never opens Jiang24 test perturbed expression. It deliberately does
not evaluate or authorize truth access; a separate remote score seal is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from run_e233_stage1_supervisor import best_checkpoint


TERMINAL = {"STAGE2_FAMILY_PASS", "STAGE2_FAMILY_COMPETENCE_BLOCKED", "BLOCKED_STAGE1",
            "BLOCKED_TRAINING_FAILED", "BLOCKED_VALIDATION_FAILED", "FAILED"}


def read_status(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def predict(args: argparse.Namespace, seed: int, checkpoint: Path, gpu: int) -> dict:
    folder = args.prediction_root / f"seed_{seed}"
    if folder.exists():
        raise RuntimeError(f"refusing to overwrite prediction folder: {folder}")
    folder.mkdir(parents=True)
    command = [
        str(args.python), str(args.prediction_script),
        "--perturbench-repo", str(args.perturbench_repo), "--architecture", "linear",
        "--seed", str(seed), "--checkpoint", str(checkpoint),
        "--cache-dir", str(args.control_cache_dir), "--tasks", str(args.task_manifest),
        "--manifest-status", str(args.manifest_status), "--output-dir", str(folder),
        "--device", "cuda:0",
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = (folder / "PREDICTION.log").open("w")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    return {"seed": seed, "gpu": gpu, "folder": folder, "process": process, "log": log,
            "checkpoint": checkpoint}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage1-status", "stage1-run-root", "stage2-status", "stage2-run-root",
                 "python", "prediction-script", "score-script", "perturbench-repo",
                 "control-cache-dir", "source-cache-dir", "task-manifest", "manifest-status",
                 "prediction-root", "score-output-dir", "supervisor-output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    result_path = args.supervisor_output_dir / "E235_PRETRUTH_SUPERVISOR_STATUS.json"
    if result_path.exists():
        raise RuntimeError("E235 pretruth supervisor already started; refusing duplicate")
    state = {"status": "WAITING_FOR_E233_STAGE2", "created_at": datetime.now().astimezone().isoformat(),
             "test_perturbed_expression_rows_read": 0, "prediction_exit_codes": {}}
    atomic_json(result_path, state)
    try:
        while True:
            stage2 = read_status(args.stage2_status)
            if stage2 and stage2.get("status") in TERMINAL:
                break
            time.sleep(args.poll_seconds)
        if stage2["status"] != "STAGE2_FAMILY_PASS":
            state.update(status="BLOCKED_UPSTREAM_COMPETENCE", stage2_result=stage2["status"])
            return
        stage1 = read_status(args.stage1_status)
        variant = stage2.get("selected_variant")
        if (not stage1 or stage1.get("status") != "STAGE1_PASS"
                or stage1.get("selected_variant") != variant
                or variant not in ("matched_control_softplus", "matched_control_linear")):
            raise RuntimeError("E233 variant/status inconsistency")
        checkpoints = {}
        for seed in (1, 2, 3, 4):
            run = ((args.stage1_run_root / variant / str(stage1["attempt"])) if seed == 1
                   else (args.stage2_run_root / variant / f"seed_{seed}"))
            training = read_status(run / "E233_RUN_STATUS.json")
            if (not training or training.get("status") != "COMPLETE"
                    or training.get("variant") != variant or training.get("seed") != seed
                    or training.get("test_perturbed_expression_rows_read") != 0):
                raise RuntimeError(f"E233 selected seed {seed} has no complete no-truth record")
            checkpoints[seed] = best_checkpoint(training)
        state.update(status="PREDICTING", selected_variant=variant,
                     checkpoint_paths={str(seed): str(path) for seed, path in checkpoints.items()})
        atomic_json(result_path, state)
        for first, second in ((1, 2), (3, 4)):
            jobs = [predict(args, first, checkpoints[first], 0),
                    predict(args, second, checkpoints[second], 1)]
            for job in jobs:
                code = job["process"].wait()
                job["log"].close()
                state["prediction_exit_codes"][str(job["seed"])] = code
                atomic_json(result_path, state)
            if any(state["prediction_exit_codes"][str(job["seed"])] != 0 for job in jobs):
                state["status"] = "BLOCKED_PREDICTION_FAILED"
                return
        state["status"] = "SCORING"
        atomic_json(result_path, state)
        command = [
            str(args.python), str(args.score_script),
            "--stage1-status", str(args.stage1_status), "--stage1-run-root", str(args.stage1_run_root),
            "--stage2-status", str(args.stage2_status), "--stage2-run-root", str(args.stage2_run_root),
            "--prediction-root", str(args.prediction_root), "--task-manifest", str(args.task_manifest),
            "--manifest-status", str(args.manifest_status),
            "--source-cache-dir", str(args.source_cache_dir),
            "--control-cache-dir", str(args.control_cache_dir),
            "--output-dir", str(args.score_output_dir),
        ]
        args.score_output_dir.mkdir(parents=True, exist_ok=True)
        with (args.score_output_dir / "E235_SCORE_WORKER.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            state.update(status="BLOCKED_SCORE_FAILED", score_exit_code=completed.returncode)
            return
        score = read_status(args.score_output_dir / "E235_PRETRUTH_SCORE_STATUS.json")
        if (not score or score.get("status") != "SCORES_READY_AWAITING_REMOTE_SEAL"
                or score.get("test_perturbed_expression_rows_read") != 0):
            raise RuntimeError("E235 score worker did not produce a no-truth seal")
        state.update(status="SCORES_READY_AWAITING_REMOTE_SEAL", score_sha256=score["score_sha256"])
    except BaseException as error:
        state.update(status="FAILED", reason=repr(error))
        raise
    finally:
        state["updated_at"] = datetime.now().astimezone().isoformat()
        atomic_json(result_path, state)


if __name__ == "__main__":
    main()
