#!/usr/bin/env python3
"""Run validation-only evaluation as soon as each McFaline training finishes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def process_has(text: str) -> bool:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if text in command:
            return True
    return False


def best_checkpoint(run: Path) -> Path:
    checkpoints = sorted(run.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint found under {run}")
    # The registered callback has save_top_k=1 and save_last disabled. Refuse
    # an ambiguous recovered directory rather than selecting after looking at
    # validation artifacts.
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one retained best checkpoint, got {checkpoints}")
    return checkpoints[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--latent-run", type=Path, required=True)
    parser.add_argument("--decoder-run", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    status_path = args.runtime_root / "VALIDATION_SUPERVISOR_STATUS.json"
    if status_path.exists():
        raise FileExistsError(f"refusing duplicate supervisor: {status_path}")
    runs = {"latent": (args.latent_run, 0), "decoder": (args.decoder_run, 1)}
    state = {
        "status": "MONITORING_FORMAL_TRAINING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "test_partition_opened": False,
        "candidates": {name: {"status": "TRAINING", "formal_run": str(run)} for name, (run, _) in runs.items()},
    }
    atomic_json(status_path, state)
    jobs: dict[str, subprocess.Popen] = {}
    logs = {}
    deadline = time.time() + 45 * 3600
    while time.time() < deadline:
        for name, (run, gpu) in runs.items():
            record = state["candidates"][name]
            if name in jobs:
                code = jobs[name].poll()
                if code is None:
                    record["status"] = "VALIDATING"
                else:
                    logs[name].close()
                    record["validation_exit_code"] = code
                    record["status"] = "VALIDATION_COMPLETE" if code == 0 else "VALIDATION_FAILED"
                    del jobs[name]
                continue
            if record["status"] in {"VALIDATION_COMPLETE", "VALIDATION_FAILED", "TRAINING_FAILED"}:
                continue
            if (run / "COMPLETED").exists() or (run / "STOPPED_BY_REGISTERED_VALIDATION_BUDGET").exists():
                checkpoint = best_checkpoint(run)
                output = args.runtime_root / f"{name}_validation"
                command = [
                    str(args.repo / "tools/scripts/evaluate_mcfaline_validation.sh"),
                    name, str(gpu), str(checkpoint), str(output),
                ]
                log = (args.runtime_root / f"{name}_validation_supervisor.log").open("w")
                jobs[name] = subprocess.Popen(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT)
                logs[name] = log
                record.update(status="VALIDATING", checkpoint=str(checkpoint), validation_output=str(output))
            elif process_has(str(run)):
                record["status"] = "TRAINING"
            else:
                record["status"] = "TRAINING_FAILED"
        terminal = {record["status"] for record in state["candidates"].values()}
        if terminal <= {"VALIDATION_COMPLETE", "VALIDATION_FAILED", "TRAINING_FAILED"}:
            state["status"] = "COMPLETE" if terminal == {"VALIDATION_COMPLETE"} else "COMPLETE_WITH_FAILURE"
            state["finished_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(status_path, state)
            return
        atomic_json(status_path, state)
        time.sleep(args.poll_seconds)
    state["status"] = "TIMEOUT"
    atomic_json(status_path, state)
    raise TimeoutError("McFaline training/validation supervisor exceeded 45 hours")


if __name__ == "__main__":
    main()
