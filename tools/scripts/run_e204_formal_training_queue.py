#!/usr/bin/env python3
"""Run the frozen E204 formal weighted-training jobs on shared GPUs.

E204 formal comparison = 4 targets x 4 seeds x 2 weighting arms
(risk_weighted, dispersion_only), 32 models, 80 epochs each, following
docs/实验结果/E204_risk_guided_training_20260830/ANALYSIS_FREEZE.md.

The supervisor is polite on a shared server: a job only starts on a GPU whose
free memory exceeds --min-free-mb AND that has no foreign compute processes
above --foreign-proc-mb.  One concurrent job per GPU.  Jobs whose run
directory already passed all gates are skipped, so the supervisor is safe to
restart.  Each job is attempted at most --max-attempts times in total; failed
run directories are archived rather than overwritten.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
SEEDS = (1, 2, 3, 4)
ARMS = ("risk", "dispersion")
WEIGHT_COLUMNS = {"risk": "task_weight", "dispersion": "dispersion_only_weight"}
EXPECTED_WEIGHT_ROWS = {"K562": 1_365, "RPE1": 1_298, "hepg2": 1_241, "jurkat": 1_308}
JOBS = tuple(
    (target, seed, arm) for seed in SEEDS for target in TARGETS for arm in ARMS
)


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
    parser.add_argument("--weight-manifest", type=Path, required=True)
    parser.add_argument("--cuda-devices", default="0,1")
    parser.add_argument("--min-free-mb", type=int, default=20480)
    parser.add_argument("--foreign-proc-mb", type=int, default=4096)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_dir_for(runs_root: Path, target: str, seed: int, arm: str) -> Path:
    return runs_root / target / f"seed_{seed}_{arm}"


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
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, mb = int(parts[0]), int(parts[1])
        if pid not in own_pids and mb >= min_mb:
            foreign.append(pid)
    return sorted(foreign)


def training_pids_for(run_dir: Path, launcher: Path) -> set[int]:
    launcher_text = str(launcher.resolve())
    run_text = str(run_dir.resolve())
    matches = set()
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            UnicodeDecodeError,
        ):
            continue
        if launcher_text in command and run_text in command:
            matches.add(int(proc_dir.name))
    return matches


def validate_complete(run_dir: Path, target: str, seed: int, arm: str) -> None:
    run_status_path = run_dir / "E201_RUN_STATUS.json"
    weight_status_path = run_dir / "E204_WEIGHTING_STATUS.json"
    if not run_status_path.is_file() or not weight_status_path.is_file():
        raise QueueFailure(f"missing status files in {run_dir}")
    run_status = read_json(run_status_path)
    weight_status = read_json(weight_status_path)
    last_model_path = Path(str(run_status.get("last_model_path", "")))
    required = {
        "run_status": run_status.get("status") == "COMPLETE",
        "kind": run_status.get("kind") == "formal",
        "target": run_status.get("target") == target,
        "seed": int(run_status.get("seed", -1)) == seed,
        "epochs": int(run_status.get("current_epoch", -1)) == 80,
        "target_access": int(run_status.get("target_perturbed_cells_accessed", -1)) == 0,
        "test_not_constructed": run_status.get("target_test_dataset_constructed") is False,
        "last_checkpoint": last_model_path.is_file() and last_model_path.stat().st_size > 0,
        "weight_status": weight_status.get("status") == "COMPLETE",
        "weight_target": weight_status.get("target") == target,
        "weight_seed": int(weight_status.get("seed", -1)) == seed,
        "weight_kind": weight_status.get("kind") == "formal",
        "weight_column": weight_status.get("weight_column") == WEIGHT_COLUMNS[arm],
        "weight_rows": int(weight_status.get("weight_manifest_rows_for_target", -1))
        == EXPECTED_WEIGHT_ROWS[target],
        "zero_weight_fallback": int(weight_status.get("unit_weight_fallback_samples", -1))
        == 0,
        "training_only": weight_status.get("weights_applied_to_training_only") is True,
        "weight_target_access": weight_status.get("target_expression_opened") is False,
        "external_source_unchanged": weight_status.get("external_txpert_modified") is False,
    }
    failed = sorted(name for name, passed in required.items() if not passed)
    if failed:
        raise QueueFailure(
            f"completed status failed gates for {target}/seed_{seed}/{arm}: {failed}"
        )


def job_state(run_dir: Path) -> str:
    """Map a run directory to pending / running / complete / failed."""
    weight_status_path = run_dir / "E204_WEIGHTING_STATUS.json"
    run_status_path = run_dir / "E201_RUN_STATUS.json"
    if weight_status_path.is_file():
        try:
            status = read_json(weight_status_path).get("status")
        except (OSError, ValueError, TypeError):
            return "failed"
        if status == "COMPLETE":
            return "complete"
        if status in ("FAILED", "FAILED_COVERAGE"):
            return "failed"
    if run_status_path.is_file():
        try:
            status = read_json(run_status_path).get("status")
        except (OSError, ValueError, TypeError):
            return "failed"
        if status == "RUNNING":
            return "running"
        if status == "FAILED":
            return "failed"
    return "pending"


def attempts_from_logs(log_root: Path, target: str, seed: int, arm: str) -> int:
    """Recover the number of starts after a supervisor restart."""
    pattern = re.compile(
        rf"^{re.escape(target)}_seed{seed}_{re.escape(arm)}_attempt(\d+)\.log$"
    )
    attempts = []
    for path in log_root.glob(f"{target}_seed{seed}_{arm}_attempt*.log"):
        match = pattern.match(path.name)
        if match:
            attempts.append(int(match.group(1)))
    return max(attempts, default=0)


def archive_failed_run(
    run_dir: Path,
    runs_root: Path,
    target: str,
    seed: int,
    arm: str,
    attempt: int,
) -> Path | None:
    """Preserve a failed/stale directory before a clean retry."""
    if not run_dir.exists():
        return None
    archive_root = runs_root / "_failed_attempts"
    archive_root.mkdir(exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    destination = archive_root / (
        f"{target}_seed{seed}_{arm}_attempt{attempt}_{stamp}"
    )
    os.replace(run_dir, destination)
    return destination


def start_job(
    launcher: Path,
    python: Path,
    txpert_repo: Path,
    manifest: Path,
    run_dir: Path,
    target: str,
    seed: int,
    arm: str,
    device: str,
    log_path: Path,
) -> subprocess.Popen:
    # The frozen adapter deliberately refuses an existing run directory.  The
    # supervisor may create the parent, but must leave the job directory for
    # the adapter to create after all preflight gates pass.
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise QueueFailure(f"refusing to start into an existing run directory: {run_dir}")
    command = [
        str(python),
        str(launcher),
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
        "--task-weight-manifest",
        str(manifest),
        "--weight-column",
        WEIGHT_COLUMNS[arm],
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = device
    log_handle = log_path.open("a", encoding="utf-8")
    try:
        log_handle.write(f"\n===== {now()} START {command} on GPU {device} =====\n")
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
        # Popen duplicated the descriptor for the child.  The supervisor must
        # close its copy or a long queue leaks one descriptor per attempt.
        log_handle.close()


def main() -> None:
    args = parse_args()
    if not 30 <= args.poll_seconds <= 1800:
        raise QueueFailure("poll-seconds must be between 30 and 1800")
    if not 1 <= args.max_attempts <= 5:
        raise QueueFailure("max-attempts must be between 1 and 5")
    safeconf_repo = args.safeconf_repo.resolve()
    txpert_repo = args.txpert_repo.resolve()
    # Do not resolve the venv python symlink: the environment depends on being
    # invoked through its own bin/python path.
    python = args.python.expanduser().absolute()
    runs_root = args.runs_root.resolve()
    manifest = args.weight_manifest.resolve()
    launcher = safeconf_repo / "tools/scripts/run_e204_weighted_training.py"
    devices = [item.strip() for item in args.cuda_devices.split(",") if item.strip()]
    if not devices:
        raise QueueFailure("no cuda devices configured")
    for path in (safeconf_repo, txpert_repo, python, manifest, launcher):
        if not path.exists():
            raise QueueFailure(f"missing required path: {path}")

    runs_root.mkdir(parents=True, exist_ok=True)
    log_root = runs_root / "_queue_logs"
    log_root.mkdir(exist_ok=True)
    lock_path = runs_root / "E204_QUEUE.lock"
    state_path = runs_root / "E204_QUEUE_STATUS.json"
    supervisor_log_path = runs_root / "E204_QUEUE_SUPERVISOR.log"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise QueueFailure("another E204 formal queue already holds the lock") from exc

    supervisor_log = supervisor_log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{now()} {message}"
        print(line, flush=True)
        supervisor_log.write(line + "\n")
        supervisor_log.flush()

    queue_status = {
        "experiment": "E204_risk_guided_training",
        "stage": "formal_weighted_training_queue",
        "status": "RUNNING",
        "started_at": now(),
        "pid": os.getpid(),
        "cuda_devices": devices,
        "min_free_mb": args.min_free_mb,
        "jobs": [
            {"target": target, "seed": seed, "arm": arm} for target, seed, arm in JOBS
        ],
        "command": sys.argv,
    }
    write_json(state_path, queue_status)
    log(f"E204 formal queue started: {len(JOBS)} jobs, devices={devices}")

    attempts: dict[tuple[str, int, str], int] = {
        job: attempts_from_logs(log_root, *job) for job in JOBS
    }
    active: dict[str, dict] = {}
    permanently_failed: list[dict] = []
    archived_attempts: list[dict] = []
    stop_requested = False

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        log(f"received signal {signum}; stop scheduling and preserve running children")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        while True:
            # 1. Reap finished jobs.
            for device, entry in list(active.items()):
                process: subprocess.Popen = entry["process"]
                if process.poll() is None:
                    continue
                target, seed, arm = entry["job"]
                run_dir = run_dir_for(runs_root, target, seed, arm)
                state = job_state(run_dir)
                if state == "complete":
                    try:
                        validate_complete(run_dir, target, seed, arm)
                    except QueueFailure as exc:
                        state = "failed"
                        log(f"GATE-FAIL {target}/seed_{seed}/{arm}: {exc}")
                    else:
                        log(f"COMPLETE {target}/seed_{seed}/{arm} on GPU {device}")
                if state != "complete":
                    used = attempts.get((target, seed, arm), 0)
                    if used >= args.max_attempts:
                        permanently_failed.append(entry["job_record"])
                        log(
                            f"FAILED-FINAL {target}/seed_{seed}/{arm} "
                            f"after {used} attempts; see {entry['log_path']}"
                        )
                    else:
                        log(
                            f"RETRY-QUEUED {target}/seed_{seed}/{arm} "
                            f"(attempt {used} failed; exit {process.returncode})"
                        )
                del active[device]

            if stop_requested:
                break

            # 2. Count remaining work.
            remaining = []
            detached_running = []
            for target, seed, arm in JOBS:
                run_dir = run_dir_for(runs_root, target, seed, arm)
                state = job_state(run_dir)
                if state == "complete":
                    try:
                        validate_complete(run_dir, target, seed, arm)
                    except QueueFailure as exc:
                        log(f"GATE-FAIL-ON-SCAN {target}/seed_{seed}/{arm}: {exc}")
                        state = "failed"
                    else:
                        continue
                if any(entry["job"] == (target, seed, arm) for entry in active.values()):
                    continue
                detached_pids = training_pids_for(run_dir, launcher)
                if detached_pids:
                    detached_running.append(
                        {
                            "target": target,
                            "seed": seed,
                            "arm": arm,
                            "pids": sorted(detached_pids),
                        }
                    )
                    continue
                record = {"target": target, "seed": seed, "arm": arm}
                if record in permanently_failed:
                    continue
                used = attempts.get((target, seed, arm), 0)
                if state in ("failed", "running"):
                    archived = archive_failed_run(
                        run_dir, runs_root, target, seed, arm, max(used, 1)
                    )
                    if archived is not None:
                        archived_record = {**record, "path": str(archived)}
                        archived_attempts.append(archived_record)
                        log(
                            f"ARCHIVE {target}/seed_{seed}/{arm} state={state} "
                            f"to {archived}"
                        )
                if used >= args.max_attempts:
                    permanently_failed.append(record)
                    log(
                        f"FAILED-FINAL {target}/seed_{seed}/{arm} "
                        f"after {used} total attempts"
                    )
                    continue
                remaining.append((target, seed, arm))
            if not remaining and not active:
                if detached_running:
                    log(
                        f"waiting for {len(detached_running)} detached jobs: "
                        f"{detached_running}"
                    )
                    queue_status.update(
                        {
                            "status": "RUNNING",
                            "updated_at": now(),
                            "active": [],
                            "detached_running": detached_running,
                            "waiting": 0,
                            "permanently_failed": permanently_failed,
                            "attempts": {
                                f"{t}/seed_{s}/{a}": n
                                for (t, s, a), n in attempts.items()
                                if n
                            },
                            "archived_attempts": archived_attempts,
                        }
                    )
                    write_json(state_path, queue_status)
                    time.sleep(args.poll_seconds)
                    continue
                break

            # 3. Launch new jobs where GPUs allow.
            for device in devices:
                if device in active or not remaining:
                    continue
                try:
                    free_mb = nvidia_free_mb(device)
                except (subprocess.CalledProcessError, ValueError, IndexError) as exc:
                    log(f"GPU {device} free-memory query failed: {exc}; skip this round")
                    continue
                own_pids = set()
                for entry in active.values():
                    if entry["process"].poll() is None:
                        own_pids.add(entry["process"].pid)
                foreign = foreign_gpu_pids(device, own_pids, args.foreign_proc_mb)
                if free_mb < args.min_free_mb or foreign:
                    log(
                        f"GPU {device} busy (free={free_mb}MB, foreign_pids={foreign}); "
                        f"{len(remaining)} jobs waiting"
                    )
                    continue
                target, seed, arm = remaining[0]
                attempt = attempts.get((target, seed, arm), 0) + 1
                attempts[(target, seed, arm)] = attempt
                run_dir = run_dir_for(runs_root, target, seed, arm)
                log_path = log_root / f"{target}_seed{seed}_{arm}_attempt{attempt}.log"
                process = start_job(
                    launcher,
                    python,
                    txpert_repo,
                    manifest,
                    run_dir,
                    target,
                    seed,
                    arm,
                    device,
                    log_path,
                )
                active[device] = {
                    "job": (target, seed, arm),
                    "job_record": {"target": target, "seed": seed, "arm": arm},
                    "process": process,
                    "log_path": str(log_path),
                    "attempt": attempt,
                }
                remaining = remaining[1:]
                log(
                    f"START {target}/seed_{seed}/{arm} attempt {attempt} "
                    f"on GPU {device} (free={free_mb}MB) pid={process.pid}"
                )

            queue_status.update(
                {
                    "status": "RUNNING",
                    "updated_at": now(),
                    "active": [
                        {
                            "device": device,
                            "target": entry["job"][0],
                            "seed": entry["job"][1],
                            "arm": entry["job"][2],
                            "pid": entry["process"].pid,
                            "attempt": entry["attempt"],
                        }
                        for device, entry in active.items()
                    ],
                    "detached_running": detached_running,
                    "waiting": len(remaining),
                    "permanently_failed": permanently_failed,
                    "attempts": {
                        f"{t}/seed_{s}/{a}": n
                        for (t, s, a), n in attempts.items()
                        if n
                    },
                    "archived_attempts": archived_attempts,
                }
            )
            write_json(state_path, queue_status)

            time.sleep(args.poll_seconds)
    finally:
        all_complete = not permanently_failed
        if all_complete:
            for target, seed, arm in JOBS:
                run_dir = run_dir_for(runs_root, target, seed, arm)
                if job_state(run_dir) != "complete":
                    all_complete = False
                    break
                try:
                    validate_complete(run_dir, target, seed, arm)
                except QueueFailure:
                    all_complete = False
                    break
        if all_complete:
            final_state = "COMPLETE"
        elif permanently_failed:
            final_state = "FAILED"
        else:
            final_state = "INTERRUPTED"
        queue_status.update(
            {
                "status": final_state,
                "finished_at": now(),
                "permanently_failed": permanently_failed,
                "attempts": {f"{t}/seed_{s}/{a}": n for (t, s, a), n in attempts.items()},
                "archived_attempts": archived_attempts,
                "running_children_preserved": [
                    entry["process"].pid
                    for entry in active.values()
                    if entry["process"].poll() is None
                ],
            }
        )
        write_json(state_path, queue_status)
        log(f"E204 formal queue exited with status {final_state}")
        supervisor_log.close()
        lock_handle.close()

    if permanently_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
