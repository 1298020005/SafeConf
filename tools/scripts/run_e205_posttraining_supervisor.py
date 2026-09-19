#!/usr/bin/env python3
"""Complete the sealed E205 post-training chain and then run E208 smoke.

The supervisor fails closed.  It never rewrites an existing scientific output,
never force-pushes, and permits target-truth evaluation only after the pretruth
risk artifacts have been committed and pushed to both configured remotes.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence


TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
SEEDS = (1, 2, 3, 4)


class PipelineFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--txpert-repo", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--queue-status", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--cuda-devices", default="0,1")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--prediction-attempts", type=int, default=2)
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    parser.add_argument("--perturbench-python", type=Path, required=True)
    parser.add_argument("--e208-data-dir", type=Path, required=True)
    parser.add_argument("--e208-output-root", type=Path, required=True)
    return parser.parse_args(argv)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def ensure_release_branch(repo: Path, expected_branch: str) -> str:
    branch = git_text(repo, "branch", "--show-current")
    if branch != expected_branch:
        raise PipelineFailure(f"expected branch {expected_branch}, found {branch}")
    head = git_text(repo, "rev-parse", "HEAD")
    for remote in ("origin", "github"):
        remote_head = git_text(repo, "rev-parse", f"{remote}/{branch}")
        if remote_head != head:
            raise PipelineFailure(f"{remote}/{branch} differs from local HEAD")
    return head


def training_gate(path: Path) -> tuple[str, str]:
    if not path.is_file():
        return "WAIT", "training queue status missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "WAIT", f"training queue status is being updated: {exc}"
    if value.get("target_truth_access") != "NOT_AUTHORIZED":
        return "FAIL", "target truth access changed before checkpoint seal"
    if value.get("permanent_failures"):
        return "FAIL", "training queue contains permanent failures"
    completed = int(value.get("completed", 0))
    status = str(value.get("status", ""))
    if status == "COMPLETE" and completed == 16:
        return "GO", "E205 formal training completed 16/16"
    if status in {"RUNNING", "WAITING", "INTERRUPTED", ""} and completed < 16:
        return "WAIT", f"E205 formal training completed {completed}/16"
    return "FAIL", f"unexpected queue state status={status}, completed={completed}"


def run_checked(
    command: list[str], cwd: Path, log_path: Path, environment: dict | None = None
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== {now()} RUN {command} =====\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise PipelineFailure(
            f"command failed with {completed.returncode}; see {log_path}"
        )


def commit_and_push(repo: Path, paths: list[Path], message: str) -> str:
    ensure_release_branch(repo, git_text(repo, "branch", "--show-current"))
    if subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--quiet"], check=False
    ).returncode != 0:
        raise PipelineFailure("Git index is not empty before automated commit")
    relative = [path.resolve().relative_to(repo.resolve()).as_posix() for path in paths]
    subprocess.run(
        ["git", "-C", str(repo), "add", "-f", "--", *relative], check=True
    )
    staged = [
        item
        for item in git_text(repo, "diff", "--cached", "--name-only").splitlines()
        if item
    ]
    allowed = tuple(item.rstrip("/") for item in relative)
    unexpected = [
        item
        for item in staged
        if not any(item == prefix or item.startswith(prefix + "/") for prefix in allowed)
    ]
    if unexpected:
        subprocess.run(["git", "-C", str(repo), "reset", "--", *staged], check=False)
        raise PipelineFailure(f"unexpected staged files: {unexpected}")
    if not staged:
        return ensure_release_branch(repo, git_text(repo, "branch", "--show-current"))
    subprocess.run(["git", "-C", str(repo), "commit", "-m", message], check=True)
    branch = git_text(repo, "branch", "--show-current")
    for remote in ("origin", "github"):
        subprocess.run(
            ["git", "-C", str(repo), "push", remote, branch], check=True
        )
    return ensure_release_branch(repo, branch)


def prediction_complete(root: Path, target: str, seed: int) -> bool:
    directory = root / target / f"seed_{seed}"
    manifest = directory / "E205_PREDICTION_RUN.json"
    array = directory / "predictions.npy"
    if not manifest.is_file() or not array.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        record = value["prediction_file"]
        return bool(
            value.get("status") == "COMPLETE"
            and value.get("target") == target
            and int(value.get("seed", -1)) == seed
            and value.get("target_truth_materialized") is False
            and int(value.get("target_expression_nonzero_values_seen", -1)) == 0
            and array.stat().st_size == int(record["bytes"])
            and sha256_file(array) == record["sha256"]
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def archive_incomplete_predictions(root: Path, state_root: Path) -> None:
    archive = state_root / "prediction_failed_attempts"
    for target in TARGETS:
        seed_one_complete = prediction_complete(root, target, 1)
        for seed in SEEDS:
            directory = root / target / f"seed_{seed}"
            if directory.exists() and not prediction_complete(root, target, seed):
                archive.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
                os.replace(directory, archive / f"{target}_seed{seed}_{stamp}")
        shared = root / target / "shared"
        if shared.exists() and not seed_one_complete:
            archive.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
            os.replace(shared, archive / f"{target}_shared_{stamp}")


def predictions_finished(root: Path) -> bool:
    return all(
        prediction_complete(root, target, seed)
        for target in TARGETS
        for seed in SEEDS
    )


def run_prediction_wave(args: argparse.Namespace, state_root: Path) -> None:
    groups = (("0", "K562,hepg2"), ("1", "RPE1,jurkat"))
    devices = [item.strip() for item in args.cuda_devices.split(",") if item.strip()]
    if len(devices) != 2:
        raise PipelineFailure("E205 post-training prediction requires exactly two GPUs")
    groups = tuple((devices[index], targets) for index, (_, targets) in enumerate(groups))
    script = args.repo / "tools/scripts/run_e205_prediction_resume.sh"
    for attempt in range(1, args.prediction_attempts + 1):
        if predictions_finished(args.prediction_root):
            return
        archive_incomplete_predictions(args.prediction_root, state_root)
        processes = []
        for device, targets in groups:
            target_names = targets.split(",")
            if all(
                prediction_complete(args.prediction_root, target, seed)
                for target in target_names
                for seed in SEEDS
            ):
                continue
            log = state_root / "logs" / f"prediction_gpu{device}_attempt{attempt}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8")
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": device,
                    "SAFECONF_REPO": str(args.repo),
                    "PYTHON_BIN": str(args.python),
                    "TXPERT_REPO": str(args.txpert_repo),
                    "DATA_ROOT": str(args.data_root),
                    "PRED_ROOT": str(args.prediction_root),
                    "TARGETS_CSV": targets,
                }
            )
            process = subprocess.Popen(
                ["bash", str(script)],
                cwd=args.repo,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((process, handle, log))
        failures = []
        for process, handle, log in processes:
            code = process.wait()
            handle.close()
            if code != 0:
                failures.append(str(log))
        if not failures and predictions_finished(args.prediction_root):
            return
    missing = [
        f"{target}/seed_{seed}"
        for target in TARGETS
        for seed in SEEDS
        if not prediction_complete(args.prediction_root, target, seed)
    ]
    raise PipelineFailure(f"sealed prediction attempts exhausted: {missing}")


def write_authorization(repo: Path, risk_commit: str, path: Path) -> None:
    e205 = repo / "docs/实验结果/E205_cross_family_disagreement_20260830"
    risk_status = e205 / "E205_PRETRUTH_RISK_STATUS.json"
    risk_table = e205 / "tables/E205_PRETRUTH_RISK_FEATURES.csv"
    e201_final = (
        repo
        / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
        / "formal_core_evaluation/E201_CORE_FINAL_STATUS.json"
    )
    payload = {
        "experiment": "E205_cross_family_exphormer",
        "stage": "TARGET_TRUTH_REUSE_AUTHORIZATION",
        "status": "AUTHORIZED",
        "authorized_at": now(),
        "allow_released_e201_target_centroids": True,
        "risk_sealed_and_pushed_before_authorization": True,
        "risk_seal_commit": risk_commit,
        "pretruth_risk_status_sha256": sha256_file(risk_status),
        "pretruth_risk_table_sha256": sha256_file(risk_table),
        "e201_core_final_status_sha256": sha256_file(e201_final),
        "scope": "reuse previously released E201 target centroids only after E205 pretruth seal",
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in payload.items() if key != "authorized_at"}
        existing_comparable = {
            key: value for key, value in existing.items() if key != "authorized_at"
        }
        if existing_comparable != comparable:
            raise PipelineFailure("existing target-truth authorization differs")
        return
    atomic_json(path, payload)


def run(args: argparse.Namespace) -> dict:
    if args.poll_seconds < 10:
        raise PipelineFailure("poll interval must be at least 10 seconds")
    if not 1 <= args.prediction_attempts <= 3:
        raise PipelineFailure("prediction attempts must be between 1 and 3")
    args.repo = args.repo.resolve()
    args.data_root = args.data_root.resolve()
    args.txpert_repo = args.txpert_repo.resolve()
    args.runs_root = args.runs_root.resolve()
    args.prediction_root = args.prediction_root.resolve()
    args.queue_status = args.queue_status.resolve()
    state_root = args.state_root.resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (state_root / "E205_POSTTRAINING_SUPERVISOR.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise PipelineFailure("another E205 post-training supervisor holds the lock") from exc
    status_path = state_root / "E205_POSTTRAINING_SUPERVISOR_STATUS.json"
    status: dict[str, object] = {
        "experiment": "E205_cross_family_exphormer",
        "stage": "WAIT_TRAINING",
        "status": "RUNNING",
        "started_at": now(),
        "target_truth_access": "NOT_AUTHORIZED",
        "expected_branch": args.expected_branch,
    }
    atomic_json(status_path, status)
    try:
        ensure_release_branch(args.repo, args.expected_branch)
        while True:
            gate, reason = training_gate(args.queue_status)
            status.update({"last_training_check_at": now(), "training_reason": reason})
            atomic_json(status_path, status)
            if gate == "GO":
                break
            if gate == "FAIL":
                raise PipelineFailure(reason)
            time.sleep(args.poll_seconds)

        e205 = args.repo / "docs/实验结果/E205_cross_family_disagreement_20260830"
        seal_json = e205 / "E205_EXPHORMER_FAMILY_SEAL.json"
        seal_csv = e205 / "tables/E205_EXPHORMER_CHECKPOINTS.csv"
        if not seal_json.exists() and not seal_csv.exists():
            status["stage"] = "CHECKPOINT_SEAL"
            atomic_json(status_path, status)
            run_checked(
                [
                    str(args.python),
                    str(args.repo / "tools/scripts/seal_e205_exphormer_checkpoint_family.py"),
                    "--runs-root", str(args.runs_root),
                    "--data-root", str(args.data_root),
                    "--output-json", str(seal_json),
                    "--output-csv", str(seal_csv),
                ],
                args.repo,
                state_root / "logs/checkpoint_seal.log",
            )
        if not seal_json.is_file() or not seal_csv.is_file():
            raise PipelineFailure("checkpoint family seal is incomplete")
        seal_commit = commit_and_push(
            args.repo,
            [seal_json, seal_csv],
            "experiment: seal E205 Exphormer checkpoint family",
        )
        status.update({"checkpoint_seal_commit": seal_commit, "stage": "SEALED_PREDICTION"})
        atomic_json(status_path, status)

        run_prediction_wave(args, state_root)
        if not predictions_finished(args.prediction_root):
            raise PipelineFailure("prediction wave returned without 16 complete predictions")

        risk_table = e205 / "tables/E205_PRETRUTH_RISK_FEATURES.csv"
        risk_status = e205 / "E205_PRETRUTH_RISK_STATUS.json"
        vectors = args.data_root / "txpert_official_20260802/e205/pretruth_vectors"
        if not risk_table.exists() and not risk_status.exists() and not vectors.exists():
            status["stage"] = "PRETRUTH_RISK_SEAL"
            atomic_json(status_path, status)
            run_checked(
                [
                    str(args.python),
                    str(args.repo / "tools/scripts/run_e205_pretruth_risk_features.py"),
                    "--data-root", str(args.data_root),
                    "--family-seal", str(seal_json),
                    "--prediction-root", str(args.prediction_root),
                    "--risk-table", str(risk_table),
                    "--risk-status", str(risk_status),
                    "--vector-output-dir", str(vectors),
                ],
                args.repo,
                state_root / "logs/pretruth_risk.log",
            )
        if not risk_table.is_file() or not risk_status.is_file() or not vectors.is_dir():
            raise PipelineFailure("pretruth risk seal is incomplete")
        risk_commit = commit_and_push(
            args.repo,
            [risk_table, risk_status],
            "experiment: seal E205 pretruth risk and certificate inputs",
        )
        status.update(
            {
                "pretruth_risk_commit": risk_commit,
                "stage": "TARGET_TRUTH_AUTHORIZATION",
                "target_truth_access": "PENDING_SEPARATE_AUTHORIZATION_COMMIT",
            }
        )
        atomic_json(status_path, status)

        authorization = e205 / "E205_TARGET_TRUTH_REUSE_AUTHORIZATION.json"
        write_authorization(args.repo, risk_commit, authorization)
        authorization_commit = commit_and_push(
            args.repo,
            [authorization],
            "experiment: authorize E205 evaluation after pretruth seal",
        )
        status.update(
            {
                "authorization_commit": authorization_commit,
                "stage": "FORMAL_EVALUATION",
                "target_truth_access": "AUTHORIZED_AFTER_PRETRUTH_SEAL",
            }
        )
        atomic_json(status_path, status)

        evaluation = e205 / "formal_evaluation"
        evaluation_status = evaluation / "E205_FORMAL_EVALUATION_STATUS.json"
        if evaluation.exists() and not evaluation_status.is_file():
            raise PipelineFailure("partial formal evaluation directory already exists")
        if not evaluation_status.is_file():
            run_checked(
                [
                    str(args.python),
                    str(args.repo / "tools/scripts/run_e205_formal_evaluation.py"),
                    "--data-root", str(args.data_root),
                    "--risk-table", str(risk_table),
                    "--risk-status", str(risk_status),
                    "--release-authorization", str(authorization),
                    "--output-dir", str(evaluation),
                    "--n-bootstrap", "5000",
                ],
                args.repo,
                state_root / "logs/formal_evaluation.log",
            )
        evaluated_status = json.loads(evaluation_status.read_text(encoding="utf-8"))
        if evaluated_status.get("execution_status") != "PASS":
            raise PipelineFailure("formal evaluation status is not PASS")

        report = evaluation / "REPORT.md"
        figure = evaluation / "E205_RESULT_OVERVIEW.svg"
        if not report.exists() and not figure.exists():
            run_checked(
                [
                    str(args.python),
                    str(args.repo / "tools/scripts/summarize_e205_formal_results.py"),
                    "--evaluation-dir", str(evaluation),
                    "--output-report", str(report),
                    "--output-figure", str(figure),
                ],
                args.repo,
                state_root / "logs/result_summary.log",
            )
        if not report.is_file() or not figure.is_file():
            raise PipelineFailure("formal result summary is incomplete")
        result_commit = commit_and_push(
            args.repo,
            [evaluation],
            "experiment: publish audited E205 cross-architecture results",
        )
        status.update(
            {
                "result_commit": result_commit,
                "stage": "E208_H5_SMOKE",
                "formal_result": {
                    "execution_status": evaluated_status.get("execution_status"),
                    "primary_increment_status": evaluated_status.get("primary_increment_status"),
                    "registered_family_certificate_status": evaluated_status.get("registered_family_certificate_status"),
                },
            }
        )
        atomic_json(status_path, status)

        e208_status_path = (
            args.e208_output_root.resolve()
            / "E208_AFTER_E205_SUPERVISOR_STATUS.json"
        )
        e208_done = False
        if e208_status_path.is_file():
            e208_state = json.loads(e208_status_path.read_text(encoding="utf-8"))
            e208_done = str(e208_state.get("status", "")).startswith("SMOKE_PASS")
        if not e208_done:
            run_checked(
                [
                    str(args.perturbench_python.expanduser().absolute()),
                    str(args.repo / "tools/scripts/run_e208_after_e205_smoke_supervisor.py"),
                    "--repo", str(args.repo),
                    "--perturbench-repo", str(args.perturbench_repo.resolve()),
                    "--python", str(args.perturbench_python.expanduser().absolute()),
                    "--data-dir", str(args.e208_data_dir.resolve()),
                    "--e205-queue-status", str(args.queue_status),
                    "--output-root", str(args.e208_output_root.resolve()),
                    "--cuda-device", "0",
                    "--poll-seconds", "60",
                    "--min-free-mb", "22000",
                ],
                args.repo,
                state_root / "logs/e208_h5_smoke.log",
            )
        status.update({"stage": "COMPLETE", "status": "COMPLETE", "finished_at": now()})
        atomic_json(status_path, status)
        return status
    except Exception as exc:
        status.update(
            {
                "status": "BLOCKED",
                "blocked_at": now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        atomic_json(status_path, status)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
