#!/usr/bin/env python3
"""Finish E208 after training: audit, dual-remote seal, authorize, evaluate, audit, persist."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


PINNED_SCRIPT_HASHES = {
    "tools/scripts/run_e208_formal_evaluation.py": "64d15009b30d6dad6507319a8dfcbc57f3ee4c7481c555afe676e9a3ea0059ce",
    "tools/scripts/create_e208_truth_authorization.py": "a4e43e85971e6df4408e14321d01a707b5ce6bed973a0381a20ffb03fd5f72d8",
    "tools/scripts/audit_e208_pretruth_risk_seal.py": "5ff77855eaea4e1e48772ff5c8b568ee47dc005855ded265480d530b5bc67df7",
    "tools/scripts/audit_e208_formal_results.py": "282cc83f1a4f8cb7bbfb39074b0ab792acf93d7e547840ed513f24bf25d4052f",
}
PROGENY_SHA256 = "a0454939734139e85a3c0c674ca5532e64f8b21eb464e88339fccec01bc29188"
GENE_AXIS_SHA256 = "ca36a3d678ea5ecccccd46a658604988262a69dee6e3eacb92db8638f9b19728"


class FinalizerFailure(RuntimeError):
    """A mandatory audit, remote persistence, authorization, or result gate failed."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], *, cwd: Path, log: Path, capture: bool = False) -> str:
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== {now()} RUN {command} =====\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE if capture else handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if capture:
            handle.write(completed.stdout)
    if completed.returncode:
        raise FinalizerFailure(f"command exited {completed.returncode}: {command}")
    return completed.stdout.strip() if capture else ""


def git(repo: Path, log: Path, *arguments: str, capture: bool = False) -> str:
    return run(["git", *arguments], cwd=repo, log=log, capture=capture)


def assert_frozen_code(repo: Path) -> None:
    for relative, expected in PINNED_SCRIPT_HASHES.items():
        if sha256_file(repo / relative) != expected:
            raise FinalizerFailure(f"frozen E208 script changed: {relative}")


def assert_clean_and_aligned(repo: Path, branch_ref: str, log: Path) -> str:
    if git(repo, log, "status", "--porcelain", "--untracked-files=all", capture=True):
        raise FinalizerFailure("finalizer requires a clean worktree")
    head = git(repo, log, "rev-parse", "HEAD", capture=True)
    for remote in ("origin", "github"):
        remote_output = git(repo, log, "ls-remote", remote, branch_ref, capture=True).split()
        if not remote_output or remote_output[0] != head:
            raise FinalizerFailure(f"{remote} branch head differs from local HEAD")
    return head


def commit_and_push(repo: Path, branch: str, branch_ref: str, message: str, paths: list[Path], log: Path) -> str:
    for path in paths:
        git(repo, log, "add", "--", str(path.relative_to(repo)))
    git(repo, log, "commit", "-m", message)
    git(repo, log, "push", "origin", f"HEAD:{branch_ref}")
    git(repo, log, "push", "github", f"HEAD:{branch_ref}")
    head = assert_clean_and_aligned(repo, branch_ref, log)
    if git(repo, log, "symbolic-ref", "--short", "HEAD", capture=True) != branch:
        raise FinalizerFailure("active branch changed during finalization")
    return head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--validation-cache-dir", type=Path, required=True)
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--risk-queue-dir", type=Path, required=True)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--progeny", type=Path, required=True)
    parser.add_argument("--gene-axis", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--repo-result-dir", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    for name in (
        "repo", "python", "training_root", "prediction_root", "validation_cache_dir",
        "source_cache_dir", "risk_queue_dir", "h5ad", "split", "progeny", "gene_axis",
        "runtime_root", "repo_result_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    if not 10 <= args.poll_seconds <= 600:
        raise FinalizerFailure("poll interval must be 10--600 seconds")
    if args.repo_result_dir.resolve().is_relative_to(args.repo.resolve()) is False:
        raise FinalizerFailure("repository result directory must live inside repository")
    branch_ref = f"refs/heads/{args.branch}"
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    log = args.runtime_root / "E208_FINALIZER.log"
    status_path = args.runtime_root / "E208_FINALIZER_STATUS.json"
    lock = (args.runtime_root / "E208_FINALIZER.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise FinalizerFailure("another E208 finalizer holds the lock") from error
    status = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_D2_AUTOMATED_SEAL_AUTHORIZE_EVALUATE_AUDIT",
        "status": "WAITING_FOR_PRETRUTH_RISK_SEAL",
        "started_at": now(),
        "pid": os.getpid(),
        "branch": args.branch,
        "test_perturbed_expression_rows_read": 0,
    }
    atomic_json(status_path, status)
    try:
        assert_frozen_code(args.repo)
        if sha256_file(args.progeny) != PROGENY_SHA256 or sha256_file(args.gene_axis) != GENE_AXIS_SHA256:
            raise FinalizerFailure("frozen output-space resource changed")
        while True:
            queue_path = args.risk_queue_dir / "E208_RISK_SEAL_QUEUE_STATUS.json"
            ready = False
            if queue_path.is_file():
                queue = read_json(queue_path)
                ready = (
                    queue.get("status") == "PRETRUTH_RISK_SEAL_READY_FOR_GIT"
                    and int(queue.get("test_perturbed_expression_rows_read", -1)) == 0
                )
                if str(queue.get("status", "")).endswith("FAILED"):
                    raise FinalizerFailure(f"risk seal queue failed: {queue}")
            training = read_json(args.training_root / "E208_FORMAL_QUEUE_STATUS.json")
            status.update({
                "updated_at": now(),
                "training_status": training.get("status"),
                "training_completed": training.get("completed", 0),
                "risk_seal_ready": ready,
            })
            atomic_json(status_path, status)
            if ready:
                break
            time.sleep(args.poll_seconds)

        assert_frozen_code(args.repo)
        assert_clean_and_aligned(args.repo, branch_ref, log)
        runtime_seal = args.risk_queue_dir / "seal"
        runtime_audit = args.runtime_root / "pretruth_audit"
        run([
            str(args.python), str(args.repo / "tools/scripts/audit_e208_pretruth_risk_seal.py"),
            "--prediction-root", str(args.prediction_root),
            "--validation-cache-dir", str(args.validation_cache_dir),
            "--source-cache-dir", str(args.source_cache_dir),
            "--seal-dir", str(runtime_seal),
            "--output-dir", str(runtime_audit),
        ], cwd=args.repo, log=log)
        audit_state = read_json(runtime_audit / "E208_PRETRUTH_RISK_SEAL_INDEPENDENT_AUDIT.json")
        if audit_state.get("status") != "PASS" or int(audit_state.get("test_perturbed_expression_rows_read", -1)) != 0:
            raise FinalizerFailure("independent pretruth risk audit did not pass")

        repo_pretruth = args.repo_result_dir / "pretruth_seal"
        if repo_pretruth.exists():
            raise FinalizerFailure("repository pretruth seal directory already exists")
        shutil.copytree(runtime_seal, repo_pretruth)
        shutil.copy2(
            runtime_audit / "E208_PRETRUTH_RISK_SEAL_INDEPENDENT_AUDIT.json",
            repo_pretruth / "E208_PRETRUTH_RISK_SEAL_INDEPENDENT_AUDIT.json",
        )
        pretruth_commit = commit_and_push(
            args.repo, args.branch, branch_ref,
            "e208: seal and independently audit external pretruth risks",
            [repo_pretruth], log,
        )
        status.update({"status": "PRETRUTH_SEALED_ON_BOTH_REMOTES", "pretruth_git_commit": pretruth_commit, "updated_at": now()})
        atomic_json(status_path, status)

        authorization = args.repo_result_dir / "E208_TEST_TRUTH_AUTHORIZATION.json"
        run([
            str(args.python), str(args.repo / "tools/scripts/create_e208_truth_authorization.py"),
            "--repo", str(args.repo),
            "--risk-table", str(repo_pretruth / "E208_PRETRUTH_RISK_FEATURES.csv"),
            "--risk-status", str(repo_pretruth / "E208_PRETRUTH_RISK_SEAL_STATUS.json"),
            "--thresholds", str(repo_pretruth / "E208_REGISTERED_FAMILY_THRESHOLDS.csv"),
            "--output", str(authorization),
            "--remote-ref", branch_ref,
        ], cwd=args.repo, log=log)
        authorization_commit = commit_and_push(
            args.repo, args.branch, branch_ref,
            "e208: authorize one-shot external truth evaluation",
            [authorization], log,
        )
        status.update({"status": "TRUTH_AUTHORIZED_ON_BOTH_REMOTES", "authorization_git_commit": authorization_commit, "updated_at": now()})
        atomic_json(status_path, status)

        formal_output = args.runtime_root / "formal_evaluation"
        status.update({"status": "RUNNING_ONE_SHOT_FORMAL_EVALUATION", "updated_at": now()})
        atomic_json(status_path, status)
        run([
            str(args.python), str(args.repo / "tools/scripts/run_e208_formal_evaluation.py"),
            "--repo", str(args.repo), "--h5ad", str(args.h5ad), "--split", str(args.split),
            "--prediction-root", str(args.prediction_root),
            "--risk-table", str(repo_pretruth / "E208_PRETRUTH_RISK_FEATURES.csv"),
            "--risk-status", str(repo_pretruth / "E208_PRETRUTH_RISK_SEAL_STATUS.json"),
            "--thresholds", str(repo_pretruth / "E208_REGISTERED_FAMILY_THRESHOLDS.csv"),
            "--authorization", str(authorization), "--progeny", str(args.progeny),
            "--gene-axis", str(args.gene_axis), "--output-dir", str(formal_output),
        ], cwd=args.repo, log=log)
        formal_state = read_json(formal_output / "E208_FORMAL_EVALUATION_STATUS.json")
        if formal_state.get("status") != "COMPLETE" or int(formal_state.get("n_truth_cells_read", -1)) != 214_901:
            raise FinalizerFailure("one-shot formal evaluation did not complete")

        result_audit = args.runtime_root / "formal_result_audit"
        status.update({"status": "RUNNING_INDEPENDENT_FORMAL_RESULT_AUDIT", "updated_at": now(), "test_perturbed_expression_rows_read": 214_901})
        atomic_json(status_path, status)
        run([
            str(args.python), str(args.repo / "tools/scripts/audit_e208_formal_results.py"),
            "--h5ad", str(args.h5ad), "--split", str(args.split),
            "--prediction-root", str(args.prediction_root),
            "--risk-table", str(repo_pretruth / "E208_PRETRUTH_RISK_FEATURES.csv"),
            "--progeny", str(args.progeny), "--gene-axis", str(args.gene_axis),
            "--formal-dir", str(formal_output), "--output-dir", str(result_audit),
        ], cwd=args.repo, log=log)
        result_audit_state = read_json(result_audit / "E208_FORMAL_RESULTS_INDEPENDENT_AUDIT.json")
        if result_audit_state.get("status") != "PASS" or int(result_audit_state.get("n_truth_cells_reread", -1)) != 214_901:
            raise FinalizerFailure("independent formal result audit did not pass")

        repo_formal = args.repo_result_dir / "formal_evaluation"
        if repo_formal.exists():
            raise FinalizerFailure("repository formal evaluation directory already exists")
        shutil.copytree(formal_output, repo_formal)
        shutil.copy2(
            result_audit / "E208_FORMAL_RESULTS_INDEPENDENT_AUDIT.json",
            repo_formal / "E208_FORMAL_RESULTS_INDEPENDENT_AUDIT.json",
        )
        result_commit = commit_and_push(
            args.repo, args.branch, branch_ref,
            "e208: add independently audited external confirmation results",
            [repo_formal], log,
        )
        status.update({
            "status": "COMPLETE",
            "finished_at": now(),
            "result_git_commit": result_commit,
            "test_perturbed_expression_rows_read": 214_901,
            "independent_truth_reread_rows": 214_901,
            "external_confirmation": formal_state.get("external_confirmation"),
            "all_practical_effect_gates": formal_state.get("all_practical_effect_gates"),
        })
        atomic_json(status_path, status)
    except Exception as error:
        status.update({
            "status": "FAILED",
            "failed_at": now(),
            "error_type": type(error).__name__,
            "error": str(error),
        })
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
