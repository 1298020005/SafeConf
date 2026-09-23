#!/usr/bin/env python3
"""Wait for E235 scores, dual-remote seal, authorize, evaluate, audit, and persist.

No test perturbation truth is opened unless both pretruth and authorization commits
are visible at the exact same branch head on GitHub and Gitee.
"""

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

import pandas as pd


TERMINAL_FAILURES = {"BLOCKED_UPSTREAM_COMPETENCE", "BLOCKED_PREDICTION_FAILED",
                     "BLOCKED_SCORE_FAILED", "FAILED"}
SCORE_COLUMNS = {"task_id", "cell_type", "treatment", "condition", "rank_M",
                 "score_M_plus_H", "score_M_plus_D", "score_M_plus_G",
                 "score_M_plus_N", "score_M_plus_C", "score_original_five_80_20",
                 "score_random_fixed"}


class FinalizerFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run(command: list[str], *, repo: Path, log: Path) -> str:
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== {now()} {command} =====\n")
        handle.flush()
        result = subprocess.run(command, cwd=repo, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        handle.write(result.stdout)
    if result.returncode:
        raise FinalizerFailure(f"exit {result.returncode}: {command}; see {log}")
    return result.stdout.strip()


def git(repo: Path, log: Path, *arguments: str) -> str:
    return run(["git", *arguments], repo=repo, log=log)


def aligned(repo: Path, ref: str, log: Path, *, clean: bool) -> str:
    if clean and git(repo, log, "status", "--porcelain", "--untracked-files=all"):
        raise FinalizerFailure("worktree contains unexpected changes")
    head = git(repo, log, "rev-parse", "HEAD")
    for remote in ("origin", "github"):
        values = git(repo, log, "ls-remote", remote, ref).split()
        if not values or values[0] != head:
            raise FinalizerFailure(f"{remote} branch head differs from local {head}")
    return head


def commit_and_push(repo: Path, ref: str, message: str, paths: list[Path], log: Path) -> str:
    for path in paths:
        git(repo, log, "add", "--", str(path.relative_to(repo)))
    git(repo, log, "commit", "-m", message)
    for remote in ("origin", "github"):
        git(repo, log, "push", remote, f"HEAD:{ref}")
    return aligned(repo, ref, log, clean=True)


def assert_only_score_outputs_are_untracked(repo: Path, pretruth: Path, log: Path) -> None:
    entries = git(repo, log, "status", "--porcelain", "--untracked-files=all").splitlines()
    allowed = pretruth.relative_to(repo).as_posix() + "/"
    if not entries or any(not row.startswith("?? ") or not row[3:].startswith(allowed) for row in entries):
        raise FinalizerFailure("other files changed while waiting; refusing automatic seal")


def validate_pretruth(pretruth: Path, supervisor: dict) -> dict:
    score_path = pretruth / "E235_PRETRUTH_SCORES.csv"
    status_path = pretruth / "E235_PRETRUTH_SCORE_STATUS.json"
    state = read_json(status_path)
    if (not state or state.get("status") != "SCORES_READY_AWAITING_REMOTE_SEAL"
            or state.get("test_perturbed_expression_rows_read") != 0
            or state.get("target_truth_access") != "NOT_AUTHORIZED"
            or state.get("score_sha256") != sha256(score_path)
            or state.get("score_sha256") != supervisor.get("score_sha256")
            or state.get("n_tasks") != 224 or state.get("n_states") != 12
            or state.get("n_target_genes") != 53):
        raise FinalizerFailure("pretruth score/status gate failed")
    header = pd.read_csv(score_path, nrows=0).columns
    if not SCORE_COLUMNS.issubset(header):
        raise FinalizerFailure("a registered comparator is missing from score table")
    forbidden = {name for name in header if any(token in name.lower() for token in
                 ("true_", "actual_", "rmse", "mse", "observed_", "error_"))}
    if forbidden:
        raise FinalizerFailure(f"truth-like fields appeared before authorization: {forbidden}")
    scores = pd.read_csv(score_path)
    if (len(scores) != 224 or scores.task_id.nunique() != 224
            or scores.condition.nunique() != 53
            or scores.test_perturbed_expression_rows_read.ne(0).any()):
        raise FinalizerFailure("pretruth table lost tasks or reports test truth access")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "python", "supervisor-status", "prediction-root", "h5ad", "split",
                 "protocol", "repo-result-dir", "runtime-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    repo = args.repo.resolve()
    result_dir = args.repo_result_dir.resolve()
    runtime = args.runtime_dir.resolve()
    if not result_dir.is_relative_to(repo) or args.poll_seconds < 10:
        raise FinalizerFailure("invalid result directory or polling interval")
    runtime.mkdir(parents=True, exist_ok=True)
    log = runtime / "E235_FINALIZER.log"
    branch = git(repo, log, "symbolic-ref", "--short", "HEAD")
    if branch != args.branch:
        raise FinalizerFailure(f"active branch {branch} differs from requested {args.branch}")
    ref = "refs/heads/" + args.branch
    lock = (runtime / "E235_FINALIZER.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise FinalizerFailure("another E235 finalizer is running") from error
    state_path = runtime / "E235_FINALIZER_STATUS.json"
    if state_path.exists():
        raise FinalizerFailure("finalizer already started; refusing a second instance")
    state = {"status": "WAITING_FOR_PRETRUTH_SCORES", "started_at": now(),
             "test_perturbed_expression_rows_read": 0, "branch": args.branch}
    atomic_json(state_path, state)
    code_files = [repo / "tools/scripts/" / name for name in
                  ("run_e235_pretruth_scores.py", "create_e235_truth_authorization.py",
                   "run_e235_formal_evaluation.py", "audit_e235_formal_results.py")]
    frozen_hashes = {path.name: sha256(path) for path in [args.protocol, *code_files]}
    try:
        while True:
            supervisor = read_json(args.supervisor_status)
            if supervisor and supervisor.get("status") in TERMINAL_FAILURES:
                state.update(status="BLOCKED_UPSTREAM", upstream_status=supervisor["status"])
                return
            if supervisor and supervisor.get("status") == "SCORES_READY_AWAITING_REMOTE_SEAL":
                break
            state["updated_at"] = now()
            atomic_json(state_path, state)
            time.sleep(args.poll_seconds)
        for path in [args.protocol, *code_files]:
            if sha256(path) != frozen_hashes[path.name]:
                raise FinalizerFailure(f"frozen E235 code or protocol changed: {path}")
        pretruth = result_dir / "pretruth"
        validate_pretruth(pretruth, supervisor)
        assert_only_score_outputs_are_untracked(repo, pretruth, log)
        aligned(repo, ref, log, clean=False)
        pretruth_commit = commit_and_push(repo, ref, "e235: seal all 224 Jiang24 scores before test truth", [pretruth], log)
        state.update(status="PRETRUTH_SEALED_ON_BOTH_REMOTES", pretruth_git_commit=pretruth_commit,
                     updated_at=now())
        atomic_json(state_path, state)

        authorization = result_dir / "E235_TEST_TRUTH_AUTHORIZATION.json"
        run([str(args.python), str(repo / "tools/scripts/create_e235_truth_authorization.py"),
             "--repo", str(repo), "--score-table", str(pretruth / "E235_PRETRUTH_SCORES.csv"),
             "--score-status", str(pretruth / "E235_PRETRUTH_SCORE_STATUS.json"),
             "--protocol", str(args.protocol), "--output", str(authorization),
             "--remote-ref", ref], repo=repo, log=log)
        auth_commit = commit_and_push(repo, ref, "e235: authorize one-shot Jiang24 truth evaluation",
                                      [authorization], log)
        state.update(status="TRUTH_AUTHORIZED_ON_BOTH_REMOTES", authorization_git_commit=auth_commit,
                     updated_at=now())
        atomic_json(state_path, state)

        formal = runtime / "formal_evaluation"
        state.update(status="RUNNING_FORMAL_EVALUATION", updated_at=now())
        atomic_json(state_path, state)
        run([str(args.python), str(repo / "tools/scripts/run_e235_formal_evaluation.py"),
             "--repo", str(repo), "--h5ad", str(args.h5ad), "--split", str(args.split),
             "--prediction-root", str(args.prediction_root),
             "--score-table", str(pretruth / "E235_PRETRUTH_SCORES.csv"),
             "--score-status", str(pretruth / "E235_PRETRUTH_SCORE_STATUS.json"),
             "--authorization", str(authorization), "--output-dir", str(formal)], repo=repo, log=log)
        result = read_json(formal / "E235_FORMAL_EVALUATION_STATUS.json")
        if not result or result.get("status") != "COMPLETE" or result.get("n_truth_cells_read") != 214_901:
            raise FinalizerFailure("formal evaluator returned an incomplete result")
        state.update(status="RUNNING_METRIC_AUDIT", test_perturbed_expression_rows_read=214_901,
                     updated_at=now())
        atomic_json(state_path, state)
        audit = runtime / "E235_FORMAL_METRIC_AUDIT.json"
        run([str(args.python), str(repo / "tools/scripts/audit_e235_formal_results.py"),
             "--formal-dir", str(formal), "--output", str(audit)], repo=repo, log=log)
        if read_json(audit).get("status") != "PASS":
            raise FinalizerFailure("formal metric audit did not pass")
        repo_formal = result_dir / "formal_evaluation"
        if repo_formal.exists():
            raise FinalizerFailure("repository formal result already exists")
        shutil.copytree(formal, repo_formal)
        shutil.copy2(audit, repo_formal / audit.name)
        result_commit = commit_and_push(repo, ref, "e235: publish full Jiang24 result and metric audit",
                                        [repo_formal], log)
        state.update(status="COMPLETE", completed_at=now(), result_git_commit=result_commit,
                     decision=result["decision"], primary_delta_utility_20=result["primary_delta_utility_20"],
                     positive_utility_states=result["positive_utility_states"])
    except BaseException as error:
        state.update(status="FAILED", failed_at=now(), error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic_json(state_path, state)


if __name__ == "__main__":
    main()
