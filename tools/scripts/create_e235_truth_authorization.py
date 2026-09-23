#!/usr/bin/env python3
"""Authorize E235 truth access only after scores are committed on both remotes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


class AuthorizationFailure(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *arguments: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=not binary)


def relative(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as error:
        raise AuthorizationFailure(f"file must be inside repository: {path}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "score-table", "score-status", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--remote-ref", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if not args.remote_ref.startswith("refs/heads/"):
        raise AuthorizationFailure("remote-ref must be a branch ref")
    if output.exists():
        raise AuthorizationFailure("authorization already exists; refusing overwrite")
    if git(repo, "status", "--porcelain", "--untracked-files=all").strip():
        raise AuthorizationFailure("worktree must be clean before authorization")

    status = json.loads(args.score_status.read_text(encoding="utf-8"))
    scores_hash = sha256(args.score_table)
    status_hash = sha256(args.score_status)
    protocol_hash = sha256(args.protocol)
    scorer_path = repo / "tools/scripts/run_e235_pretruth_scores.py"
    evaluator_path = repo / "tools/scripts/run_e235_formal_evaluation.py"
    scorer_hash = sha256(scorer_path)
    evaluator_hash = sha256(evaluator_path)
    if (status.get("status") != "SCORES_READY_AWAITING_REMOTE_SEAL"
            or status.get("score_sha256") != scores_hash
            or status.get("n_tasks") != 224 or status.get("n_states") != 12
            or status.get("n_target_genes") != 53
            or status.get("test_perturbed_expression_rows_read") != 0
            or status.get("target_truth_access") != "NOT_AUTHORIZED"
            or status.get("registered_primary_score") != "score_M_plus_H"
            or status.get("registered_comparator") != "rank_M"):
        raise AuthorizationFailure("pretruth score status did not pass E235 gates")

    paths = {
        "score_table_repo_path": relative(repo, args.score_table),
        "score_status_repo_path": relative(repo, args.score_status),
        "protocol_repo_path": relative(repo, args.protocol),
        "scorer_repo_path": relative(repo, scorer_path),
        "evaluator_repo_path": relative(repo, evaluator_path),
    }
    hashes = {
        "score_table_sha256": scores_hash,
        "score_status_sha256": status_hash,
        "protocol_sha256": protocol_hash,
        "scorer_sha256": scorer_hash,
        "evaluator_sha256": evaluator_hash,
    }
    head = git(repo, "rev-parse", "HEAD").strip()
    for label, path in paths.items():
        expected = hashes[label.replace("_repo_path", "_sha256")]
        actual = hashlib.sha256(git(repo, "show", f"{head}:{path}", binary=True)).hexdigest()
        if actual != expected:
            raise AuthorizationFailure(f"pretruth committed {label} differs from working copy")
    for remote in ("origin", "github"):
        result = git(repo, "ls-remote", remote, args.remote_ref).strip().split()
        if not result or result[0] != head:
            raise AuthorizationFailure(f"{remote} does not contain pretruth commit {head}")

    authorization = {
        "experiment": "E235_jiang24_fixed_history_dispersion",
        "stage": "TEST_TRUTH_AUTHORIZATION",
        "status": "AUTHORIZED_AWAITING_REMOTE_PERSISTENCE",
        "authorized_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pretruth_git_commit": head,
        "remote_ref": args.remote_ref,
        **paths,
        **hashes,
        "test_perturbed_expression_rows_read_before_authorization": 0,
        "authorized_action": "one-shot evaluation of all 224 fixed Jiang24 E235 tasks",
        "next_gate": "commit and push this authorization to origin and github before reading truth",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(authorization, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(authorization, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
