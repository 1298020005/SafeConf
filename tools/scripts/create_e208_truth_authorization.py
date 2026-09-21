#!/usr/bin/env python3
"""Create E208 truth authorization only after the pretruth seal is on both remotes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


class AuthorizationFailure(RuntimeError):
    """The pretruth seal has not satisfied the remote persistence contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *arguments: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=text)


def repository_path(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as error:
        raise AuthorizationFailure(f"sealed file is outside repository: {path}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--risk-table", type=Path, required=True)
    parser.add_argument("--risk-status", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-ref", required=True)
    args = parser.parse_args()
    for name in ("repo", "risk_table", "risk_status", "thresholds", "output"):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    if not args.remote_ref.startswith("refs/heads/"):
        raise AuthorizationFailure("remote-ref must be refs/heads/<branch>")
    if args.output.exists():
        raise AuthorizationFailure("authorization output already exists")
    if git(args.repo, "status", "--porcelain", "--untracked-files=all").strip():
        raise AuthorizationFailure("repository must be clean before authorization is created")

    status = json.loads(args.risk_status.read_text(encoding="utf-8"))
    risk_hash = sha256_file(args.risk_table)
    status_hash = sha256_file(args.risk_status)
    thresholds_hash = sha256_file(args.thresholds)
    if (
        status.get("experiment") != "E208_jiang24_external_confirmation"
        or status.get("status") != "PASS_AWAITING_REMOTE_SEAL"
        or int(status.get("n_tasks", -1)) != 224
        or int(status.get("test_perturbed_expression_rows_read", -1)) != 0
        or status.get("risk_table", {}).get("sha256") != risk_hash
        or status.get("family_thresholds", {}).get("sha256") != thresholds_hash
    ):
        raise AuthorizationFailure("pretruth risk status failed its frozen gates")

    paths = {
        "risk_table_repo_path": repository_path(args.repo, args.risk_table),
        "risk_status_repo_path": repository_path(args.repo, args.risk_status),
        "thresholds_repo_path": repository_path(args.repo, args.thresholds),
    }
    head = git(args.repo, "rev-parse", "HEAD").strip()
    for label, path, expected in (
        ("risk table", paths["risk_table_repo_path"], risk_hash),
        ("risk status", paths["risk_status_repo_path"], status_hash),
        ("thresholds", paths["thresholds_repo_path"], thresholds_hash),
    ):
        try:
            sealed = git(args.repo, "show", f"{head}:{path}", text=False)
        except subprocess.CalledProcessError as error:
            raise AuthorizationFailure(f"{label} is not committed at HEAD") from error
        if hashlib.sha256(sealed).hexdigest() != expected:
            raise AuthorizationFailure(f"{label} differs from the committed pretruth copy")
    for remote in ("origin", "github"):
        values = git(args.repo, "ls-remote", remote, args.remote_ref).strip().split()
        observed = values[0] if values else ""
        if observed != head:
            raise AuthorizationFailure(f"{remote} does not contain the pretruth HEAD")

    authorization = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D2_TEST_TRUTH_AUTHORIZATION",
        "status": "AUTHORIZED",
        "authorized_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pretruth_git_commit": head,
        "remote_ref": args.remote_ref,
        **paths,
        "risk_table_sha256": risk_hash,
        "risk_status_sha256": status_hash,
        "thresholds_sha256": thresholds_hash,
        "authorized_action": "run the frozen one-shot E208 formal evaluation exactly once",
        "test_perturbed_expression_rows_read_before_authorization": 0,
        "next_required_gate": "commit and push this authorization to origin and github before evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(authorization, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(authorization, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
