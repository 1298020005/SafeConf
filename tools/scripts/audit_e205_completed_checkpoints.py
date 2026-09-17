#!/usr/bin/env python3
"""Audit every currently completed E205 checkpoint before the queue drains."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_sealer(script: Path):
    spec = importlib.util.spec_from_file_location("e205_checkpoint_sealer", script)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load sealer: {script}")
    spec.loader.exec_module(module)
    return module


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    repo = args.repo.resolve()
    runs_root = args.runs_root.resolve()
    data_root = args.data_root.resolve()
    sealer = load_sealer(repo / "tools/scripts/seal_e205_exphormer_checkpoint_family.py")
    records = []
    failures = []
    for target in sealer.TARGETS:
        for seed in sealer.SEEDS:
            status_path = runs_root / target / f"seed_{seed}/E205_RUN_STATUS.json"
            if not status_path.is_file():
                continue
            try:
                state = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                failures.append(
                    {"target": target, "seed": seed, "error": f"status read: {exc}"}
                )
                continue
            if state.get("status") != "COMPLETE":
                continue
            try:
                records.append(sealer.inspect_run(runs_root, data_root, target, seed))
            except Exception as exc:
                failures.append(
                    {
                        "target": target,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    schema_hashes = sorted({record["last"]["state_schema_sha256"] for record in records})
    result = {
        "experiment": "E205_cross_family_exphormer",
        "stage": "PARTIAL_CHECKPOINT_AUDIT",
        "status": "PASS" if records and not failures and len(schema_hashes) == 1 else "FAIL",
        "audited_at": now(),
        "target_truth_access": "NOT_AUTHORIZED",
        "completed_checkpoints_audited": len(records),
        "state_schema_sha256": schema_hashes[0] if len(schema_hashes) == 1 else None,
        "failures": failures,
        "records": records,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
