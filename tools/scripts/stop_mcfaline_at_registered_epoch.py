#!/usr/bin/env python3
"""Stop both formal McFaline runs after the registered validation epoch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def process_for(run: Path) -> int | None:
    target = str(run)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if target in cmd and "/bin/train " in cmd and "timeout " not in cmd:
            return int(proc.name)
    return None


def completed_validation_epoch(run: Path) -> int:
    path = run / "csv/version_0/metrics.csv"
    if not path.exists():
        return -1
    frame = pd.read_csv(path, usecols=["epoch", "val_loss_epoch"])
    valid = frame[frame.val_loss_epoch.notna() & frame.epoch.notna()]
    return int(valid.epoch.max()) if not valid.empty else -1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--stop-after-epoch", type=int, default=15)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    pending = set(args.run)
    while pending:
        for run in list(pending):
            epoch = completed_validation_epoch(run)
            if epoch < args.stop_after_epoch:
                continue
            pid = process_for(run)
            if pid is not None:
                os.kill(pid, signal.SIGINT)
                for _ in range(120):
                    if not Path(f"/proc/{pid}").exists():
                        break
                    time.sleep(5)
                if Path(f"/proc/{pid}").exists():
                    os.kill(pid, signal.SIGTERM)
            checkpoints = sorted((run / "checkpoints").glob("*.ckpt"))
            if len(checkpoints) != 1:
                raise RuntimeError(f"expected one retained checkpoint: {run}: {checkpoints}")
            marker = {
                "contract": "registered validation budget epoch 15",
                "completed_validation_epoch": epoch,
                "test_partition_opened": False,
                "checkpoint": str(checkpoints[0]),
                "checkpoint_sha256": sha256_file(checkpoints[0]),
                "stopped_utc": datetime.now(timezone.utc).isoformat(),
            }
            (run / "STOPPED_BY_REGISTERED_VALIDATION_BUDGET").write_text(
                json.dumps(marker, indent=2) + "\n"
            )
            pending.remove(run)
            print(json.dumps({"run": str(run), **marker}), flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
