#!/usr/bin/env python3
"""Wait for E60 seed runs, package with the latest audit code, and push.

This is intentionally conservative: it exits on a failed seed and never
creates a result commit unless all three recorded GEARS runs report success.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "实验结果" / "E60_gears_fixed_panel_formal_20260711"
SEEDS = (11, 22, 33)
LOG = OUT / "E60_AUTOMATIC_FINALIZER.log"


def log(message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def status(seed: int) -> dict | None:
    path = OUT / "raw_gears" / f"seed_{seed}" / "E60_SEED_STATUS.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status_read_error": repr(exc)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--poll-seconds", type=int, default=60)
    args = p.parse_args()
    if not 10 <= args.poll_seconds <= 60:
        raise ValueError("--poll-seconds must be between 10 and 60")
    log("watching E60 GEARS seed status")
    while True:
        rows = {seed: status(seed) for seed in SEEDS}
        failures = {seed: row for seed, row in rows.items() if row and (row.get("returncode") != 0 or row.get("gears_status", {}).get("status") == "failed")}
        if failures:
            log("seed failure detected; finalizer exits without packaging: " + json.dumps(failures, ensure_ascii=False))
            raise SystemExit(1)
        complete = all(row and row.get("returncode") == 0 and row.get("gears_status", {}).get("status") == "ok" for row in rows.values())
        if complete:
            break
        present = [str(seed) for seed, row in rows.items() if row]
        log("waiting; completed status files=" + ",".join(present or ["none"]))
        time.sleep(args.poll_seconds)

    log("all seed runs succeeded; repackaging E60 with current code")
    subprocess.run([sys.executable, "tools/scripts/run_e60_gears_fixed_panel_formal.py", "--mode", "package", "--n-test", "24", "--n-boot", "2000"], cwd=ROOT, check=True)
    stage = [
        "docs/实验结果/E60_gears_fixed_panel_formal_20260711/README_先看这个.md",
        "docs/实验结果/E60_gears_fixed_panel_formal_20260711/RUN_STATUS.json",
        "docs/实验结果/E60_gears_fixed_panel_formal_20260711/tables",
        "docs/实验结果/E60_gears_fixed_panel_formal_20260711/figures",
        "docs/实验结果/E60_gears_fixed_panel_formal_20260711/reports",
    ]
    subprocess.run(["git", "add", *stage], cwd=ROOT, check=True)
    cached = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if cached.returncode == 0:
        log("E60 outputs already staged or unchanged; no commit needed")
        return
    if cached.returncode != 1:
        raise RuntimeError("git diff --cached failed")
    subprocess.run(["git", "commit", "-m", "experiments: finalize GEARS fixed-task ensemble audit"], cwd=ROOT, check=True)
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    for remote, refs in (("github", [branch, "main", "master"]), ("origin", [branch, "master"])):
        for ref in refs:
            subprocess.run(["git", "push", remote, f"HEAD:{ref}"], cwd=ROOT, check=True)
    log("E60 packaged, committed, and pushed to GitHub and Gitee")


if __name__ == "__main__":
    main()
