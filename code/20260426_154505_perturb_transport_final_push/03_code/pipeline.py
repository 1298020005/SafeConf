from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], log: Path, cwd: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\nexit_code={proc.returncode}\n")
        return proc.returncode


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--zip-path", required=True)
    args = p.parse_args()
    root = Path(args.root)
    code = root / "03_code"
    py = sys.executable
    run([py, "run_gate.py", "--root", str(root), "--seeds", "11,22,33", "--max-datasets", "3"], root / "04_logs" / "gate.log", code)
    gate = read_json(root / "05_gate_runs" / "gate_status.json")
    if gate.get("gate_label") == "GATE_PASS":
        run([py, "run_full.py", "--root", str(root), "--seeds", "11,22,33,44,55,66,77,88,99,101", "--max-datasets", "6"], root / "04_logs" / "full.log", code)
    run([py, "finalize_handoff.py", "--root", str(root), "--archive", args.archive, "--zip-path", args.zip_path], root / "04_logs" / "finalize.log", code)


if __name__ == "__main__":
    main()
