#!/usr/bin/env bash
set -euo pipefail
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python -m safetrans_confidence.cli.run_benchmark \
  --input-dir "$ROOT/outputs/confidence_task_mvp_v2_1" \
  --out-dir "$ROOT/outputs/benchmark_protocol_v0_2_pkg"
python -m pytest safetrans_confidence/tests -q
