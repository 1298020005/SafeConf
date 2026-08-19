#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="$ROOT/outputs/confidence_task_mvp_v2_1"
mkdir -p "$OUT/logs"

cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/yyf/.conda/envs/scgpt_env/bin/python}"

{
  echo "[run] $(date '+%F %T') starting confidence_task MVP v2.1"
  echo "[run] ROOT=$ROOT"
  echo "[run] OUT=$OUT"
  "$PYTHON_BIN" confidence_task/run_confidence_mvp_v2_1.py \
    --project-root "$ROOT" \
    --out-dir "$OUT" \
    "$@"
  echo "[run] $(date '+%F %T') completed confidence_task MVP v2.1"
} 2>&1 | tee "$OUT/logs/run_mvp_v2_1.log"
