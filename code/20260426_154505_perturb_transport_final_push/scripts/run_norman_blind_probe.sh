#!/usr/bin/env bash
# Phase 3 探针：仅 Norman，冻结 protocol v0.2，不改公式
set -euo pipefail
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="$ROOT/outputs/confidence_task_norman_blind_probe"
PKG_OUT="$ROOT/outputs/benchmark_norman_blind_probe"
PYTHON="${PYTHON_BIN:-/home/yyf/.conda/envs/scgpt_env/bin/python}"
export PYTHONPATH="$ROOT:$PYTHONPATH"

mkdir -p "$OUT/logs" "$PKG_OUT/logs"
cd "$ROOT"

echo "[$(date '+%F %T')] Norman blind probe: build records (v2_1 pipeline, Norman only)"
"$PYTHON" scripts/run_norman_blind_probe.py \
  --out-dir "$OUT" \
  2>&1 | tee "$OUT/logs/norman_pipeline.log"

echo "[$(date '+%F %T')] Norman blind probe: protocol v0.2 scoring (frozen)"
"$PYTHON" -m safetrans_confidence.cli.run_benchmark \
  --input-dir "$OUT" \
  --out-dir "$PKG_OUT" \
  2>&1 | tee "$PKG_OUT/logs/scoring.log"

echo "[$(date '+%F %T')] DONE. See $PKG_OUT/RUN_REPORT.md"
