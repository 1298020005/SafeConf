#!/usr/bin/env bash
# 小型 blind 扩展：KaggleCrossPatient（chem_robust，未参与 v0.2 调参的三数据集之一）
set -euo pipefail
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="$ROOT/outputs/confidence_task_kcp_blind_probe"
PKG_OUT="$ROOT/outputs/benchmark_kcp_blind_probe"
PYTHON="${PYTHON_BIN:-/home/yyf/.conda/envs/scgpt_env/bin/python}"
export PYTHONPATH="$ROOT:$PYTHONPATH"
mkdir -p "$OUT/logs" "$PKG_OUT/logs"
cd "$ROOT"

echo "[$(date '+%F %T')] KCP blind probe start"
"$PYTHON" scripts/run_kcp_blind_probe.py --out-dir "$OUT" 2>&1 | tee "$OUT/logs/pipeline.log"
"$PYTHON" -m safetrans_confidence.cli.run_benchmark --input-dir "$OUT" --out-dir "$PKG_OUT" 2>&1 | tee "$PKG_OUT/logs/scoring.log"
echo "[$(date '+%F %T')] DONE → $PKG_OUT/RUN_REPORT.md"
