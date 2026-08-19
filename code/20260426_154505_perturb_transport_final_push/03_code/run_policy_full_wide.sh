#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
OUT="$ROOT/31_policy_full_20260519"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

python -u "$ROOT/03_code/run_full.py" \
  --root "$OUT" \
  --atlas-root "/home/yyf/datasets/singlecell_perturbation_atlas" \
  --seeds "11,22,33,44,55" \
  --max-datasets 6 \
  --n-genes 1600 \
  --n-programs 96 \
  --external-seed-count 3 \
  | tee "$LOG_DIR/run_full.log"
