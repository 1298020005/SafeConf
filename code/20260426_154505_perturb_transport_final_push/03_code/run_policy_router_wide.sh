#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
OUT="$ROOT/30_policy_router_wide_20260519"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

python -u "$ROOT/03_code/run_safety_abstention_evidence.py" \
  --root "$OUT" \
  --atlas-root "/home/yyf/datasets/singlecell_perturbation_atlas" \
  --seeds "11,22,33,44,55" \
  --max-datasets 6 \
  --max-external-datasets 4 \
  --n-genes 1600 \
  --n-programs 96 \
  --split-per-type 2 \
  | tee "$LOG_DIR/run_safety_abstention_evidence.log"
