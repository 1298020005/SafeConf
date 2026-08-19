#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
OUT="$ROOT/33_policy_router_hybrid_20260519"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

python -u "$ROOT/03_code/run_safety_abstention_evidence.py" \
  --root "$OUT" \
  --atlas-root "/home/yyf/datasets/singlecell_perturbation_atlas" \
  --seeds "11,22,33" \
  --max-datasets 3 \
  --max-external-datasets 2 \
  --n-genes 1600 \
  --n-programs 96 \
  --split-per-type 2 \
  --policy-routing-mode hybrid \
  | tee "$LOG_DIR/run_safety_abstention_evidence.log"
