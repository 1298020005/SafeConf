#!/usr/bin/env bash
# Score every recent run folder that has a summary CSV.
set -euo pipefail

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
PY="$ROOT/03_code/evaluate_q1_readiness.py"

for d in \
  "$ROOT/46_q1_cpu_push_20260520/results" \
  "$ROOT/39_gpu_calibrated_main_20260520/results" \
  "$ROOT/40_gpu_calibrated_tian_20260520/results" \
  "$ROOT/41_effect_objective_smoke_20260520/results" \
  "$ROOT/32_policy_router_soft_20260519/results" \
  "$ROOT/31_policy_full_20260519/06_full_runs"; do
  if [[ -f "$d/SAFETY_SUMMARY.csv" || -f "$d/GPU_DEEP_SUMMARY.csv" ]]; then
    echo "=== $d ==="
    python -u "$PY" --results-dir "$d" --write-md || true
  fi
done
