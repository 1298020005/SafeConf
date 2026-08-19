#!/usr/bin/env bash
# Q1 / Q2-top CPU evidence push — wide main + external, PolicySafeTrans focus.
set -euo pipefail

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
OUT="$ROOT/46_q1_cpu_push_20260520"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

source "${HOME}/.bashrc" 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate scgpt_env
fi

export PAIRDELTA_V2_BANK_MODE="pca_nmf_hvg"
export PAIRDELTA_ROUTING_MODE="hybrid"

echo "[q1-cpu] start $(date '+%F %T')" | tee "$LOG_DIR/master.log"

python -u "$ROOT/03_code/run_safety_abstention_evidence.py" \
  --root "$OUT" \
  --atlas-root "/home/yyf/datasets/singlecell_perturbation_atlas" \
  --seeds "4101,4111,4121,4131" \
  --external-seed-count 3 \
  --max-datasets 6 \
  --max-external-datasets 5 \
  --n-genes 1800 \
  --n-programs 128 \
  --split-per-type 3 \
  --eval-bank "pca_nmf_hvg" \
  --policy-routing-mode "hybrid" \
  2>&1 | tee "$LOG_DIR/run_safety.log"

python -u "$ROOT/03_code/evaluate_q1_readiness.py" \
  --results-dir "$OUT/results" \
  --write-md \
  2>&1 | tee "$LOG_DIR/q1_readiness.log"

echo "[q1-cpu] done $(date '+%F %T')" | tee -a "$LOG_DIR/master.log"
