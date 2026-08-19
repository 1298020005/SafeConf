#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
OUT="$ROOT/40_gpu_calibrated_tian_20260520"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0 /home/yyf/.conda/envs/scgpt_env/bin/python -u "$ROOT/03_code/run_deep_gpu_transport.py" \
  --root "$OUT" \
  --atlas-root "/home/yyf/datasets/singlecell_perturbation_atlas" \
  --studies "TianKampmann2019" \
  --seeds "11,22,33,44,55" \
  --max-datasets 1 \
  --n-genes 3000 \
  --n-programs 160 \
  --split-per-type 2 \
  --hidden 1792 \
  --epochs 320 \
  --patience 65 \
  --batch-size 128 \
  --lr 5e-4 \
  --dropout 0.18 \
  --default-blend 0.25 \
  --mc-samples 12 \
  | tee "$LOG_DIR/run_gpu_calibrated_tian.log"
