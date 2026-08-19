#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
OUT="$ROOT/35_gpu_deep_gpu0_20260519"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0 /home/yyf/.conda/envs/scgpt_env/bin/python -u "$ROOT/03_code/run_deep_gpu_transport.py" \
  --root "$OUT" \
  --atlas-root "/home/yyf/datasets/singlecell_perturbation_atlas" \
  --studies "TianKampmann2019" \
  --seeds "11,22,33" \
  --max-datasets 1 \
  --n-genes 2500 \
  --n-programs 128 \
  --split-per-type 2 \
  --hidden 1536 \
  --epochs 280 \
  --patience 50 \
  --batch-size 128 \
  --lr 7e-4 \
  --dropout 0.12 \
  --default-blend 0.35 \
  --mc-samples 10 \
  | tee "$LOG_DIR/run_gpu_deep.log"
