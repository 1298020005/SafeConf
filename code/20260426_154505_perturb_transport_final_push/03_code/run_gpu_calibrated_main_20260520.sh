#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
OUT="$ROOT/39_gpu_calibrated_main_20260520"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=1 /home/yyf/.conda/envs/scgpt_env/bin/python -u "$ROOT/03_code/run_deep_gpu_transport.py" \
  --root "$OUT" \
  --atlas-root "/home/yyf/datasets/singlecell_perturbation_atlas" \
  --studies "KaggleCrossCell,Haber,Parekh,McFarland,KaggleCrossPatient" \
  --seeds "11,22,33,44,55" \
  --max-datasets 4 \
  --n-genes 2600 \
  --n-programs 128 \
  --split-per-type 2 \
  --hidden 1536 \
  --epochs 260 \
  --patience 55 \
  --batch-size 128 \
  --lr 6e-4 \
  --dropout 0.16 \
  --default-blend 0.25 \
  --mc-samples 12 \
  | tee "$LOG_DIR/run_gpu_calibrated_main.log"
