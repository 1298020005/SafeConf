#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
OUT="$ROOT/43_gpu_effect_objective_main_20260520"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=1 /home/yyf/.conda/envs/scgpt_env/bin/python -u "$ROOT/03_code/run_deep_gpu_transport.py" \
  --root "$OUT" \
  --atlas-root "/home/yyf/datasets/singlecell_perturbation_atlas" \
  --studies "KaggleCrossCell,Haber,Parekh,McFarland,KaggleCrossPatient" \
  --seeds "11,22,33,44,55" \
  --max-datasets 5 \
  --n-genes 2600 \
  --n-programs 160 \
  --split-per-type 2 \
  --hidden 1280 \
  --epochs 220 \
  --patience 45 \
  --batch-size 128 \
  --lr 6e-4 \
  --dropout 0.16 \
  --default-blend 0.25 \
  --mc-samples 10 \
  --rank-loss-weight 0.55 \
  --cosine-loss-weight 0.45 \
  --residual-loss-weight 0.15 \
  --sign-loss-weight 0.06 \
  --blend-objective effect \
  --v2-expert-blends "0,0.05,0.1,0.18,0.25,0.35,0.5,0.75,1.0" \
  --expert-min-gain 0.0 \
  | tee "$LOG_DIR/run_gpu_effect_objective_main.log"
