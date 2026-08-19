#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
OUT="$ROOT/53_gpu_policy_fix_external_20260520"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0 /home/yyf/.conda/envs/scgpt_env/bin/python -u "$ROOT/03_code/run_deep_gpu_transport.py" \
  --root "$OUT" \
  --atlas-root "/home/yyf/datasets/singlecell_perturbation_atlas" \
  --studies "TianKampmann2019,PapalexiSatija2021,Frangieh,SrivatsanTrapnell2020" \
  --seeds "5201,5211,5221,5231,5241" \
  --max-datasets 4 \
  --n-genes 3000 \
  --n-programs 192 \
  --split-per-type 3 \
  --inner-bank-mode pca \
  --hidden 1280 \
  --epochs 240 \
  --patience 50 \
  --batch-size 128 \
  --lr 5e-4 \
  --dropout 0.18 \
  --default-blend 0.25 \
  --mc-samples 10 \
  --rank-loss-weight 0.60 \
  --cosine-loss-weight 0.45 \
  --residual-loss-weight 0.14 \
  --sign-loss-weight 0.07 \
  --blend-objective effect \
  --v2-expert-blends "0,0.05,0.1,0.18,0.25,0.35,0.5,0.75,1.0" \
  --expert-min-gain 0.0 \
  --graft-blend-grid "0,0.1,0.18,0.25,0.35,0.5,0.75" \
  --graft-topk-grid "20,35,50,80,120" \
  --graft-min-gain 0.0 \
  2>&1 | tee "$LOG_DIR/run_gpu_external.log"

/home/yyf/.conda/envs/scgpt_env/bin/python -u "$ROOT/03_code/evaluate_q1_readiness.py" \
  --results-dir "$OUT/results" \
  --primary-model DeepCalibratedSafeTransport \
  --write-md \
  2>&1 | tee "$LOG_DIR/q1_readiness_deep.log" || true
