#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="/home/yyf/.conda/envs/scgpt_env/bin/python"
OUT="$ROOT/outputs/gears_prediction_records_formal"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "===== GEARS per-prediction formal run started $(date '+%F %T') =====" | tee -a "$LOG_DIR/run.log"

for dataset in norman adamson dixit; do
  for seed in 1 2 3; do
    echo "===== dataset=$dataset seed=$seed $(date '+%F %T') =====" | tee -a "$LOG_DIR/run.log"
    "$PY" -m safetrans_confidence.cli.run_gears_prediction_records \
      --dataset "$dataset" \
      --seed "$seed" \
      --epochs 8 \
      --max-genes 6000 \
      --hidden-size 48 \
      --decoder-hidden-size 16 \
      --num-similar-genes 10 \
      --device cuda:0 \
      --out-dir "$OUT" 2>&1 | tee -a "$LOG_DIR/${dataset}_seed${seed}.log"
  done
done

"$PY" -m safetrans_confidence.cli.run_gears_confidence_eval \
  --input-root "$OUT" \
  --out-dir "$ROOT/outputs/gears_confidence_eval_formal" 2>&1 | tee -a "$LOG_DIR/gears_confidence_eval.log"

echo "===== GEARS per-prediction formal run finished $(date '+%F %T') =====" | tee -a "$LOG_DIR/run.log"
