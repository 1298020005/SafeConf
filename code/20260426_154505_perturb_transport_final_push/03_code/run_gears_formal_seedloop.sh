#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?dataset name required}"
CUDA_DEV="${2:?cuda device id required}"
DATA_PATH="${3:-/home/yyf/datasets/gears_formal_baselines_v2}"
ROOT_BASE="${4:-/home/yyf/codex_cout/SAFE_TRANS_PT_WEEKEND_Q1_PUSH_20260514/results/gears_formal_runs}"
LOG_BASE="${5:-/home/yyf/codex_cout/SAFE_TRANS_PT_WEEKEND_Q1_PUSH_20260514/logs/gears_formal}"
SEEDS="${SEEDS:-1 2 3}"
EPOCHS="${EPOCHS:-8}"
MAX_GENES="${MAX_GENES:-6000}"

PYTHON_BIN="${PYTHON_BIN:-/home/yyf/.conda/envs/scgpt_env/bin/python}"
CODE_DIR="${CODE_DIR:-/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push/03_code}"

mkdir -p "${ROOT_BASE}/${DATASET}" "${LOG_BASE}/${DATASET}"

for seed in ${SEEDS}; do
  run_root="${ROOT_BASE}/${DATASET}/seed_${seed}"
  run_log="${LOG_BASE}/${DATASET}/seed_${seed}.log"
  mkdir -p "$run_root"
  {
    echo "[$(date '+%F %T')] start dataset=${DATASET} seed=${seed} cuda=${CUDA_DEV}"
    PYTHONPATH="${CODE_DIR}" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${CUDA_DEV}" \
      "${PYTHON_BIN}" -u "${CODE_DIR}/run_gears_formal_baseline.py" \
      --root "${run_root}" \
      --data-path "${DATA_PATH}" \
      --datasets "${DATASET}" \
      --split single \
      --seed "${seed}" \
      --max-genes "${MAX_GENES}" \
      --epochs "${EPOCHS}" \
      --batch-size 16 \
      --test-batch-size 32 \
      --device cuda:0
    echo "[$(date '+%F %T')] done dataset=${DATASET} seed=${seed}"
  } | tee "${run_log}"
done
