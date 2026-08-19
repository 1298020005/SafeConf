#!/usr/bin/env bash
# Calibrated PolicySafeTransPT rerun after Opus7/Codex strict-review fixes.
set -euo pipefail

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
OUT="$ROOT/51_policy_calibrated_q1_20260520"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

source "${HOME}/.bashrc" 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate scgpt_env
fi

echo "[policy-calibrated] start $(date '+%F %T')" | tee "$LOG_DIR/master.log"

python -u "$ROOT/03_code/run_safety_abstention_evidence.py" \
  --root "$OUT" \
  --atlas-root "/home/yyf/datasets/singlecell_perturbation_atlas" \
  --seeds "5201,5211,5221,5231,5241" \
  --external-seed-count 4 \
  --max-datasets 6 \
  --max-external-datasets 5 \
  --n-genes 2200 \
  --n-programs 160 \
  --split-per-type 3 \
  --eval-bank "pca_nmf_hvg" \
  --v2-blend 0.12 \
  --max-blend 0.24 \
  --unsafe-threshold 0.42 \
  --policy-routing-mode "hard" \
  --main-studies "KaggleCrossCell,Haber,Parekh,Wessels,NormanWeissman2019,DixitRegev2016,AdamsonWeissman2016" \
  --external-studies "TianKampmann2019,PapalexiSatija2021,Frangieh,SrivatsanTrapnell2020,KaggleCrossPatient,McFarland,crossPatient,TCDD" \
  2>&1 | tee "$LOG_DIR/run_safety.log"

python -u "$ROOT/03_code/evaluate_q1_readiness.py" \
  --results-dir "$OUT/results" \
  --write-md \
  2>&1 | tee "$LOG_DIR/q1_readiness.log"

echo "[policy-calibrated] done $(date '+%F %T')" | tee -a "$LOG_DIR/master.log"
