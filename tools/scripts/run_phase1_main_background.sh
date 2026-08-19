#!/usr/bin/env bash
set -euo pipefail

PROJ="/home/yyf/proj"
CODE="$PROJ/code/20260426_154505_perturb_transport_final_push"
PYTHON="/home/yyf/.conda/envs/scgpt_env/bin/python"
ATLAS="/home/yyf/data/singlecell_perturbation_atlas"
OUT_ROOT="$CODE/outputs/safeconf_phase1_main"
LOG_DIR="$OUT_ROOT/logs"
STATUS="$OUT_ROOT/RUN_STATUS.tsv"

mkdir -p "$LOG_DIR"
cd "$CODE"
export PYTHONPATH="$CODE:$CODE/03_code:${PYTHONPATH:-}"

echo -e "dataset\tstatus\tstarted_at\tfinished_at\tout_dir\tlog" > "$STATUS"

run_one() {
  local dataset="$1"
  local h5ad="$2"
  local context_col="$3"
  local perturb_col="$4"
  local family="$5"
  local out_dir="$OUT_ROOT/$dataset"
  local log="$LOG_DIR/${dataset}.log"
  local start
  start="$(date '+%F %T')"
  echo "[START] $start dataset=$dataset family=$family h5ad=$h5ad" | tee -a "$log"

  set +e
  timeout 18h "$PYTHON" scripts/run_blind_dataset.py \
    --dataset-name "$dataset" \
    --h5ad-path "$h5ad" \
    --context-col "$context_col" \
    --perturbation-col "$perturb_col" \
    --dataset-family "$family" \
    --atlas-root "$ATLAS" \
    --out-dir "$out_dir" \
    --n-genes 5000 \
    --seed 5201 \
    >> "$log" 2>&1
  rc=$?
  set -e

  local finish
  finish="$(date '+%F %T')"
  if [ "$rc" -eq 0 ]; then
    echo "[DONE] $finish dataset=$dataset" | tee -a "$log"
    echo -e "$dataset\tDONE\t$start\t$finish\t$out_dir\t$log" >> "$STATUS"
  elif [ "$rc" -eq 124 ]; then
    echo "[TIMEOUT] $finish dataset=$dataset after 18h" | tee -a "$log"
    echo -e "$dataset\tTIMEOUT\t$start\t$finish\t$out_dir\t$log" >> "$STATUS"
  else
    echo "[FAILED] $finish dataset=$dataset rc=$rc" | tee -a "$log"
    echo -e "$dataset\tFAILED_$rc\t$start\t$finish\t$out_dir\t$log" >> "$STATUS"
  fi
}

{
  echo "[INFO] Phase1 main started at $(date '+%F %T')"
  echo "[INFO] Tahoe prefetch may run in parallel; this script is CPU-oriented."
  echo "[SKIP] McFarlandTsherniak2020 requires a drug-only adapter before formal run."
} | tee -a "$LOG_DIR/phase1_main.log"

run_one "SrivatsanTrapnell2020_sciplex3" \
  "$ATLAS/official_scperturb/SrivatsanTrapnell2020_sciplex3.h5ad" \
  "celltype" "perturbation" "chem_robust"

run_one "SantinhaPlatt2023" \
  "$ATLAS/official_scperturb/SantinhaPlatt2023.h5ad" \
  "cell_types" "perturbation" "chem_robust"

run_one "LaraAstiasoHuntly2023_invivo" \
  "$ATLAS/official_scperturb/LaraAstiasoHuntly2023_invivo.h5ad" \
  "celltype" "perturbation" "gene_main"

run_one "LaraAstiasoHuntly2023_exvivo" \
  "$ATLAS/official_scperturb/LaraAstiasoHuntly2023_exvivo.h5ad" \
  "celltype" "perturbation" "gene_main"

run_one "Frangieh" \
  "$ATLAS/official_generalization/Frangieh.h5ad" \
  "condition" "perturbation" "gene_main"

{
  echo "[INFO] Phase1 main finished at $(date '+%F %T')"
  echo "[INFO] status=$STATUS"
} | tee -a "$LOG_DIR/phase1_main.log"
