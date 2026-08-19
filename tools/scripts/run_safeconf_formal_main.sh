#!/usr/bin/env bash
set -euo pipefail

PROJ="/home/yyf/proj"
CODE="$PROJ/code/20260426_154505_perturb_transport_final_push"
PYTHON="/home/yyf/.conda/envs/scgpt_env/bin/python"
ATLAS="/home/yyf/data/singlecell_perturbation_atlas"
OUT_ROOT="$CODE/outputs/safeconf_formal_main_20260604"
LOG_DIR="$OUT_ROOT/logs"
STATUS="$OUT_ROOT/RUN_STATUS.tsv"

mkdir -p "$LOG_DIR"
cd "$CODE"
export PYTHONPATH="$CODE:$CODE/03_code:${PYTHONPATH:-}"

echo -e "step\tstatus\tstarted_at\tfinished_at\tpath\tlog" > "$STATUS"

log_step() {
  local step="$1"
  local status="$2"
  local started="$3"
  local finished="$4"
  local path="$5"
  local log="$6"
  echo -e "${step}\t${status}\t${started}\t${finished}\t${path}\t${log}" >> "$STATUS"
}

run_mcfarland() {
  local dataset="McFarlandTsherniak2020"
  local out_dir="$OUT_ROOT/$dataset"
  local log="$LOG_DIR/${dataset}.log"
  local start
  start="$(date '+%F %T')"
  echo "[START] $start dataset=$dataset drug-only" | tee -a "$log"
  if [ -s "$out_dir/tables/PREDICTION_RECORDS.csv" ]; then
    local finish
    finish="$(date '+%F %T')"
    echo "[SKIP] $finish dataset=$dataset existing output found" | tee -a "$log"
    log_step "$dataset" "SKIPPED_EXISTING" "$start" "$finish" "$out_dir" "$log"
    return 0
  fi

  set +e
  timeout 18h "$PYTHON" scripts/run_blind_dataset.py \
    --dataset-name "$dataset" \
    --h5ad-path "$ATLAS/official_scperturb/McFarlandTsherniak2020.h5ad" \
    --context-col "cell_line" \
    --perturbation-col "perturbation" \
    --dataset-family "chem_robust" \
    --filter-col "perturbation_type" \
    --filter-value "drug" \
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
    log_step "$dataset" "DONE" "$start" "$finish" "$out_dir" "$log"
  elif [ "$rc" -eq 124 ]; then
    echo "[TIMEOUT] $finish dataset=$dataset after 18h" | tee -a "$log"
    log_step "$dataset" "TIMEOUT" "$start" "$finish" "$out_dir" "$log"
    return 124
  else
    echo "[FAILED] $finish dataset=$dataset rc=$rc" | tee -a "$log"
    log_step "$dataset" "FAILED_$rc" "$start" "$finish" "$out_dir" "$log"
    return "$rc"
  fi
}

run_formal_audit() {
  local log="$LOG_DIR/formal_audit.log"
  local start
  start="$(date '+%F %T')"
  echo "[START] $start formal audit" | tee -a "$log"
  set +e
  "$PYTHON" -m safetrans_confidence.cli.run_formal_main_audit \
    --run-dir "$CODE/outputs/safeconf_cui_go_nogo_probe" \
    --run-dir "$OUT_ROOT/McFarlandTsherniak2020" \
    --run-dir "$CODE/outputs/safeconf_phase1_main/Frangieh" \
    --run-dir "$CODE/outputs/safeconf_phase1_main/SrivatsanTrapnell2020_sciplex3" \
    --run-dir "$CODE/outputs/safeconf_phase1_main/SantinhaPlatt2023" \
    --run-dir "$CODE/outputs/safeconf_phase1_main/LaraAstiasoHuntly2023_invivo" \
    --run-dir "$CODE/outputs/safeconf_phase1_main/LaraAstiasoHuntly2023_exvivo" \
    --out-dir "$OUT_ROOT/formal_audit" \
    --bootstrap 1000 \
    --seed 5201 \
    >> "$log" 2>&1
  rc=$?
  set -e
  local finish
  finish="$(date '+%F %T')"
  if [ "$rc" -eq 0 ]; then
    echo "[DONE] $finish formal audit" | tee -a "$log"
    log_step "formal_audit" "DONE" "$start" "$finish" "$OUT_ROOT/formal_audit" "$log"
  else
    echo "[FAILED] $finish formal audit rc=$rc" | tee -a "$log"
    log_step "formal_audit" "FAILED_$rc" "$start" "$finish" "$OUT_ROOT/formal_audit" "$log"
    return "$rc"
  fi
}

{
  echo "[INFO] SafeConf formal main started at $(date '+%F %T')"
  echo "[INFO] Output root: $OUT_ROOT"
  echo "[INFO] Tahoe prefetch may continue in safeconf_data_prefetch."
} | tee -a "$LOG_DIR/formal_main.log"

run_mcfarland
run_formal_audit

{
  echo "[INFO] SafeConf formal main finished at $(date '+%F %T')"
  echo "[INFO] status=$STATUS"
} | tee -a "$LOG_DIR/formal_main.log"
