#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yyf/proj"
CODE_ROOT="$ROOT/code/20260426_154505_perturb_transport_final_push"
PY="/home/yyf/.conda/envs/scgpt_env/bin/python"
OUT_ROOT="$CODE_ROOT/outputs/safeconf_supplement_v02_20260605"
LOG_DIR="$OUT_ROOT/logs"
STATUS="$OUT_ROOT/SUPPLEMENT_RUN_STATUS.csv"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"

echo "[START] $(date '+%F %T') SafeConf supplement v0.2 runs" | tee -a "$LOG_DIR/run.log"
echo "dataset,status,exit_code,log_path,run_status_path,note" > "$STATUS"

run_one() {
  local dataset="$1"
  local h5ad="$2"
  local context="$3"
  local perturb="$4"
  local family="$5"
  local out="$OUT_ROOT/$dataset"
  echo "[RUN] $(date '+%F %T') dataset=$dataset context=$context perturbation=$perturb family=$family" | tee -a "$LOG_DIR/run.log"
  if [ -f "$out/RUN_STATUS.json" ]; then
    echo "[SKIP] $dataset already has RUN_STATUS.json" | tee -a "$LOG_DIR/run.log"
    echo "$dataset,skipped_existing,0,$LOG_DIR/${dataset}.log,$out/RUN_STATUS.json,already_completed" >> "$STATUS"
    return 0
  fi
  set +e
  "$PY" "$CODE_ROOT/scripts/run_blind_dataset.py" \
    --dataset-name "$dataset" \
    --h5ad-path "$h5ad" \
    --context-col "$context" \
    --perturbation-col "$perturb" \
    --dataset-family "$family" \
    --out-dir "$out" \
    --n-genes 5000 \
    --seed 5201 \
    2>&1 | tee -a "$LOG_DIR/${dataset}.log"
  local code=${PIPESTATUS[0]}
  set -e

  if [ "$code" -eq 0 ] && [ -f "$out/RUN_STATUS.json" ]; then
    echo "[OK] $(date '+%F %T') dataset=$dataset" | tee -a "$LOG_DIR/run.log"
    echo "$dataset,ok,0,$LOG_DIR/${dataset}.log,$out/RUN_STATUS.json,completed" >> "$STATUS"
  else
    echo "[FAIL] $(date '+%F %T') dataset=$dataset exit_code=$code; continuing" | tee -a "$LOG_DIR/run.log"
    echo "$dataset,failed,$code,$LOG_DIR/${dataset}.log,$out/RUN_STATUS.json,see_log" >> "$STATUS"
  fi
}

# These are supplement candidates, not current main-table datasets.
# Selection rule: prefer biological/sample context over pure batch/replicate proxies.
run_one \
  "LaraAstiasoHuntly2023_leukemia" \
  "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/LaraAstiasoHuntly2023_leukemia.h5ad" \
  "celltype" \
  "guide_id" \
  "gene_main"

run_one \
  "XieHon2017" \
  "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/XieHon2017.h5ad" \
  "sample" \
  "perturbation" \
  "gene_main"

run_one \
  "ShifrutMarson2018" \
  "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/ShifrutMarson2018.h5ad" \
  "patient" \
  "perturbation" \
  "gene_main"

run_one \
  "sciplex3_small" \
  "/home/yyf/data/singlecell_perturbation_atlas/extra_official/cellular_context_generalization/sciplex3.h5ad" \
  "cell_type" \
  "perturbation" \
  "chem_robust"

run_one \
  "SrivatsanTrapnell2020_sciplex4" \
  "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/SrivatsanTrapnell2020_sciplex4.h5ad" \
  "celltype" \
  "perturbation" \
  "chem_robust"

echo "[DONE] $(date '+%F %T') SafeConf supplement v0.2 runs" | tee -a "$LOG_DIR/run.log"
