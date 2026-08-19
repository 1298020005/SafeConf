#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="/home/yyf/.conda/envs/scgpt_env/bin/python"
OUT="$ROOT/outputs/safeconf_sprint1_lodo_20260604"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

exec > >(tee -a "$LOG_DIR/run_lodo.log") 2>&1

echo "[START] $(date '+%F %T') SafeConf Sprint 1 group-LODO"
echo "[INFO] out=$OUT"

"$PYTHON" -m safetrans_confidence.cli.run_lodo \
  --group-table "$ROOT/configs/dataset_groups.csv" \
  --n-bootstrap 200 \
  --run-dir "$ROOT/outputs/safeconf_cui_go_nogo_probe" \
  --run-dir "$ROOT/outputs/safeconf_formal_main_20260604/McFarlandTsherniak2020" \
  --run-dir "$ROOT/outputs/safeconf_phase1_main/Frangieh" \
  --run-dir "$ROOT/outputs/safeconf_phase1_main/SrivatsanTrapnell2020_sciplex3" \
  --run-dir "$ROOT/outputs/safeconf_phase1_main/SantinhaPlatt2023" \
  --run-dir "$ROOT/outputs/safeconf_phase1_main/LaraAstiasoHuntly2023_invivo" \
  --run-dir "$ROOT/outputs/safeconf_phase1_main/LaraAstiasoHuntly2023_exvivo" \
  --out-dir "$OUT"

echo "[DONE] $(date '+%F %T') SafeConf Sprint 1 group-LODO"
