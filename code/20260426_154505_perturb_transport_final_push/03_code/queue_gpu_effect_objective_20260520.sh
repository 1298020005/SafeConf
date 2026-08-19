#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
LOG_DIR="$ROOT/45_gpu_effect_queue_20260520/logs"
mkdir -p "$LOG_DIR"

wait_for_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    echo "[queue] waiting for $session at $(date '+%F %T')" | tee -a "$LOG_DIR/queue.log"
    sleep 300
  done
}

(
  wait_for_session gpu_calibrated_main_20260520
  echo "[queue] starting effect-objective main at $(date '+%F %T')" | tee -a "$LOG_DIR/queue.log"
  bash "$ROOT/03_code/run_gpu_effect_objective_main_20260520.sh"
) > "$LOG_DIR/main_queue.log" 2>&1 &

(
  wait_for_session gpu_calibrated_tian_20260520
  echo "[queue] starting effect-objective tian at $(date '+%F %T')" | tee -a "$LOG_DIR/queue.log"
  bash "$ROOT/03_code/run_gpu_effect_objective_tian_20260520.sh"
) > "$LOG_DIR/tian_queue.log" 2>&1 &

wait
