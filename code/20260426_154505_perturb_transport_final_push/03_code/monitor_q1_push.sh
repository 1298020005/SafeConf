#!/usr/bin/env bash
# One-shot status for Q1 CPU push + GPU + readiness score.
set -euo pipefail

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
CPU_OUT="$ROOT/46_q1_cpu_push_20260520"
PY="/home/yyf/.conda/envs/scgpt_env/bin/python"
EVAL="$ROOT/03_code/evaluate_q1_readiness.py"

echo "=== Q1 push monitor $(date '+%F %T') ==="

if pgrep -f "run_safety_abstention_evidence.py.*46_q1_cpu_push" >/dev/null; then
  echo "[CPU] RUNNING (pid $(pgrep -f 'run_safety_abstention_evidence.py.*46_q1_cpu_push' | head -1))"
else
  echo "[CPU] not running"
fi

INC="$CPU_OUT/results/SAFETY_TASK_METRICS_INCREMENTAL.csv"
if [[ -f "$INC" ]]; then
  echo "[CPU] incremental rows: $(($(wc -l < "$INC") - 1))  size=$(stat -c%s "$INC")  mtime=$(stat -c%y "$INC" | cut -d. -f1)"
else
  echo "[CPU] no incremental file yet"
fi

if [[ -f "$CPU_OUT/results/SAFETY_SUMMARY.csv" ]]; then
  echo "[CPU] SAFETY_SUMMARY ready — scoring..."
  "$PY" -u "$EVAL" --results-dir "$CPU_OUT/results" --write-md 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('[CPU] label:', d.get('label')); [print('  gap:', g) for g in d.get('gaps',[])[:5]]" 2>/dev/null || true
fi

echo "--- GPU ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

for sess in gpu_graft_main_20260520 gpu_graft_tian_20260520 gpu_calibrated_main_20260520 gpu_calibrated_tian_20260520; do
  if tmux has-session -t "$sess" 2>/dev/null; then
    echo "[tmux] $sess: active"
  fi
done

for run in 39_gpu_calibrated_main_20260520 43_gpu_effect_objective_main_20260520; do
  summ="$ROOT/$run/results/GPU_DEEP_SUMMARY.csv"
  if [[ -f "$summ" ]]; then
    echo "[GPU] $run: $(wc -l < "$summ") summary lines"
  fi
done

echo "=== done ==="
