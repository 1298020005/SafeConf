#!/usr/bin/env bash
# Wait for sealed raw aggregation, then use both GPUs for the fixed validation gate.
set -euo pipefail

base="/home/yyf/data/feng2025_candidate/raw_dev"
model_dir="/home/yyf/data/feng2025_candidate/raw_dev_models"
scripts="$(cd "$(dirname "$0")" && pwd)"
python="/home/yyf/.conda/envs/prescribe_env/bin/python"
prefix="$base/E258_RAW_ALLOWED_GROUP_SUMS"
view="$base/E258_RAW_DEV_VIEW.npz"

for attempt in {1..720}; do
  if [[ -f "$prefix.u32" && -f "$prefix.library.u64" && -f "$prefix.genes.txt" ]]; then
    break
  fi
  if ! tmux has-session -t e258_feng_raw_20260924 2>/dev/null; then
    printf 'raw aggregator stopped without complete outputs\n' >&2
    exit 1
  fi
  sleep 30
done
[[ -f "$prefix.u32" && -f "$prefix.library.u64" && -f "$prefix.genes.txt" ]] || {
  printf 'raw aggregate deadline reached without complete outputs\n' >&2
  exit 1
}
mkdir -p "$model_dir"
"$python" "$scripts/build_e258_raw_dev_view.py" --output "$view" \
  > "$model_dir/build_raw_view.log" 2>&1
"$python" "$scripts/run_e258_train_only_shrinkage_baseline.py" \
  --view "$view" --output "$model_dir/shrinkage_status.json" --mode raw \
  > "$model_dir/shrinkage.log" 2>&1

for base_mode in mean train_shrunk; do
  CUDA_VISIBLE_DEVICES=0 "$python" "$scripts/run_e258_official_dev_predictor.py" \
    --view "$view" --output-dir "$model_dir" --variant small --device cuda:0 \
    --mode raw --base-mode "$base_mode" > "$model_dir/${base_mode}_small.log" 2>&1 &
  pid0=$!
  CUDA_VISIBLE_DEVICES=1 "$python" "$scripts/run_e258_official_dev_predictor.py" \
    --view "$view" --output-dir "$model_dir" --variant medium --device cuda:0 \
    --mode raw --base-mode "$base_mode" > "$model_dir/${base_mode}_medium.log" 2>&1 &
  pid1=$!
  status=0
  wait "$pid0" || status=1
  wait "$pid1" || status=1
  if (( status != 0 )); then
    printf 'raw-count %s predictor failed; inspect %s/*.log\n' \
      "$base_mode" "$model_dir" >&2
    exit "$status"
  fi
done
"$python" "$scripts/analyze_e258_official_dev_subgroups.py" \
  --view "$view" --model-dir "$model_dir" --mode raw \
  > "$model_dir/subgroup_diagnostic.json"
printf 'E258 raw train/validation predictor gate complete; test donor truth unopened.\n'
