#!/usr/bin/env bash
# Two-GPU upstream engineering screen; not the formal raw-count confirmation.
set -euo pipefail

base="/home/yyf/data/feng2025_candidate"
scripts="$(cd "$(dirname "$0")" && pwd)"
python="/home/yyf/.conda/envs/prescribe_env/bin/python"
lfc="$base/TargetedScreen_LFC_byGene-perLine.tsv.gz"
view="$base/E258_OFFICIAL_DEV_VIEW.npz"
out="$base/official_dev_models"

for attempt in {1..720}; do
  [[ -f "$lfc" ]] && break
  sleep 30
done
[[ -f "$lfc" ]] || { printf 'verified official LFC did not arrive\n' >&2; exit 1; }
mkdir -p "$out"
if [[ ! -f "$view" ]]; then
  "$python" "$scripts/build_e258_official_dev_view.py" --output "$view" > "$out/prepare_view.log" 2>&1
fi

CUDA_VISIBLE_DEVICES=0 "$python" "$scripts/run_e258_official_dev_predictor.py" \
  --view "$view" --output-dir "$out" --variant small --device cuda:0 \
  > "$out/small.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$python" "$scripts/run_e258_official_dev_predictor.py" \
  --view "$view" --output-dir "$out" --variant medium --device cuda:0 \
  > "$out/medium.log" 2>&1 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if (( status != 0 )); then
  printf 'E258 engineering screen failed; inspect %s/*.log\n' "$out" >&2
  exit "$status"
fi
printf 'E258 two-GPU engineering screen finished; formal raw-count confirmation remains separate.\n'
