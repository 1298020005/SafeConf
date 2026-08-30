#!/usr/bin/env bash
set -euo pipefail

# Resume the E201 blind-prediction chain without overwriting completed runs.
# The sealed runner refuses overwrite by design; this wrapper makes the
# intended order explicit and skips only runs whose manifest and array are
# already complete.  It never materializes target perturbed expression.

PYTHON_BIN="${PYTHON_BIN:-/home/yyf/.venvs/txpert-08d82eea/bin/python}"
TXPERT_REPO="${TXPERT_REPO:-/home/yyf/archive/external/TxPert}"
DATA_ROOT="${DATA_ROOT:-/home/yyf/data}"
FAMILY_SEAL="${FAMILY_SEAL:-/home/yyf/proj/docs/实验结果/E201_txpert_multitarget_retraining_20260802/E201_FAMILY_SEAL.json}"
PRED_ROOT="${PRED_ROOT:-/home/yyf/data/txpert_official_20260802/e201/formal/predictions}"
BATCH_SIZE="${BATCH_SIZE:-16}"

RUNNER="/home/yyf/proj/tools/scripts/run_e201_txpert_sealed_prediction.py"
TARGETS=(K562 RPE1 hepg2 jurkat)

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$FAMILY_SEAL" ]]; then
  echo "Missing family seal: $FAMILY_SEAL" >&2
  exit 2
fi

is_complete() {
  local manifest="$1"
  local array="${2:-}"
  [[ -s "$manifest" ]] || return 1
  [[ -s "$array" ]] || return 1
  "$PYTHON_BIN" - "$manifest" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as handle:
    record = json.load(handle)
if record.get('status') != 'COMPLETE':
    raise SystemExit(1)
if record.get('target_expression_nonzero_values_seen') not in (0, 0.0):
    raise SystemExit('target expression was observed; refusing to continue')
PY
}

for target in "${TARGETS[@]}"; do
  shared_dir="$PRED_ROOT/$target/shared"
  # The sealed runner creates shared/ atomically during seed 1.  Creating it
  # here would make seed 1 (correctly) treat the directory as a stale run.
  mkdir -p "$PRED_ROOT/$target"
  for seed in 1 2 3 4; do
    output_dir="$PRED_ROOT/$target/seed_$seed"
    manifest="$output_dir/E201_PREDICTION_RUN.json"
    array="$output_dir/predictions.npy"
    if is_complete "$manifest" "$array"; then
      echo "[SKIP] $target seed $seed already complete"
      continue
    fi
    if [[ -e "$array" || -e "$manifest" ]]; then
      echo "Refusing to overwrite incomplete run: $output_dir" >&2
      exit 3
    fi
    echo "[RUN ] $target seed $seed"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
      "$PYTHON_BIN" "$RUNNER" \
      --txpert-repo "$TXPERT_REPO" \
      --data-root "$DATA_ROOT" \
      --family-seal "$FAMILY_SEAL" \
      --target "$target" \
      --seed "$seed" \
      --output-dir "$output_dir" \
      --shared-dir "$shared_dir" \
      --batch-size "$BATCH_SIZE"
  done
done

echo "E201 blind prediction chain complete; truth remains sealed."
