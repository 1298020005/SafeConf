#!/usr/bin/env bash
set -euo pipefail

# Resume sealed E205 Exphormer prediction for selected targets.  Run at most
# one copy per GPU.  Target truth remains absent from the prediction cache.

PYTHON_BIN="${PYTHON_BIN:-/home/yyf/.venvs/txpert-08d82eea/bin/python}"
TXPERT_REPO="${TXPERT_REPO:-/home/yyf/archive/external/TxPert}"
DATA_ROOT="${DATA_ROOT:-/home/yyf/data}"
FAMILY_SEAL="${FAMILY_SEAL:-/home/yyf/proj/docs/实验结果/E205_cross_family_disagreement_20260830/E205_EXPHORMER_FAMILY_SEAL.json}"
PRED_ROOT="${PRED_ROOT:-/home/yyf/data/txpert_official_20260802/e205/formal/predictions}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TARGETS_CSV="${TARGETS_CSV:-K562,RPE1,hepg2,jurkat}"

RUNNER="/home/yyf/proj/tools/scripts/run_e201_txpert_sealed_prediction.py"
IFS=',' read -r -a TARGETS <<< "$TARGETS_CSV"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$FAMILY_SEAL" ]]; then
  echo "Missing E205 family seal: $FAMILY_SEAL" >&2
  exit 2
fi

is_complete() {
  local manifest="$1"
  local array="$2"
  [[ -s "$manifest" && -s "$array" ]] || return 1
  "$PYTHON_BIN" - "$manifest" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    record = json.load(handle)
required = {
    "status": "COMPLETE",
    "experiment": "E205_cross_family_exphormer",
    "model_family": "TxPert-Exphormer",
    "architecture": "exphormer",
    "target_truth_materialized": False,
    "target_expression_nonzero_values_seen": 0,
}
for key, expected in required.items():
    if record.get(key) != expected:
        raise SystemExit(f"prediction manifest gate failed: {key}")
PY
}

for target in "${TARGETS[@]}"; do
  case "$target" in
    K562|RPE1|hepg2|jurkat) ;;
    *) echo "Unknown target: $target" >&2; exit 2 ;;
  esac
  shared_dir="$PRED_ROOT/$target/shared"
  mkdir -p "$PRED_ROOT/$target"
  for seed in 1 2 3 4; do
    output_dir="$PRED_ROOT/$target/seed_$seed"
    manifest="$output_dir/E205_PREDICTION_RUN.json"
    array="$output_dir/predictions.npy"
    if is_complete "$manifest" "$array"; then
      echo "[SKIP] $target seed $seed already complete"
      continue
    fi
    if [[ -e "$output_dir" ]]; then
      echo "Refusing to overwrite incomplete run: $output_dir" >&2
      exit 3
    fi
    echo "[RUN ] $target seed $seed on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    "$PYTHON_BIN" "$RUNNER" \
      --txpert-repo "$TXPERT_REPO" \
      --data-root "$DATA_ROOT" \
      --family-seal "$FAMILY_SEAL" \
      --architecture exphormer \
      --target "$target" \
      --seed "$seed" \
      --output-dir "$output_dir" \
      --shared-dir "$shared_dir" \
      --batch-size "$BATCH_SIZE"
  done
done

echo "E205 Exphormer prediction subset complete; target truth remains sealed."
