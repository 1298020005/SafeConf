#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Run the final CPU-only confidence scoring MVP.

This script:
  1. Copies prior Step 1-4 inputs from outputs/confidence_task_mvp/ into outputs/confidence_task_mvp_final/input/
  2. Computes confidence features
  3. Builds confidence/risk scores
  4. Evaluates scores against true_error_rmse
  5. Plots figures
  6. Writes MVP_REPORT.md
  7. Creates outputs/confidence_task_mvp_final.zip

Environment variables:
  PROJECT_ROOT   default: parent directory of this script
  PYTHON_BIN     default: /home/yyf/.conda/envs/scgpt_env/bin/python
EOF
  exit 0
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/yyf/.conda/envs/scgpt_env/bin/python}"
SRC="$PROJECT_ROOT/outputs/confidence_task_mvp"
OUT="$PROJECT_ROOT/outputs/confidence_task_mvp_final"
INPUT="$OUT/input"
TABLES="$OUT/tables"
FIGURES="$OUT/figures"
REPORTS="$OUT/reports"
LOGS="$OUT/logs"
SCRIPTS="$OUT/scripts"
mkdir -p "$INPUT" "$TABLES" "$FIGURES" "$REPORTS" "$LOGS" "$SCRIPTS"
LOG="$LOGS/run_confidence_mvp_final.log"

exec > >(tee -a "$LOG") 2>&1
echo "[confidence_mvp_final] start $(date)"
echo "[confidence_mvp_final] project=$PROJECT_ROOT"

required=(
  "$SRC/splits/kagglecrosscell_heldout_pair_split.csv"
  "$SRC/splits/kagglecrosscell_heldout_pair_split_summary.json"
  "$SRC/predictions/kagglecrosscell_v0_contextsim_predictions.csv"
  "$SRC/arrays/kagglecrosscell_predicted_effects.npz"
  "$SRC/arrays/kagglecrosscell_true_effects.npz"
  "$SRC/PREDICTION_RECORDS.csv"
  "$SRC/PREDICTION_RECORDS_SCHEMA.md"
)
for f in "${required[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[confidence_mvp_final] missing input: $f" >&2
    exit 2
  fi
done

cp -f "$SRC/splits/kagglecrosscell_heldout_pair_split.csv" "$INPUT/"
cp -f "$SRC/splits/kagglecrosscell_heldout_pair_split_summary.json" "$INPUT/"
cp -f "$SRC/predictions/kagglecrosscell_v0_contextsim_predictions.csv" "$INPUT/"
cp -f "$SRC/arrays/kagglecrosscell_predicted_effects.npz" "$INPUT/"
cp -f "$SRC/arrays/kagglecrosscell_true_effects.npz" "$INPUT/"
cp -f "$SRC/PREDICTION_RECORDS.csv" "$INPUT/"
cp -f "$SRC/PREDICTION_RECORDS_SCHEMA.md" "$INPUT/"
cp -f "$SRC/PREDICTION_RECORDS_STATUS.json" "$INPUT/" 2>/dev/null || true
cp -f "$SRC/reports/heldout_pair_split_report.md" "$INPUT/" 2>/dev/null || true
cp -f "$SRC/reports/predictor_run_report.md" "$INPUT/" 2>/dev/null || true
cp -f "$SRC/reports/step1_to_step4_status.md" "$INPUT/" 2>/dev/null || true

PROJECT_ROOT="$PROJECT_ROOT" "$PYTHON_BIN" - <<'PY'
from __future__ import annotations
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

root = Path(os.environ["PROJECT_ROOT"]) / "outputs/confidence_task_mvp_final"
inp = root / "input"
reports = root / "reports"
reports.mkdir(parents=True, exist_ok=True)
checks = []

def check_csv(name: str, required_cols: list[str]) -> None:
    path = inp / name
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    checks.append({
        "file": name,
        "path": str(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_required_columns": missing,
    })
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")

check_csv("kagglecrosscell_heldout_pair_split.csv", ["task_id", "dataset_name", "context", "perturbation", "fold_id", "split", "pair_seen_in_train", "perturbation_seen_in_train", "context_seen_in_train"])
check_csv("kagglecrosscell_v0_contextsim_predictions.csv", ["record_id", "task_id", "dataset_name", "fold_id", "split", "context", "perturbation", "predictor_name", "predicted_effect_key", "true_effect_key", "true_error_rmse", "true_error_cosine"])
check_csv("PREDICTION_RECORDS.csv", ["record_id", "task_id", "dataset_name", "fold_id", "split", "context", "perturbation", "predictor_name", "predicted_effect_key", "true_effect_key", "true_error_rmse", "true_error_cosine"])

for name in ["kagglecrosscell_predicted_effects.npz", "kagglecrosscell_true_effects.npz"]:
    path = inp / name
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path)
    shapes = {k: tuple(arr[k].shape) for k in arr.files[:5]}
    checks.append({"file": name, "path": str(path), "arrays": int(len(arr.files)), "first_shapes": str(shapes), "missing_required_columns": []})

summary_path = inp / "kagglecrosscell_heldout_pair_split_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
checks.append({"file": summary_path.name, "path": str(summary_path), "rows": "json", "columns": len(summary), "missing_required_columns": []})

lines = [
    "# Input File Check",
    "",
    "All required MVP Step 1-4 input files were found and copied into `input/`.",
    "",
    "| file | rows/arrays | columns/shape | missing_required_columns | path |",
    "| --- | ---: | --- | --- | --- |",
]
for row in checks:
    rows = row.get("rows", row.get("arrays", "NA"))
    cols = row.get("columns", row.get("first_shapes", "NA"))
    miss = ",".join(row.get("missing_required_columns", [])) or "none"
    lines.append(f"| `{row['file']}` | {rows} | {cols} | {miss} | `{row['path']}` |")
lines.extend([
    "",
    "## Split Summary",
    "",
    f"- dataset: `{summary.get('dataset_name')}`",
    f"- n_tasks: {summary.get('n_tasks')}",
    f"- n_contexts: {summary.get('n_contexts')}",
    f"- n_perturbations: {summary.get('n_perturbations')}",
    f"- actual_folds: {summary.get('actual_folds')}",
])
(reports / "input_file_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("[input_check] ok")
PY

"$PYTHON_BIN" "$PROJECT_ROOT/confidence_task/05_compute_confidence_features.py"
"$PYTHON_BIN" "$PROJECT_ROOT/confidence_task/06_run_confidence_scores.py"
"$PYTHON_BIN" "$PROJECT_ROOT/confidence_task/07_evaluate_confidence_scores.py"
"$PYTHON_BIN" "$PROJECT_ROOT/confidence_task/08_plot_confidence_results.py"

cp -f "$PROJECT_ROOT/confidence_task/05_compute_confidence_features.py" "$SCRIPTS/"
cp -f "$PROJECT_ROOT/confidence_task/06_run_confidence_scores.py" "$SCRIPTS/"
cp -f "$PROJECT_ROOT/confidence_task/07_evaluate_confidence_scores.py" "$SCRIPTS/"
cp -f "$PROJECT_ROOT/confidence_task/08_plot_confidence_results.py" "$SCRIPTS/"
cp -f "$PROJECT_ROOT/confidence_task/run_confidence_mvp_final.sh" "$SCRIPTS/"

ZIP="$PROJECT_ROOT/outputs/confidence_task_mvp_final.zip"
rm -f "$ZIP"
(cd "$PROJECT_ROOT/outputs" && zip -qr "$ZIP" "confidence_task_mvp_final")
zip -T "$ZIP"
echo "[confidence_mvp_final] zip=$ZIP"
echo "[confidence_mvp_final] done $(date)"
