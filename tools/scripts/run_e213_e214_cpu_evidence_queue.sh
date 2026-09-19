#!/usr/bin/env bash
set -euo pipefail

# Run the two CPU-only evidence jobs concurrently in a dedicated clean
# worktree.  NumPy/BLAS threads are capped because concurrency is provided by
# the experiment-level process pools.

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO=${REPO:-/home/yyf/runtime_worktrees/cpu_evidence_20260918}
PYTHON=${PYTHON:-/home/yyf/.venvs/txpert-08d82eea/bin/python}
STATE=${STATE:-/home/yyf/data/safeconf_cpu/queue_20260918}
BRANCH=${BRANCH:-exp/cpu-evidence-20260918}

mkdir -p "$STATE/logs"
STATUS="$STATE/CPU_EVIDENCE_QUEUE_STATUS.json"
PIDS=()

write_status() {
  local stage=$1
  local state=$2
  local detail=$3
  STAGE_VALUE="$stage" STATE_VALUE="$state" DETAIL_VALUE="$detail" \
    STATUS_PATH="$STATUS" "$PYTHON" - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path

path = Path(os.environ["STATUS_PATH"])
temporary = path.with_name(f".{path.name}.tmp")
value = {
    "experiment": "E213_E214_cpu_evidence_queue",
    "stage": os.environ["STAGE_VALUE"],
    "status": os.environ["STATE_VALUE"],
    "detail": os.environ["DETAIL_VALUE"],
    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "target_truth_access": "NOT_APPLICABLE_RELEASED_DATA_ONLY",
}
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
os.replace(temporary, path)
PY
}

cleanup() {
  local exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    for pid in "${PIDS[@]:-}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
      fi
    done
    write_status "FAILED" "FAILED" "see logs under $STATE/logs"
  fi
}
trap cleanup EXIT

cd "$REPO"
if [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
  write_status "PREFLIGHT" "FAILED" "unexpected branch"
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  write_status "PREFLIGHT" "FAILED" "worktree is not clean"
  exit 2
fi

write_status "RUNNING" "RUNNING" "E213 and E214 executing concurrently"
"$PYTHON" tools/scripts/run_e213_selective_prediction_endpoints.py \
  --input /home/yyf/proj/docs/实验结果/E153_eight_study_formal_meta_20260714/tables/E153_ABSOLUTE_TASK_INPUT.csv \
  --bootstrap 5000 \
  --workers 32 \
  --scratch /home/yyf/data/safeconf_cpu/e213_20260918 \
  > "$STATE/logs/E213.log" 2>&1 &
PIDS+=("$!")

"$PYTHON" tools/scripts/run_e214_runtime_scaling.py \
  --repeats 5 \
  --parallel-batches 32 \
  --parallel-batch-size 250000 \
  --max-workers 32 \
  > "$STATE/logs/E214.log" 2>&1 &
PIDS+=("$!")

wait "${PIDS[0]}"
wait "${PIDS[1]}"
PIDS=()

write_status "VERIFY" "RUNNING" "running complete safeconf_audit test suite"
"$PYTHON" -m unittest discover -s code/safeconf_audit/tests -p 'test_*.py' \
  > "$STATE/logs/tests.log" 2>&1

git add -- \
  docs/实验结果/E213_selective_prediction_endpoints_20260918 \
  docs/实验结果/E214_runtime_scaling_20260918
# Project-wide policy ignores newly generated CSV files by default.  These are
# small audited summaries (raw bootstrap draws stay under /home/yyf/data), so
# add only the named result tables explicitly.
git add -f -- \
  docs/实验结果/E213_selective_prediction_endpoints_20260918/tables/*.csv \
  docs/实验结果/E214_runtime_scaling_20260918/tables/*.csv
if git diff --cached --quiet; then
  write_status "PUBLISH" "FAILED" "no result files were generated"
  exit 3
fi
git commit -m "experiment: publish E213 selective-risk and E214 scaling results" \
  > "$STATE/logs/git_commit.log" 2>&1

write_status "PUBLISH" "RUNNING" "pushing audited CPU evidence to both remotes"
git push origin "$BRANCH" > "$STATE/logs/push_gitee.log" 2>&1
GIT_SSH_COMMAND='ssh -i /home/yyf/.ssh/id_ed25519_github_safeconf -o IdentitiesOnly=yes -o ConnectTimeout=15' \
  git push git@github.com:1298020005/SafeConf.git "$BRANCH" \
  > "$STATE/logs/push_github.log" 2>&1

write_status "COMPLETE" "PASS" "results tested, committed, and pushed"
trap - EXIT
