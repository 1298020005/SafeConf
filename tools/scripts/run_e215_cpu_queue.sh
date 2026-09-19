#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO=${REPO:-/home/yyf/runtime_worktrees/cpu_stability_20260918}
PYTHON=${PYTHON:-/home/yyf/.venvs/txpert-08d82eea/bin/python}
STATE=${STATE:-/home/yyf/data/safeconf_cpu/e215_queue_20260918}
BRANCH=${BRANCH:-exp/cpu-stability-20260918}

mkdir -p "$STATE/logs"
STATUS="$STATE/E215_QUEUE_STATUS.json"

write_status() {
  local stage=$1 state=$2 detail=$3
  STAGE_VALUE="$stage" STATE_VALUE="$state" DETAIL_VALUE="$detail" \
    STATUS_PATH="$STATUS" "$PYTHON" - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path

path = Path(os.environ["STATUS_PATH"])
temporary = path.with_name(f".{path.name}.tmp")
value = {
    "experiment": "E215_candidate_set_stability",
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

trap 'write_status "FAILED" "FAILED" "see logs under $STATE/logs"' ERR
cd "$REPO"
if [[ "$(git branch --show-current)" != "$BRANCH" ]] || [[ -n "$(git status --porcelain)" ]]; then
  write_status "PREFLIGHT" "FAILED" "branch mismatch or dirty worktree"
  exit 2
fi

write_status "RUNNING" "RUNNING" "34 folds x 3 fractions x 1000 draws on 24 workers"
"$PYTHON" tools/scripts/run_e215_candidate_set_stability.py \
  --input /home/yyf/proj/docs/实验结果/E153_eight_study_formal_meta_20260714/tables/E153_ABSOLUTE_TASK_INPUT.csv \
  --replicates 1000 \
  --workers 24 \
  --scratch /home/yyf/data/safeconf_cpu/e215_20260918 \
  > "$STATE/logs/E215.log" 2>&1

write_status "VERIFY" "RUNNING" "running E215 unit tests"
"$PYTHON" code/safeconf_audit/tests/test_e215_candidate_set_stability.py \
  > "$STATE/logs/tests.log" 2>&1

git add -- docs/实验结果/E215_candidate_set_stability_20260918
git add -f -- docs/实验结果/E215_candidate_set_stability_20260918/tables/*.csv
git commit -m "experiment: publish E215 candidate-set stability audit" \
  > "$STATE/logs/git_commit.log" 2>&1
write_status "PUBLISH" "RUNNING" "pushing E215 result branch"
git push origin "$BRANCH" > "$STATE/logs/push_gitee.log" 2>&1
GIT_SSH_COMMAND='ssh -i /home/yyf/.ssh/id_ed25519_github_safeconf -o IdentitiesOnly=yes -o ConnectTimeout=15' \
  git push git@github.com:1298020005/SafeConf.git "$BRANCH" \
  > "$STATE/logs/push_github.log" 2>&1
write_status "COMPLETE" "PASS" "results tested, committed, and pushed"
trap - ERR
