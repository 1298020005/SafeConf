#!/usr/bin/env bash
set -euo pipefail

# Cron @reboot entry for the September E205 unattended run.  It does not
# duplicate live tmux sessions and waits for the data mount and NVIDIA driver
# before restoring checkpoint-aware supervisors.

export HOME=/home/yyf
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

REPO=/home/yyf/proj
RELEASE_REPO=/home/yyf/runtime_worktrees/e205_postprocess_20260915
RUNS=/home/yyf/data/txpert_official_20260802/e205/formal
STATE=/home/yyf/data/txpert_official_20260802/e205/posttraining_supervisor_20260915
E208_SMOKE=/home/yyf/data/perturbench_e208/after_e205_postprocess_20260915
E208_RUNS=/home/yyf/data/perturbench_e208/formal_20260917
RESTORE_LOG="$STATE/reboot_restore.log"

mkdir -p "$STATE" "$E208_RUNS"
exec 9>"$STATE/reboot_restore.lock"
if ! flock -n 9; then
  exit 0
fi

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >> "$RESTORE_LOG"
}

ready=0
for _ in $(seq 1 120); do
  if [[ -d "$REPO/.git" || -f "$REPO/.git" ]] \
    && [[ -d "$RELEASE_REPO" ]] \
    && [[ -f "$RUNS/E205_FORMAL_QUEUE_STATUS.json" ]] \
    && nvidia-smi -L >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 10
done
if [[ "$ready" -ne 1 ]]; then
  log "BLOCKED prerequisites unavailable after 20 minutes"
  exit 2
fi

if ! tmux has-session -t safeconf_e205_formal_recovery_supervisor 2>/dev/null; then
  tmux new-session -d -s safeconf_e205_formal_recovery_supervisor -c "$REPO" \
    "/usr/bin/python3 tools/scripts/run_e205_exphormer_formal_queue.py --safeconf-repo /home/yyf/proj --txpert-repo /home/yyf/archive/external/TxPert --python /home/yyf/.venvs/txpert-08d82eea/bin/python --runs-root /home/yyf/data/txpert_official_20260802/e205/formal --profile-run-dir /home/yyf/data/txpert_official_20260802/e205/profile/RPE1/seed_1 --cuda-devices 0,1 --min-free-mb 20480 --foreign-proc-mb 1024 --poll-seconds 30 --max-attempts 3 --batch-size 64 >> /home/yyf/data/txpert_official_20260802/e205/formal/E205_FORMAL_QUEUE_REBOOT_RESTORE.log 2>&1"
  log "STARTED training queue supervisor"
else
  log "SKIP training queue supervisor already exists"
fi

if ! tmux has-session -t safeconf_e205_posttraining_release 2>/dev/null; then
  tmux new-session -d -s safeconf_e205_posttraining_release -c "$RELEASE_REPO" \
    "/usr/bin/python3 tools/scripts/run_e205_posttraining_supervisor.py --repo /home/yyf/runtime_worktrees/e205_postprocess_20260915 --expected-branch exp/e205-postprocess-20260915 --python /home/yyf/.venvs/txpert-08d82eea/bin/python --data-root /home/yyf/data --txpert-repo /home/yyf/archive/external/TxPert --runs-root /home/yyf/data/txpert_official_20260802/e205/formal --prediction-root /home/yyf/data/txpert_official_20260802/e205/formal/predictions --queue-status /home/yyf/data/txpert_official_20260802/e205/formal/E205_FORMAL_QUEUE_STATUS.json --state-root /home/yyf/data/txpert_official_20260802/e205/posttraining_supervisor_20260915 --cuda-devices 0,1 --poll-seconds 60 --prediction-attempts 2 --perturbench-repo /home/yyf/archive/external/PerturBench --perturbench-python /home/yyf/.venvs/perturbench-c84038bc/bin/python --e208-data-dir /home/yyf/data/external/perturbench_2025/jiang24 --e208-output-root /home/yyf/data/perturbench_e208/after_e205_postprocess_20260915 >> /home/yyf/data/txpert_official_20260802/e205/posttraining_supervisor_20260915/supervisor.log 2>&1"
  log "STARTED post-training release supervisor"
else
  log "SKIP post-training release supervisor already exists"
fi

if ! tmux has-session -t safeconf_e208_formal_queue 2>/dev/null; then
  tmux new-session -d -s safeconf_e208_formal_queue -c "$REPO" \
    "/home/yyf/.venvs/perturbench-c84038bc/bin/python tools/scripts/run_e208_formal_training_queue.py --repo /home/yyf/proj --perturbench-repo /home/yyf/archive/external/PerturBench --python /home/yyf/.venvs/perturbench-c84038bc/bin/python --data-dir /home/yyf/data/external/perturbench_2025/jiang24 --smoke-supervisor-status $E208_SMOKE/E208_AFTER_E205_SUPERVISOR_STATUS.json --runs-root $E208_RUNS --cuda-devices 0,1 --min-free-mb 22000 --foreign-proc-mb 1024 --poll-seconds 60 --max-attempts 3 >> $E208_RUNS/E208_FORMAL_QUEUE_CONSOLE.log 2>&1"
  log "STARTED E208 formal queue supervisor (waiting for smoke PASS)"
else
  log "SKIP E208 formal queue supervisor already exists"
fi
