#!/usr/bin/env bash
# Run a Codex CLI task against AGENTS.md (non-interactive).
set -euo pipefail

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
# shellcheck disable=SC1091
source "$ROOT/03_code/setup_research_env.sh"

PROMPT="${1:-Read ~/.codex/AGENTS.md and 00_meta/Q1_TOP_Q2_MASTER_STANDARD.md. Then write 00_meta/GEARS_HEAD_TO_HEAD.md comparing formal GEARS results with PolicySafeTrans held-out metrics. Be honest about split differences.}"

LOG="$ROOT/00_meta/codex_last_run.log"
mkdir -p "$ROOT/00_meta"

echo "[codex] start $(date '+%F %T')" | tee "$LOG"
cd "$ROOT"

# Codex non-interactive: use exec if available, else print instruction
if codex exec --help >/dev/null 2>&1; then
  codex exec "$PROMPT" 2>&1 | tee -a "$LOG"
else
  echo "codex exec not available; run manually in Codex app with AGENTS.md" | tee -a "$LOG"
  echo "PROMPT: $PROMPT" | tee -a "$LOG"
fi

echo "[codex] done $(date '+%F %T')" | tee -a "$LOG"
