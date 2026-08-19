#!/usr/bin/env bash
set -euo pipefail

GROK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WRITE_VARIANTS="${GROK_DIR}/scripts/write_copy_variants.py"
COPY_SERVER="${GROK_DIR}/scripts/copy_server.py"
PID_FILE="${GROK_DIR}/.copy_server.pid"
PORT="${COPY_SERVER_PORT:-18765}"
LOG="${GROK_DIR}/.copy_server.log"

ensure_server() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  nohup python3 "$COPY_SERVER" >>"$LOG" 2>&1 &
  echo $! >"$PID_FILE"
  sleep 0.3
}

push_to_server() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS -X POST --data-binary @"${GROK_DIR}/COPY.txt" \
      -H "Content-Type: text/plain; charset=utf-8" \
      "http://127.0.0.1:${PORT}/copy" >/dev/null 2>&1 && echo "bridge:POST-ok" || echo "bridge:POST-fail"
  else
    echo "bridge:no-curl"
  fi
}

main() {
  python3 "$WRITE_VARIANTS"
  ensure_server
  push_to_server
  echo "COPY_ADAPTER_OK"
  echo "file: ${GROK_DIR}/COPY.txt"
  echo "bridge: http://127.0.0.1:${PORT}/copy"
  echo "ACTION: If clip_sync.ps1 running on Windows -> Ctrl+V in 1 second"
  echo "SETUP_ONCE: Run agents/grok/windows/setup_once.ps1 on Windows if not done"
}

main "$@"