#!/usr/bin/env bash
set -euo pipefail

# Wait for the single clean aria2 download, then verify, decompress and run D0.
# This process uses CPU/disk only and never reads the H5AD expression matrix X.

DOWNLOAD_SESSION="${DOWNLOAD_SESSION:-safeconf_e208_download}"
DATA_DIR="${DATA_DIR:-/home/yyf/data/external/perturbench_2025/jiang24}"
GZIP_FILE="$DATA_DIR/jiang24_processed.h5ad.gz"
H5AD_FILE="$DATA_DIR/jiang24_processed.h5ad"
H5AD_PARTIAL="$DATA_DIR/jiang24_processed.h5ad.partial"
SPLIT_FILE="$DATA_DIR/jiang24_split.csv"
AUDIT_FILE="$DATA_DIR/E208_D0_RAW_AUDIT.json"
LOG_FILE="$DATA_DIR/E208_POSTDOWNLOAD_D0.log"
PYTHON_BIN="${PYTHON_BIN:-/home/yyf/.venvs/txpert-08d82eea/bin/python}"
AUDITOR="/home/yyf/proj/tools/scripts/audit_e208_jiang24_contract.py"
EXPECTED_BYTES=15232554616

exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date --iso-8601=seconds)] waiting for $DOWNLOAD_SESSION"
while tmux has-session -t "$DOWNLOAD_SESSION" 2>/dev/null; do
  sleep 30
done

echo "[$(date --iso-8601=seconds)] download session ended; starting gates"
[[ -f "$GZIP_FILE" && -f "$SPLIT_FILE" ]]
observed_bytes="$(stat -c %s "$GZIP_FILE")"
if [[ "$observed_bytes" != "$EXPECTED_BYTES" ]]; then
  echo "gzip size mismatch: $observed_bytes != $EXPECTED_BYTES" >&2
  exit 10
fi
gzip -t "$GZIP_FILE"
sha256sum "$GZIP_FILE" "$SPLIT_FILE"

if [[ -e "$H5AD_FILE" || -e "$H5AD_PARTIAL" || -e "$AUDIT_FILE" ]]; then
  echo "refusing to overwrite an existing D0 output" >&2
  exit 11
fi
echo "[$(date --iso-8601=seconds)] decompressing to a partial path"
gzip -dc "$GZIP_FILE" > "$H5AD_PARTIAL"
mv "$H5AD_PARTIAL" "$H5AD_FILE"

echo "[$(date --iso-8601=seconds)] running metadata and barcode audit"
"$PYTHON_BIN" "$AUDITOR" \
  --gzip "$GZIP_FILE" \
  --split "$SPLIT_FILE" \
  --h5ad "$H5AD_FILE" \
  --output "$AUDIT_FILE"
echo "[$(date --iso-8601=seconds)] D0 raw audit complete: $AUDIT_FILE"
