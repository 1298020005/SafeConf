#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yyf/proj"
OUT_ROOT="$ROOT/code/20260426_154505_perturb_transport_final_push/outputs/data_prefetch"
DATA_ROOT="/home/yyf/data/singlecell_perturbation_atlas/mega_external/Tahoe-100M"
LOG_DIR="$OUT_ROOT/logs"
MANIFEST_DIR="$OUT_ROOT/tahoe_manifest"
ARIA_INPUT="$MANIFEST_DIR/aria2_tahoe_phasec.txt"
GENERATED_MANIFEST="$MANIFEST_DIR/DOWNLOAD_MANIFEST_GENERATED.csv"
HF_BASE="https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main"

mkdir -p "$LOG_DIR" "$MANIFEST_DIR" "$DATA_ROOT/metadata/pseudobulk_differential_expression"

exec > >(tee -a "$LOG_DIR/tahoe_prefetch.log") 2>&1

echo "[START] $(date '+%F %T') SafeConf Tahoe PhaseC prefetch"
echo "[INFO] data_root=$DATA_ROOT"
echo "[INFO] output_root=$OUT_ROOT"

PROXY_SNAPSHOT="${http_proxy:-}${https_proxy:-}${all_proxy:-}${HTTP_PROXY:-}${HTTPS_PROXY:-}${ALL_PROXY:-}"
if echo "$PROXY_SNAPSHOT" | grep -Eq '(:|//[^/]*:)202([/$]|$)'; then
  if [ "${SAFECONF_ALLOW_PROXY_202:-0}" != "1" ]; then
    echo "[NETWORK] detected proxy on port 202; disabling proxy env to avoid using 202."
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
  else
    echo "[NETWORK] proxy port 202 explicitly allowed by SAFECONF_ALLOW_PROXY_202=1"
  fi
else
  echo "[NETWORK] server-side proxy is not port 202; keeping current proxy entry."
  echo "[NETWORK] proxy summary: ${HTTPS_PROXY:-${https_proxy:-none}}"
  echo "[NETWORK] note: if this proxy entry forwards to a local router, downstream 202/non-202 routing is controlled by the local router rules."
  echo "[NETWORK] Tahoe uses huggingface.co / hf.co / xethub.hf.co domains; verify final route in local proxy logs if needed."
fi

python - <<'PY'
from pathlib import Path

hf_base = "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main"
manifest_dir = Path("/home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/data_prefetch/tahoe_manifest")
data_root = Path("/home/yyf/data/singlecell_perturbation_atlas/mega_external/Tahoe-100M")
aria_input = manifest_dir / "aria2_tahoe_phasec.txt"
csv_path = manifest_dir / "DOWNLOAD_MANIFEST_GENERATED.csv"

core_files = [
    ("README.md", 18826, "README.md", "dataset_card"),
    ("LICENSE.md", 6884, "LICENSE.md", "license"),
    ("metadata/cell_line_metadata.parquet", 19040, "metadata/cell_line_metadata.parquet", "cell_line_metadata"),
    ("metadata/drug_metadata.parquet", 40475, "metadata/drug_metadata.parquet", "drug_metadata"),
    ("metadata/gene_metadata.parquet", 1326799, "metadata/gene_metadata.parquet", "gene_metadata"),
    ("metadata/gene_vocabulary.json", 1744784, "metadata/gene_vocabulary.json", "gene_vocabulary"),
    ("metadata/gene_vocabulary.jsonl", 4790043, "metadata/gene_vocabulary.jsonl", "gene_vocabulary_jsonl"),
    ("metadata/obs_metadata.parquet", 2293981573, "metadata/obs_metadata.parquet", "obs_metadata"),
]

rows = []
aria_lines = []
to_download = 0

def add_file(role, rel, url, size, out):
    global to_download
    local_path = data_root / out
    present = local_path.exists() and local_path.stat().st_size > 0
    status = "present" if present else "planned"
    rows.append((role, rel, url, size, out, status, "yes"))
    if not present:
        to_download += 1
        aria_lines.extend([url, f"  out={out}"])

for rel, size, out, role in core_files:
    url = f"{hf_base}/{rel}"
    add_file(role, rel, url, size, out)

for i in range(1026):
    name = f"metadata/pseudobulk_differential_expression/train-{i:05d}-of-01026.parquet"
    url = f"{hf_base}/{name}"
    add_file("pseudobulk_de", name, url, "", name)

csv_path.write_text(
    "role,remote_path,source_url,expected_size_bytes,local_relative_path,status,resume_supported\n"
    + "\n".join(",".join(map(str, r)) for r in rows)
    + "\n",
    encoding="utf-8",
)
aria_input.write_text("\n".join(aria_lines) + ("\n" if aria_lines else ""), encoding="utf-8")
print(f"[MANIFEST] wrote {csv_path} rows={len(rows)} to_download={to_download}")
print(f"[ARIA] wrote {aria_input}")
PY

if [ ! -s "$ARIA_INPUT" ]; then
  echo "[DONE] no missing Tahoe PhaseC files to download"
  du -sh "$DATA_ROOT" || true
  exit 0
fi

echo "[INFO] starting aria2c with resume enabled"
aria2c \
  --no-conf=true \
  --continue=true \
  --max-connection-per-server=4 \
  --split=4 \
  --max-concurrent-downloads=2 \
  --max-tries=20 \
  --retry-wait=15 \
  --timeout=120 \
  --connect-timeout=60 \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  --summary-interval=120 \
  --dir="$DATA_ROOT" \
  --input-file="$ARIA_INPUT"

echo "[DONE] $(date '+%F %T') SafeConf Tahoe PhaseC prefetch"
du -sh "$DATA_ROOT" || true
