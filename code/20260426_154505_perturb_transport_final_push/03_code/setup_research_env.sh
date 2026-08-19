#!/usr/bin/env bash
# Research + Codex CLI environment (non-interactive safe).
set -euo pipefail

export http_proxy="${http_proxy:-http://127.0.0.1:17897}"
export https_proxy="${https_proxy:-$http_proxy}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export PATH="$HOME/.local/node/bin:$HOME/.local/bin:$HOME/miniconda/bin:$PATH"

# Conda without interactive bashrc return
if [[ -f /home/miniconda/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/miniconda/etc/profile.d/conda.sh
  conda activate scgpt_env
fi

ROOT="/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push"
export PYTHONPATH="$ROOT/03_code:${PYTHONPATH:-}"
export PAIRDELTA_V2_BANK_MODE="${PAIRDELTA_V2_BANK_MODE:-pca_nmf_hvg}"
export PAIRDELTA_ROUTING_MODE="${PAIRDELTA_ROUTING_MODE:-hybrid}"

echo "=== SafeTrans research env ==="
echo "python: $(which python)"
python -c "import torch, scanpy, pandas, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
echo "codex: $(command -v codex) $(codex --version 2>/dev/null || true)"
echo "ROOT=$ROOT"
echo "PYTHONPATH=$PYTHONPATH"
