#!/usr/bin/env bash
set -euo pipefail
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
/home/yyf/.conda/envs/scgpt_env/bin/python -m safetrans_confidence.cli.run_phase4_experiments --out-dir "outputs/benchmark_phase4_experiments"
