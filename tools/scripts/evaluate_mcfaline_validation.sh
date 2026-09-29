#!/usr/bin/env bash
set -euo pipefail

candidate="${1:-}"
gpu="${2:-}"
checkpoint="${3:-}"
output="${4:-}"

case "$candidate" in
  latent) experiment="neurips2025/mcfaline23/latent_additive_best_params_mcfaline23_full" ;;
  decoder) experiment="neurips2025/mcfaline23/decoder_only_best_params_mcfaline23_full" ;;
  *) echo "candidate must be latent or decoder" >&2; exit 2 ;;
esac
case "$gpu" in 0|1) ;; *) echo "GPU must be 0 or 1" >&2; exit 2 ;; esac
test -f "$checkpoint"
test ! -e "$output/COMPLETED"

data_root=/home/yyf/data/perturbench_mcfaline23_official
repo=/home/yyf/archive/external/PerturBench
split="$data_root/splits/mcfaline23_gxe_splits/full_covariate_split.csv"
mkdir -p "$output"

echo '34db710ad850b5b5fd478d123f1c10ae3d5aaa013fce6061be8216ea27aed215  '"$data_root/mcfaline23_gxe_processed.h5ad.gz" | sha256sum -c -
echo '08a6558a68599b3703f626c8da72a1603486edd94a3f34dfc9fa7c9b6181b091  '"$split" | sha256sum -c -
checkpoint_sha="$(sha256sum "$checkpoint" | cut -d' ' -f1)"

cat >"$output/EVALUATION_CONTRACT.txt" <<EOF
candidate=$candidate
gpu=$gpu
partition=validation
test_partition_opened=false
checkpoint=$checkpoint
checkpoint_sha256=$checkpoint_sha
split_sha256=08a6558a68599b3703f626c8da72a1603486edd94a3f34dfc9fa7c9b6181b091
EOF

common=(
  "experiment=$experiment"
  "paths.data_dir=$data_root"
  "paths.log_dir=/home/yyf/runtime_artifacts/safeconf_mcfaline23"
  "hydra.run.dir=$output"
  "data.splitter.split_path=$split"
  "data.evaluation.split_value_to_evaluate=val"
  "data.loader.num_workers=0"
  "train=false"
  "test=true"
  "ckpt_path=$checkpoint"
)

cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu"
export MLFLOW_DISABLE_AGENT_HINT=1
export HDF5_USE_FILE_LOCKING=FALSE
printf '%q ' /home/yyf/.venvs/perturbench-c84038bc/bin/train "${common[@]}" >"$output/COMMAND.sh"
printf '\n' >>"$output/COMMAND.sh"

/home/yyf/.venvs/perturbench-c84038bc/bin/train "${common[@]}" 2>&1 | tee "$output/console.log"
touch "$output/COMPLETED"
