#!/usr/bin/env bash
set -euo pipefail

candidate="${1:-}"
gpu="${2:-}"
mode="${3:-formal}"

case "$candidate" in
  latent)
    experiment="neurips2025/mcfaline23/latent_additive_best_params_mcfaline23_full"
    ;;
  decoder)
    experiment="neurips2025/mcfaline23/decoder_only_best_params_mcfaline23_full"
    ;;
  *)
    echo "usage: $0 {latent|decoder} {0|1} [preflight|formal]" >&2
    exit 2
    ;;
esac

case "$gpu" in 0|1) ;; *) echo "GPU must be 0 or 1" >&2; exit 2 ;; esac
case "$mode" in preflight|formal) ;; *) echo "mode must be preflight or formal" >&2; exit 2 ;; esac

data_root=/home/yyf/data/perturbench_mcfaline23_official
repo=/home/yyf/archive/external/PerturBench
run_root=/home/yyf/runtime_artifacts/safeconf_mcfaline23
split="$data_root/splits/mcfaline23_gxe_splits/full_covariate_split.csv"
data_gz="$data_root/mcfaline23_gxe_processed.h5ad.gz"
data_h5ad="$data_root/mcfaline23_gxe_processed.h5ad"

test -f "$data_h5ad"
echo '34db710ad850b5b5fd478d123f1c10ae3d5aaa013fce6061be8216ea27aed215  '"$data_gz" | sha256sum -c -
echo '08a6558a68599b3703f626c8da72a1603486edd94a3f34dfc9fa7c9b6181b091  '"$split" | sha256sum -c -

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$run_root/${candidate}_${mode}_${stamp}"
mkdir -p "$run_dir"

cat >"$run_dir/ATTEMPT_CONTRACT.txt" <<EOF
candidate=$candidate
mode=$mode
gpu=$gpu
experiment=$experiment
test_truth_opened=false
evaluation_split=validation
data_lfs_sha256=34db710ad850b5b5fd478d123f1c10ae3d5aaa013fce6061be8216ea27aed215
split_sha256=08a6558a68599b3703f626c8da72a1603486edd94a3f34dfc9fa7c9b6181b091
started_utc=$stamp
EOF

common=(
  "experiment=$experiment"
  "paths.data_dir=$data_root"
  "paths.log_dir=$run_root"
  "hydra.run.dir=$run_dir"
  "data.splitter.split_path=$split"
  "data.evaluation.split_value_to_evaluate=val"
  # Registered engineering repair after the first dual preflight: official
  # num_workers=12 caused worker processes to be OOM-killed before a GPU batch
  # was produced. Keep the official batch size/model/optimizer unchanged and
  # read batches in the main process so workers do not duplicate HDF5 state.
  "data.loader.num_workers=0"
  "test=false"
)

if [[ "$mode" == preflight ]]; then
  # One train and one validation batch.  This checks schema, forward/backward,
  # and 24GB-memory feasibility; it is not a reported competence result.
  common+=("+trainer.fast_dev_run=true")
fi

cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu"
export MLFLOW_DISABLE_AGENT_HINT=1
export HDF5_USE_FILE_LOCKING=FALSE

printf '%q ' /home/yyf/.venvs/perturbench-c84038bc/bin/train "${common[@]}" >"$run_dir/COMMAND.sh"
printf '\n' >>"$run_dir/COMMAND.sh"

if [[ "$mode" == formal ]]; then
  timeout --signal=INT --kill-after=10m 44h \
    /home/yyf/.venvs/perturbench-c84038bc/bin/train "${common[@]}" 2>&1 | tee "$run_dir/console.log"
else
  /home/yyf/.venvs/perturbench-c84038bc/bin/train "${common[@]}" 2>&1 | tee "$run_dir/console.log"
fi

touch "$run_dir/COMPLETED"
