#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ACT_ENV="${ACT_ENV:-roboassemblybench-act}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/outputs/fabrica_plumbers_block_ur5e_right_base_prepare_2k_lerobot_v3}"
DATASET_REPO_ID="${DATASET_REPO_ID:-baiyu858/roboassemblybench_fabrica_plumbers_block_ur5e_2k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/fabrica_plumbers_block_ur5e_right_base_prepare_act}"
RESUME="${RESUME:-false}"

RESUME_ARGS=(--resume=false)
if [[ "${RESUME}" == "true" ]]; then
  RESUME_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
  if [[ ! -f "${RESUME_CONFIG}" ]]; then
    echo "ACT resume config is missing: ${RESUME_CONFIG}" >&2
    exit 2
  fi
  RESUME_ARGS=(--config_path="${RESUME_CONFIG}" --resume=true)
fi

cd "${REPO_ROOT}"
exec conda run --no-capture-output -n "${ACT_ENV}" bash -c '
  set -euo pipefail
  repo_root="$1"
  shift
  export PYTHONNOUSERSITE=1
  export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
  exec lerobot-train "$@"
' _ "${REPO_ROOT}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp="${USE_AMP:-true}" \
  --policy.push_to_hub=false \
  --policy.chunk_size="${CHUNK_SIZE:-100}" \
  --policy.n_action_steps="${ACTION_STEPS:-25}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=fabrica_plumbers_block_ur5e_act \
  "${RESUME_ARGS[@]}" \
  --batch_size="${BATCH_SIZE:-4}" \
  --num_workers="${NUM_WORKERS:-2}" \
  --steps="${STEPS:-100000}" \
  --eval_freq=0 \
  --log_freq="${LOG_FREQ:-100}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ:-10000}" \
  --wandb.enable="${WANDB_ENABLE:-false}" \
  "$@"
