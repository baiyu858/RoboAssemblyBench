#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FABRICA_ROOT="${REPO_ROOT}/third_part/Fabrica"
GYM_ENV="${GYM_ENV:-fabrica_isaacgym_py38}"
GYM_ENV_PREFIX="${GYM_ENV_PREFIX:-/home/baiyu24/APP/miniconda3/envs/${GYM_ENV}}"

ASSEMBLY="${ASSEMBLY:-cooling_manifold}"
PLUG="${PLUG:-6}"
SOCKET="${SOCKET:-1}"
NUM_ENVS="${NUM_ENVS:-1}"
HEADLESS="${HEADLESS:-False}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
RL_DEVICE="${RL_DEVICE:-cuda:0}"
PIPELINE="${PIPELINE:-gpu}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"

cd "${FABRICA_ROOT}/learning/isaacgymenvs"
echo "Opening official FabricaTaskAsset for ${ASSEMBLY} plug=${PLUG} socket=${SOCKET}."
echo "This loads official Isaac Gym assets only; it does not run the RL insertion policy."

conda run -n "${GYM_ENV}" env LD_LIBRARY_PATH="${GYM_ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  python train.py \
    task=FabricaTaskAsset \
    "task.env.assemblies=[${ASSEMBLY}]" \
    "task.env.part_plug=${PLUG}" \
    "task.env.part_socket=${SOCKET}" \
    "task.env.numEnvs=${NUM_ENVS}" \
    "headless=${HEADLESS}" \
    "sim_device=${SIM_DEVICE}" \
    "rl_device=${RL_DEVICE}" \
    "pipeline=${PIPELINE}" \
    "max_iterations=${MAX_ITERATIONS}" \
    "$@"
