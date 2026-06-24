#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FABRICA_IG="${REPO_ROOT}/third_part/Fabrica/learning/isaacgymenvs"

CONDA_ENV="${CONDA_ENV:-fabrica_isaacgym_py38}"
PLUG="${PLUG:-6}"
SOCKET="${SOCKET:-1}"
NUM_ENVS="${NUM_ENVS:-1}"
GPU="${GPU:-0}"
HEADLESS="${HEADLESS:-False}"
TEST="${TEST:-False}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
FRICTION="${FRICTION:-1}"
CHECKPOINT="${CHECKPOINT:-}"
OPENLOOP="${OPENLOOP:-False}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the official Fabrica FixPlug cooling-manifold insertion task for one plug/socket pair.

This uses third_part/Fabrica's Isaac Gym task:
  task=FabricaFixPlugTaskAssemble

It does not load the current InternUtopia dual-arm recipe, does not modify board size,
does not move cameras, and does not attempt table-top friction grasping.

Defaults:
  CONDA_ENV=fabrica_isaacgym_py38
  PLUG=6
  SOCKET=1
  NUM_ENVS=1
  HEADLESS=False
  TEST=False
  OPENLOOP=False
  MAX_ITERATIONS=1

Examples:
  roboassemblybench/scripts/run_fabrica_fixplug_cooling_manifold_pair.sh
  PLUG=0 SOCKET=1 HEADLESS=True OPENLOOP=True roboassemblybench/scripts/run_fabrica_fixplug_cooling_manifold_pair.sh
  TEST=True CHECKPOINT=/path/to/policy.pth HEADLESS=False roboassemblybench/scripts/run_fabrica_fixplug_cooling_manifold_pair.sh
EOF
  exit 0
fi

if [[ ! -d "${FABRICA_IG}" ]]; then
  echo "Missing Fabrica Isaac Gym dir: ${FABRICA_IG}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${REPO_ROOT}/third_part/Fabrica/learning:${PYTHONPATH:-}"

args=(
  python train.py
  task=FabricaFixPlugTaskAssemble
  'task.env.assemblies=["cooling_manifold"]'
  "task.env.part_plug=${PLUG}"
  "task.env.part_socket=${SOCKET}"
  "task.env.numEnvs=${NUM_ENVS}"
  "max_iterations=${MAX_ITERATIONS}"
  "headless=${HEADLESS}"
  "task.env.franka_friction=${FRICTION}"
)

if [[ "${OPENLOOP}" == "True" || "${OPENLOOP}" == "true" || "${OPENLOOP}" == "1" ]]; then
  args+=("task.env.openloop=True" "task.env.residual_action=False" "test=True")
else
  args+=("test=${TEST}")
fi

if [[ -n "${CHECKPOINT}" ]]; then
  args+=("checkpoint=${CHECKPOINT}")
fi

cd "${FABRICA_IG}"
echo "Running official Fabrica FixPlug cooling_manifold ${PLUG}->${SOCKET}"
echo "Env: ${CONDA_ENV}; num_envs=${NUM_ENVS}; headless=${HEADLESS}; test=${TEST}; openloop=${OPENLOOP}"
conda run -n "${CONDA_ENV}" bash -lc 'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; exec "$@"' bash "${args[@]}"
