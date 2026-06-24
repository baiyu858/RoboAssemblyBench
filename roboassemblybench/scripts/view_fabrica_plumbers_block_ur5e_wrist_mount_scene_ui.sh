#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
RECIPE="${RECIPE:-fabrica_plumbers_block_ur5e_wrist_mount}"
SCENE_PROFILE="${SCENE_PROFILE:-taoyuan_grscenes_tabletop}"
SEED="${SEED:-0}"
WARMUP_RENDER_STEPS="${WARMUP_RENDER_STEPS:-8}"
ATTACH_RUNTIME_CAMERAS="${ATTACH_RUNTIME_CAMERAS:-0}"

args=(
  python roboassemblybench/scripts/view_task_scene.py
  --recipe "${RECIPE}"
  --scene-profile "${SCENE_PROFILE}"
  --seed "${SEED}"
  --warmup-render-steps "${WARMUP_RENDER_STEPS}"
)

if [[ "${ATTACH_RUNTIME_CAMERAS}" == "1" ]]; then
  args+=(--attach-runtime-cameras)
fi

cd "${REPO_ROOT}"
echo "Opening ${RECIPE} in Isaac Sim UI..."
echo "Viewer mode only: no policy, no demo generation."
echo "Expected gripper parent: /World/env_0/robots/ur5e_left/wrist_3_link/Gripper and right equivalent."
conda run -n "${CONDA_ENV}" "${args[@]}" "$@"
