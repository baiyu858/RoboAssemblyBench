#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
RECIPE="${RECIPE:-fabrica_plumbers_block_ur5e_fabrica_online_plan}"
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
echo "Opening ${RECIPE} scene with profile ${SCENE_PROFILE} in Isaac Sim UI..."
echo "Viewer mode only: no demo policy, no robot actions, no data collection."
conda run -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 "${args[@]}" "$@"
