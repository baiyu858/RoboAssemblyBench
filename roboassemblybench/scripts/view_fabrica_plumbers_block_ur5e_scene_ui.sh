#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
RECIPE="${RECIPE:-fabrica_plumbers_block_ur5e}"
SCENE_PROFILE="${SCENE_PROFILE:-taoyuan_grscenes_tabletop}"
SEED="${SEED:-0}"
WARMUP_RENDER_STEPS="${WARMUP_RENDER_STEPS:-8}"
ATTACH_RUNTIME_CAMERAS="${ATTACH_RUNTIME_CAMERAS:-0}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Open only the RoboAssemblyBench Fabrica plumbers-block UR5e task scene in Isaac Sim UI.

This viewer loads the task scene and then idles. It does not run the demo policy,
does not step robot actions, and does not collect data.

Defaults:
  RECIPE=fabrica_plumbers_block_ur5e
  SCENE_PROFILE=taoyuan_grscenes_tabletop
  CONDA_ENV=internutopia311

Examples:
  roboassemblybench/scripts/view_fabrica_plumbers_block_ur5e_scene_ui.sh
  SEED=3 roboassemblybench/scripts/view_fabrica_plumbers_block_ur5e_scene_ui.sh
  ATTACH_RUNTIME_CAMERAS=1 roboassemblybench/scripts/view_fabrica_plumbers_block_ur5e_scene_ui.sh

Any extra arguments are forwarded to view_task_scene.py.
EOF
  exit 0
fi

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
echo "Robots: logical names franka_left/franka_right, actual type UR5eRobot at prims /ur5e_left and /ur5e_right."
echo "Plumbers assets: roboassemblybench/assets/Fabrica/fabrica_franka_plumbers_block_optical_board_black_fullbundle_sdf001"
echo "UR5e robot asset: roboassemblybench/assets/Fabrica/fabrica_ur5e_cooling_optical_board_black_fullbundle_sdf001"
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "Warning: DISPLAY/WAYLAND_DISPLAY is empty, so Isaac may not open a local UI." >&2
fi
conda run -n "${CONDA_ENV}" "${args[@]}" "$@"
