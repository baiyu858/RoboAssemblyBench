#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
RECIPE="${RECIPE:-fabrica_ur5e_d405_plumbers_block_minimal_workcell}"
SCENE_PROFILE="${SCENE_PROFILE:-}"
SEED="${SEED:-0}"
WARMUP_RENDER_STEPS="${WARMUP_RENDER_STEPS:-8}"
ATTACH_RUNTIME_CAMERAS="${ATTACH_RUNTIME_CAMERAS:-0}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Open the standalone Fabrica UR5e+D405 plumbers-block minimal workcell task in Isaac Sim UI.

Defaults:
  RECIPE=fabrica_ur5e_d405_plumbers_block_minimal_workcell
  SCENE_PROFILE is unset, so the recipe's scene_profile is used
  CONDA_ENV=internutopia311

Examples:
  roboassemblybench/scripts/view_fabrica_ur5e_d405_plumbers_block_minimal_workcell_scene_ui.sh
  SEED=3 roboassemblybench/scripts/view_fabrica_ur5e_d405_plumbers_block_minimal_workcell_scene_ui.sh
EOF
  exit 0
fi

args=(
  python roboassemblybench/scripts/view_task_scene.py
  --recipe "${RECIPE}"
  --seed "${SEED}"
  --warmup-render-steps "${WARMUP_RENDER_STEPS}"
)

if [[ -n "${SCENE_PROFILE}" ]]; then
  args+=(--scene-profile "${SCENE_PROFILE}")
fi

if [[ "${ATTACH_RUNTIME_CAMERAS}" == "1" ]]; then
  args+=(--attach-runtime-cameras)
fi

cd "${REPO_ROOT}"
if [[ -n "${SCENE_PROFILE}" ]]; then
  echo "Opening ${RECIPE} scene with profile ${SCENE_PROFILE} in Isaac Sim UI..."
else
  echo "Opening ${RECIPE} scene with its recipe-defined scene profile in Isaac Sim UI..."
fi
echo "Standalone bundle: roboassemblybench/assets/Fabrica/fabrica_ur5e_d405_plumbers_block_minimal_workcell_fullbundle_sdf001_v1"
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "Warning: DISPLAY/WAYLAND_DISPLAY is empty, so Isaac may not open a local UI." >&2
fi
conda run -n "${CONDA_ENV}" "${args[@]}" "$@"
