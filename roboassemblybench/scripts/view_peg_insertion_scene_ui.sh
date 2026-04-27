#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
RECIPE="${RECIPE:-peg_insertion}"
SCENE_PROFILE="${SCENE_PROFILE:-taoyuan_grscenes_tabletop}"
SEED="${SEED:-0}"
WARMUP_RENDER_STEPS="${WARMUP_RENDER_STEPS:-8}"
ATTACH_RUNTIME_CAMERAS="${ATTACH_RUNTIME_CAMERAS:-0}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Open only the RoboAssemblyBench task scene in Isaac Sim UI.

This viewer loads the task scene and then idles. It does not run the demo policy,
does not step robot actions, and does not collect data.

Defaults:
  RECIPE=peg_insertion
  SCENE_PROFILE=taoyuan_grscenes_tabletop
  CONDA_ENV=internutopia311

Examples:
  roboassemblybench/scripts/view_peg_insertion_scene_ui.sh
  SEED=3 roboassemblybench/scripts/view_peg_insertion_scene_ui.sh
  SCENE_PROFILE=proxy_factory_cell roboassemblybench/scripts/view_peg_insertion_scene_ui.sh
  ATTACH_RUNTIME_CAMERAS=1 roboassemblybench/scripts/view_peg_insertion_scene_ui.sh

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
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "Warning: DISPLAY/WAYLAND_DISPLAY is empty, so Isaac may not open a local UI." >&2
fi
"${args[@]}" "$@"
