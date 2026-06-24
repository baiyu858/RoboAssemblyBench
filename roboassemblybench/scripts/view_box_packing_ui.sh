#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
RECIPE="${RECIPE:-box_packing}"
SCENE_PROFILE="${SCENE_PROFILE:-taoyuan_grscenes_tabletop}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-roboassemblybench/outputs/${RECIPE}_ui_preview}"
LIVE_VIDEO_FPS="${LIVE_VIDEO_FPS:-30}"
LIVE_VIDEO_FRAME_STRIDE="${LIVE_VIDEO_FRAME_STRIDE:-1}"
RECORD_LIVE_VIDEO="${RECORD_LIVE_VIDEO:-0}"
KEEP_VIDEO_FRAMES="${KEEP_VIDEO_FRAMES:-0}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Open the RoboAssemblyBench box packing task in the Isaac Sim UI.

Defaults:
  RECIPE=box_packing
  SCENE_PROFILE=taoyuan_grscenes_tabletop
  CONDA_ENV=internutopia311

Examples:
  roboassemblybench/scripts/view_box_packing_ui.sh
  RECORD_LIVE_VIDEO=1 KEEP_VIDEO_FRAMES=1 roboassemblybench/scripts/view_box_packing_ui.sh
  SEED=3 roboassemblybench/scripts/view_box_packing_ui.sh
  SCENE_PROFILE=proxy_factory_cell OUTPUT_DIR=/tmp/box_packing_ui roboassemblybench/scripts/view_box_packing_ui.sh

Any extra arguments are forwarded to generate_demos.py.
EOF
  exit 0
fi

args=(
  python roboassemblybench/scripts/generate_demos.py
  --worker-mode collect
  --worker-recipe "${RECIPE}"
  --worker-scene-profile "${SCENE_PROFILE}"
  --worker-results-path "${OUTPUT_DIR}/ui_results.json"
  --worker-seeds "${SEED}"
  --start-seed "${SEED}"
  --max-trials 1
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${RECORD_LIVE_VIDEO}" == "1" ]]; then
  args+=(
    --record-live-video
    --live-video-fps "${LIVE_VIDEO_FPS}"
    --live-video-frame-stride "${LIVE_VIDEO_FRAME_STRIDE}"
  )
fi

if [[ "${KEEP_VIDEO_FRAMES}" == "1" ]]; then
  args+=(--keep-video-frames)
fi

cd "${REPO_ROOT}"
echo "Opening ${RECIPE} with scene profile ${SCENE_PROFILE} in Isaac Sim UI..."
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "Warning: DISPLAY/WAYLAND_DISPLAY is empty, so Isaac may still fall back to headless." >&2
fi
echo "Output dir: ${OUTPUT_DIR}"
conda run -n "${CONDA_ENV}" "${args[@]}" "$@"
