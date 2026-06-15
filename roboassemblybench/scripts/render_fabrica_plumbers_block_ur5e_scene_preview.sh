#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
RECIPE="${RECIPE:-fabrica_plumbers_block_ur5e}"
SCENE_PROFILE="${SCENE_PROFILE:-taoyuan_grscenes_tabletop}"
SEED="${SEED:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/outputs/fabrica_plumbers_block_ur5e/plumbers_block_ur5e_scene_preview.mp4}"
FRAMES_DIR="${FRAMES_DIR:-${REPO_ROOT}/outputs/fabrica_plumbers_block_ur5e/plumbers_block_ur5e_scene_preview_frames}"
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-544}"
FPS="${FPS:-30}"
FRAME_COUNT="${FRAME_COUNT:-120}"
WARMUP_STEPS="${WARMUP_STEPS:-8}"
CAMERA_OPTION="${CAMERA_OPTION:-official_like}"
OBJECT_PREFIX="${OBJECT_PREFIX:-fabrica_plumbers_block}"
INCLUDE_ROBOTS_IN_CAMERA="${INCLUDE_ROBOTS_IN_CAMERA:-1}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

cd "${REPO_ROOT}"
echo "Rendering Isaac Sim scene preview for ${RECIPE}."
echo "Output: ${OUTPUT_PATH}"
echo "Note: this is not a UR5e retargeted assembly-motion replay."

conda run -n "${CONDA_ENV}" env \
  PYTHONNOUSERSITE=1 \
  python toolkits/factory_dual_franka_assembly/render_task_scene_preview.py \
    --recipe "${RECIPE}" \
    --scene-profile "${SCENE_PROFILE}" \
    --seed "${SEED}" \
    --output "${OUTPUT_PATH}" \
    --frames-dir "${FRAMES_DIR}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --fps "${FPS}" \
    --frame-count "${FRAME_COUNT}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --camera-option "${CAMERA_OPTION}" \
    --object-prefix "${OBJECT_PREFIX}" \
    --headless \
    $(if [[ "${INCLUDE_ROBOTS_IN_CAMERA}" == "1" ]]; then printf '%s' "--include-robots-in-camera"; fi) \
    "$@"
