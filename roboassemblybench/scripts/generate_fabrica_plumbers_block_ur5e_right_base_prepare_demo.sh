#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
SCENE_PROFILE="${SCENE_PROFILE:-taoyuan_grscenes_tabletop}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fabrica_plumbers_block_ur5e_right_base_prepare_demo}"
NUM_DEMOS="${NUM_DEMOS:-1}"
MAX_TRIALS="${MAX_TRIALS:-1}"
START_SEED="${START_SEED:-0}"
LIVE_VIDEO_FPS="${LIVE_VIDEO_FPS:-30}"
LIVE_VIDEO_FRAME_STRIDE="${LIVE_VIDEO_FRAME_STRIDE:-8}"
RESULTS_PATH="${RESULTS_PATH:-${OUTPUT_DIR}/collect_results.json}"
export UR5E_DEBUG_GRASP="${UR5E_DEBUG_GRASP:-1}"
export UR5E_DEBUG_TRANSPORT_EVERY="${UR5E_DEBUG_TRANSPORT_EVERY:-10}"
HEADLESS_ARG=()
if [[ "${HEADLESS:-1}" == "1" || "${HEADLESS:-true}" == "true" ]]; then
  HEADLESS_ARG=(--headless)
fi
KEEP_VIDEO_FRAMES_ARG=()
if [[ "${KEEP_VIDEO_FRAMES:-0}" == "1" || "${KEEP_VIDEO_FRAMES:-false}" == "true" ]]; then
  KEEP_VIDEO_FRAMES_ARG=(--keep-video-frames)
fi

SEEDS=()
for ((i = 0; i < NUM_DEMOS; i++)); do
  SEEDS+=("$((START_SEED + i))")
done

cd "${REPO_ROOT}"
echo "Generating UR5e plumbers_block right-base-preparation atomic-skill demo."
echo "Mode: headless collect worker; failed rollouts are saved for debug."
echo "Seeds: ${SEEDS[*]}"
echo "Output directory: ${REPO_ROOT}/${OUTPUT_DIR}"
echo "UR5E debug logs: UR5E_DEBUG_GRASP=${UR5E_DEBUG_GRASP}, UR5E_DEBUG_TRANSPORT_EVERY=${UR5E_DEBUG_TRANSPORT_EVERY}"
echo "Expected videos per episode:"
echo "  episode_XXXX_live_videos/observation_images_front.mp4"
echo "  episode_XXXX_live_videos/observation_images_left_wrist.mp4"
echo "  episode_XXXX_live_videos/observation_images_right_wrist.mp4"
conda run -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python roboassemblybench/scripts/generate_demos.py \
  --worker-mode collect \
  --worker-recipe fabrica_plumbers_block_ur5e_right_base_prepare \
  --worker-scene-profile "${SCENE_PROFILE}" \
  --worker-results-path "${RESULTS_PATH}" \
  --worker-seeds "${SEEDS[@]}" \
  --max-trials "${MAX_TRIALS}" \
  --start-seed "${START_SEED}" \
  --record-live-video \
  --live-video-fps "${LIVE_VIDEO_FPS}" \
  --live-video-frame-stride "${LIVE_VIDEO_FRAME_STRIDE}" \
  --output-dir "${OUTPUT_DIR}" \
  "${KEEP_VIDEO_FRAMES_ARG[@]}" \
  "${HEADLESS_ARG[@]}" \
  "$@"
