#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
SCENE_PROFILE="${SCENE_PROFILE:-taoyuan_grscenes_tabletop}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fabrica_plumbers_block_online_plan_fixplug_rl_demo}"
NUM_DEMOS="${NUM_DEMOS:-1}"
MAX_TRIALS="${MAX_TRIALS:-1}"
START_SEED="${START_SEED:-0}"
LIVE_VIDEO_FPS="${LIVE_VIDEO_FPS:-30}"
LIVE_VIDEO_FRAME_STRIDE="${LIVE_VIDEO_FRAME_STRIDE:-2}"
HEADLESS_ARG=()
if [[ "${HEADLESS:-0}" == "1" || "${HEADLESS:-false}" == "true" ]]; then
  HEADLESS_ARG=(--headless)
fi

cd "${REPO_ROOT}"
echo "Generating Fabrica plumbers_block demo with Isaac Sim online planning and local FixPlug RL insertion."
echo "Output directory: ${REPO_ROOT}/${OUTPUT_DIR}"
conda run -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python roboassemblybench/scripts/generate_demos.py \
  --recipes fabrica_plumbers_block_online_plan_fixplug_rl \
  --scene-profiles "${SCENE_PROFILE}" \
  --num-demos "${NUM_DEMOS}" \
  --max-trials "${MAX_TRIALS}" \
  --start-seed "${START_SEED}" \
  --record-live-video \
  --live-video-fps "${LIVE_VIDEO_FPS}" \
  --live-video-frame-stride "${LIVE_VIDEO_FRAME_STRIDE}" \
  --output-dir "${OUTPUT_DIR}" \
  "${HEADLESS_ARG[@]}" \
  "$@"
