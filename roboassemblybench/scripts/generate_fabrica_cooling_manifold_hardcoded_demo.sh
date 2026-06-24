#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
SCENE_PROFILE="${SCENE_PROFILE:-taoyuan_grscenes_tabletop}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fabrica_cooling_manifold_hardcoded_demo}"
NUM_DEMOS="${NUM_DEMOS:-1}"
MAX_TRIALS="${MAX_TRIALS:-3}"
START_SEED="${START_SEED:-0}"
LIVE_VIDEO_FPS="${LIVE_VIDEO_FPS:-30}"
LIVE_VIDEO_FRAME_STRIDE="${LIVE_VIDEO_FRAME_STRIDE:-8}"
HEADLESS_ARG=()
if [[ "${HEADLESS:-0}" == "1" || "${HEADLESS:-false}" == "true" ]]; then
  HEADLESS_ARG=(--headless)
fi

cd "${REPO_ROOT}"
echo "Generating hardcoded automatic Fabrica cooling-manifold demo."
conda run -n "${CONDA_ENV}" python roboassemblybench/scripts/generate_demos.py \
  --recipes fabrica_cooling_manifold_hardcoded \
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

echo "Output: ${REPO_ROOT}/${OUTPUT_DIR}"
