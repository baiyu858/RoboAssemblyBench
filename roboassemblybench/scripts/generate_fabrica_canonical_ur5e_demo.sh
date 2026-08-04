#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASK="${TASK:-${1:-}}"
if [[ -z "${TASK}" ]]; then
  echo "Usage: $0 <canonical-assembly> [generate_demos arguments...]" >&2
  exit 2
fi
if [[ "${1:-}" == "${TASK}" ]]; then
  shift
fi

CONDA_ENV="${CONDA_ENV:-internutopia311}"
SCENE_PROFILE="${SCENE_PROFILE:-taoyuan_grscenes_tabletop}"
RECIPE="fabrica_${TASK}_ur5e_staged"
RECIPE_PATH="${REPO_ROOT}/roboassemblybench/tasks/${RECIPE}/recipe.yaml"
if [[ ! -f "${RECIPE_PATH}" ]]; then
  echo "Canonical Fabrica recipe not found: ${RECIPE_PATH}" >&2
  exit 2
fi
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RECIPE}_demo}"
NUM_DEMOS="${NUM_DEMOS:-1}"
START_SEED="${START_SEED:-0}"
MAX_TRIALS="${MAX_TRIALS:-1}"
RESULTS_PATH="${RESULTS_PATH:-${OUTPUT_DIR}/collect_results.json}"
LIVE_VIDEO_FPS="${LIVE_VIDEO_FPS:-30}"
LIVE_VIDEO_FRAME_STRIDE="${LIVE_VIDEO_FRAME_STRIDE:-8}"

SEEDS=()
for ((i = 0; i < NUM_DEMOS; i++)); do
  SEEDS+=("$((START_SEED + i))")
done

HEADLESS_ARG=()
if [[ "${HEADLESS:-1}" == "1" || "${HEADLESS:-true}" == "true" ]]; then
  HEADLESS_ARG=(--headless)
fi
RANDOMIZATION_ARG=()
if [[ "${DOMAIN_RANDOMIZATION:-1}" == "1" || "${DOMAIN_RANDOMIZATION:-true}" == "true" ]]; then
  RANDOMIZATION_ARG=(--domain-randomization)
fi
SKIP_EPISODE_STEPS_ARG=()
if [[ "${SKIP_EPISODE_STEPS:-1}" == "1" || "${SKIP_EPISODE_STEPS:-true}" == "true" ]]; then
  SKIP_EPISODE_STEPS_ARG=(--skip-episode-steps)
fi

cd "${REPO_ROOT}"
echo "Generating ${RECIPE} with seeds: ${SEEDS[*]}"
if ((${#RANDOMIZATION_ARG[@]})); then
  echo "Optical board: fixed; pickup/assembly layouts and scene appearance: randomized."
else
  echo "Optical board: fixed; domain randomization: disabled."
fi
conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python roboassemblybench/scripts/generate_demos.py \
  --worker-mode collect \
  --worker-recipe "${RECIPE}" \
  --worker-scene-profile "${SCENE_PROFILE}" \
  --worker-results-path "${RESULTS_PATH}" \
  --worker-seeds "${SEEDS[@]}" \
  --max-trials "${MAX_TRIALS}" \
  --start-seed "${START_SEED}" \
  --record-live-video \
  --live-video-fps "${LIVE_VIDEO_FPS}" \
  --live-video-frame-stride "${LIVE_VIDEO_FRAME_STRIDE}" \
  --output-dir "${OUTPUT_DIR}" \
  "${RANDOMIZATION_ARG[@]}" \
  "${SKIP_EPISODE_STEPS_ARG[@]}" \
  "${HEADLESS_ARG[@]}" \
  "$@"
