#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FABRICA_ROOT="${REPO_ROOT}/third_part/Fabrica"

CONDA_ENV="${CONDA_ENV:-RoboFactory}"
EXP_NAME="${EXP_NAME:-codex_cooling_manifold_official}"
ASSEMBLY="${ASSEMBLY:-cooling_manifold}"
ASSEMBLY_DIR="${ASSEMBLY_DIR:-fabrica}"
CAMERA_OPTION="${CAMERA_OPTION:-0}"
FPS="${FPS:-30}"
OUTPUT_PATH="${OUTPUT_PATH:-${FABRICA_ROOT}/records/opengl/codex_official_nonrl/${ASSEMBLY}.mp4}"
REDMAX_BUILD="${REDMAX_BUILD:-${FABRICA_ROOT}/simulation/build/lib.linux-x86_64-cpython-39}"
FABRICA_PYTHONPATH="${FABRICA_PYTHONPATH:-${REDMAX_BUILD}}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

cd "${FABRICA_ROOT}"
echo "Rendering official Fabrica non-RL motion plan for ${ASSEMBLY}."
echo "Input log: ${FABRICA_ROOT}/logs/${EXP_NAME}/${ASSEMBLY}"
echo "Output: ${OUTPUT_PATH}"

conda run -n "${CONDA_ENV}" env \
  PYTHONPATH="${FABRICA_PYTHONPATH}" \
  PYTHONNOUSERSITE=1 \
  MPLBACKEND=Agg \
  python rendering/render_motion_plan.py \
    --assembly-dir "assets/${ASSEMBLY_DIR}/${ASSEMBLY}" \
    --log-dir "logs/${EXP_NAME}/${ASSEMBLY}" \
    --record-path "${OUTPUT_PATH}" \
    --camera-option "${CAMERA_OPTION}" \
    --fps "${FPS}" \
    "$@"
