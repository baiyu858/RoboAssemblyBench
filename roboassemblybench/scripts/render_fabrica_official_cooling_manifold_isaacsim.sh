#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/outputs/fabrica_official_isaacsim/cooling_manifold_official_replay.mp4}"
FRAMES_DIR="${FRAMES_DIR:-${REPO_ROOT}/outputs/fabrica_official_isaacsim/cooling_manifold_official_replay_frames}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/third_part/Fabrica/logs/codex_cooling_manifold_official/cooling_manifold}"
MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/roboassemblybench/assets/Fabrica/fabrica_franka_cooling_optical_board_black_fullbundle_sdf001/assets/fabrica_original_usd_sdf_margin_001/aligned/cooling_manifold/manifest.json}"
FIXTURE_USD="${FIXTURE_USD:-${REPO_ROOT}/roboassemblybench/assets/Fabrica/fabrica_franka_cooling_optical_board_black_fullbundle_sdf001/assets/fabrica_fixture/cooling_manifold/fixture_pickup_tray.usda}"
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-540}"
FPS="${FPS:-30}"
STRIDE="${STRIDE:-6}"
MAPPING_MODE="${MAPPING_MODE:-scene_spec_raw_center}"
CAMERA_OPTION="${CAMERA_OPTION:-official_like}"
ROBOT_LAYOUT="${ROBOT_LAYOUT:-fabrica_workcell}"
PART_REPLAY_MODE="${PART_REPLAY_MODE:-isaac_gripper_attach}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

cd "${REPO_ROOT}"
echo "Rendering Fabrica official cooling_manifold motion replay in Isaac Sim."
echo "Input log: ${LOG_DIR}"
echo "Output: ${OUTPUT_PATH}"

conda run -n "${CONDA_ENV}" env \
  PYTHONNOUSERSITE=1 \
  python toolkits/factory_dual_franka_assembly/render_fabrica_official_motion_isaac.py \
    --log-dir "${LOG_DIR}" \
    --manifest "${MANIFEST_PATH}" \
    --fixture-usd "${FIXTURE_USD}" \
    --output "${OUTPUT_PATH}" \
    --frames-dir "${FRAMES_DIR}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --fps "${FPS}" \
    --stride "${STRIDE}" \
    --mapping-mode "${MAPPING_MODE}" \
    --camera-option "${CAMERA_OPTION}" \
    --robot-layout "${ROBOT_LAYOUT}" \
    --part-replay-mode "${PART_REPLAY_MODE}" \
    --headless \
    "$@"
