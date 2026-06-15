#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/outputs/fabrica_official_hybrid_fixplug_rl/plumbers_block_official_hybrid_fixplug_rl.mp4}"
FRAMES_DIR="${FRAMES_DIR:-${REPO_ROOT}/outputs/fabrica_official_hybrid_fixplug_rl/plumbers_block_official_hybrid_fixplug_rl_frames}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/third_part/Fabrica/logs/codex_plumbers_block_official/plumbers_block}"
ASSET_ROOT="${ASSET_ROOT:-${REPO_ROOT}/roboassemblybench/assets/Fabrica/fabrica_franka_plumbers_block_optical_board_black_fullbundle_sdf001}"
MANIFEST_PATH="${MANIFEST_PATH:-${ASSET_ROOT}/assets/fabrica_original_usd_sdf_margin_001/aligned/plumbers_block/manifest.json}"
SCENE_SPEC="${SCENE_SPEC:-${ASSET_ROOT}/scene/scene_spec.json}"
FIXTURE_USD="${FIXTURE_USD:-${ASSET_ROOT}/assets/fabrica_fixture/plumbers_block/fixture_pickup_tray.usda}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/roboassemblybench/assets/Fabrica/checkpoints/plumbers_block_fixplug_rl/sr_gen_plumbers_block.pth}"
PLAN_INFO="${PLAN_INFO:-${REPO_ROOT}/roboassemblybench/assets/Fabrica/checkpoints/plumbers_block_fixplug_rl/plumbers_block_plan_info.pkl}"
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-544}"
FPS="${FPS:-30}"
STRIDE="${STRIDE:-6}"
TRACKING_SUBSTEPS="${TRACKING_SUBSTEPS:-4}"
ALLOW_NONCONTACT_ATTACH="${ALLOW_NONCONTACT_ATTACH:-0}"

ALLOW_ARG=()
if [[ "${ALLOW_NONCONTACT_ATTACH}" == "1" || "${ALLOW_NONCONTACT_ATTACH}" == "true" ]]; then
  ALLOW_ARG=(--allow-noncontact-attach)
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

cd "${REPO_ROOT}"
echo "Running Fabrica plumbers_block official trajectory with FixPlug RL windows in Isaac Sim."
echo "Output: ${OUTPUT_PATH}"

conda run -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python \
  toolkits/factory_dual_franka_assembly/run_fabrica_official_hybrid_isaac.py \
    --output "${OUTPUT_PATH}" \
    --frames-dir "${FRAMES_DIR}" \
    --log-dir "${LOG_DIR}" \
    --manifest "${MANIFEST_PATH}" \
    --scene-spec "${SCENE_SPEC}" \
    --fixture-usd "${FIXTURE_USD}" \
    --checkpoint "${CHECKPOINT}" \
    --plan-info "${PLAN_INFO}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --fps "${FPS}" \
    --stride "${STRIDE}" \
    --tracking-substeps "${TRACKING_SUBSTEPS}" \
    --headless \
    "${ALLOW_ARG[@]}" \
    "$@"
