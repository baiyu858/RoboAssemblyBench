#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-544}"
FPS="${FPS:-30}"
STRIDE="${STRIDE:-6}"
CAMERA_OPTION="${CAMERA_OPTION:-close}"
MAX_FRAMES="${MAX_FRAMES:-}"

FACTORY_SCENE="${FACTORY_SCENE:-taoyuan_grscenes_tabletop}"
INCLUDE_PROFILE_OBJECTS="${INCLUDE_PROFILE_OBJECTS:-1}"
WORLD_OFFSET="${WORLD_OFFSET:-0.47,0,1.012}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/roboassemblybench/assets/Fabrica/official_logs/codex_plumbers_block_ur5e_official/plumbers_block}"
ASSEMBLY_DIR="${ASSEMBLY_DIR:-$REPO_ROOT/third_part/Fabrica/assets/fabrica/plumbers_block}"
ASSET_DIR="${ASSET_DIR:-$REPO_ROOT/third_part/Fabrica/assets}"
OUTPUT="${OUTPUT:-$REPO_ROOT/outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_replay.mp4}"
FRAMES_DIR="${FRAMES_DIR:-$REPO_ROOT/outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_replay_frames}"

echo "Rendering official Fabrica UR5e plumbers_block traj replay in Isaac Sim factory scene."
echo "Input log: $LOG_DIR"
echo "Factory scene: $FACTORY_SCENE"
echo "Replay world offset: $WORLD_OFFSET"
echo "Output: $OUTPUT"

args=(
  python toolkits/factory_dual_franka_assembly/render_fabrica_traj_replay_isaac.py
  --log-dir "$LOG_DIR"
  --assembly-dir "$ASSEMBLY_DIR"
  --asset-dir "$ASSET_DIR"
  --output "$OUTPUT"
  --frames-dir "$FRAMES_DIR"
  --width "$WIDTH"
  --height "$HEIGHT"
  --fps "$FPS"
  --stride "$STRIDE"
  --camera-option "$CAMERA_OPTION"
  --factory-scene "$FACTORY_SCENE"
  --world-offset "$WORLD_OFFSET"
  --headless
)

if [[ "$INCLUDE_PROFILE_OBJECTS" == "1" || "$INCLUDE_PROFILE_OBJECTS" == "true" ]]; then
  args+=(--include-profile-objects)
fi

if [[ -n "$MAX_FRAMES" ]]; then
  args+=(--max-frames "$MAX_FRAMES")
fi

conda run -n "$CONDA_ENV" env PYTHONNOUSERSITE=1 "${args[@]}"
