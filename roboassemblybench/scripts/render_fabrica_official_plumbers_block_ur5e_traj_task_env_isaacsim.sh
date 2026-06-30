#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
RECIPE="${RECIPE:-fabrica_plumbers_block_ur5e}"
SCENE_PROFILE="${SCENE_PROFILE:-taoyuan_grscenes_tabletop}"
SEED="${SEED:-0}"
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-544}"
FPS="${FPS:-30}"
STRIDE="${STRIDE:-6}"
CAMERA_OPTION="${CAMERA_OPTION:-close}"
MAX_FRAMES="${MAX_FRAMES:-}"
WARMUP_STEPS="${WARMUP_STEPS:-8}"
WORLD_OFFSET="${WORLD_OFFSET:-0.47,0,1.012}"
KEEP_TASK_REPLAY_OVERLAPS="${KEEP_TASK_REPLAY_OVERLAPS:-0}"
HEADLESS="${HEADLESS:-1}"
WEBRTC="${WEBRTC:-0}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/roboassemblybench/assets/Fabrica/official_logs/codex_plumbers_block_ur5e_official/plumbers_block}"
ASSEMBLY_DIR="${ASSEMBLY_DIR:-$REPO_ROOT/roboassemblybench/assets/Fabrica/official_replay_assets/fabrica/plumbers_block}"
ASSET_DIR="${ASSET_DIR:-$REPO_ROOT/roboassemblybench/assets/Fabrica/official_replay_assets}"
OUTPUT="${OUTPUT:-$REPO_ROOT/outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_task_env_replay.mp4}"
FRAMES_DIR="${FRAMES_DIR:-$REPO_ROOT/outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_taoyuan_task_env_replay_frames}"

echo "Rendering official Fabrica UR5e plumbers_block traj replay inside RoboAssemblyBench task env."
echo "Recipe: $RECIPE"
echo "Scene profile: $SCENE_PROFILE"
echo "Input log: $LOG_DIR"
echo "Replay world offset: $WORLD_OFFSET"
echo "Output: $OUTPUT"
echo "Headless: $HEADLESS"
echo "WebRTC: $WEBRTC"

args=(
  python toolkits/factory_dual_franka_assembly/render_fabrica_traj_replay_in_task_env.py
  --recipe "$RECIPE"
  --scene-profile "$SCENE_PROFILE"
  --seed "$SEED"
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
  --world-offset "$WORLD_OFFSET"
  --warmup-steps "$WARMUP_STEPS"
)

if [[ "$HEADLESS" == "1" || "$HEADLESS" == "true" ]]; then
  args+=(--headless)
else
  args+=(--no-headless)
  if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "Warning: DISPLAY/WAYLAND_DISPLAY is empty, so Isaac Sim may still be unable to open a GUI window." >&2
  fi
fi

if [[ "$WEBRTC" == "1" || "$WEBRTC" == "true" ]]; then
  args+=(--webrtc)
fi

if [[ -n "$MAX_FRAMES" ]]; then
  args+=(--max-frames "$MAX_FRAMES")
fi

if [[ "$KEEP_TASK_REPLAY_OVERLAPS" == "1" || "$KEEP_TASK_REPLAY_OVERLAPS" == "true" ]]; then
  args+=(--keep-task-replay-overlaps)
fi

conda run -n "$CONDA_ENV" env PYTHONNOUSERSITE=1 "${args[@]}"
