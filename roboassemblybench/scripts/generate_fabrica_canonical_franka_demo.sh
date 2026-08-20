#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
TASK="${1:-car}"

case "$TASK" in
  beam|car|cooling_manifold|duct|gamepad|plumbers_block|stool_circular) ;;
  *)
    echo "Unknown Fabrica Franka task: $TASK" >&2
    exit 2
    ;;
esac

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${ISAAC_SIM_ROOT:+$ISAAC_SIM_ROOT/python.sh}}"
if [[ -z "$ISAAC_PYTHON" || ! -x "$ISAAC_PYTHON" ]]; then
  echo "Set ISAAC_SIM_ROOT or ISAAC_PYTHON to the Isaac Sim 5.1 Python launcher." >&2
  exit 1
fi

CONTROL_FPS="${CONTROL_FPS:-80}"
DATASET_FPS="${DATASET_FPS:-20}"
DATASET_FRAME_STRIDE="${DATASET_FRAME_STRIDE:-4}"
LIVE_VIDEO_FPS="${LIVE_VIDEO_FPS:-$DATASET_FPS}"
LIVE_VIDEO_FRAME_STRIDE="${LIVE_VIDEO_FRAME_STRIDE:-$DATASET_FRAME_STRIDE}"
if (( CONTROL_FPS != DATASET_FPS * DATASET_FRAME_STRIDE )); then
  echo "Timing must satisfy CONTROL_FPS = DATASET_FPS * DATASET_FRAME_STRIDE." >&2
  exit 2
fi

SEED="${SEED:-5412}"
LAYOUT_SEED="${LAYOUT_SEED:-$SEED}"
SCENE_PROFILE="${SCENE_PROFILE:-taoyuan_grscenes_tabletop}"
DOMAIN_RANDOMIZATION="${DOMAIN_RANDOMIZATION:-0}"
RANDOMIZATION_PROFILE="${RANDOMIZATION_PROFILE:-position}"
RECORD_LIVE_VIDEO="${RECORD_LIVE_VIDEO:-1}"
HEADLESS="${HEADLESS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/fabrica_franka_validation/$TASK}"
RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH:-$REPO_ROOT/.runtime_python}"
ISAAC_ASSETS_ROOT="${ISAAC_ASSETS_ROOT:-$REPO_ROOT/roboassemblybench/assets/isaac_sim_5.1}"

mkdir -p "$OUTPUT_DIR"
args=(
  "$REPO_ROOT/toolkits/factory_dual_franka_assembly/generate_demos.py"
  --worker-mode collect
  --worker-recipe "fabrica_${TASK}_franka_staged"
  --worker-seeds "$SEED"
  --worker-layout-seeds "$LAYOUT_SEED"
  --worker-scene-profile "$SCENE_PROFILE"
  --worker-results-path "$OUTPUT_DIR/results.json"
  --output-dir "$OUTPUT_DIR"
  --rendering-fps "$CONTROL_FPS"
  --record-lerobot-raw
  --record-trajectory-only
  --dataset-fps "$DATASET_FPS"
  --dataset-frame-stride "$DATASET_FRAME_STRIDE"
)

if [[ "$HEADLESS" == "1" ]]; then
  args+=(--headless)
fi
if [[ "$RECORD_LIVE_VIDEO" == "1" ]]; then
  args+=(
    --record-live-video
    --live-video-fps "$LIVE_VIDEO_FPS"
    --live-video-frame-stride "$LIVE_VIDEO_FRAME_STRIDE"
  )
fi
if [[ "$DOMAIN_RANDOMIZATION" == "1" ]]; then
  args+=(--domain-randomization --randomization-profile "$RANDOMIZATION_PROFILE")
fi

cd "$REPO_ROOT"
env -u CUDA_VISIBLE_DEVICES \
  PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 \
  ISAAC_ASSETS_ROOT="$ISAAC_ASSETS_ROOT" \
  ISAACSIM_ACTIVE_GPU="${GPU_ID:-0}" \
  ISAACSIM_PHYSICS_GPU="${GPU_ID:-0}" \
  ISAACSIM_PORTABLE_ROOT="${ISAACSIM_PORTABLE_ROOT:-/tmp/roboassemblybench_${USER}_franka_${TASK}}" \
  PYTHONPATH="$RUNTIME_PYTHONPATH:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$ISAAC_PYTHON" "${args[@]}" 2>&1 | tee "$OUTPUT_DIR/rollout.log"
