#!/usr/bin/env bash
set -euo pipefail

if [[ -f /root/.bashrc ]]; then
  set +u
  source /root/.bashrc
  set -u
fi

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/baiyongjie/projectA17}"
REPO_ROOT="${REPO_ROOT:-$PROJECT_ROOT/source/InternUtopia_ur5e_production_20260820}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-$PROJECT_ROOT/runtime/isaac-sim-5.1.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/fabrica_ur5e_60060_twostage_80hz_front640_depth352_renderfix_20260820}"
LOG_PATH="${LOG_PATH:-$PROJECT_ROOT/logs/fabrica_ur5e_60060_launcher.log}"
PID_PATH="${PID_PATH:-$PROJECT_ROOT/logs/fabrica_ur5e_60060_launcher.pid}"

for path in \
  "$REPO_ROOT" \
  "$ISAAC_SIM_ROOT/python.sh" \
  "$REPO_ROOT/roboassemblybench/assets/Fabrica/canonical_7_bundles/canonical_tasks.json" \
  "$REPO_ROOT/roboassemblybench/assets/Fabrica/canonical_7_bundles/task_bundles"; do
  if [[ ! -e "$path" ]]; then
    echo "Required deployment path does not exist: $path" >&2
    exit 1
  fi
done
ASSET_PREFLIGHT_PYTHON="${CONTROL_PYTHON:-/root/miniconda3/envs/xvla/bin/python}"
if [[ ! -x "$ASSET_PREFLIGHT_PYTHON" ]]; then
  echo "Asset preflight Python does not exist or is not executable: $ASSET_PREFLIGHT_PYTHON" >&2
  exit 1
fi
"$ASSET_PREFLIGHT_PYTHON" - "$REPO_ROOT" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
metadata_path = repo_root / 'roboassemblybench/assets/Fabrica/canonical_7_bundles/canonical_tasks.json'
metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
asset_paths = []


def collect_usd_paths(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'usd_path':
                asset_paths.append(Path(str(child)))
            collect_usd_paths(child)
    elif isinstance(value, list):
        for child in value:
            collect_usd_paths(child)


collect_usd_paths(metadata)
missing = sorted(str(path) for path in set(asset_paths) if not (repo_root / path).is_file())
if missing:
    print('Canonical Fabrica asset preflight failed; missing USD files:', file=sys.stderr)
    for path in missing:
        print(f'  {path}', file=sys.stderr)
    raise SystemExit(1)
print(f'Canonical Fabrica asset preflight passed: {len(set(asset_paths))} USD files.')
PY
if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
  echo "Collection is already running with PID $(cat "$PID_PATH")."
  exit 0
fi

MAX_EXISTING_GPU_MEMORY_MIB="${MAX_EXISTING_GPU_MEMORY_MIB:-8192}"
MAX_EXISTING_GPU_UTILIZATION="${MAX_EXISTING_GPU_UTILIZATION:-25}"
if [[ "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
  while IFS=',' read -r index memory_used utilization; do
    index="${index//[[:space:]]/}"
    memory_used="${memory_used//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    if (( memory_used > MAX_EXISTING_GPU_MEMORY_MIB || utilization > MAX_EXISTING_GPU_UTILIZATION )); then
      echo "Refusing to start: GPU $index is busy (${memory_used} MiB, ${utilization}% utilization)." >&2
      echo 'Wait for training to finish, or set ALLOW_BUSY_GPUS=1 only after confirming resource ownership.' >&2
      exit 3
    fi
  done < <(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
fi

mkdir -p "$OUTPUT_ROOT" "$(dirname "$LOG_PATH")"
cd "$REPO_ROOT"
export PROJECT_ROOT REPO_ROOT OUTPUT_ROOT
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export INITIAL_GPU_WORKERS="${INITIAL_GPU_WORKERS:-3}"
export GPU_WORKERS_PER_GPU="${GPU_WORKERS_PER_GPU:-3}"
export WORKER_RAMP_SECONDS="${WORKER_RAMP_SECONDS:-300}"
export GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-8192}"
export TARGET_PER_TASK="${TARGET_PER_TASK:-1430}"
export STAGE1_COLLECTION_ENABLED="${STAGE1_COLLECTION_ENABLED:-0}"
export STAGE1_MAX_ATTEMPT_FACTOR="${STAGE1_MAX_ATTEMPT_FACTOR:-128}"
export STAGE1_SEED_BLOCK="${STAGE1_SEED_BLOCK:-1000000}"
# Four queued episodes per process keep all 12 GPU workers active without
# exhausting host memory when each recipe expands into a large USD task graph.
export STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-4}"
export REPLAY_BATCH_SIZE="${REPLAY_BATCH_SIZE:-4}"
export PIPELINED_REPLAY="${PIPELINED_REPLAY:-1}"
export PIPELINE_REPLAY_GPU_IDS="${PIPELINE_REPLAY_GPU_IDS:-$GPU_IDS}"
export PIPELINE_REPLAY_WORKERS_PER_GPU="${PIPELINE_REPLAY_WORKERS_PER_GPU:-3}"
export PIPELINE_REPLAY_START_DELAY_SECONDS="${PIPELINE_REPLAY_START_DELAY_SECONDS:-0}"
export PIPELINE_REPLAY_RAMP_SECONDS="${PIPELINE_REPLAY_RAMP_SECONDS:-30}"
export PIPELINE_REPLAY_POLL_SECONDS="${PIPELINE_REPLAY_POLL_SECONDS:-30}"
export PIPELINE_REPLAY_NICE_INCREMENT="${PIPELINE_REPLAY_NICE_INCREMENT:-0}"
export VIDEO_CODEC="${VIDEO_CODEC:-h265}"
export VIDEO_CRF="${VIDEO_CRF:-30}"
export VIDEO_PRESET="${VIDEO_PRESET:-veryfast}"
export DEPTH_ZSTD_LEVEL="${DEPTH_ZSTD_LEVEL:-8}"
export DATASET_OUTPUT_WIDTH="${DATASET_OUTPUT_WIDTH:-352}"
export DATASET_OUTPUT_HEIGHT="${DATASET_OUTPUT_HEIGHT:-198}"
export FRONT_OUTPUT_WIDTH="${FRONT_OUTPUT_WIDTH:-640}"
export FRONT_OUTPUT_HEIGHT="${FRONT_OUTPUT_HEIGHT:-360}"
export FFMPEG_THREADS="${FFMPEG_THREADS:-1}"
export ISAAC_SIM_ROOT
export ISAAC_PYTHON="${ISAAC_PYTHON:-$ISAAC_SIM_ROOT/python.sh}"
export ISAAC_ASSETS_ROOT="${ISAAC_ASSETS_ROOT:-$REPO_ROOT/roboassemblybench/assets/isaac_sim_5.1}"
export ISAACSIM_PORTABLE_BASE="${ISAACSIM_PORTABLE_BASE:-$PROJECT_ROOT/runtime/isaacsim_portable}"
export RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH:-$REPO_ROOT/.runtime_python:$ISAAC_SIM_ROOT/python_packages:$REPO_ROOT}"
export EXPORT_PYTHONPATH="${EXPORT_PYTHONPATH:-$REPO_ROOT}"
export LEROBOT_PYTHON="${LEROBOT_PYTHON:-/root/miniconda3/envs/xvla/bin/python}"
export EXPORT_LEROBOT="${EXPORT_LEROBOT:-1}"
export CONTROL_PYTHON="${CONTROL_PYTHON:-/root/miniconda3/envs/xvla/bin/python}"

nohup setsid nice -n "${COLLECTOR_NICE:-10}" \
  bash "$REPO_ROOT/roboassemblybench/scripts/collect_fabrica_7tasks_50k_twostage_multigpu.sh" \
  >>"$LOG_PATH" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$PID_PATH"
echo "Started Fabrica UR5e 60,060-episode collection with PID $pid."
echo "Log: $LOG_PATH"
echo "Data: $OUTPUT_ROOT"
