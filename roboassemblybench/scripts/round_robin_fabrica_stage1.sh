#!/usr/bin/env bash
set -Eeuo pipefail
umask 0000

# Run a sequence of stage-1 Fabrica collectors on one GPU.  Each collection
# invocation resumes its own manifest; a finite time slice prevents a task
# with a poor acceptance rate from monopolizing the GPU indefinitely.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:?ISAAC_SIM_ROOT is required}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
TASK_QUEUE="${TASK_QUEUE:?TASK_QUEUE is required}"
TIME_SLICE_SECONDS="${TIME_SLICE_SECONDS:-2700}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-60}"
MAX_ATTEMPT_FACTOR="${MAX_ATTEMPT_FACTOR:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ISAACSIM_THREAD_COUNT="${ISAACSIM_THREAD_COUNT:-8}"
ISAACSIM_PORTABLE_BASE="${ISAACSIM_PORTABLE_BASE:-/tmp/roboassemblybench_${USER}}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/round_robin_gpu_${GPU_ID}.log}"

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo 'GPU_ID must be a non-negative integer.' >&2
  exit 2
fi
if [[ ! "$TIME_SLICE_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo 'TIME_SLICE_SECONDS must be a positive integer.' >&2
  exit 2
fi
if [[ ! -x "$ISAAC_SIM_ROOT/python.sh" ]]; then
  echo "Isaac Sim python.sh is not executable: $ISAAC_SIM_ROOT/python.sh" >&2
  exit 2
fi

IFS=',' read -r -a TASKS <<<"$TASK_QUEUE"
if (( ${#TASKS[@]} == 0 )); then
  echo 'TASK_QUEUE is empty.' >&2
  exit 2
fi
for task in "${TASKS[@]}"; do
  case "$task" in
    beam|car|cooling_manifold|duct|gamepad|plumbers_block|stool_circular) ;;
    *) echo "Unknown Fabrica task: $task" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR" "$ISAACSIM_PORTABLE_BASE"

log() {
  printf '%s gpu=%s %s\n' "$(date -Is)" "$GPU_ID" "$*" | tee -a "$LOG_PATH"
}

read_contract() {
  local task="$1"
  local manifest="$OUTPUT_ROOT/stage1/$task/shards/shard_000/collection_manifest.json"
  python3 - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f'Missing manifest: {path}')
payload = json.loads(path.read_text(encoding='utf-8'))
target = int(payload['target_successful_episodes'])
start_seed = int(payload['start_seed'])
complete = bool(payload.get('complete', False)) and int(payload.get('num_successful', 0)) >= target
print(target, start_seed, int(complete))
PY
}

collector_alive() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

stop_collector_group() {
  local pid="$1"
  collector_alive "$pid" || return
  log "event=time_slice_stop pgid=$pid"
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do
    collector_alive "$pid" || return
    sleep 1
  done
  log "event=time_slice_kill pgid=$pid"
  kill -KILL -- "-$pid" 2>/dev/null || true
}

run_task_slice() {
  local task="$1"
  local target start_seed complete
  read -r target start_seed complete < <(read_contract "$task")
  if (( complete )); then
    log "task=$task event=skip_complete target=$target"
    return
  fi

  local task_output="$OUTPUT_ROOT/stage1/$task/shards/shard_000"
  local task_log="$LOG_DIR/round_robin_${task}_gpu_${GPU_ID}.collector.log"
  log "task=$task event=slice_start target=$target start_seed=$start_seed duration=${TIME_SLICE_SECONDS}s"

  setsid env -u CUDA_VISIBLE_DEVICES \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    ISAACSIM_ACTIVE_GPU="$GPU_ID" \
    ISAACSIM_PHYSICS_GPU="$GPU_ID" \
    ISAACSIM_THREAD_COUNT="$ISAACSIM_THREAD_COUNT" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OPENCV_FOR_THREADS_NUM=1 \
    RAB_FFMPEG_THREADS=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    ISAAC_ASSETS_ROOT="$REPO_ROOT/roboassemblybench/assets/isaac_sim_5.1" \
    ISAAC_SIM_ROOT="$ISAAC_SIM_ROOT" \
    ISAACSIM_PORTABLE_ROOT="$ISAACSIM_PORTABLE_BASE/round_robin_${task}_gpu_${GPU_ID}" \
    PYTHONPATH="$REPO_ROOT/.runtime_python:$ISAAC_SIM_ROOT/python_packages:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$ISAAC_SIM_ROOT/python.sh" \
      "$REPO_ROOT/roboassemblybench/scripts/collect_fabrica_plumbers_block_2k.py" \
        --output-dir "$task_output" \
        --num-episodes "$target" \
        --start-seed "$start_seed" \
        --max-attempts "$((target * MAX_ATTEMPT_FACTOR))" \
        --batch-size "$BATCH_SIZE" \
        --isaac-python "$ISAAC_SIM_ROOT/python.sh" \
        --recipe "fabrica_${task}_ur5e_staged" \
        --scene-profile taoyuan_grscenes_tabletop \
        --randomization-profile position \
        --dataset-fps 10 \
        --dataset-frame-stride 8 \
        --rendering-fps 80 \
        --video-codec h265 \
        --video-crf 30 \
        --video-preset veryfast \
        --depth-compression-level 8 \
        --unique-layout-seeds \
        --skip-qualification \
        --require-extended-observations \
        --require-visual-quality \
        --prune-failed-raw \
        --min-available-memory-gib 48 \
        --abort-available-memory-gib 32 \
        --worker-timeout-seconds 7200 \
        --worker-stall-timeout-seconds 900 \
        --estimated-episode-mib 32 \
        --disk-reserve-gib 256 \
      >>"$task_log" 2>&1 < /dev/null &
  local pid=$!
  local deadline=$(( $(date +%s) + TIME_SLICE_SECONDS ))
  while collector_alive "$pid" && (( $(date +%s) < deadline )); do
    sleep 15
  done
  if collector_alive "$pid"; then
    stop_collector_group "$pid"
  fi
  wait "$pid" 2>/dev/null || true
  log "task=$task event=slice_end"
}

trap 'exit 0' INT TERM
log "event=queue_start tasks=$TASK_QUEUE time_slice_seconds=$TIME_SLICE_SECONDS"
while true; do
  for task in "${TASKS[@]}"; do
    run_task_slice "$task"
    sleep "$COOLDOWN_SECONDS"
  done
done
