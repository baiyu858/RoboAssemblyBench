#!/usr/bin/env bash
set -euo pipefail
umask 0000

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ROBOT_PLATFORM="${ROBOT_PLATFORM:-ur5e}"
if [[ "$ROBOT_PLATFORM" != "ur5e" && "$ROBOT_PLATFORM" != "franka" ]]; then
  echo "ROBOT_PLATFORM must be ur5e or franka." >&2
  exit 2
fi
FRANKA_DATA_ROOT="${FRANKA_DATA_ROOT:-$REPO_ROOT/outputs/franka}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$FRANKA_DATA_ROOT/fabrica_${ROBOT_PLATFORM}_twostage_80hz}"
MACHINE_ID="${MACHINE_ID:-$(hostname -s)}"
NODE_INDEX="${NODE_INDEX:-0}"
NODE_COUNT="${NODE_COUNT:-1}"
GPU_IDS="${GPU_IDS:-0,1}"
INITIAL_GPU_WORKERS="${INITIAL_GPU_WORKERS:-1}"
GPU_WORKERS_PER_GPU="${GPU_WORKERS_PER_GPU:-2}"
WORKER_RAMP_SECONDS="${WORKER_RAMP_SECONDS:-300}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-4096}"
ISAAC_ENV="${ISAAC_ENV:-internutopia311}"
ISAAC_PYTHON="${ISAAC_PYTHON:-}"
ISAAC_ASSETS_ROOT="${ISAAC_ASSETS_ROOT:-$REPO_ROOT/roboassemblybench/assets/isaac_sim_5.1}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-}"
RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH:-$REPO_ROOT/.runtime_python:$(dirname "$REPO_ROOT")/python_packages:$REPO_ROOT}"
EXPORT_PYTHONPATH="${EXPORT_PYTHONPATH:-$REPO_ROOT}"
TARGET_PER_TASK="${TARGET_PER_TASK:-1430}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-8}"
STAGE1_MAX_ATTEMPT_FACTOR="${STAGE1_MAX_ATTEMPT_FACTOR:-32}"
REPLAY_BATCH_SIZE="${REPLAY_BATCH_SIZE:-8}"
STAGE1_SEED_BLOCK="${STAGE1_SEED_BLOCK:-100000}"
CONTROL_PYTHON="${CONTROL_PYTHON:-$(command -v python || true)}"
VIDEO_CODEC="${VIDEO_CODEC:-h265}"
VIDEO_CRF="${VIDEO_CRF:-30}"
VIDEO_PRESET="${VIDEO_PRESET:-veryfast}"
DEPTH_ZSTD_LEVEL="${DEPTH_ZSTD_LEVEL:-8}"
DATASET_OUTPUT_WIDTH="${DATASET_OUTPUT_WIDTH:-352}"
DATASET_OUTPUT_HEIGHT="${DATASET_OUTPUT_HEIGHT:-198}"
FRONT_OUTPUT_WIDTH="${FRONT_OUTPUT_WIDTH:-640}"
FRONT_OUTPUT_HEIGHT="${FRONT_OUTPUT_HEIGHT:-360}"
FFMPEG_THREADS="${FFMPEG_THREADS:-1}"
EXPORT_LEROBOT="${EXPORT_LEROBOT:-1}"
EXPORT_NODE_INDEX="${EXPORT_NODE_INDEX:-0}"
EXPORT_WORKERS="${EXPORT_WORKERS:-4}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-python}"
WORKER_TIMEOUT_SECONDS="${WORKER_TIMEOUT_SECONDS:-7200}"
WORKER_STALL_TIMEOUT_SECONDS="${WORKER_STALL_TIMEOUT_SECONDS:-900}"
MAX_RESTARTS="${MAX_RESTARTS:-100}"

IFS=',' read -r -a GPU_ID_LIST <<<"$GPU_IDS"
if (( INITIAL_GPU_WORKERS < 1 || INITIAL_GPU_WORKERS > GPU_WORKERS_PER_GPU )); then
  echo "INITIAL_GPU_WORKERS must be in [1, GPU_WORKERS_PER_GPU]." >&2
  exit 2
fi
if (( NODE_INDEX < 0 || NODE_INDEX >= NODE_COUNT )); then
  echo "NODE_INDEX must be in [0, NODE_COUNT)." >&2
  exit 2
fi
WORKER_SLOTS_PER_NODE=$((${#GPU_ID_LIST[@]} * GPU_WORKERS_PER_GPU))
TOTAL_WORKER_SLOTS=$((NODE_COUNT * WORKER_SLOTS_PER_NODE))
STAGE1_SHARDS="${STAGE1_SHARDS:-$TOTAL_WORKER_SLOTS}"
if ((
  (EXPORT_LEROBOT == 1 && (EXPORT_NODE_INDEX < 0 || EXPORT_NODE_INDEX >= NODE_COUNT))
  || EXPORT_WORKERS < 1
  || EXPORT_WORKERS > WORKER_SLOTS_PER_NODE
)); then
  echo "EXPORT_NODE_INDEX/EXPORT_WORKERS do not describe a valid node-local export pool." >&2
  exit 2
fi
if ((
  STAGE1_SHARDS < 1
  || STAGE1_MAX_ATTEMPT_FACTOR < 1
  || STAGE1_SEED_BLOCK <= TARGET_PER_TASK * STAGE1_MAX_ATTEMPT_FACTOR
)); then
  echo "Stage-1 shard count, attempt factor, or disjoint seed-block range is invalid." >&2
  exit 2
fi

TASKS=(beam car cooling_manifold duct gamepad plumbers_block stool_circular)
REPLAY_PROFILES=(object_distractors texture lighting table_color scene)
DATA_PROFILES=(position "${REPLAY_PROFILES[@]}")
EXPECTED_TOTAL_EPISODES=$((${#TASKS[@]} * ${#DATA_PROFILES[@]} * TARGET_PER_TASK))
declare -A RECIPES=(
  [beam]="fabrica_beam_${ROBOT_PLATFORM}_staged"
  [car]="fabrica_car_${ROBOT_PLATFORM}_staged"
  [cooling_manifold]="fabrica_cooling_manifold_${ROBOT_PLATFORM}_staged"
  [duct]="fabrica_duct_${ROBOT_PLATFORM}_staged"
  [gamepad]="fabrica_gamepad_${ROBOT_PLATFORM}_staged"
  [plumbers_block]="fabrica_plumbers_block_${ROBOT_PLATFORM}_staged"
  [stool_circular]="fabrica_stool_circular_${ROBOT_PLATFORM}_staged"
)

mkdir -p "$OUTPUT_ROOT/stage1" "$OUTPUT_ROOT/rendered" "$OUTPUT_ROOT/logs/$MACHINE_ID"
printf 'Collection target: %d tasks x %d groups x %d episodes = %d episodes\n' \
  "${#TASKS[@]}" "${#DATA_PROFILES[@]}" "$TARGET_PER_TASK" "$EXPECTED_TOTAL_EPISODES"
printf 'GPU concurrency: %d initial workers/GPU, ramping to %d workers/GPU every %ss\n' \
  "$INITIAL_GPU_WORKERS" "$GPU_WORKERS_PER_GPU" "$WORKER_RAMP_SECONDS"

isaac_runtime() {
  if [[ -n "$ISAAC_PYTHON" ]]; then
    printf '%s\n' "$ISAAC_PYTHON"
  elif [[ -x "$ISAAC_SIM_ROOT/python.sh" ]]; then
    printf '%s\n' "$ISAAC_SIM_ROOT/python.sh"
  else
    printf '%s\n' "$(command -v conda)" run --no-capture-output -n "$ISAAC_ENV" python
  fi
}

wait_for_gpu_capacity() {
  local gpu="$1"
  local free_mib
  while true; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | head -n 1 | tr -d ' ')"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= GPU_MIN_FREE_MIB )); then
      return
    fi
    sleep 30
  done
}

run_with_restarts() {
  local gpu="$1"
  local worker_id="$2"
  local log_path="$3"
  shift 3
  local restart=0
  while true; do
    wait_for_gpu_capacity "$gpu"
    restart=$((restart + 1))
    echo "$(date -Is) gpu=$gpu worker=$worker_id restart=$restart command=$*" >>"$log_path"
    if env -u CUDA_VISIBLE_DEVICES \
      PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
      ISAACSIM_ACTIVE_GPU="$gpu" ISAACSIM_PHYSICS_GPU="$gpu" \
      ISAAC_ASSETS_ROOT="$ISAAC_ASSETS_ROOT" \
      ISAAC_SIM_ROOT="$ISAAC_SIM_ROOT" \
      RAB_DATASET_OUTPUT_WIDTH="$DATASET_OUTPUT_WIDTH" \
      RAB_DATASET_OUTPUT_HEIGHT="$DATASET_OUTPUT_HEIGHT" \
      RAB_FRONT_OUTPUT_WIDTH="$FRONT_OUTPUT_WIDTH" \
      RAB_FRONT_OUTPUT_HEIGHT="$FRONT_OUTPUT_HEIGHT" \
      RAB_FRONT_RUNTIME_CAMERA_MAX_WIDTH="$FRONT_OUTPUT_WIDTH" \
      RAB_FFMPEG_THREADS="$FFMPEG_THREADS" \
      ISAACSIM_PORTABLE_ROOT="${ISAACSIM_PORTABLE_BASE:-/tmp/roboassemblybench_isaacsim_${USER}}/twostage_${MACHINE_ID}_gpu_${gpu}_worker_${worker_id}" \
      PYTHONPATH="$RUNTIME_PYTHONPATH:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "$@" >>"$log_path" 2>&1; then
      return
    fi
    if (( restart >= MAX_RESTARTS )); then
      echo "$(date -Is) gpu=$gpu worker=$worker_id status=failed restart_limit=$MAX_RESTARTS" >>"$log_path"
      return 1
    fi
    sleep 60
  done
}

stage1_shard_target() {
  local shard_id="$1"
  local base=$((TARGET_PER_TASK / STAGE1_SHARDS))
  local remainder=$((TARGET_PER_TASK % STAGE1_SHARDS))
  if (( shard_id < remainder )); then
    echo $((base + 1))
  else
    echo "$base"
  fi
}

stage1_shard_complete() {
  local task="$1"
  local shard_id="$2"
  local target
  target="$(stage1_shard_target "$shard_id")"
  local manifest="$OUTPUT_ROOT/stage1/$task/shards/shard_$(printf '%03d' "$shard_id")/collection_manifest.json"
  [[ -f "$manifest" ]] && "$CONTROL_PYTHON" - "$manifest" "$target" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding='utf-8'))
raise SystemExit(0 if payload.get('complete') and int(payload.get('num_successful', 0)) >= int(sys.argv[2]) else 1)
PY
}

stage1_shard_process_alive() {
  local output="$1"
  ps -eo comm=,args= | awk -v output="$output" '
    $1 ~ /^python/ &&
    index($0, "/collect_fabrica_plumbers_block_2k.py") &&
    index($0, "--output-dir " output) {
      found = 1
    }
    END { exit(found ? 0 : 1) }
  '
}

wait_for_existing_stage1_shard() {
  local task="$1"
  local shard_id="$2"
  local output="$3"
  while stage1_shard_process_alive "$output"; do
    if stage1_shard_complete "$task" "$shard_id"; then
      return
    fi
    sleep 30
  done
}

run_stage1_shard() {
  local gpu="$1"
  local worker_id="$2"
  local task="$3"
  local shard_id="$4"
  local shard_name
  shard_name="shard_$(printf '%03d' "$shard_id")"
  local target
  target="$(stage1_shard_target "$shard_id")"
  if (( target == 0 )); then
    return
  fi
  local start_seed=$((shard_id * STAGE1_SEED_BLOCK))
  local output="$OUTPUT_ROOT/stage1/$task/shards/$shard_name"
  local log="$OUTPUT_ROOT/logs/$MACHINE_ID/stage1_${task}_${shard_name}.log"
  # A restarted scheduler may find an Isaac worker orphaned by its former
  # parent shell.  Let that worker finish, then continue the same slot without
  # duplicating seeds or writing concurrently into the shard.
  wait_for_existing_stage1_shard "$task" "$shard_id" "$output"
  if stage1_shard_complete "$task" "$shard_id"; then
    return
  fi
  local worker_runtime_args=()
  if [[ -n "$ISAAC_PYTHON" ]]; then
    worker_runtime_args+=(--isaac-python "$ISAAC_PYTHON")
  fi
  mapfile -t runtime < <(isaac_runtime)
  run_with_restarts "$gpu" "$worker_id" "$log" "${runtime[@]}" \
    "$REPO_ROOT/roboassemblybench/scripts/collect_fabrica_plumbers_block_2k.py" \
      --output-dir "$output" \
      --num-episodes "$target" \
      --start-seed "$start_seed" \
      --max-attempts "$((target * STAGE1_MAX_ATTEMPT_FACTOR))" \
      --batch-size "$STAGE1_BATCH_SIZE" \
      "${worker_runtime_args[@]}" \
      --recipe "${RECIPES[$task]}" \
      --scene-profile taoyuan_grscenes_tabletop \
      --randomization-profile position \
      --dataset-fps 10 \
      --dataset-frame-stride 8 \
      --rendering-fps 80 \
      --video-codec "$VIDEO_CODEC" \
      --video-crf "$VIDEO_CRF" \
      --video-preset "$VIDEO_PRESET" \
      --depth-compression-level "$DEPTH_ZSTD_LEVEL" \
      --unique-layout-seeds \
      --skip-qualification \
      --require-extended-observations \
      --require-visual-quality \
      --prune-failed-raw \
      --min-available-memory-gib 48 \
      --abort-available-memory-gib 32 \
      --worker-timeout-seconds "$WORKER_TIMEOUT_SECONDS" \
      --worker-stall-timeout-seconds "$WORKER_STALL_TIMEOUT_SECONDS" \
      --estimated-episode-mib 32 \
      --disk-reserve-gib 256
}

wait_for_stage1_barrier() {
  local task shard_id
  while true; do
    for task in "${TASKS[@]}"; do
      for ((shard_id = 0; shard_id < STAGE1_SHARDS; shard_id++)); do
        if ! stage1_shard_complete "$task" "$shard_id"; then
          sleep 60
          continue 3
        fi
      done
    done
    return
  done
}

write_stage1_aggregate_manifest() {
  local task="$1"
  "$CONTROL_PYTHON" - "$OUTPUT_ROOT/stage1/$task" "$TARGET_PER_TASK" <<'PY'
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
target = int(sys.argv[2])
manifests = sorted((root / 'shards').glob('shard_*/collection_manifest.json'))
successful = {}
for path in manifests:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not payload.get('complete'):
        raise RuntimeError(f'Incomplete Stage-1 shard: {path}')
    for seed, episode in (payload.get('successful_episodes') or {}).items():
        if seed in successful:
            raise RuntimeError(f'Duplicate Stage-1 seed {seed} in {path}')
        successful[seed] = episode
if len(successful) != target:
    raise RuntimeError(f'Stage-1 aggregate has {len(successful)} episodes, expected {target}.')
payload = {
    'schema_version': 'roboassemblybench_position_trajectory_aggregate_v1',
    'target_successful_episodes': target,
    'num_successful': len(successful),
    'randomization_profile': 'position',
    'shard_manifests': [str(path.resolve()) for path in manifests],
    'successful_episodes': dict(sorted(successful.items(), key=lambda item: int(item[0]))),
    'complete': True,
    'finished_at_unix': time.time(),
}
temporary = root / 'collection_manifest.json.tmp'
temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
temporary.replace(root / 'collection_manifest.json')
PY
}

wait_for_stage1_aggregate_manifests() {
  local task
  while true; do
    for task in "${TASKS[@]}"; do
      [[ -f "$OUTPUT_ROOT/stage1/$task/collection_manifest.json" ]] || {
        sleep 30
        continue 2
      }
    done
    return
  done
}

run_replay_job() {
  local gpu="$1"
  local worker_id="$2"
  local task="$3"
  local profile="$4"
  local shard_id="$5"
  local shard_name
  shard_name="shard_$(printf '%03d' "$shard_id")"
  local target
  target="$(stage1_shard_target "$shard_id")"
  if (( target == 0 )); then
    return
  fi
  local source="$OUTPUT_ROOT/stage1/$task/shards/$shard_name"
  local raw="$OUTPUT_ROOT/rendered/$task/$profile/shards/$shard_name"
  local log="$OUTPUT_ROOT/logs/$MACHINE_ID/replay_${task}_${profile}_${shard_name}.log"
  if replay_shard_complete "$task" "$profile" "$shard_id"; then
    return
  fi
  local worker_runtime_args=()
  if [[ -n "$ISAAC_PYTHON" ]]; then
    worker_runtime_args+=(--isaac-python "$ISAAC_PYTHON")
  fi
  mapfile -t runtime < <(isaac_runtime)
  run_with_restarts "$gpu" "$worker_id" "$log" "${runtime[@]}" \
    "$REPO_ROOT/roboassemblybench/scripts/replay_fabrica_successful_trajectories.py" \
      --source-dir "$source" \
      --output-dir "$raw" \
      --num-episodes "$target" \
      --batch-size "$REPLAY_BATCH_SIZE" \
      "${worker_runtime_args[@]}" \
      --recipe "${RECIPES[$task]}" \
      --scene-profile taoyuan_grscenes_tabletop \
      --randomization-profile "$profile" \
      --rendering-fps 80 \
      --video-codec "$VIDEO_CODEC" \
      --video-crf "$VIDEO_CRF" \
      --video-preset "$VIDEO_PRESET" \
      --depth-compression-level "$DEPTH_ZSTD_LEVEL" \
      --require-visual-quality \
      --min-available-memory-gib 48 \
      --abort-available-memory-gib 32 \
      --worker-timeout-seconds "$WORKER_TIMEOUT_SECONDS" \
      --worker-stall-timeout-seconds "$WORKER_STALL_TIMEOUT_SECONDS"
}

replay_shard_complete() {
  local task="$1"
  local profile="$2"
  local shard_id="$3"
  local target
  target="$(stage1_shard_target "$shard_id")"
  local manifest="$OUTPUT_ROOT/rendered/$task/$profile/shards/shard_$(printf '%03d' "$shard_id")/replay_manifest.json"
  [[ -f "$manifest" ]] && "$CONTROL_PYTHON" - "$manifest" "$target" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding='utf-8'))
raise SystemExit(0 if payload.get('complete') and int(payload.get('num_successful', 0)) >= int(sys.argv[2]) else 1)
PY
}

wait_for_replay_barrier() {
  local task profile shard_id
  while true; do
    for profile in "${REPLAY_PROFILES[@]}"; do
      for task in "${TASKS[@]}"; do
        for ((shard_id = 0; shard_id < STAGE1_SHARDS; shard_id++)); do
          if ! replay_shard_complete "$task" "$profile" "$shard_id"; then
            sleep 60
            continue 4
          fi
        done
      done
    done
    return
  done
}

lerobot_complete() {
  local task="$1"
  local profile="$2"
  local info="$OUTPUT_ROOT/rendered/$task/$profile/lerobot_v3/meta/info.json"
  [[ -f "$info" ]] && "$CONTROL_PYTHON" - "$info" "$TARGET_PER_TASK" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding='utf-8'))
raise SystemExit(0 if int(payload.get('total_episodes', 0)) >= int(sys.argv[2]) else 1)
PY
}

run_export_job() {
  local task="$1"
  local profile="$2"
  local raw
  if [[ "$profile" == "position" ]]; then
    raw="$OUTPUT_ROOT/stage1/$task"
  else
    raw="$OUTPUT_ROOT/rendered/$task/$profile"
  fi
  local lerobot="$OUTPUT_ROOT/rendered/$task/$profile/lerobot_v3"
  local log="$OUTPUT_ROOT/logs/$MACHINE_ID/export_${task}_${profile}.log"
  if lerobot_complete "$task" "$profile"; then
    return
  fi
  local resume=()
  [[ -f "$lerobot/meta/info.json" ]] && resume+=(--resume)
  PYTHONPATH="$EXPORT_PYTHONPATH:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$LEROBOT_PYTHON" \
    "$REPO_ROOT/roboassemblybench/scripts/export_fabrica_lerobot_v3.py" \
      --input-dir "$raw" \
      --output-dir "$lerobot" \
      --repo-id "baiyu858/roboassemblybench_fabrica_${task}_${ROBOT_PLATFORM}_${profile}" \
      --encoder-threads 2 \
      --vcodec h264 \
      "${resume[@]}" >>"$log" 2>&1
}

worker_loop() {
  local local_slot="$1"
  local gpu="$2"
  local worker_id="$3"
  local launch_delay="$4"
  local global_slot=$((NODE_INDEX * WORKER_SLOTS_PER_NODE + local_slot))
  local task profile shard_id job_index owner_global_slot
  sleep "$launch_delay"

  job_index=0
  for ((shard_id = 0; shard_id < STAGE1_SHARDS; shard_id++)); do
    for task in "${TASKS[@]}"; do
      owner_global_slot=$((job_index % TOTAL_WORKER_SLOTS))
      if (( owner_global_slot == global_slot )); then
        run_stage1_shard "$gpu" "$worker_id" "$task" "$shard_id"
      fi
      job_index=$((job_index + 1))
    done
  done

  wait_for_stage1_barrier
  if (( NODE_INDEX == 0 && local_slot == 0 )); then
    for task in "${TASKS[@]}"; do
      write_stage1_aggregate_manifest "$task"
    done
  fi
  wait_for_stage1_aggregate_manifests
  job_index=0
  for ((shard_id = 0; shard_id < STAGE1_SHARDS; shard_id++)); do
    for profile in "${REPLAY_PROFILES[@]}"; do
      for task in "${TASKS[@]}"; do
        owner_global_slot=$((job_index % TOTAL_WORKER_SLOTS))
        if (( owner_global_slot == global_slot )); then
          run_replay_job "$gpu" "$worker_id" "$task" "$profile" "$shard_id"
        fi
        job_index=$((job_index + 1))
      done
    done
  done

  wait_for_replay_barrier
  if [[ "$EXPORT_LEROBOT" == "1" ]] && (( NODE_INDEX == EXPORT_NODE_INDEX && local_slot < EXPORT_WORKERS )); then
    job_index=0
    for profile in "${DATA_PROFILES[@]}"; do
      for task in "${TASKS[@]}"; do
        if (( job_index % EXPORT_WORKERS == local_slot )); then
          run_export_job "$task" "$profile"
        fi
        job_index=$((job_index + 1))
      done
    done
  fi
}

cd "$REPO_ROOT"
pids=()
local_slot=0
for ((worker_id = 0; worker_id < GPU_WORKERS_PER_GPU; worker_id++)); do
  for gpu in "${GPU_ID_LIST[@]}"; do
    delay=0
    if (( worker_id >= INITIAL_GPU_WORKERS )); then
      delay=$(((worker_id - INITIAL_GPU_WORKERS + 1) * WORKER_RAMP_SECONDS))
    fi
    worker_loop "$local_slot" "$gpu" "$worker_id" "$delay" &
    pids+=("$!")
    local_slot=$((local_slot + 1))
  done
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
