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
STAGE2_PAUSE_MARKER="${STAGE2_PAUSE_MARKER:-$OUTPUT_ROOT/STAGE2_PAUSED_BY_OPERATOR.json}"
MACHINE_ID="${MACHINE_ID:-$(hostname -s)}"
NODE_INDEX="${NODE_INDEX:-0}"
NODE_COUNT="${NODE_COUNT:-1}"
GPU_IDS="${GPU_IDS:-0,1}"
INITIAL_GPU_WORKERS="${INITIAL_GPU_WORKERS:-1}"
GPU_WORKERS_PER_GPU="${GPU_WORKERS_PER_GPU:-2}"
# Optional comma-separated worker counts matching GPU_IDS.  This lets a node
# stay within a host thread/process quota without giving up an entire GPU.
GPU_WORKER_COUNTS="${GPU_WORKER_COUNTS:-}"
WORKER_RAMP_SECONDS="${WORKER_RAMP_SECONDS:-300}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-4096}"
ISAAC_ENV="${ISAAC_ENV:-internutopia311}"
ISAAC_PYTHON="${ISAAC_PYTHON:-}"
ISAAC_ASSETS_ROOT="${ISAAC_ASSETS_ROOT:-$REPO_ROOT/roboassemblybench/assets/isaac_sim_5.1}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-}"
RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH:-$REPO_ROOT/.runtime_python:$(dirname "$REPO_ROOT")/python_packages:$REPO_ROOT}"
EXPORT_PYTHONPATH="${EXPORT_PYTHONPATH:-$REPO_ROOT}"
TARGET_PER_TASK="${TARGET_PER_TASK:-1430}"
STAGE1_COLLECTION_ENABLED="${STAGE1_COLLECTION_ENABLED:-1}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-8}"
STAGE1_MAX_ATTEMPT_FACTOR="${STAGE1_MAX_ATTEMPT_FACTOR:-32}"
REPLAY_BATCH_SIZE="${REPLAY_BATCH_SIZE:-8}"
PIPELINED_REPLAY="${PIPELINED_REPLAY:-1}"
PIPELINE_REPLAY_GPU_IDS="${PIPELINE_REPLAY_GPU_IDS:-$GPU_IDS}"
PIPELINE_REPLAY_WORKERS_PER_GPU="${PIPELINE_REPLAY_WORKERS_PER_GPU:-1}"
PIPELINE_REPLAY_START_DELAY_SECONDS="${PIPELINE_REPLAY_START_DELAY_SECONDS:-120}"
PIPELINE_REPLAY_RAMP_SECONDS="${PIPELINE_REPLAY_RAMP_SECONDS:-120}"
PIPELINE_REPLAY_POLL_SECONDS="${PIPELINE_REPLAY_POLL_SECONDS:-30}"
PIPELINE_REPLAY_NICE_INCREMENT="${PIPELINE_REPLAY_NICE_INCREMENT:-5}"
# Keep every shard's retry seed range disjoint.  The default scales with the
# requested target so changing collection size cannot invalidate the launch.
STAGE1_SEED_BLOCK="${STAGE1_SEED_BLOCK:-$((TARGET_PER_TASK * STAGE1_MAX_ATTEMPT_FACTOR + 1))}"
CONTROL_PYTHON="${CONTROL_PYTHON:-$(command -v python3 || command -v python || true)}"
VIDEO_CODEC="${VIDEO_CODEC:-h265}"
VIDEO_CRF="${VIDEO_CRF:-30}"
VIDEO_PRESET="${VIDEO_PRESET:-veryfast}"
DEPTH_ZSTD_LEVEL="${DEPTH_ZSTD_LEVEL:-8}"
FFMPEG_THREADS="${FFMPEG_THREADS:-1}"
ISAACSIM_OMP_NUM_THREADS="${ISAACSIM_OMP_NUM_THREADS:-1}"
ISAACSIM_THREAD_COUNT="${ISAACSIM_THREAD_COUNT:-}"
EXPORT_LEROBOT="${EXPORT_LEROBOT:-1}"
EXPORT_NODE_INDEX="${EXPORT_NODE_INDEX:-0}"
EXPORT_WORKERS="${EXPORT_WORKERS:-4}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-$(command -v python3 || command -v python || true)}"
WORKER_TIMEOUT_SECONDS="${WORKER_TIMEOUT_SECONDS:-7200}"
WORKER_STALL_TIMEOUT_SECONDS="${WORKER_STALL_TIMEOUT_SECONDS:-900}"
MAX_RESTARTS="${MAX_RESTARTS:-100}"

IFS=',' read -r -a GPU_ID_LIST <<<"$GPU_IDS"
IFS=',' read -r -a PIPELINE_REPLAY_GPU_ID_LIST <<<"$PIPELINE_REPLAY_GPU_IDS"
GPU_WORKER_COUNT_LIST=()
if [[ -n "$GPU_WORKER_COUNTS" ]]; then
  IFS=',' read -r -a GPU_WORKER_COUNT_LIST <<<"$GPU_WORKER_COUNTS"
  if (( ${#GPU_WORKER_COUNT_LIST[@]} != ${#GPU_ID_LIST[@]} )); then
    echo "GPU_WORKER_COUNTS must provide one positive count for each GPU_IDS entry." >&2
    exit 2
  fi
else
  for _ in "${GPU_ID_LIST[@]}"; do
    GPU_WORKER_COUNT_LIST+=("$GPU_WORKERS_PER_GPU")
  done
fi
WORKER_SLOTS_PER_NODE=0
MAX_GPU_WORKERS=0
for worker_count in "${GPU_WORKER_COUNT_LIST[@]}"; do
  if [[ ! "$worker_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU worker counts must be positive integers." >&2
    exit 2
  fi
  WORKER_SLOTS_PER_NODE=$((WORKER_SLOTS_PER_NODE + worker_count))
  if (( worker_count > MAX_GPU_WORKERS )); then
    MAX_GPU_WORKERS="$worker_count"
  fi
done
if (( INITIAL_GPU_WORKERS < 1 || INITIAL_GPU_WORKERS > MAX_GPU_WORKERS )); then
  echo "INITIAL_GPU_WORKERS must be in [1, max(GPU_WORKER_COUNTS)]." >&2
  exit 2
fi
if (( NODE_INDEX < 0 || NODE_INDEX >= NODE_COUNT )); then
  echo "NODE_INDEX must be in [0, NODE_COUNT)." >&2
  exit 2
fi
TOTAL_WORKER_SLOTS=$((NODE_COUNT * WORKER_SLOTS_PER_NODE))
STAGE1_SHARDS="${STAGE1_SHARDS:-$TOTAL_WORKER_SLOTS}"
if [[ ! "$PIPELINED_REPLAY" =~ ^[01]$ ]]; then
  echo "PIPELINED_REPLAY must be 0 or 1." >&2
  exit 2
fi
if [[ ! "$STAGE1_COLLECTION_ENABLED" =~ ^[01]$ ]]; then
  echo "STAGE1_COLLECTION_ENABLED must be 0 or 1." >&2
  exit 2
fi
if (( STAGE1_COLLECTION_ENABLED == 0 && PIPELINED_REPLAY == 0 )); then
  echo "PIPELINED_REPLAY must be 1 when Stage-1 collection is disabled." >&2
  exit 2
fi
if (( PIPELINED_REPLAY == 1 )); then
  if [[ ! "$PIPELINE_REPLAY_WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "PIPELINE_REPLAY_WORKERS_PER_GPU must be a positive integer." >&2
    exit 2
  fi
  if (( ${#PIPELINE_REPLAY_GPU_ID_LIST[@]} == 0 )); then
    echo "PIPELINE_REPLAY_GPU_IDS must contain at least one GPU." >&2
    exit 2
  fi
  for value in \
    "$PIPELINE_REPLAY_START_DELAY_SECONDS" \
    "$PIPELINE_REPLAY_RAMP_SECONDS" \
    "$PIPELINE_REPLAY_NICE_INCREMENT"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
      echo "Pipeline replay delays and nice increment must be non-negative integers." >&2
      exit 2
    fi
  done
  if [[ ! "$PIPELINE_REPLAY_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "PIPELINE_REPLAY_POLL_SECONDS must be a positive integer." >&2
    exit 2
  fi
fi
PIPELINE_REPLAY_SLOTS_PER_NODE=$((${#PIPELINE_REPLAY_GPU_ID_LIST[@]} * PIPELINE_REPLAY_WORKERS_PER_GPU))
TOTAL_PIPELINE_REPLAY_SLOTS=$((NODE_COUNT * PIPELINE_REPLAY_SLOTS_PER_NODE))
if (( EXPORT_LEROBOT == 1 && (
  EXPORT_NODE_INDEX < 0
  || EXPORT_NODE_INDEX >= NODE_COUNT
  || EXPORT_WORKERS < 1
  || EXPORT_WORKERS > WORKER_SLOTS_PER_NODE
) )); then
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

ALL_TASKS=(beam car cooling_manifold duct gamepad plumbers_block stool_circular)
# A node can collect an independent task subset into its own output root.  This
# avoids cross-node locks while retaining the all-task default for local runs.
if [[ -n "${FABRICA_TASKS:-}" ]]; then
  IFS=',' read -r -a REQUESTED_TASKS <<<"$FABRICA_TASKS"
  TASKS=()
  for task in "${REQUESTED_TASKS[@]}"; do
    task="${task//[[:space:]]/}"
    if [[ -z "$task" ]]; then
      continue
    fi
    case " ${ALL_TASKS[*]} " in
      *" $task "*) TASKS+=("$task") ;;
      *)
        echo "Unknown Fabrica task in FABRICA_TASKS: $task" >&2
        exit 2
        ;;
    esac
  done
  if (( ${#TASKS[@]} == 0 )); then
    echo 'FABRICA_TASKS did not contain a valid task.' >&2
    exit 2
  fi
else
  TASKS=("${ALL_TASKS[@]}")
fi
declare -A SEEN_TASKS=()
for task in "${TASKS[@]}"; do
  if [[ -n "${SEEN_TASKS[$task]:-}" ]]; then
    echo "FABRICA_TASKS contains a duplicate task: $task" >&2
    exit 2
  fi
  SEEN_TASKS["$task"]=1
done
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
printf 'GPU concurrency: counts=[%s], %d worker slots/node; workers >= %d ramp every %ss\n' \
  "$(IFS=,; echo "${GPU_WORKER_COUNT_LIST[*]}")" "$WORKER_SLOTS_PER_NODE" \
  "$INITIAL_GPU_WORKERS" "$WORKER_RAMP_SECONDS"
printf 'Tasks: %s\n' "${TASKS[*]}"
if (( STAGE1_COLLECTION_ENABLED == 1 )); then
  printf 'Stage 1 recording: full RGB-D/raw observations (not trajectory-only)\n'
else
  printf 'Stage 1 recording: paused; replaying all successful manifest entries\n'
fi
if (( PIPELINED_REPLAY == 1 )); then
  printf 'Pipelined replay: %d workers/node on GPUs [%s], start delay %ss, ramp %ss\n' \
    "$PIPELINE_REPLAY_SLOTS_PER_NODE" "$PIPELINE_REPLAY_GPU_IDS" \
    "$PIPELINE_REPLAY_START_DELAY_SECONDS" "$PIPELINE_REPLAY_RAMP_SECONDS"
fi

if [[ "${PIPELINE_DRY_RUN:-0}" == "1" ]]; then
  printf 'Dry run complete: no Isaac workers were started.\n'
  exit 0
fi

isaac_runtime() {
  if [[ -n "$ISAAC_PYTHON" ]]; then
    printf '%s\n' "$ISAAC_PYTHON"
  elif [[ -x "$ISAAC_SIM_ROOT/python.sh" ]]; then
    printf '%s\n' "$ISAAC_SIM_ROOT/python.sh"
  else
    printf '%s\n' "$(command -v conda)" run --no-capture-output -n "$ISAAC_ENV" python
  fi
}

stage2_paused() {
  [[ -f "$STAGE2_PAUSE_MARKER" ]]
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
    if [[ "${RAB_STAGE2_CONTEXT:-0}" == "1" ]] && stage2_paused; then
      echo "$(date -Is) gpu=$gpu worker=$worker_id status=stage2_paused" >>"$log_path"
      return
    fi
    wait_for_gpu_capacity "$gpu"
    restart=$((restart + 1))
    echo "$(date -Is) gpu=$gpu worker=$worker_id restart=$restart command=$*" >>"$log_path"
    if env -u CUDA_VISIBLE_DEVICES \
      PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS="$ISAACSIM_OMP_NUM_THREADS" \
      MKL_NUM_THREADS="$ISAACSIM_OMP_NUM_THREADS" \
      OPENBLAS_NUM_THREADS="$ISAACSIM_OMP_NUM_THREADS" \
      NUMEXPR_NUM_THREADS="$ISAACSIM_OMP_NUM_THREADS" \
      ISAACSIM_ACTIVE_GPU="$gpu" ISAACSIM_PHYSICS_GPU="$gpu" \
      ISAACSIM_THREAD_COUNT="$ISAACSIM_THREAD_COUNT" \
      ISAAC_ASSETS_ROOT="$ISAAC_ASSETS_ROOT" \
      ISAAC_SIM_ROOT="$ISAAC_SIM_ROOT" \
      RAB_FFMPEG_THREADS="$FFMPEG_THREADS" \
      ISAACSIM_PORTABLE_ROOT="${ISAACSIM_PORTABLE_BASE:-/tmp/roboassemblybench_isaacsim_${USER}}/twostage_${MACHINE_ID}_gpu_${gpu}_worker_${worker_id}" \
      PYTHONPATH="$RUNTIME_PYTHONPATH:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "$@" >>"$log_path" 2>&1; then
      return
    fi
    if [[ "${RAB_STAGE2_CONTEXT:-0}" == "1" ]] && stage2_paused; then
      echo "$(date -Is) gpu=$gpu worker=$worker_id status=stage2_paused" >>"$log_path"
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
  if stage2_paused; then
    echo "$(date -Is) task=$task profile=$profile shard=$shard_name status=stage2_paused" >>"$log"
    return
  fi
  if replay_shard_complete "$task" "$profile" "$shard_id"; then
    return
  fi
  local worker_runtime_args=()
  if [[ -n "$ISAAC_PYTHON" ]]; then
    worker_runtime_args+=(--isaac-python "$ISAAC_PYTHON")
  fi
  mapfile -t runtime < <(isaac_runtime)
  local replay_runtime=("${runtime[@]}")
  if (( PIPELINED_REPLAY == 1 && PIPELINE_REPLAY_NICE_INCREMENT > 0 )); then
    replay_runtime=(nice -n "$PIPELINE_REPLAY_NICE_INCREMENT" "${runtime[@]}")
  fi
  RAB_STAGE2_CONTEXT=1 run_with_restarts "$gpu" "$worker_id" "$log" "${replay_runtime[@]}" \
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

pipeline_replay_loop() {
  local local_replay_slot="$1"
  local gpu="$2"
  local launch_delay="$3"
  local global_replay_slot=$((NODE_INDEX * PIPELINE_REPLAY_SLOTS_PER_NODE + local_replay_slot))
  local worker_id="pipeline_replay_${local_replay_slot}"
  local scheduler_log="$OUTPUT_ROOT/logs/$MACHINE_ID/${worker_id}.log"
  local task profile shard_id job_index owner_global_slot pending ready
  if stage2_paused; then
    echo "$(date -Is) event=pipeline_replay_paused slot=$global_replay_slot" >>"$scheduler_log"
    return
  fi
  sleep "$launch_delay"
  echo "$(date -Is) event=pipeline_replay_start gpu=$gpu slot=$global_replay_slot" >>"$scheduler_log"

  while true; do
    if stage2_paused; then
      echo "$(date -Is) event=pipeline_replay_paused slot=$global_replay_slot" >>"$scheduler_log"
      return
    fi
    pending=0
    ready=0
    job_index=0
    for ((shard_id = 0; shard_id < STAGE1_SHARDS; shard_id++)); do
      for profile in "${REPLAY_PROFILES[@]}"; do
        for task in "${TASKS[@]}"; do
          owner_global_slot=$((job_index % TOTAL_PIPELINE_REPLAY_SLOTS))
          job_index=$((job_index + 1))
          if (( owner_global_slot != global_replay_slot )); then
            continue
          fi
          if replay_shard_complete "$task" "$profile" "$shard_id"; then
            continue
          fi
          pending=1
          if ! stage1_shard_has_unreplayed_success "$task" "$profile" "$shard_id"; then
            continue
          fi
          ready=1
          echo "$(date -Is) event=pipeline_replay_job task=$task profile=$profile shard=$shard_id gpu=$gpu" \
            >>"$scheduler_log"
          if ! run_replay_job "$gpu" "$worker_id" "$task" "$profile" "$shard_id"; then
            echo "$(date -Is) event=pipeline_replay_job_failed task=$task profile=$profile shard=$shard_id" \
              >>"$scheduler_log"
            sleep "$PIPELINE_REPLAY_POLL_SECONDS"
          fi
        done
      done
    done
    if (( pending == 0 )); then
      echo "$(date -Is) event=pipeline_replay_complete slot=$global_replay_slot" >>"$scheduler_log"
      return
    fi
    if (( ready == 0 )); then
      sleep "$PIPELINE_REPLAY_POLL_SECONDS"
    fi
  done
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

stage1_shard_has_unreplayed_success() {
  local task="$1"
  local profile="$2"
  local shard_id="$3"
  local shard_name
  shard_name="shard_$(printf '%03d' "$shard_id")"
  local source_manifest="$OUTPUT_ROOT/stage1/$task/shards/$shard_name/collection_manifest.json"
  local replay_manifest="$OUTPUT_ROOT/rendered/$task/$profile/shards/$shard_name/replay_manifest.json"
  [[ -f "$source_manifest" ]] || return 1
  "$CONTROL_PYTHON" - "$source_manifest" "$replay_manifest" <<'PY'
import json
import sys
from pathlib import Path

source = json.load(open(sys.argv[1], encoding='utf-8'))
source_seeds = set((source.get('successful_episodes') or {}).keys())
if not source_seeds:
    raise SystemExit(1)

replay_path = Path(sys.argv[2])
replayed_seeds = set()
if replay_path.is_file():
    replay = json.load(replay_path.open(encoding='utf-8'))
    replayed_seeds = set((replay.get('successful_episodes') or {}).keys())
raise SystemExit(0 if source_seeds - replayed_seeds else 1)
PY
}

wait_for_replay_barrier() {
  local task profile shard_id
  while true; do
    if stage2_paused; then
      return
    fi
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
  if (( PIPELINED_REPLAY == 0 )); then
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
  fi

  wait_for_replay_barrier
  if [[ "$EXPORT_LEROBOT" == "1" ]] && ! stage2_paused && (( NODE_INDEX == EXPORT_NODE_INDEX && local_slot < EXPORT_WORKERS )); then
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
if (( STAGE1_COLLECTION_ENABLED == 1 )); then
  for gpu_index in "${!GPU_ID_LIST[@]}"; do
    gpu="${GPU_ID_LIST[$gpu_index]}"
    worker_count="${GPU_WORKER_COUNT_LIST[$gpu_index]}"
    for ((worker_id = 0; worker_id < worker_count; worker_id++)); do
      delay=0
      if (( worker_id >= INITIAL_GPU_WORKERS )); then
        delay=$(((worker_id - INITIAL_GPU_WORKERS + 1) * WORKER_RAMP_SECONDS))
      fi
      worker_loop "$local_slot" "$gpu" "$worker_id" "$delay" &
      pids+=("$!")
      local_slot=$((local_slot + 1))
    done
  done
fi

if (( PIPELINED_REPLAY == 1 )); then
  local_replay_slot=0
  for gpu in "${PIPELINE_REPLAY_GPU_ID_LIST[@]}"; do
    for ((replay_worker_id = 0; replay_worker_id < PIPELINE_REPLAY_WORKERS_PER_GPU; replay_worker_id++)); do
      delay=$((PIPELINE_REPLAY_START_DELAY_SECONDS + local_replay_slot * PIPELINE_REPLAY_RAMP_SECONDS))
      pipeline_replay_loop "$local_replay_slot" "$gpu" "$delay" &
      pids+=("$!")
      local_replay_slot=$((local_replay_slot + 1))
    done
  done
fi

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
