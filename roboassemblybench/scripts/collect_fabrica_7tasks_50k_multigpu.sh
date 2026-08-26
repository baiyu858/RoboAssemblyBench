#!/usr/bin/env bash
set -euo pipefail
umask 0000

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
ISAAC_ENV="${ISAAC_ENV:-internutopia311}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-$(command -v python3 || command -v python || true)}"
ISAAC_ASSETS_ROOT="${ISAAC_ASSETS_ROOT:-$REPO_ROOT/roboassemblybench/assets/isaac_sim_5.1}"
ISAAC_PYTHON="${ISAAC_PYTHON:-}"
EXPORT_LEROBOT="${EXPORT_LEROBOT:-1}"
RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH:-$REPO_ROOT/.runtime_python}"
NODE_INDEX="${NODE_INDEX:-0}"
NODE_COUNT="${NODE_COUNT:-1}"
ROBOT_PLATFORM="${ROBOT_PLATFORM:-ur5e}"
FRANKA_DATA_ROOT="${FRANKA_DATA_ROOT:-$REPO_ROOT/outputs/franka}"
FRANKA_MACHINE_ID="${FRANKA_MACHINE_ID:-}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_WORKERS_PER_GPU="${GPU_WORKERS_PER_GPU:-1}"
GLOBAL_SLOT_OFFSET="${GLOBAL_SLOT_OFFSET:-}"
GLOBAL_SLOT_COUNT="${GLOBAL_SLOT_COUNT:-}"
WORKER_TIMEOUT_SECONDS="${WORKER_TIMEOUT_SECONDS:-28800}"
WORKER_STALL_TIMEOUT_SECONDS="${WORKER_STALL_TIMEOUT_SECONDS:-600}"
GPU_WORKER_RESTART_LIMIT="${GPU_WORKER_RESTART_LIMIT:-1000}"
GPU_WORKER_RESTART_DELAY_SECONDS="${GPU_WORKER_RESTART_DELAY_SECONDS:-30}"
RENDERING_FPS="${RENDERING_FPS:-240}"
DATASET_FPS="${DATASET_FPS:-10}"
DATASET_FRAME_STRIDE="${DATASET_FRAME_STRIDE:-24}"
TARGET_PER_SUBSET="${TARGET_PER_SUBSET:-}"
MODE="${1:-formal}"

if (( NODE_COUNT < 1 || NODE_INDEX < 0 || NODE_INDEX >= NODE_COUNT )); then
  echo "Invalid node shard NODE_INDEX=$NODE_INDEX NODE_COUNT=$NODE_COUNT" >&2
  exit 2
fi
if (( GPU_WORKERS_PER_GPU < 1 )); then
  echo "GPU_WORKERS_PER_GPU must be at least 1" >&2
  exit 2
fi
if ((
  RENDERING_FPS < 1
  || DATASET_FPS < 1
  || DATASET_FRAME_STRIDE < 1
  || RENDERING_FPS != DATASET_FPS * DATASET_FRAME_STRIDE
)); then
  echo "Timing must satisfy RENDERING_FPS = DATASET_FPS * DATASET_FRAME_STRIDE." >&2
  exit 2
fi
if [[ -n "$TARGET_PER_SUBSET" && ! "$TARGET_PER_SUBSET" =~ ^[1-9][0-9]*$ ]]; then
  echo "TARGET_PER_SUBSET must be a positive integer when set." >&2
  exit 2
fi
if (( GPU_WORKER_RESTART_LIMIT < 1 || GPU_WORKER_RESTART_DELAY_SECONDS < 0 )); then
  echo "GPU worker restart limits must be non-negative, with at least one allowed restart" >&2
  exit 2
fi
case "$ROBOT_PLATFORM" in
  ur5e|franka) ;;
  *)
    echo "ROBOT_PLATFORM must be ur5e or franka, got $ROBOT_PLATFORM" >&2
    exit 2
    ;;
esac
if [[ "$ROBOT_PLATFORM" == "franka" && ! "$FRANKA_MACHINE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Set FRANKA_MACHINE_ID to a machine-specific directory name for Franka collection." >&2
  exit 2
fi
IFS=',' read -r -a GPU_ID_LIST <<<"$GPU_IDS"
for index in "${!GPU_ID_LIST[@]}"; do
  GPU_ID_LIST[$index]="${GPU_ID_LIST[$index]// /}"
  if [[ ! "${GPU_ID_LIST[$index]}" =~ ^[0-9]+$ ]]; then
    echo "GPU_IDS must be a comma-separated list of physical GPU indices, got $GPU_IDS" >&2
    exit 2
  fi
done
if (( ${#GPU_ID_LIST[@]} < 1 )); then
  echo "GPU_IDS must select at least one GPU" >&2
  exit 2
fi
if [[ -z "$ISAAC_PYTHON" && ! -x "$CONDA_BIN" ]]; then
  echo "Conda was not found at $CONDA_BIN; set ISAAC_PYTHON to a direct Isaac Python launcher." >&2
  exit 1
fi
if [[ "$EXPORT_LEROBOT" == "1" && ! -x "$LEROBOT_PYTHON" ]]; then
  echo "LeRobot Python was not found at $LEROBOT_PYTHON" >&2
  exit 1
fi

WAREHOUSE_ASSET="$ISAAC_ASSETS_ROOT/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
if [[ ! -f "$WAREHOUSE_ASSET" ]]; then
  echo "Required offline Isaac Warehouse asset is missing: $WAREHOUSE_ASSET" >&2
  exit 1
fi

ALL_TASKS=(beam car cooling_manifold duct gamepad plumbers_block stool_circular)
if [[ -n "${TASK_FILTER:-}" ]]; then
  IFS=',' read -r -a TASKS <<<"$TASK_FILTER"
  for index in "${!TASKS[@]}"; do
    TASKS[$index]="${TASKS[$index]// /}"
  done
else
  TASKS=("${ALL_TASKS[@]}")
fi
for task in "${TASKS[@]}"; do
  if [[ ! " ${ALL_TASKS[*]} " =~ " $task " ]]; then
    echo "Unsupported task in TASK_FILTER: $task" >&2
    exit 2
  fi
done
if [[ -n "${PROFILE_FILTER:-}" ]]; then
  IFS=',' read -r -a PROFILES <<<"$PROFILE_FILTER"
  for index in "${!PROFILES[@]}"; do
    PROFILES[$index]="${PROFILES[$index]// /}"
  done
else
  PROFILES=(object_distractors texture lighting table_color scene)
fi
for profile in "${PROFILES[@]}"; do
  case "$profile" in
    object_distractors|texture|lighting|table_color|scene) ;;
    *)
      echo "Unsupported profile in PROFILE_FILTER: $profile" >&2
      exit 2
      ;;
  esac
done
declare -A RECIPES=(
  [beam]="fabrica_beam_${ROBOT_PLATFORM}_staged"
  [car]="fabrica_car_${ROBOT_PLATFORM}_staged"
  [cooling_manifold]="fabrica_cooling_manifold_${ROBOT_PLATFORM}_staged"
  [duct]="fabrica_duct_${ROBOT_PLATFORM}_staged"
  [gamepad]="fabrica_gamepad_${ROBOT_PLATFORM}_staged"
  [plumbers_block]="fabrica_plumbers_block_${ROBOT_PLATFORM}_staged"
  [stool_circular]="fabrica_stool_circular_${ROBOT_PLATFORM}_staged"
)

target_for() {
  local task="$1"
  local profile="$2"
  if [[ -n "$TARGET_PER_SUBSET" ]]; then
    echo "$TARGET_PER_SUBSET"
  elif [[ "$MODE" == "smoke" ]]; then
    echo 1
  elif [[ "$task" == "stool_circular" ]]; then
    case "$profile" in
      object_distractors|texture) echo 1429 ;;
      *) echo 1428 ;;
    esac
  else
    case "$profile" in
      object_distractors|texture|lighting) echo 1429 ;;
      *) echo 1428 ;;
    esac
  fi
}

if [[ "$MODE" == "smoke" ]]; then
  if [[ "$ROBOT_PLATFORM" == "franka" ]]; then
    OUTPUT_ROOT="${OUTPUT_ROOT:-$FRANKA_DATA_ROOT/$FRANKA_MACHINE_ID/fabrica_7tasks_franka_5profiles_smoke}"
  else
    OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/fabrica_7tasks_${ROBOT_PLATFORM}_5profiles_smoke}"
  fi
  BATCH_SIZE="${BATCH_SIZE:-1}"
  MAX_COLLECTOR_RESTARTS="${MAX_COLLECTOR_RESTARTS:-3}"
elif [[ "$MODE" == "formal" ]]; then
  if [[ "$ROBOT_PLATFORM" == "franka" ]]; then
    OUTPUT_ROOT="${OUTPUT_ROOT:-$FRANKA_DATA_ROOT/$FRANKA_MACHINE_ID/fabrica_7tasks_franka_50k_lerobot_v3}"
  else
    OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/fabrica_7tasks_${ROBOT_PLATFORM}_50k_lerobot_v3}"
  fi
  BATCH_SIZE="${BATCH_SIZE:-8}"
  MAX_COLLECTOR_RESTARTS="${MAX_COLLECTOR_RESTARTS:-100}"
else
  echo "Usage: $0 [smoke|formal]" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/status"

run_subset() (
  local gpu="$1"
  local worker_id="$2"
  local task="$3"
  local profile="$4"
  local target
  target="$(target_for "$task" "$profile")"
  local subset="$task/$profile"
  local raw_dir="$OUTPUT_ROOT/$subset/raw"
  local dataset_dir="$OUTPUT_ROOT/$subset/lerobot_v3"
  local log_path="$OUTPUT_ROOT/logs/${task}__${profile}.log"
  local status_path="$OUTPUT_ROOT/status/${task}__${profile}.status"
  local lock_path="$OUTPUT_ROOT/status/${task}__${profile}.lock"
  local max_attempts=$((target * 5))
  local restart=0
  local lock_fd

  mkdir -p "$raw_dir"
  exec {lock_fd}>"$lock_path"
  flock "$lock_fd"
  exec >>"$log_path" 2>&1
  echo "$(date -Is) subset=$subset gpu=$gpu worker=$worker_id target=$target mode=$MODE status=starting"

  while true; do
    restart=$((restart + 1))
    echo "$(date -Is) subset=$subset gpu=$gpu worker=$worker_id target=$target status=collecting restart=$restart" | tee "$status_path"
    local collector_runtime_args=(--conda-env "$ISAAC_ENV")
    local collector_command=()
    if [[ -n "$ISAAC_PYTHON" ]]; then
      collector_runtime_args+=(--isaac-python "$ISAAC_PYTHON")
      collector_command=("$ISAAC_PYTHON")
    else
      collector_command=("$CONDA_BIN" run --no-capture-output -n "$ISAAC_ENV" python)
    fi
    # Isaac's active/physics GPU settings use physical device ordinals. Hiding
    # the other GPUs renumbers the selected device to logical ordinal 0 and
    # makes a physical ordinal such as 1 invalid inside PhysX.
    if env -u CUDA_VISIBLE_DEVICES PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
      ISAACSIM_ACTIVE_GPU="$gpu" \
      ISAACSIM_PHYSICS_GPU="$gpu" \
      ISAAC_ASSETS_ROOT="$ISAAC_ASSETS_ROOT" \
      ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-$ISAAC_ASSETS_ROOT}" \
      ISAACSIM_PORTABLE_ROOT="${ISAACSIM_PORTABLE_BASE:-/tmp/roboassemblybench_isaacsim_${USER}}/node_${NODE_INDEX}_gpu_${gpu}_worker_${worker_id}" \
      PYTHONPATH="$RUNTIME_PYTHONPATH:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "${collector_command[@]}" \
      "$REPO_ROOT/roboassemblybench/scripts/collect_fabrica_plumbers_block_2k.py" \
        --output-dir "$raw_dir" \
        --num-episodes "$target" \
        --max-attempts "$max_attempts" \
        --batch-size "$BATCH_SIZE" \
        "${collector_runtime_args[@]}" \
        --recipe "${RECIPES[$task]}" \
        --scene-profile taoyuan_grscenes_tabletop \
        --randomization-profile "$profile" \
        --dataset-fps "$DATASET_FPS" \
        --dataset-frame-stride "$DATASET_FRAME_STRIDE" \
        --rendering-fps "$RENDERING_FPS" \
        --unique-layout-seeds \
        --skip-qualification \
        --require-extended-observations \
        --require-visual-quality \
        --prune-failed-raw \
        --min-available-memory-gib 32 \
        --abort-available-memory-gib 16 \
        --resource-poll-seconds 10 \
        --resource-wait-seconds 60 \
        --low-memory-grace-polls 3 \
        --worker-timeout-seconds "$WORKER_TIMEOUT_SECONDS" \
        --worker-stall-timeout-seconds "$WORKER_STALL_TIMEOUT_SECONDS" \
        --estimated-episode-mib "${ESTIMATED_EPISODE_MIB:-512}" \
        --disk-reserve-gib "${DISK_RESERVE_GIB:-256}"; then
      break
    fi
    if (( restart >= MAX_COLLECTOR_RESTARTS )); then
      echo "$(date -Is) subset=$subset gpu=$gpu status=failed reason=collector_restart_limit" | tee "$status_path"
      return 1
    fi
    echo "$(date -Is) subset=$subset gpu=$gpu status=collector_restart_wait" | tee "$status_path"
    sleep 60
  done

  if [[ "$EXPORT_LEROBOT" != "1" ]]; then
    echo "$(date -Is) subset=$subset gpu=$gpu status=raw_complete" | tee "$status_path"
    return 0
  fi

  echo "$(date -Is) subset=$subset gpu=$gpu status=exporting" | tee "$status_path"
  local resume_args=()
  if [[ -f "$dataset_dir/meta/info.json" ]]; then
    resume_args+=(--resume)
  elif [[ -d "$dataset_dir" ]]; then
    if [[ -n "$(find "$dataset_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to overwrite incomplete non-empty LeRobot directory: $dataset_dir" >&2
      return 1
    fi
    rmdir "$dataset_dir"
  fi
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="$RUNTIME_PYTHONPATH:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$LEROBOT_PYTHON" \
    "$REPO_ROOT/roboassemblybench/scripts/export_fabrica_lerobot_v3.py" \
      --input-dir "$raw_dir" \
      --output-dir "$dataset_dir" \
      --repo-id "baiyu858/roboassemblybench_fabrica_${task}_${ROBOT_PLATFORM}_${profile}" \
      --encoder-threads 2 \
      "${resume_args[@]}"
  echo "$(date -Is) subset=$subset gpu=$gpu status=complete" | tee "$status_path"
)

gpu_worker() {
  local gpu="$1"
  local worker_id="$2"
  shift 2
  local job task profile
  for job in "$@"; do
    task="${job%%:*}"
    profile="${job#*:}"
    run_subset "$gpu" "$worker_id" "$task" "$profile"
  done
}

supervise_gpu_worker() {
  local gpu="$1"
  local worker_id="$2"
  shift 2
  local restart=0
  local worker_pid
  local worker_status

  while true; do
    gpu_worker "$gpu" "$worker_id" "$@" &
    worker_pid=$!
    if wait "$worker_pid"; then
      return 0
    else
      worker_status=$?
    fi
    restart=$((restart + 1))
    echo "$(date -Is) gpu=$gpu worker=$worker_id status=gpu_worker_restart restart=$restart exit=$worker_status" >&2
    if (( restart >= GPU_WORKER_RESTART_LIMIT )); then
      echo "$(date -Is) gpu=$gpu worker=$worker_id status=failed reason=gpu_worker_restart_limit" >&2
      return "$worker_status"
    fi
    sleep "$GPU_WORKER_RESTART_DELAY_SECONDS"
  done
}

cd "$REPO_ROOT"
pids=()
total_worker_slots=$((${#GPU_ID_LIST[@]} * GPU_WORKERS_PER_GPU))
if [[ -n "$GLOBAL_SLOT_OFFSET" || -n "$GLOBAL_SLOT_COUNT" ]]; then
  if [[ ! "$GLOBAL_SLOT_OFFSET" =~ ^[0-9]+$ || ! "$GLOBAL_SLOT_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "GLOBAL_SLOT_OFFSET and GLOBAL_SLOT_COUNT must both be non-negative integer shard values" >&2
    exit 2
  fi
  if (( GLOBAL_SLOT_OFFSET + total_worker_slots > GLOBAL_SLOT_COUNT )); then
    echo "Local slots [$GLOBAL_SLOT_OFFSET,$((GLOBAL_SLOT_OFFSET + total_worker_slots))) exceed GLOBAL_SLOT_COUNT=$GLOBAL_SLOT_COUNT" >&2
    exit 2
  fi
fi
for ((worker_id = 0; worker_id < total_worker_slots; worker_id++)); do
  gpu="${GPU_ID_LIST[$((worker_id / GPU_WORKERS_PER_GPU))]}"
  jobs=()
  job_index=0
  node_job_index=0
  for profile in "${PROFILES[@]}"; do
    for task in "${TASKS[@]}"; do
      if [[ -n "$GLOBAL_SLOT_COUNT" ]]; then
        if (( job_index % GLOBAL_SLOT_COUNT == GLOBAL_SLOT_OFFSET + worker_id )); then
          jobs+=("$task:$profile")
        fi
      elif (( job_index % NODE_COUNT == NODE_INDEX )); then
        if (( node_job_index % total_worker_slots == worker_id )); then
          jobs+=("$task:$profile")
        fi
        node_job_index=$((node_job_index + 1))
      fi
      job_index=$((job_index + 1))
    done
  done
  if (( ${#jobs[@]} )); then
    echo "Node shard $NODE_INDEX/$NODE_COUNT assigned ${#jobs[@]} ${ROBOT_PLATFORM} subsets to GPU $gpu worker $worker_id: ${jobs[*]}"
    supervise_gpu_worker "$gpu" "$worker_id" "${jobs[@]}" &
    pids+=("$!")
  fi
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed )); then
  echo "One or more Fabrica profile collectors failed; inspect $OUTPUT_ROOT/logs." >&2
  exit 1
fi
echo "Node shard $NODE_INDEX/$NODE_COUNT is complete under $OUTPUT_ROOT."
