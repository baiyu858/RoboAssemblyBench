#!/usr/bin/env bash
set -euo pipefail
umask 0000

# Persistent, node-local supervisor for an independent subset of Fabrica tasks.
# The collection pipeline has per-worker retries; this wrapper handles a dead
# scheduler, long log stalls, low disk space, and SSH session disconnects.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must point to a node-specific data directory}"
FABRICA_TASKS="${FABRICA_TASKS:?FABRICA_TASKS must be a comma-separated task subset}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-$REPO_ROOT/roboassemblybench/scripts/collect_fabrica_7tasks_50k_twostage_multigpu.sh}"
CONTROL_PYTHON="${CONTROL_PYTHON:-$(command -v python3 || command -v python || true)}"
TARGET_PER_TASK="${TARGET_PER_TASK:-1430}"
STAGE1_SHARDS="${STAGE1_SHARDS:-}"
MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-256}"
CACHE_GUARD_PATH="${CACHE_GUARD_PATH:-}"
MIN_CACHE_FREE_GIB="${MIN_CACHE_FREE_GIB:-4}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
SNAPSHOT_INTERVAL_SECONDS="${SNAPSHOT_INTERVAL_SECONDS:-300}"
STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-10800}"
RESTART_BACKOFF_SECONDS="${RESTART_BACKOFF_SECONDS:-120}"
COLLECTOR_NICE="${COLLECTOR_NICE:-10}"
STATE_DIR="${STATE_DIR:-$OUTPUT_ROOT/supervisor}"
COLLECTOR_PID_PATH="$STATE_DIR/collector.pid"
SUPERVISOR_PID_PATH="$STATE_DIR/supervisor.pid"
STATUS_PATH="$STATE_DIR/watchdog.status"
SUPERVISOR_LOG="$STATE_DIR/supervisor.log"
COLLECTOR_LOG="$OUTPUT_ROOT/logs/collector.launcher.log"
LOCK_DIR="$STATE_DIR/lock"

if [[ -z "$CONTROL_PYTHON" || ! -x "$CONTROL_PYTHON" ]]; then
  echo 'A usable Python 3 interpreter is required by the collection supervisor.' >&2
  exit 2
fi
if [[ ! -x "$PIPELINE_SCRIPT" ]]; then
  echo "Pipeline script is not executable: $PIPELINE_SCRIPT" >&2
  exit 2
fi

# The pipeline is launched as a child process. Export the validated interpreter
# so nodes that provide only `python3` do not enter the scheduler with an empty
# CONTROL_PYTHON command.
export CONTROL_PYTHON

mkdir -p "$STATE_DIR" "$OUTPUT_ROOT/logs"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [[ -f "$SUPERVISOR_PID_PATH" ]] && kill -0 "$(<"$SUPERVISOR_PID_PATH")" 2>/dev/null; then
    echo "Supervisor is already running with PID $(<"$SUPERVISOR_PID_PATH")."
    exit 0
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
fi
printf '%s\n' "$$" >"$SUPERVISOR_PID_PATH"
trap 'rm -rf "$LOCK_DIR"; rm -f "$SUPERVISOR_PID_PATH"' EXIT

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$SUPERVISOR_LOG"
}

collector_pid() {
  [[ -f "$COLLECTOR_PID_PATH" ]] && <"$COLLECTOR_PID_PATH"
}

collector_worker_pgids() {
  # Isaac workers launch generate_demos.py beneath the collection/replay
  # scheduler.  Track process groups instead of only the scheduler's Python
  # process so a restart cannot leave an orphaned Isaac/Kit process writing to
  # this node's output tree.
  ps -eo pgid=,args= | awk -v output_root="$OUTPUT_ROOT" '
    index($0, output_root) &&
    ($0 ~ /collect_fabrica_plumbers_block_2k[.]py/ ||
     $0 ~ /replay_fabrica_successful_trajectories[.]py/ ||
     $0 ~ /generate_demos[.]py/) { print $1 }
  ' | sort -nu
}

collector_orphaned_encoder_pids() {
  # Video encoders inherit the worker's output directory in argv.  A crashed
  # Kit process can leave them reparented to init, where they otherwise keep a
  # raw-video pipe and CPU core alive indefinitely.
  ps -eo pid=,ppid=,args= | awk -v output_root="$OUTPUT_ROOT" '
    $2 == 1 && index($0, output_root) && $0 ~ /\/ffmpeg([[:space:]]|$)/ { print $1 }
  '
}

reap_orphaned_encoders() {
  local orphaned_pids pid
  orphaned_pids="$(collector_orphaned_encoder_pids || true)"
  [[ -n "$orphaned_pids" ]] || return 0
  while IFS= read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    log "event=orphaned_encoder_stop pid=$pid"
    kill -TERM "$pid" 2>/dev/null || true
  done <<<"$orphaned_pids"
  sleep 3
  while IFS= read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    kill -0 "$pid" 2>/dev/null || continue
    log "event=orphaned_encoder_kill pid=$pid"
    kill -KILL "$pid" 2>/dev/null || true
  done <<<"$orphaned_pids"
}

collector_alive() {
  local pid
  pid="$(collector_pid || true)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  local pgid
  while IFS= read -r pgid; do
    [[ "$pgid" =~ ^[0-9]+$ ]] && kill -0 -- "-$pgid" 2>/dev/null && return 0
  done < <(collector_worker_pgids)
  return 1
}

stop_collector() {
  local pid worker_pgids pgid
  pid="$(collector_pid || true)"
  worker_pgids="$(collector_worker_pgids || true)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    log "event=collector_stop pid=$pid"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  fi
  while IFS= read -r pgid; do
    [[ "$pgid" =~ ^[0-9]+$ ]] || continue
    log "event=collector_stop worker_pgid=$pgid"
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done <<<"$worker_pgids"
  for _ in $(seq 1 45); do
    collector_alive || break
    sleep 1
  done
  if collector_alive; then
    worker_pgids="$(collector_worker_pgids || true)"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -KILL -- "-$pid" 2>/dev/null || true
    while IFS= read -r pgid; do
      [[ "$pgid" =~ ^[0-9]+$ ]] && kill -KILL -- "-$pgid" 2>/dev/null || true
    done <<<"$worker_pgids"
  fi
  rm -f "$COLLECTOR_PID_PATH"
}

free_disk_gib() {
  df -Pk "$1" | awk 'NR == 2 {printf "%d\n", $4 / 1024 / 1024}'
}

latest_log_epoch() {
  local latest
  latest="$(find "$OUTPUT_ROOT/logs" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -n 1 || true)"
  [[ -n "$latest" ]] && printf '%.0f\n' "$latest" || date +%s
}

write_status() {
  local now="$1" free_gib="$2" state="$3" detail="$4"
  "$CONTROL_PYTHON" - "$OUTPUT_ROOT" "$FABRICA_TASKS" "$TARGET_PER_TASK" "$now" "$free_gib" "$state" "$detail" >"$STATUS_PATH.tmp" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
tasks = [task for task in sys.argv[2].split(',') if task]
target = int(sys.argv[3])
payload = {
    'timestamp_unix': int(sys.argv[4]),
    'free_disk_gib': int(sys.argv[5]),
    'state': sys.argv[6],
    'detail': sys.argv[7],
    'target_per_task_profile': target,
    'tasks': {},
}
for task in tasks:
    groups = {}
    for profile in ('position', 'object_distractors', 'texture', 'lighting', 'table_color', 'scene'):
        root_dir = root / ('stage1' if profile == 'position' else 'rendered') / task
        if profile != 'position':
            root_dir /= profile
        name = 'collection_manifest.json' if profile == 'position' else 'replay_manifest.json'
        manifests = sorted(root_dir.glob(f'shards/shard_*/{name}'))
        successful = 0
        complete = bool(manifests)
        trajectory_only = False
        for path in manifests:
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError, json.JSONDecodeError):
                complete = False
                continue
            successful += int(data.get('num_successful', 0))
            complete = complete and bool(data.get('complete'))
            trajectory_only = trajectory_only or bool(data.get('trajectory_only', False))
        groups[profile] = {
            'successful': successful,
            'target': target,
            'complete': complete and successful >= target,
            'trajectory_only': trajectory_only,
        }
    payload['tasks'][task] = groups
print(json.dumps(payload, indent=2, sort_keys=True))
PY
  mv "$STATUS_PATH.tmp" "$STATUS_PATH"
}

collection_complete() {
  "$CONTROL_PYTHON" - "$STATUS_PATH" <<'PY'
import json
import sys
from pathlib import Path
try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
except (OSError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
groups = [group for task in payload['tasks'].values() for group in task.values()]
raise SystemExit(0 if groups and all(group['complete'] and not group['trajectory_only'] for group in groups) else 1)
PY
}

launch_collector() {
  # A distinct output root per machine prevents duplicate writers and locks.
  nohup setsid nice -n "$COLLECTOR_NICE" bash "$PIPELINE_SCRIPT" \
    >>"$COLLECTOR_LOG" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" >"$COLLECTOR_PID_PATH"
  log "event=collector_launch pid=$pid tasks=$FABRICA_TASKS output=$OUTPUT_ROOT"
}

restart_count=0
last_snapshot=0
log "event=supervisor_start pid=$$ tasks=$FABRICA_TASKS output=$OUTPUT_ROOT"
while true; do
  now="$(date +%s)"
  free_gib="$(free_disk_gib "$OUTPUT_ROOT")"
  reap_orphaned_encoders
  if (( now - last_snapshot >= SNAPSHOT_INTERVAL_SECONDS )); then
    write_status "$now" "$free_gib" "running" "periodic_snapshot"
    last_snapshot="$now"
  fi
  if collection_complete; then
    write_status "$now" "$free_gib" "complete" "all_full_rgbd_groups_complete"
    log 'event=collection_complete'
    exit 0
  fi
  if (( free_gib < MIN_FREE_DISK_GIB )); then
    collector_alive && stop_collector
    write_status "$now" "$free_gib" "paused" "storage_guard"
    log "event=storage_guard free_gib=$free_gib minimum=$MIN_FREE_DISK_GIB"
    sleep "$SNAPSHOT_INTERVAL_SECONDS"
    continue
  fi
  if [[ -n "$CACHE_GUARD_PATH" ]]; then
    cache_free_gib="$(free_disk_gib "$CACHE_GUARD_PATH")"
    if (( cache_free_gib < MIN_CACHE_FREE_GIB )); then
      collector_alive && stop_collector
      write_status "$now" "$free_gib" "paused" "cache_storage_guard_free_gib=${cache_free_gib}"
      log "event=cache_storage_guard free_gib=$cache_free_gib minimum=$MIN_CACHE_FREE_GIB path=$CACHE_GUARD_PATH"
      sleep "$SNAPSHOT_INTERVAL_SECONDS"
      continue
    fi
  fi
  if collector_alive; then
    latest_epoch="$(latest_log_epoch)"
    stalled_seconds=$((now - latest_epoch))
    if (( stalled_seconds >= STALL_TIMEOUT_SECONDS )); then
      log "event=collector_stalled seconds=$stalled_seconds action=restart"
      stop_collector
      sleep "$RESTART_BACKOFF_SECONDS"
    fi
  else
    rm -f "$COLLECTOR_PID_PATH"
    restart_count=$((restart_count + 1))
    launch_collector
    sleep 15
  fi
  sleep "$CHECK_INTERVAL_SECONDS"
done
