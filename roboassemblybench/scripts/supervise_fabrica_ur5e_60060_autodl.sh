#!/usr/bin/env bash
set -euo pipefail

if [[ -f /root/.bashrc ]]; then
  set +u
  source /root/.bashrc
  set -u
fi

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/baiyongjie/projectA17}"
REPO_ROOT="${REPO_ROOT:-$PROJECT_ROOT/source/InternUtopia_ur5e_production_20260820}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/fabrica_ur5e_60060_twostage_80hz_front640_depth352_renderfix_20260820}"
COLLECTOR_PID_PATH="${COLLECTOR_PID_PATH:-$PROJECT_ROOT/logs/fabrica_ur5e_60060_launcher.pid}"
SUPERVISOR_PID_PATH="${SUPERVISOR_PID_PATH:-$PROJECT_ROOT/logs/fabrica_ur5e_60060_supervisor.pid}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-$PROJECT_ROOT/logs/fabrica_ur5e_60060_supervisor.log}"
MONITOR_LOG="${MONITOR_LOG:-$PROJECT_ROOT/logs/fabrica_ur5e_60060_monitor.log}"
LOCK_DIR="${LOCK_DIR:-$PROJECT_ROOT/logs/fabrica_ur5e_60060_supervisor.lock}"
TARGET_PER_TASK="${TARGET_PER_TASK:-1430}"
MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-180}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
SNAPSHOT_INTERVAL_SECONDS="${SNAPSHOT_INTERVAL_SECONDS:-300}"
# The collectors have their own 15-minute worker watchdog.  Keep the outer
# watchdog conservative because a cold Isaac startup can take more than one
# collection batch before the worker log is updated.
STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-10800}"
RESTART_BACKOFF_SECONDS="${RESTART_BACKOFF_SECONDS:-120}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" && -x /root/miniconda3/envs/xvla/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/envs/xvla/bin/python
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo 'A Python 3 interpreter is required by the collection supervisor.' >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs" "$OUTPUT_ROOT"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [[ -f "$SUPERVISOR_PID_PATH" ]] && kill -0 "$(cat "$SUPERVISOR_PID_PATH")" 2>/dev/null; then
    echo "Supervisor is already running with PID $(cat "$SUPERVISOR_PID_PATH")."
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
  if [[ -f "$COLLECTOR_PID_PATH" ]]; then
    cat "$COLLECTOR_PID_PATH"
  fi
}

collector_worker_pids() {
  ps -eo pid=,args= | awk -v output_root="$OUTPUT_ROOT" '
    index($0, output_root) &&
    ($0 ~ /collect_fabrica_plumbers_block_2k[.]py/ ||
     $0 ~ /replay_fabrica_successful_trajectories[.]py/) {
      print $1
    }
  '
}

collector_alive() {
  local pid
  pid="$(collector_pid)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  [[ -n "$(collector_worker_pids)" ]]
}

stop_collector() {
  local pid
  local worker_pids
  pid="$(collector_pid)"
  worker_pids="$(collector_worker_pids)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    log "event=collector_stop pid=$pid"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  elif [[ -n "$worker_pids" ]]; then
    log "event=collector_stop orphan_workers=$(tr '\n' ',' <<<"$worker_pids" | sed 's/,$//')"
  fi
  if [[ -n "$worker_pids" ]]; then
    # shellcheck disable=SC2086
    kill -TERM $worker_pids 2>/dev/null || true
  fi
  for _ in $(seq 1 30); do
    collector_alive || break
    sleep 1
  done
  if collector_alive; then
    worker_pids="$(collector_worker_pids)"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    if [[ -n "$worker_pids" ]]; then
      # shellcheck disable=SC2086
      kill -KILL $worker_pids 2>/dev/null || true
    fi
  fi
  rm -f "$COLLECTOR_PID_PATH"
}

free_disk_gib() {
  df -Pk "$OUTPUT_ROOT" | awk 'NR == 2 {printf "%d\n", $4 / 1024 / 1024}'
}

latest_log_epoch() {
  local latest
  latest="$(find "$OUTPUT_ROOT/logs" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -n 1 || true)"
  if [[ -n "$latest" ]]; then
    printf '%.0f\n' "$latest"
  else
    date +%s
  fi
}

collection_complete() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT" "$TARGET_PER_TASK" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
target = int(sys.argv[2])
tasks = ('beam', 'car', 'cooling_manifold', 'duct', 'gamepad', 'plumbers_block', 'stool_circular')
profiles = ('position', 'object_distractors', 'texture', 'lighting', 'table_color', 'scene')
for task in tasks:
    for profile in profiles:
        info = root / 'rendered' / task / profile / 'lerobot_v3' / 'meta' / 'info.json'
        if not info.is_file():
            raise SystemExit(1)
        try:
            if int(json.loads(info.read_text(encoding='utf-8')).get('total_episodes', 0)) < target:
                raise SystemExit(1)
        except (OSError, ValueError, json.JSONDecodeError):
            raise SystemExit(1)
raise SystemExit(0)
PY
}

snapshot() {
  {
    printf '\n===== %s =====\n' "$(date -Is)"
    bash "$REPO_ROOT/roboassemblybench/scripts/monitor_fabrica_7tasks_twostage.sh" "$OUTPUT_ROOT"
  } >>"$MONITOR_LOG" 2>&1 || true
}

last_snapshot=0
restart_count=0
log "event=supervisor_start pid=$$ output=$OUTPUT_ROOT"

while true; do
  now="$(date +%s)"
  free_gib="$(free_disk_gib)"

  if collection_complete; then
    snapshot
    log "event=collection_complete target=60060"
    exit 0
  fi

  if (( free_gib < MIN_FREE_DISK_GIB )); then
    collector_alive && stop_collector
    log "event=storage_guard free_gib=$free_gib minimum_gib=$MIN_FREE_DISK_GIB status=paused"
    sleep "$SNAPSHOT_INTERVAL_SECONDS"
    continue
  fi

  if collector_alive; then
    latest_epoch="$(latest_log_epoch)"
    stalled_seconds=$((now - latest_epoch))
    if (( stalled_seconds >= STALL_TIMEOUT_SECONDS )); then
      log "event=collector_stalled stalled_seconds=$stalled_seconds action=restart"
      stop_collector
      sleep "$RESTART_BACKOFF_SECONDS"
      continue
    fi
  else
    rm -f "$COLLECTOR_PID_PATH"
    restart_count=$((restart_count + 1))
    log "event=collector_launch attempt=$restart_count free_gib=$free_gib"
    if ! bash "$REPO_ROOT/roboassemblybench/scripts/launch_fabrica_ur5e_60060_autodl.sh" >>"$SUPERVISOR_LOG" 2>&1; then
      log "event=collector_launch_failed attempt=$restart_count"
      sleep "$RESTART_BACKOFF_SECONDS"
      continue
    fi
    sleep 15
  fi

  if (( now - last_snapshot >= SNAPSHOT_INTERVAL_SECONDS )); then
    snapshot
    last_snapshot="$now"
  fi
  sleep "$CHECK_INTERVAL_SECONDS"
done
