#!/usr/bin/env bash
set -u

# Run this script on the destination host.  Workers own disjoint hash
# partitions, so changing the worker count does not duplicate completed data.
DESTINATION_ROOT="${DESTINATION_ROOT:?DESTINATION_ROOT is required}"
SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
STATE_ROOT="${STATE_ROOT:-$DESTINATION_ROOT/.stream_state}"
WORKER_ROOT="${WORKER_ROOT:-$DESTINATION_ROOT/.stream_workers}"
SOURCE_KEY="${SOURCE_KEY:-$STATE_ROOT/source_pull_ed25519}"
SOURCE_PORT_BASE="${SOURCE_PORT_BASE:-45217}"
TUNNEL_COUNT="${TUNNEL_COUNT:-1}"
WORKERS="${WORKERS:-5}"
POLL_SECONDS="${POLL_SECONDS:-45}"
SSH_COMMAND_TIMEOUT="${SSH_COMMAND_TIMEOUT:-180}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STREAM_SCRIPT="${STREAM_SCRIPT:-$STATE_ROOT/stream_successful_episodes_to_shared.py}"
RUNNER="${RUNNER:-$STATE_ROOT/run_successful_episode_stream_daemon.sh}"

if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be a positive integer." >&2
  exit 2
fi
if [[ ! "$SOURCE_PORT_BASE" =~ ^[1-9][0-9]*$ || ! "$TUNNEL_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "SOURCE_PORT_BASE and TUNNEL_COUNT must be positive integers." >&2
  exit 2
fi

mkdir -p "$STATE_ROOT" "$WORKER_ROOT"
DATABASE_PATH="$DESTINATION_ROOT/.stream_state/transfers.sqlite3"

stop_worker() {
  local worker_state="$1"
  local pid
  for pid_file in "$worker_state/watchdog.pid" "$worker_state/daemon.pid"; do
    [[ -s "$pid_file" ]] || continue
    pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      for _ in {1..20}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

for worker_state in "$WORKER_ROOT"/worker_*; do
  [[ -d "$worker_state" ]] || continue
  stop_worker "$worker_state"
done

for ((index = 0; index < WORKERS; index += 1)); do
  worker_state="$WORKER_ROOT/worker_${index}"
  mkdir -p "$worker_state"
  chmod 700 "$worker_state"
  PARTITION_INDEX="$index" \
  PARTITION_COUNT="$WORKERS" \
  SOURCE_PORT="$((SOURCE_PORT_BASE + index % TUNNEL_COUNT))" \
  SOURCE_ROOT="$SOURCE_ROOT" \
  DESTINATION_ROOT="$DESTINATION_ROOT" \
  STATE_ROOT="$worker_state" \
  SOURCE_KEY="$SOURCE_KEY" \
  DATABASE_PATH="$DATABASE_PATH" \
  LOCK_PATH="$worker_state/daemon.lock" \
  POLL_SECONDS="$POLL_SECONDS" \
  SSH_COMMAND_TIMEOUT="$SSH_COMMAND_TIMEOUT" \
  PYTHON_BIN="$PYTHON_BIN" \
  STREAM_SCRIPT="$STREAM_SCRIPT" \
  nohup "$RUNNER" >>"$worker_state/watchdog.stdout.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"$worker_state/watchdog.pid"
done

printf 'started %s stream workers; ledger=%s\n' "$WORKERS" "$DATABASE_PATH"
