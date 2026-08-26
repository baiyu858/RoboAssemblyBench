#!/usr/bin/env bash
set -u

SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
DESTINATION_ROOT="${DESTINATION_ROOT:?DESTINATION_ROOT is required}"
STATE_ROOT="${STATE_ROOT:-$DESTINATION_ROOT/.stream_state}"
SOURCE_KEY="${SOURCE_KEY:-$STATE_ROOT/source_pull_ed25519}"
SOURCE_PORT="${SOURCE_PORT:-45217}"
POLL_SECONDS="${POLL_SECONDS:-45}"
SSH_COMMAND_TIMEOUT="${SSH_COMMAND_TIMEOUT:-180}"
DATABASE_PATH="${DATABASE_PATH:-$DESTINATION_ROOT/.stream_state/transfers.sqlite3}"
LOCK_PATH="${LOCK_PATH:-$STATE_ROOT/daemon.lock}"
PARTITION_INDEX="${PARTITION_INDEX:-0}"
PARTITION_COUNT="${PARTITION_COUNT:-1}"
RESTART_SECONDS="${RESTART_SECONDS:-10}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STREAM_SCRIPT="${STREAM_SCRIPT:-$STATE_ROOT/stream_successful_episodes_to_shared.py}"

mkdir -p "$STATE_ROOT"
printf '%s\n' "$$" >"$STATE_ROOT/watchdog.pid"
child_pid=""

stop() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill "$child_pid"
    wait "$child_pid" 2>/dev/null || true
  fi
  exit 0
}

trap stop INT TERM

while true; do
  "$PYTHON_BIN" "$STREAM_SCRIPT" \
    --source-root "$SOURCE_ROOT" \
    --destination-root "$DESTINATION_ROOT" \
    --state-root "$STATE_ROOT" \
    --source-key "$SOURCE_KEY" \
    --source-port "$SOURCE_PORT" \
    --poll-seconds "$POLL_SECONDS" \
    --ssh-command-timeout "$SSH_COMMAND_TIMEOUT" \
    --database-path "$DATABASE_PATH" \
    --lock-path "$LOCK_PATH" \
    --partition-index "$PARTITION_INDEX" \
    --partition-count "$PARTITION_COUNT" &
  child_pid="$!"
  printf '%s\n' "$child_pid" >"$STATE_ROOT/daemon.pid"
  wait "$child_pid"
  exit_code="$?"
  child_pid=""
  printf '%s stream_exit=%s; restarting in %ss\n' \
    "$(date -Is)" "$exit_code" "$RESTART_SECONDS"
  sleep "$RESTART_SECONDS"
done
