#!/usr/bin/env bash
set -euo pipefail
umask 0000

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
ROBOT_PLATFORM="${ROBOT_PLATFORM:-ur5e}"
FRANKA_DATA_ROOT="${FRANKA_DATA_ROOT:-/data/a17/baiyongjie/data/franka}"
FRANKA_MACHINE_ID="${FRANKA_MACHINE_ID:-}"
if [[ "$ROBOT_PLATFORM" == "franka" && ! "$FRANKA_MACHINE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Set FRANKA_MACHINE_ID to a machine-specific directory name for Franka export." >&2
  exit 2
fi
if [[ "$ROBOT_PLATFORM" == "franka" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-$FRANKA_DATA_ROOT/$FRANKA_MACHINE_ID/fabrica_7tasks_franka_50k_lerobot_v3}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-/data/a17/baiyongjie/data/fabrica_7tasks_50k_lerobot_v3}"
fi
LEROBOT_PYTHON="${LEROBOT_PYTHON:-/data/baiyongjie/.venvs/roboassemblybench-act/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-120}"
RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH:-$REPO_ROOT/.runtime_python}"

TASKS=(beam car cooling_manifold duct gamepad plumbers_block stool_circular)
PROFILES=(object_distractors texture lighting table_color scene)

if [[ ! -x "$LEROBOT_PYTHON" ]]; then
  echo "LeRobot Python was not found at $LEROBOT_PYTHON" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/export_logs" "$OUTPUT_ROOT/export_locks" "$OUTPUT_ROOT/status"

manifest_complete() {
  "$LEROBOT_PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding='utf-8'))
raise SystemExit(0 if payload.get('complete') is True else 1)
PY
}

dataset_complete() {
  local raw_manifest="$1"
  local dataset_dir="$2"
  [[ -f "$dataset_dir/.roboassemblybench_export_complete" ]] || return 1
  "$LEROBOT_PYTHON" - "$raw_manifest" "$dataset_dir/roboassemblybench_conversion_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

raw = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
converted = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
target = int(raw.get('target_successful_episodes', 0))
actual = int(converted.get('total_episodes', len(converted.get('episodes') or [])))
raise SystemExit(0 if target > 0 and actual == target else 1)
PY
}

export_subset() {
  local task="$1"
  local profile="$2"
  local raw_dir="$OUTPUT_ROOT/$task/$profile/raw"
  local dataset_dir="$OUTPUT_ROOT/$task/$profile/lerobot_v3"
  local status_path="$OUTPUT_ROOT/status/${task}__${profile}.status"
  local log_path="$OUTPUT_ROOT/export_logs/${task}__${profile}.log"
  local lock_dir="$OUTPUT_ROOT/export_locks/${task}__${profile}.lock"

  if dataset_complete "$raw_dir/collection_manifest.json" "$dataset_dir"; then
    return 0
  fi
  manifest_complete "$raw_dir/collection_manifest.json" || return 1
  mkdir "$lock_dir" 2>/dev/null || return 1

  echo "$(date -Is) subset=$task/$profile status=exporting" | tee "$status_path" >>"$log_path"
  local resume_args=()
  if [[ -f "$dataset_dir/meta/info.json" ]]; then
    resume_args+=(--resume)
  elif [[ -d "$dataset_dir" && -n "$(find "$dataset_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    mv "$dataset_dir" "${dataset_dir}.incomplete.$(date +%s)"
  fi

  if env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="$RUNTIME_PYTHONPATH:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$LEROBOT_PYTHON" "$REPO_ROOT/roboassemblybench/scripts/export_fabrica_lerobot_v3.py" \
      --input-dir "$raw_dir" \
      --output-dir "$dataset_dir" \
      --repo-id "baiyu858/roboassemblybench_fabrica_${task}_${ROBOT_PLATFORM}_${profile}" \
      --encoder-threads 2 \
      "${resume_args[@]}" >>"$log_path" 2>&1; then
    if "$LEROBOT_PYTHON" - "$raw_dir/collection_manifest.json" "$dataset_dir/roboassemblybench_conversion_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

raw = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
converted = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
target = int(raw.get('target_successful_episodes', 0))
actual = int(converted.get('total_episodes', len(converted.get('episodes') or [])))
raise SystemExit(0 if target > 0 and actual == target else 1)
PY
    then
      touch "$dataset_dir/.roboassemblybench_export_complete"
      echo "$(date -Is) subset=$task/$profile status=complete" | tee "$status_path" >>"$log_path"
    else
      echo "$(date -Is) subset=$task/$profile status=export_count_mismatch" | tee "$status_path" >>"$log_path"
    fi
  else
    echo "$(date -Is) subset=$task/$profile status=export_failed" | tee "$status_path" >>"$log_path"
  fi
  rmdir "$lock_dir"
}

while true; do
  exported=0
  for task in "${TASKS[@]}"; do
    for profile in "${PROFILES[@]}"; do
      dataset_dir="$OUTPUT_ROOT/$task/$profile/lerobot_v3"
      if dataset_complete "$OUTPUT_ROOT/$task/$profile/raw/collection_manifest.json" "$dataset_dir"; then
        exported=$((exported + 1))
        continue
      fi
      export_subset "$task" "$profile" || true
      if dataset_complete "$OUTPUT_ROOT/$task/$profile/raw/collection_manifest.json" "$dataset_dir"; then
        exported=$((exported + 1))
      fi
    done
  done
  if (( exported == 35 )); then
    echo "All 35 Fabrica subsets were exported to LeRobot v3 under $OUTPUT_ROOT."
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
