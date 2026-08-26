#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/a17/baiyongjie/data/fabrica_7tasks_50k_lerobot_v3}"
TASKS=(beam car cooling_manifold duct gamepad plumbers_block stool_circular)
PROFILES=(object_distractors texture lighting table_color scene)

target_for() {
  local task="$1"
  local profile="$2"
  if [[ "$task" == "stool_circular" ]]; then
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

printf '%-20s %-20s %8s %8s %8s %-18s\n' task profile success target failed status
total_success=0
total_target=0
total_failed=0
for task in "${TASKS[@]}"; do
  for profile in "${PROFILES[@]}"; do
    manifest="$ROOT/$task/$profile/raw/collection_manifest.json"
    status_file="$ROOT/status/${task}__${profile}.status"
    values="0 $(target_for "$task" "$profile") 0"
    if [[ -f "$manifest" ]]; then
      values="$(python - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(
    int(payload.get('num_successful', len(payload.get('successful_episodes') or {}))),
    int(payload.get('target_successful_episodes', 0)),
    int(payload.get('num_failed_attempts', len(payload.get('failed_attempts') or []))),
)
PY
)"
    fi
    status="not_started"
    if [[ -f "$status_file" ]]; then
      status="$(tail -n 1 "$status_file" | sed -n 's/.*status=\([^ ]*\).*/\1/p')"
    fi
    read -r success target failed <<<"$values"
    total_success=$((total_success + success))
    total_target=$((total_target + target))
    total_failed=$((total_failed + failed))
    printf '%-20s %-20s %8s %8s %8s %-18s\n' "$task" "$profile" "$success" "$target" "$failed" "$status"
  done
done
printf '%-20s %-20s %8s %8s %8s\n' TOTAL all "$total_success" "$total_target" "$total_failed"
du -sh "$ROOT" 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
