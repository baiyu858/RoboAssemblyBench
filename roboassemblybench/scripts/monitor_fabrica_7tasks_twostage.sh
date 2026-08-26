#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/baiyongjie/projectA17/data/fabrica_ur5e_60060_twostage_80hz_front640_depth352_renderfix_20260820}"
TARGET_PER_TASK="${TARGET_PER_TASK:-1430}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" && -x /root/miniconda3/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/bin/python
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo 'A Python 3 interpreter is required to read collection manifests.' >&2
  exit 1
fi
TASKS=(beam car cooling_manifold duct gamepad plumbers_block stool_circular)
PROFILES=(position object_distractors texture lighting table_color scene)

count_manifest() {
  "$PYTHON_BIN" - "$ROOT" "$1" "$2" "$TARGET_PER_TASK" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
task = sys.argv[2]
profile = sys.argv[3]
target = int(sys.argv[4])
if profile == 'position':
    primary = root / 'stage1' / task / 'collection_manifest.json'
    shards = root / 'stage1' / task / 'shards'
    pattern = 'shard_*/collection_manifest.json'
else:
    primary = root / 'rendered' / task / profile / 'replay_manifest.json'
    shards = root / 'rendered' / task / profile / 'shards'
    pattern = 'shard_*/replay_manifest.json'

if primary.is_file():
    payloads = [json.loads(primary.read_text(encoding='utf-8'))]
else:
    payloads = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(shards.glob(pattern))]
manifest_success = sum(int(payload.get('num_successful', len(payload.get('successful_episodes') or {}))) for payload in payloads)
manifest_failed = sum(int(payload.get('num_failed_attempts', len(payload.get('failed_attempts') or []))) for payload in payloads)

# Manifests lag behind in-flight batches.  Raw successful metadata is already
# durable, so include it in the live total even when a shard manifest exists.
if profile == 'position':
    raw_root = root / 'stage1' / task
else:
    raw_root = root / 'rendered' / task / profile
raw_success = 0
raw_failed = 0
for path in raw_root.glob('shards/*/batches/*/episode_*_cartesian_raw/metadata.json'):
    try:
        metadata = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, json.JSONDecodeError):
        raw_failed += 1
        continue
    if metadata.get('metrics', {}).get('success') is True:
        raw_success += 1
    else:
        raw_failed += 1
success = max(manifest_success, raw_success)
failed = max(manifest_failed, raw_failed)
if success >= target and payloads and all(payload.get('complete') for payload in payloads):
    status = 'complete'
elif success or failed:
    status = 'running'
else:
    status = 'pending'
print(success, failed, status)
PY
}

printf '%-20s %-20s %8s %8s %8s %-10s\n' task group success target failed status
total_success=0
total_failed=0
for task in "${TASKS[@]}"; do
  for profile in "${PROFILES[@]}"; do
    read -r success failed status < <(count_manifest "$task" "$profile")
    total_success=$((total_success + success))
    total_failed=$((total_failed + failed))
    printf '%-20s %-20s %8d %8d %8d %-10s\n' \
      "$task" "$profile" "$success" "$TARGET_PER_TASK" "$failed" "$status"
  done
done
total_target=$((${#TASKS[@]} * ${#PROFILES[@]} * TARGET_PER_TASK))
printf '%-20s %-20s %8d %8d %8d\n' TOTAL all "$total_success" "$total_target" "$total_failed"
du -sh "$ROOT" 2>/dev/null || true
df -h "$ROOT" 2>/dev/null | tail -n 1 || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader 2>/dev/null || true
if [[ -r /sys/fs/cgroup/memory.max && -r /sys/fs/cgroup/memory.current ]]; then
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path

raw_limit = Path('/sys/fs/cgroup/memory.max').read_text(encoding='utf-8').strip()
current = int(Path('/sys/fs/cgroup/memory.current').read_text(encoding='utf-8').strip())
if raw_limit != 'max':
    limit = int(raw_limit)
    gib = 1024**3
    print(
        f'cgroup_memory current={current / gib:.1f}GiB '
        f'limit={limit / gib:.1f}GiB available={(limit - current) / gib:.1f}GiB'
    )
PY
fi
