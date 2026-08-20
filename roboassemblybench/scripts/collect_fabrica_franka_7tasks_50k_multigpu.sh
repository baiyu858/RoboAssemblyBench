#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ROBOT_PLATFORM=franka
exec "$SCRIPT_DIR/collect_fabrica_7tasks_50k_multigpu.sh" "$@"
