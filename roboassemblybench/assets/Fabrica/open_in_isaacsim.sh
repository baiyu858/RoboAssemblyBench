#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/isaac-sim}"
PYTHON_SH="${ISAAC_SIM_ROOT}/python.sh"

if [[ ! -x "${PYTHON_SH}" ]]; then
  echo "Could not find Isaac Sim python.sh at: ${PYTHON_SH}" >&2
  echo "Set ISAAC_SIM_ROOT, for example:" >&2
  echo "  ISAAC_SIM_ROOT=/path/to/isaac-sim ${SCRIPT_DIR}/open_in_isaacsim.sh" >&2
  exit 1
fi

cd "${PACKAGE_ROOT}"
exec "${PYTHON_SH}" "${SCRIPT_DIR}/open_in_isaacsim.py" "$@"
