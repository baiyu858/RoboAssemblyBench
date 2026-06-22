#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
export ROBOBRAIN_AGENT_HOST="${ROBOBRAIN_AGENT_HOST:-127.0.0.1}"
export ROBOBRAIN_AGENT_PORT="${ROBOBRAIN_AGENT_PORT:-7861}"

cd "${REPO_ROOT}"
echo "Starting RoboBrain Agent App at http://${ROBOBRAIN_AGENT_HOST}:${ROBOBRAIN_AGENT_PORT}"
echo "Conda env: ${CONDA_ENV}"

if [[ -z "${CONDA_ENV}" || "${CONDA_ENV}" == "none" ]]; then
  env PYTHONNOUSERSITE=1 python -m roboassemblybench.robobrain.webapp
else
  conda run -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python -m roboassemblybench.robobrain.webapp
fi
