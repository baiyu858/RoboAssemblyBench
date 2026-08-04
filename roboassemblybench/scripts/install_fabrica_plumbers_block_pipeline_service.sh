#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE="${REPO_ROOT}/roboassemblybench/systemd/roboassemblybench-fabrica-pipeline.service.in"
USER_UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_PATH="${USER_UNIT_DIR}/roboassemblybench-fabrica-pipeline.service"

mkdir -p "${USER_UNIT_DIR}"
sed "s|@REPO_ROOT@|${REPO_ROOT}|g" "${TEMPLATE}" > "${UNIT_PATH}"
systemctl --user daemon-reload
systemctl --user enable --now roboassemblybench-fabrica-pipeline.service
systemctl --user --no-pager status roboassemblybench-fabrica-pipeline.service
