#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXP_NAME="${EXP_NAME:-codex_plumbers_block_official}" \
ASSEMBLY="${ASSEMBLY:-plumbers_block}" \
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/third_part/Fabrica/records/opengl/codex_official_nonrl/plumbers_block.mp4}" \
"${SCRIPT_DIR}/render_fabrica_official_cooling_manifold_nonrl.sh" "$@"
