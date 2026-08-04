#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec conda run --no-capture-output -n "${ORCHESTRATOR_ENV:-internutopia311}" env PYTHONNOUSERSITE=1 \
  python roboassemblybench/scripts/collect_fabrica_plumbers_block_2k.py \
  --num-episodes "${NUM_EPISODES:-2000}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --start-seed "${START_SEED:-0}" \
  --max-attempts "${MAX_ATTEMPTS:-10000}" \
  --layout-seeds ${LAYOUT_SEEDS:-4906 485 34 12} \
  --abort-available-memory-gib "${ABORT_AVAILABLE_MEMORY_GIB:-1.5}" \
  --output-dir "${OUTPUT_DIR:-outputs/fabrica_plumbers_block_ur5e_right_base_prepare_2k_raw_v3}" \
  "$@"
