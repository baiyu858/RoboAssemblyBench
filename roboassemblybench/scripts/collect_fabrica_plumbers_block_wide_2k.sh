#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec conda run --no-capture-output -n "${ORCHESTRATOR_ENV:-internutopia311}" env \
  PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 \
  python roboassemblybench/scripts/collect_fabrica_plumbers_block_2k.py \
  --recipe "${RECIPE:-roboassemblybench/tasks/fabrica_plumbers_block_ur5e_right_base_prepare/recipe_wide_30cm_showcase.yaml}" \
  --scene-profile "${SCENE_PROFILE:-taoyuan_grscenes_tabletop}" \
  --num-episodes "${NUM_EPISODES:-2000}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --start-seed "${START_SEED:-0}" \
  --max-attempts "${MAX_ATTEMPTS:-10000}" \
  --layout-seeds ${LAYOUT_SEEDS:-439314 2831800 915333 4667522 1683518 2896846 310134 1875394 675630 518498 1705799 662755 1252392 2020953 741847 2170245} \
  --min-available-memory-gib "${MIN_AVAILABLE_MEMORY_GIB:-5.5}" \
  --abort-available-memory-gib "${ABORT_AVAILABLE_MEMORY_GIB:-1.5}" \
  --worker-timeout-seconds "${WORKER_TIMEOUT_SECONDS:-1800}" \
  --estimated-episode-mib "${ESTIMATED_EPISODE_MIB:-64}" \
  --disk-reserve-gib "${DISK_RESERVE_GIB:-80}" \
  --output-dir "${OUTPUT_DIR:-outputs/fabrica_plumbers_block_ur5e_right_base_prepare_2k_raw_wide_v1}" \
  "$@"
