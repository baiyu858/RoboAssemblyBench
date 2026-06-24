#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
SCENE_PROFILE="${SCENE_PROFILE:-taoyuan_grscenes_tabletop}"
SEED="${SEED:-0}"
WARMUP_RENDER_STEPS="${WARMUP_RENDER_STEPS:-8}"

cd "${REPO_ROOT}"
echo "Opening Fabrica plumbers_block online-plan FixPlug-RL scene only."
echo "Viewer mode only: no demo policy, no robot actions, no data collection."
conda run -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python roboassemblybench/scripts/view_task_scene.py \
  --recipe fabrica_plumbers_block_online_plan_fixplug_rl \
  --scene-profile "${SCENE_PROFILE}" \
  --seed "${SEED}" \
  --warmup-render-steps "${WARMUP_RENDER_STEPS}" "$@"
