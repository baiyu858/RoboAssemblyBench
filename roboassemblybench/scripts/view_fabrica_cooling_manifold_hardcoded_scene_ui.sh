#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-internutopia311}"
SCENE_PROFILE="${SCENE_PROFILE:-taoyuan_grscenes_tabletop}"
SEED="${SEED:-0}"
WARMUP_RENDER_STEPS="${WARMUP_RENDER_STEPS:-8}"

cd "${REPO_ROOT}"
echo "Opening hardcoded Fabrica cooling-manifold scene only."
conda run -n "${CONDA_ENV}" python roboassemblybench/scripts/view_task_scene.py \
  --recipe fabrica_cooling_manifold_hardcoded \
  --scene-profile "${SCENE_PROFILE}" \
  --seed "${SEED}" \
  --warmup-render-steps "${WARMUP_RENDER_STEPS}" "$@"
