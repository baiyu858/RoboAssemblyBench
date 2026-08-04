#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASK="${TASK:-${1:-}}"
if [[ -z "${TASK}" ]]; then
  echo "Usage: $0 <canonical-assembly> [viewer arguments...]" >&2
  exit 2
fi
if [[ "${1:-}" == "${TASK}" ]]; then
  shift
fi

CONDA_ENV="${CONDA_ENV:-internutopia311}"
SCENE_PROFILE="${SCENE_PROFILE:-taoyuan_grscenes_tabletop}"
SEED="${SEED:-0}"
RECIPE="fabrica_${TASK}_ur5e_staged"
RECIPE_PATH="${REPO_ROOT}/roboassemblybench/tasks/${RECIPE}/recipe.yaml"
if [[ ! -f "${RECIPE_PATH}" ]]; then
  echo "Canonical Fabrica recipe not found: ${RECIPE_PATH}" >&2
  exit 2
fi
RANDOMIZATION_ARG=()
if [[ "${DOMAIN_RANDOMIZATION:-1}" == "1" || "${DOMAIN_RANDOMIZATION:-true}" == "true" ]]; then
  RANDOMIZATION_ARG=(--domain-randomization)
fi

cd "${REPO_ROOT}"
echo "Opening ${RECIPE}; optical board remains fixed for every seed."
conda run -n "${CONDA_ENV}" env PYTHONNOUSERSITE=1 python roboassemblybench/scripts/view_task_scene.py \
  --recipe "${RECIPE}" \
  --scene-profile "${SCENE_PROFILE}" \
  --seed "${SEED}" \
  "${RANDOMIZATION_ARG[@]}" \
  "$@"
