#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACT_ENV="${ACT_ENV:-roboassemblybench-act}"
ISAAC_ENV="${ISAAC_ENV:-internutopia311}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/outputs/fabrica_plumbers_block_ur5e_right_base_prepare_act/checkpoints/last/pretrained_model}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/fabrica_plumbers_block_ur5e_act_eval_50ep}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
NUM_EPISODES="${NUM_EPISODES:-50}"
START_SEED="${START_SEED:-10000}"
LAYOUT_SEEDS="${LAYOUT_SEEDS:-4906 485 34 12}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-180}"
SERVER_LOG="${SERVER_LOG:-${OUTPUT_DIR}/policy_server.log}"
EPISODES_DIR="${EPISODES_DIR:-${OUTPUT_DIR}/episodes}"
read -r -a LAYOUT_SEED_ARRAY <<< "${LAYOUT_SEEDS}"

if (( ${#LAYOUT_SEED_ARRAY[@]} == 0 )); then
  echo "LAYOUT_SEEDS must contain at least one integer seed." >&2
  exit 2
fi

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "ACT checkpoint is missing config.json: ${CHECKPOINT}" >&2
  exit 2
fi
if (exec 3<>"/dev/tcp/${HOST}/${PORT}") 2>/dev/null; then
  echo "Refusing to start because ${HOST}:${PORT} is already in use." >&2
  exit 2
fi

AMP_FLAG=(--use-amp)
if [[ "${USE_AMP:-1}" == "0" ]]; then
  AMP_FLAG=(--no-use-amp)
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${EPISODES_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

setsid conda run --no-capture-output -n "${ACT_ENV}" env \
  PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" \
  python -u roboassemblybench/scripts/serve_fabrica_plumbers_block_act.py \
    --checkpoint "${CHECKPOINT}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "${AMP_FLAG[@]}" \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while ! (exec 3<>"/dev/tcp/${HOST}/${PORT}") 2>/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ACT policy server exited during startup. See ${SERVER_LOG}" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for ACT policy server at ${HOST}:${PORT}. See ${SERVER_LOG}" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done

echo "ACT policy server ready at ${HOST}:${PORT}"
echo "Evaluation output: ${OUTPUT_DIR}"

for ((episode_index = 0; episode_index < NUM_EPISODES; episode_index++)); do
  seed=$((START_SEED + episode_index))
  layout_seed="${LAYOUT_SEED_ARRAY[$((episode_index % ${#LAYOUT_SEED_ARRAY[@]}))]}"
  episode_dir="$(printf '%s/episode_%04d_seed_%06d_layout_%06d' "${EPISODES_DIR}" "${episode_index}" "${seed}" "${layout_seed}")"
  summary_path="${episode_dir}/success_rate.json"
  if [[ -f "${summary_path}" ]] && python - "${summary_path}" "${seed}" "${layout_seed}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding='utf-8'))
valid = (
    summary.get('complete')
    and summary.get('num_episodes') == 1
    and summary.get('start_seed') == int(sys.argv[2])
    and summary.get('layout_seeds') == [int(sys.argv[3])]
)
raise SystemExit(0 if valid else 1)
PY
  then
    echo "Skipping completed evaluation seed ${seed}"
    continue
  fi

  echo "Evaluating episode $((episode_index + 1))/${NUM_EPISODES}, seed=${seed}, layout_seed=${layout_seed}"
  conda run --no-capture-output -n "${ISAAC_ENV}" env \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS="${ISAAC_OMP_NUM_THREADS:-4}" \
    MKL_NUM_THREADS="${ISAAC_MKL_NUM_THREADS:-4}" \
    OPENBLAS_NUM_THREADS="${ISAAC_OPENBLAS_NUM_THREADS:-4}" \
    python roboassemblybench/scripts/evaluate_fabrica_plumbers_block_act.py \
      --host "${HOST}" \
      --port "${PORT}" \
      --num-episodes 1 \
      --start-seed "${seed}" \
      --layout-seeds "${layout_seed}" \
      --output-dir "${episode_dir}" \
      --headless \
      "$@"
done

python roboassemblybench/scripts/aggregate_fabrica_plumbers_block_act_eval.py \
  --episodes-dir "${EPISODES_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --expected-episodes "${NUM_EPISODES}" \
  --start-seed "${START_SEED}" \
  --layout-seeds "${LAYOUT_SEED_ARRAY[@]}"
