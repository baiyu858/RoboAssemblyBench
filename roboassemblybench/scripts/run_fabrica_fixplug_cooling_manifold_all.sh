#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pairs=("0:1" "2:1" "3:1" "4:1" "5:1" "6:1")

for pair in "${pairs[@]}"; do
  plug="${pair%%:*}"
  socket="${pair##*:}"
  echo "=== cooling_manifold ${plug}->${socket} ==="
  PLUG="${plug}" SOCKET="${socket}" "${SCRIPT_DIR}/run_fabrica_fixplug_cooling_manifold_pair.sh"
done
