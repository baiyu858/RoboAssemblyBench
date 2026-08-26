#!/usr/bin/env bash
# Pause only Stage-2 replay on one node-specific Fabrica output directory.
set -euo pipefail

OUTPUT_ROOT="${1:?usage: pause_fabrica_stage2.sh OUTPUT_ROOT}"
mkdir -p "$OUTPUT_ROOT"
printf '{"paused_at":"%s","reason":"operator_requested_stage2_pause"}\n' "$(date -Is)" \
  >"$OUTPUT_ROOT/STAGE2_PAUSED_BY_OPERATOR.json"
echo "marker=$OUTPUT_ROOT/STAGE2_PAUSED_BY_OPERATOR.json"

targets=()
for pid_file in \
  "$OUTPUT_ROOT/stage2_pipeline.pid" \
  "$OUTPUT_ROOT/supervisor/collector.pid" \
  "$OUTPUT_ROOT/supervisor/supervisor.pid"; do
  [[ -f "$pid_file" ]] || continue
  pid="$(tr -dc '0-9' <"$pid_file")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    targets+=("$pid")
    echo "pid_file_target=$pid_file:$pid"
  else
    rm -f "$pid_file"
  fi
done

while IFS= read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] && targets+=("$pid")
done < <(
  ps -eo pid=,args= | awk -v output_root="$OUTPUT_ROOT" '
    index($0, output_root) &&
    ($0 ~ /pipeline_fabrica_stage2_replay[.]py/ ||
     $0 ~ /replay_fabrica_successful_trajectories[.]py/ ||
     $0 ~ /generate_demos[.]py.*--worker-mode replay/ ||
     $0 ~ /\/ffmpeg([[:space:]]|$)/) { print $1 }
  '
)

mapfile -t targets < <(printf '%s\n' "${targets[@]:-}" | awk '/^[0-9]+$/' | sort -nu)
echo "term_targets=${targets[*]:-none}"
for pid in "${targets[@]:-}"; do
  kill -TERM "$pid" 2>/dev/null || true
done

remaining=()
for _ in $(seq 1 35); do
  remaining=()
  for pid in "${targets[@]:-}"; do
    kill -0 "$pid" 2>/dev/null && remaining+=("$pid")
  done
  [[ "${#remaining[@]}" -eq 0 ]] && break
  sleep 1
done

for pid in "${remaining[@]:-}"; do
  echo "force_kill=$pid"
  kill -KILL "$pid" 2>/dev/null || true
done

rm -f \
  "$OUTPUT_ROOT/stage2_pipeline.pid" \
  "$OUTPUT_ROOT/supervisor/collector.pid" \
  "$OUTPUT_ROOT/supervisor/supervisor.pid"

sleep 2
echo "remaining_stage2_processes:"
ps -eo pid=,args= | awk -v output_root="$OUTPUT_ROOT" '
  index($0, output_root) &&
  ($0 ~ /pipeline_fabrica_stage2_replay[.]py/ ||
   $0 ~ /replay_fabrica_successful_trajectories[.]py/ ||
   $0 ~ /generate_demos[.]py.*--worker-mode replay/ ||
   $0 ~ /\/ffmpeg([[:space:]]|$)/) { print }
' || true
