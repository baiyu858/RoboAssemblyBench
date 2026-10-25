cd /home/baiyu24/model/InternUtopia

PLUMBERS_RUN_DIR="$PWD/outputs/plumbers_physics_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
mkdir -- "$PLUMBERS_RUN_DIR"
echo "$PLUMBERS_RUN_DIR"

/home/baiyu24/APP/miniconda3/bin/conda run \
  --no-capture-output -n internutopia311 env \
  BEAM_CONTACT_TRACE_PATH="$PLUMBERS_RUN_DIR/contact_trace.jsonl" \
  BEAM_CLOSE_CONTROL_TRACE_PATH="$PLUMBERS_RUN_DIR/close_control.jsonl" \
  BEAM_HOLD_TRACE_PATH="$PLUMBERS_RUN_DIR/hold_trace.jsonl" \
  python roboassemblybench/scripts/validate_fabrica_canonical_ur5e.py \
  --tasks plumbers_block \
  --seed 4906 \
  --layout-seed 4906 \
  --output-dir "$PLUMBERS_RUN_DIR" \
  --worker-timeout-seconds 1800 \
  --record-live-video \
  --live-video-fps 30 \
  --live-video-frame-stride 8
