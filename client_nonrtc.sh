# Usage: bash client.sh [run_name]
# run_name names the recording folder under zed_recordings/ (default: timestamp).
# The ZED recorder subscribes to the server's viewer PUB stream (5559), which is
# separate from the inference REP socket (5558), so recording never blocks the
# policy client.

RUN_NAME="${1:-recordings}"
OUT_DIR="zed_recordings/${RUN_NAME}"
echo "[client.sh] recording into $OUT_DIR"

# Pin the sonic env's python so a stray venv on PATH (e.g. .venv_sim) can't
# shadow it — `python` alone picks up whatever the terminal inherited.
PY="$HOME/miniforge3/envs/sonic/bin/python"
ZED_SERVER="${ZED_SERVER:-192.168.123.164}"
# IMAGE_FIT=pad    -> pad ZED 672x376 to 672x384 with black rows at the bottom (matches LeRobot conversion)
# IMAGE_FIT=resize -> stretch to 672x384 (legacy, default)
IMAGE_FIT="${IMAGE_FIT:-pad}"

"$PY" zed_recorder.py --server "$ZED_SERVER" --out-dir "$OUT_DIR" &
REC_PID=$!

# cleanup() {
#     if kill -0 "$REC_PID" 2>/dev/null; then
#         kill -INT "$REC_PID" 2>/dev/null
#         wait "$REC_PID" 2>/dev/null
#     fi
#     # The client has exited by now, so port 5570 is free for reset_neck.py
#     # to bind and drive the neck back to (0, 0). It closes the loop over the
#     # 5560 state stream and exits as soon as the head actually sits at zero.
#     "$PY" reset_neck.py
# }
# trap cleanup EXIT

"$PY" g1_sonic_client.py --action-only --include-neck --image-fit "$IMAGE_FIT"
# --action-only