#!/usr/bin/env bash
SCRIPT_DIR=$(dirname $(realpath $0))

# Runtime configuration
hand_side="left"           # "left" or "right"
serial_number=""           # e.g. "337238793233"; leave empty to use first found
tracking_host="localhost"  # IP of machine running pico_manus_thread_server.py
                           # change to laptop IP when running this script on G1
tracking_port=5559
state_port=5560
target_fps=50
retarget_config="${SCRIPT_DIR}/wuji-retargeting/example/config/retarget_manus_${hand_side}.yaml"

# Start controller
# Ensure we use the correct Python from conda environment
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
    PYTHON_BIN="python"
fi

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN "${SCRIPT_DIR}/wuji_hand_server.py" \
    --hand_side ${hand_side} \
    --config ${retarget_config} \
    --tracking_host ${tracking_host} \
    --tracking_port ${tracking_port} \
    --state_port ${state_port} \
    --target_fps ${target_fps} \
    --no_smooth \
    # --serial_number ${serial_number} \
    # --smooth_steps 5 \
    # --clamp_min -1.5 \
    # --clamp_max 1.5
