#!/bin/bash
# Launch the WBC deploy binary against the MuJoCo sim (DDS over loopback).
#
# gear_sonic_deploy/deploy.sh cannot be used unattended here: it rebuilds via
# `just` and prompts for confirmation. It also assumes the binary's runtime deps
# are on the default loader path, which on this host they are not - the binary
# needs GLIBC >= 2.38 (system has 2.35) and libcudart.so.13 (system CUDA is
# 11.8/12.8). Both exist elsewhere, so it is launched through the nix glibc
# loader with an explicit library path. Override the paths below if your machine
# resolves them normally, in which case you can call the binary directly.
set -eu

REPO="$(cd "$(dirname "$(dirname "$(readlink -f "$0")")")" && pwd)"
DEPLOY_DIR="$REPO/gear_sonic_deploy"

# Newer glibc than the system one, needed by the prebuilt binary.
GLIBC="${GLIBC_ROOT:-/nix/store/5m9amsvvh2z8sl7jrnc87hzy21glw6k1-glibc-2.40-66}"
# libcudart.so.13 ships inside the venvs' nvidia/cu13 wheel.
CUDA13="${CUDA13_LIB:-$REPO/.venv_sim/lib/python3.10/site-packages/nvidia/cu13/lib}"

LIBS="$GLIBC/lib"
LIBS="$LIBS:${TensorRT_ROOT:-$HOME/TensorRT}/lib"
LIBS="$LIBS:/usr/local/cuda/lib64"
LIBS="$LIBS:$DEPLOY_DIR/thirdparty/unitree_sdk2/thirdparty/lib/x86_64"
LIBS="$LIBS:/opt/onnxruntime/lib"
LIBS="$LIBS:$CUDA13"
LIBS="$LIBS:/usr/lib/x86_64-linux-gnu"

# Policy checkpoint. POLICY_DIR points at a directory holding model_decoder.onnx,
# model_encoder.onnx and observation_config.yaml (the layout of policy/release and
# of the GEAR-SONIC checkpoint drops); the three can also be set individually.
# Paths are relative to gear_sonic_deploy/ unless absolute.
POLICY_DIR="${POLICY_DIR:-policy/release}"
POLICY_DECODER="${POLICY_DECODER:-$POLICY_DIR/model_decoder.onnx}"
POLICY_ENCODER="${POLICY_ENCODER:-$POLICY_DIR/model_encoder.onnx}"
OBS_CONFIG="${OBS_CONFIG:-$POLICY_DIR/observation_config.yaml}"
PLANNER="${PLANNER:-planner/target_vel/V2/planner_sonic.onnx}"

cd "$DEPLOY_DIR"

for f in "$POLICY_DECODER" "$POLICY_ENCODER" "$OBS_CONFIG" "$PLANNER"; do
  [ -f "$f" ] || { echo "missing: $f (cwd $PWD)" >&2; exit 1; }
done
echo "[run_wbc_deploy] policy: $POLICY_DECODER"
echo "[run_wbc_deploy] obs:    $OBS_CONFIG"

# --input-type zmq_manager (not plain zmq): only ZMQManager subscribes to the
# 'command' topic, which is how the replay starts and stops control.
exec "$GLIBC/lib/ld-linux-x86-64.so.2" --library-path "$LIBS" \
  ./target/release/g1_deploy_onnx_ref \
  lo "$POLICY_DECODER" reference/example/ \
  --obs-config "$OBS_CONFIG" \
  --encoder-file "$POLICY_ENCODER" \
  --planner-file "$PLANNER" \
  --input-type zmq_manager \
  --output-type all \
  --zmq-host localhost \
  --default-motion neutral_kick_R_001__A543 \
  --disable-crc-check
