#!/usr/bin/env python3
"""
walk_then_replay.py — auto-switch between data replay and planner walking.

Pipeline @ 50Hz:
  - Compute body-frame velocity from data odometry: v_local = R(quat).T @ v_world
  - Threshold (smoothed, hysteresis) splits each tick into:
        phase A: window from data, encode, publish (no planner)
        phase B: planner walks lower body + arms overridden from data
  - At first A→B, lock heading from real WBC IMU (so phase B continues from
    whatever direction the robot is *actually* facing, not startup heading).

Both phases publish the same v4 token protocol — server is unaware of phase.

Tweak EPISODE_DIR + thresholds in the USER CONFIG block below.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

# ── USER CONFIG ────────────────────────────────────────────────────────────────
# EPISODE_DIR = "/home/xiawei/hongyi/Unitree_Robotics/Humanoid-Teleop/teleop/data/g1_1001/Basic/Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new/episode_40"
EPISODE_DIR = "/home/hongyi/data/real_psi0_g1/Rotate_to_pour_ham_into_plate_and_push_the_cart_forward/episode_20"
# EPISODE_DIR = "/home/xiawei/data/HE_RAW/Locomanip/walk_towards_a_desk_and_place_a_cube_on_a_tray/episode_4"
# EPISODE_DIR = "/home/xiawei/hongyi/Unitree_Robotics/Humanoid-Teleop/teleop/data/g1_1001/Basic/Spray_the_bowl_and_wipe_it_and_stack_it_up/episode_10"

# Body-frame walk thresholds (m/s, max(|vx|,|vy|)). Hysteresis: enter walk
# above ENTER, exit below EXIT.
ENTER_THR = 0.10
EXIT_THR  = 0.05

# Fixed walking speed sent to planner (chosen at load time by sol_q dim:
# 14 → 0.3, 29 → 0.2, else 0.3). Dominant local-vel axis → ±WALK_SPEED, the
# other axis scales proportionally (e.g. (0.5,0.1) → planner gets (0.3,0.06)).
WALK_SPEED_BY_SOL_DIM = {14: 0.3, 29: 0.2}
WALK_SPEED_DEFAULT    = 0.3

# Yaw-rate (deg/s) thresholds for the turn-in-place phase: planner gets
# target_vel=0 + facing_dir tracking data yaw delta in real time, so feet
# actually step around instead of dragging.
TURN_RATE_ENTER = 8.0
TURN_RATE_EXIT  = 4.0

# Absolute cap on waist_pitch (states.leg_state[14], mujoco order, degrees) when
# loading the episode. Teleop data often has sustained 8-14° forward lean for
# manipulation, which destabilizes the WBC controller (robot takes small
# compensating forward steps in phase A). Set to None to disable.
WAIST_PITCH_CAP_DEG = 2.0

# ───────────────────────────────────────────────────────────────────────────────

# ── Internal constants (don't usually need tuning) ─────────────────────────────
# Body-frame vy (lateral) oscillates ±~0.3 m/s at ~1Hz from natural step cycle
# (CoM swings left/right with each foot). To extract the underlying drift
# direction we need a window covering ≥2 step cycles. 2.0s kills oscillation
# (std drops 6× vs raw); shorter windows leak step osc into mvmt_dir and the
# robot ends up swerving right/left/right/left as the planner chases it.
SMOOTH_SECS    = 2.0
MIN_DWELL_SECS = 0.2   # debounce: must hold above/below threshold this long
HEIGHT         = 0.788740
WARMUP_SECS    = 3.0   # WBC settle time before reading init quat
PLANNER_MODE   = 1     # SLOW_WALK
ZMQ_HOST, ZMQ_PORT, ZMQ_TOPIC = "*", 5556, "pose"
WBC_HOST, WBC_PORT, WBC_TOPIC = "localhost", 5557, "g1_debug"

_GROOT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _GROOT_ROOT)

from encoder_client import EncoderClient
from walk_forward_token import (
    CONTROL_HZ, ENCODER_NUM_FRAMES, ENCODER_STEP_50HZ,
    MOTION_LOOK_AHEAD_50HZ, NUM_JOINTS,
    ENCODER_MODEL, PLANNER_MODEL,
    G1_DEFAULT_ANGLES_MUJOCO, _MUJOCO_TO_ISAACLAB_DOF, ARM_ISAAC_INDICES,
    fsq_quantize, quat_wxyz_to_yaw,
    resample_30hz_to_50hz, resample_pred_30hz_to_50hz,
    PlannerOnnx, Motion50HzBuffer, AsyncPlannerThread,
    WBCStateReader, TokenPublisher,
)

# sonic v1.1 checkpoint (pass --v1.1 to use it). Encoder input layout differs
# from release (1751-dim, heading-normalized anchors) — EncoderClient handles
# that via version="v1_1". Must match the checkpoint the WBC sim runs
# (deploy.sh --cp policy/sonic_v1_1/model).
ENCODER_MODEL_V1_1 = os.path.join(
    _GROOT_ROOT, "gear_sonic_deploy/policy/sonic_v1_1/model_encoder.onnx")


# ── Episode loader ────────────────────────────────────────────────────────────

def load_full_episode(episode_dir):
    """Returns (qpos_isaac, hands_or_None, imu_quat, body_vel, sol_dim, cmd_or_None).

    Joint sources by sol_q dimension (probed from frame 0):
      - sol_q is 14-d → arm command only. Use it for upper body (overrides
        states.arm_state); legs come from states.leg_state.
      - sol_q is 29-d → full body command. Use it for both legs and arms
        (cleaner than states; no controller tracking error).
      - sol_q missing/other → fall back to states for everything.

    IMU quaternion always comes from states (needed for the encoder window).

    body_vel is `states.odometry.velocity`, already body-frame (Unitree
    convention): vx = forward, vy = lateral left, vz = up.

    cmd: if every frame has numeric `actions.torso_vx`, `actions.torso_vy`,
    `actions.target_yaw`, returns (cmd_vel (n,3), cmd_yaw (n,)). Phase detection
    then uses these clean teleop-command signals (no IMU noise / step osc).
    Otherwise None and the caller falls back to body_vel + IMU yaw.
    """
    with open(os.path.join(episode_dir, "data.json")) as f:
        d = json.load(f)
    frames = d if isinstance(d, list) else d["frames"]
    n = len(frames)

    a0 = frames[0].get("actions") or {}
    sol0 = a0.get("sol_q")
    sol_dim = len(sol0) if sol0 is not None else 0
    if sol_dim in (14, 29):
        body_src = "arm-from-action, leg-from-states"
    else:
        body_src = "full-body-from-states"
    has_action_hands = (
        a0.get("left_angles") is not None and a0.get("right_angles") is not None
        and len(a0["left_angles"]) == 7 and len(a0["right_angles"]) == 7
    )
    hand_src = "action(left+right_angles)" if has_action_hands else "states.hand_state"
    print(f"[load] sol_q dim={sol_dim} → {body_src}; hands → {hand_src}")

    qpos_mj = np.zeros((n, 29), dtype=np.float32)
    hands = np.zeros((n, 14), dtype=np.float32)
    imu_quat = np.zeros((n, 4), dtype=np.float32)
    body_vel = np.zeros((n, 3), dtype=np.float32)
    cmd_vel = np.zeros((n, 3), dtype=np.float32)
    cmd_yaw = np.zeros(n, dtype=np.float32)
    hands_valid = False
    cmd_valid = True

    for i, fr in enumerate(frames):
        s = fr["states"]
        a = fr.get("actions") or {}
        sol_q = a.get("sol_q")

        leg = np.asarray(s.get("leg_states", s.get("leg_state")), dtype=np.float32).reshape(-1)[:15]
        if WAIST_PITCH_CAP_DEG is not None:
            cap = np.radians(WAIST_PITCH_CAP_DEG)
            leg[14] = np.clip(leg[14], -cap, +cap)   # waist_pitch (mujoco order)
        qpos_mj[i, :15] = leg
        if sol_dim in (14, 29) and sol_q is not None:
            # 14-d sol_q = arm only; 29-d sol_q = full body — take the arm slice (last 14).
            arm_action = np.asarray(sol_q, dtype=np.float32).reshape(-1)[-14:]
            qpos_mj[i, 15:] = arm_action
        else:
            arm = np.asarray(s.get("arm_states", s.get("arm_state")), dtype=np.float32).reshape(-1)[:14]
            qpos_mj[i, 15:] = arm

        if has_action_hands:
            la = np.asarray(a.get("left_angles"), dtype=np.float32).reshape(-1)[:7]
            ra = np.asarray(a.get("right_angles"), dtype=np.float32).reshape(-1)[:7]
            hands[i, :7] = la
            hands[i, 7:] = ra
            hands_valid = True
        else:
            h = s.get("hand_state")
            if h is not None and len(h) >= 14:
                hands[i] = np.asarray(h[:14], dtype=np.float32)
                hands_valid = True

        q = np.asarray(s["imu"]["quaternion"], dtype=np.float32).reshape(4)
        q = q / max(np.linalg.norm(q), 1e-8)
        if i > 0 and float(np.dot(q, imu_quat[i-1])) < 0.0:
            q = -q
        imu_quat[i] = q

        body_vel[i] = np.asarray(s["odometry"]["velocity"], dtype=np.float32).reshape(3)

        if cmd_valid:
            tvx = a.get("torso_vx"); tvy = a.get("torso_vy"); ty = a.get("target_yaw")
            if isinstance(tvx, (int, float)) and isinstance(tvy, (int, float)) and isinstance(ty, (int, float)):
                cmd_vel[i, 0] = tvx
                cmd_vel[i, 1] = tvy
                cmd_yaw[i] = ty
            else:
                cmd_valid = False

    qpos_isaac = qpos_mj[:, _MUJOCO_TO_ISAACLAB_DOF].astype(np.float32)
    cmd = (cmd_vel, cmd_yaw) if cmd_valid else None
    return qpos_isaac, hands if hands_valid else None, imu_quat, body_vel, sol_dim, cmd


# ── Helpers ───────────────────────────────────────────────────────────────────

def smooth_1d(x, win):
    win = max(1, int(win) | 1)  # force odd
    half = win // 2
    pad = np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])])
    return np.convolve(pad, np.ones(win, dtype=np.float32) / win, mode="valid").astype(np.float32)


def hysteresis_mask(metric, enter_thr, exit_thr, min_dwell):
    """Boolean mask with enter/exit hysteresis + min-dwell debounce.
    `metric[t]` is the scalar to threshold (typically |·|). Returns array of bools."""
    n = len(metric)
    out = np.zeros(n, dtype=bool)
    state, dwell = False, 0
    for i in range(n):
        s = float(metric[i])
        crossed = (s > enter_thr) if not state else (s < exit_thr)
        dwell = dwell + 1 if crossed else 0
        if dwell >= min_dwell:
            state, dwell = (not state), 0
        out[i] = state
    return out


def scale_velocity_xy(v_xy, walk_speed):
    """Scale local-frame velocity so the dominant axis = ±walk_speed; the
    other axis is scaled proportionally. Returns (speed, dir_unit_xy)."""
    m = float(max(abs(v_xy[0]), abs(v_xy[1])))
    if m < 1e-6:
        return walk_speed, np.array([1.0, 0.0], dtype=np.float32)
    v_scaled = (v_xy * (walk_speed / m)).astype(np.float32)
    speed = float(np.linalg.norm(v_scaled))
    return speed, (v_scaled / speed).astype(np.float32)


def build_data_window(qpos_50, quat_50, pub_frame):
    n = len(qpos_50)
    idx = [min(pub_frame + k * ENCODER_STEP_50HZ, n - 1) for k in range(ENCODER_NUM_FRAMES)]
    joint_pos = qpos_50[idx].copy()
    body_quat = quat_50[idx].copy()
    vel = np.zeros((ENCODER_NUM_FRAMES, NUM_JOINTS), dtype=np.float32)
    for k, j in enumerate(idx):
        vel[k] = (qpos_50[min(j + 1, n - 1)] - qpos_50[j]) * 50.0
    return joint_pos, vel, body_quat


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # CLI is intentionally minimal — tweak EPISODE_DIR / thresholds at top of file
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    episode_dir = args[0] if args else EPISODE_DIR
    no_wbc = "--no-wbc" in sys.argv
    # --v1.1 : encode with the sonic v1.1 checkpoint (sim side must run the
    # matching decoder: deploy.sh --cp policy/sonic_v1_1/model).
    use_v1_1 = "--v1.1" in sys.argv or "--v1_1" in sys.argv

    # --record-tokens [path] : write data_sonic.json after replay finishes.
    # Saves every published 50Hz token, then samples back to 30Hz to align with
    # the original data.json frames. If no path given, writes to
    # <episode_dir>/data_sonic.json.
    record_tokens_path = None
    if "--record-tokens" in sys.argv:
        i = sys.argv.index("--record-tokens")
        nxt = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else None
        record_tokens_path = nxt or os.path.join(episode_dir, "data_sonic.json")
        print(f"[main] recording tokens → {record_tokens_path}")

    print(f"[main] loading {episode_dir}")
    qpos_30, hands_30, imu_q_30, body_vel_30, sol_dim, cmd_30 = load_full_episode(episode_dir)
    n_30 = len(qpos_30)
    walk_speed = WALK_SPEED_BY_SOL_DIM.get(sol_dim, WALK_SPEED_DEFAULT)
    print(f"[main] WALK_SPEED={walk_speed} (sol_q dim={sol_dim})")

    qpos_50 = resample_30hz_to_50hz(qpos_30)
    quat_50 = resample_30hz_to_50hz(imu_q_30)
    quat_50 /= np.linalg.norm(quat_50, axis=1, keepdims=True).clip(1e-8)
    hand_50 = resample_30hz_to_50hz(hands_30) if hands_30 is not None else None
    n_50 = len(qpos_50)

    # Detection signal source: prefer teleop commands (torso_vx/vy + target_yaw)
    # when present — clean step signals, no IMU/odometry noise. Fall back to
    # measured states.odometry.velocity + IMU yaw when commands are missing.
    if cmd_30 is not None:
        cmd_vel_30, cmd_yaw_30 = cmd_30
        body_vel_50 = resample_30hz_to_50hz(cmd_vel_30)
        det_src = "cmd(torso_vx,torso_vy,target_yaw)"
    else:
        body_vel_50 = resample_30hz_to_50hz(body_vel_30)
        det_src = "states(odometry.velocity, imu.yaw)"
    print(f"[main] detection signals ← {det_src}")

    dwell = max(1, int(MIN_DWELL_SECS * CONTROL_HZ))
    vx_smooth = smooth_1d(body_vel_50[:, 0], int(SMOOTH_SECS * CONTROL_HZ))
    vy_smooth = smooth_1d(body_vel_50[:, 1], int(SMOOTH_SECS * CONTROL_HZ))
    body_vel_xy_smooth = np.stack([vx_smooth, vy_smooth], axis=1).astype(np.float32)
    walking_mask = hysteresis_mask(
        np.maximum(np.abs(vx_smooth), np.abs(vy_smooth)),
        ENTER_THR, EXIT_THR, dwell,
    )

    # Yaw + yaw-rate. Source matches detection: target_yaw (cmd) or IMU yaw.
    # Walk takes priority — planner can't cleanly do walk+turn together.
    if cmd_30 is not None:
        yaw_50 = np.unwrap(resample_30hz_to_50hz(cmd_yaw_30)).astype(np.float32)
    else:
        yaw_50 = np.unwrap([quat_wxyz_to_yaw(q) for q in quat_50]).astype(np.float32)
    yaw_rate_50 = (np.gradient(smooth_1d(yaw_50, CONTROL_HZ)) * CONTROL_HZ).astype(np.float32)
    turning_mask = hysteresis_mask(
        np.abs(yaw_rate_50),
        np.radians(TURN_RATE_ENTER), np.radians(TURN_RATE_EXIT), dwell,
    ) & ~walking_mask

    print(f"[main] {n_30}f@30Hz → {n_50}f@50Hz; "
          f"smoothed |vx|∈[{abs(vx_smooth).min():.2f},{abs(vx_smooth).max():.2f}], "
          f"|vy|∈[{abs(vy_smooth).min():.2f},{abs(vy_smooth).max():.2f}] m/s; "
          f"|yaw_rate|max={np.degrees(np.abs(yaw_rate_50).max()):.0f}°/s; "
          f"walking {walking_mask.sum()}/{n_50} ({100*walking_mask.mean():.0f}%); "
          f"turning {turning_mask.sum()}/{n_50} ({100*turning_mask.mean():.0f}%)")

    arms_isaac_50 = qpos_50[:, ARM_ISAAC_INDICES]

    # Setup
    pub = TokenPublisher(host=ZMQ_HOST, port=ZMQ_PORT, topic=ZMQ_TOPIC)
    time.sleep(1.0)
    planner = PlannerOnnx(PLANNER_MODEL, default_height=HEIGHT)
    if use_v1_1:
        print("[main] using sonic v1.1 encoder")
        encoder = EncoderClient(ENCODER_MODEL_V1_1, mode=0, version="v1_1")
    else:
        encoder = EncoderClient(ENCODER_MODEL, mode=0)
    wbc = None if no_wbc else WBCStateReader(WBC_HOST, WBC_PORT, WBC_TOPIC)
    if wbc is not None and not wbc.wait_until_ready(10.0):
        wbc.close(); wbc = None
    if wbc is not None:
        time.sleep(WARMUP_SECS)
    pub.send_command(start=True, stop=False, planner=False)

    # ── Planner state ────────────────────────────────────────────────────────
    # Eager init: planner is initialized BEFORE the main loop (during WBC settle
    # time, so the sync warmup blocking is free). Async planner runs in IDLE
    # mode throughout phase A — its internal context stays primed, so the first
    # A→active transition has no ramp-from-standing artifact, no sync warmup
    # blocking, and no need for a warmup pause. The buffer is indexed by
    # `pub_frame` directly (continuous since startup).
    motion_buffer = None
    async_planner = None
    last_replan_local = 0

    # current_target_yaw is the absolute world yaw the planner is being told
    # to face. Updated:
    #   - Set at startup from initial WBC quat (used for phase A IDLE)
    #   - At every A→active transition: re-anchored to current WBC quat
    #   - During phase C: incrementally += data yaw delta
    #   - During phase B: held constant (lock)
    current_target_yaw = None
    c_data_yaw_anchor = None    # data yaw at the moment we entered C
    c_locked_yaw      = None    # current_target_yaw at the moment we entered C

    def _wbc_quat():
        if wbc is None: return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        s = wbc.get_state()
        return s["base_quat"].astype(np.float32) if s is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _wbc_qpos():
        if wbc is None: return G1_DEFAULT_ANGLES_MUJOCO.copy()
        s = wbc.get_state()
        return s["qpos"] if s is not None else G1_DEFAULT_ANGLES_MUJOCO.copy()

    def _init_planner():
        """Pre-init planner at startup. Fills motion_buffer with 4 generations
        of stand-in-place content (target_vel=0) and starts AsyncPlannerThread
        in IDLE mode. By the time the first A→active transition happens, the
        planner has been running async for several seconds in IDLE — its
        internal context is primed and replans pick up new mode params (B or C)
        immediately, no ramp-from-standing artifact.

        Yaw at startup is used only for the warmup direction. The actual lock
        yaw for active phases is set at each A→active transition (so it
        reflects current WBC orientation, not stale startup yaw).
        """
        nonlocal motion_buffer, async_planner, current_target_yaw
        cur_quat = _wbc_quat()
        current_target_yaw = quat_wxyz_to_yaw(cur_quat)
        fwd = np.array([np.cos(current_target_yaw), np.sin(current_target_yaw), 0.0], dtype=np.float32)
        print(f"[main] pre-init planner: startup yaw={np.degrees(current_target_yaw):.1f}°, warmup vel=0 (idle)")
        planner.initialize_context(cur_quat, _wbc_qpos())
        motion_buffer = Motion50HzBuffer()
        for w in range(4):
            gen = 0 if w == 0 else max(0, motion_buffer.length - 10)
            if w > 0:
                planner.set_context(motion_buffer.build_context_locked(gen))
            pred = planner.step(mode=PLANNER_MODE, target_vel=0.0,
                                movement_dir=fwd, facing_dir=fwd, height=HEIGHT)
            motion_buffer.splice_at(gen, *resample_pred_30hz_to_50hz(pred))
        async_planner = AsyncPlannerThread(
            planner, motion_buffer,
            mode=PlannerOnnx.IDLE, speed=0.0, height=HEIGHT,
            mvmt_dir=fwd, facing_dir=fwd,
        )

    _init_planner()

    REPLAN_INTERVAL = int(round(CONTROL_HZ / 10))
    pub_frame = 0
    hand_joints = np.zeros(14, dtype=np.float32)
    phase = "A"
    stop = {"f": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(f=True))
    dt = 1.0 / CONTROL_HZ
    n_pub = 0

    # Token recording: capture every published 50Hz token; downsample to 30Hz at end.
    tokens_50 = np.zeros((n_50, 64), dtype=np.float32) if record_tokens_path else None

    print(f"[main] start phase A. walk thr={ENTER_THR}/{EXIT_THR} m/s, "
          f"turn thr={TURN_RATE_ENTER}/{TURN_RATE_EXIT} °/s")

    try:
        while not stop["f"] and pub_frame < n_50:
            t0 = time.perf_counter()
            walking = bool(walking_mask[pub_frame])
            turning = bool(turning_mask[pub_frame])
            new_phase = "B" if walking else ("C" if turning else "A")

            # ── Phase transitions ────────────────────────────────────────────
            if new_phase != phase:
                if phase == "A" and new_phase != "A":
                    # Re-anchor target yaw to current WBC at every A→active.
                    # Planner is pre-initialized at startup, so no init here.
                    # Force-trigger a replan next tick so async overwrites the
                    # tail IDLE content with new-phase (walk/turn) content ASAP
                    # — saves ~100ms vs waiting for the next REPLAN_INTERVAL.
                    current_target_yaw = quat_wxyz_to_yaw(_wbc_quat())
                    last_replan_local = pub_frame - REPLAN_INTERVAL

                if new_phase == "C":
                    c_data_yaw_anchor = float(yaw_50[pub_frame])
                    c_locked_yaw      = float(current_target_yaw)

                print(f"[main] {phase}→{new_phase} at t={pub_frame/CONTROL_HZ:.2f}s")
                phase = new_phase

            # ── During phase C, target yaw tracks data yaw delta ────────────
            if phase == "C":
                delta = float(yaw_50[pub_frame] - c_data_yaw_anchor)
                current_target_yaw = c_locked_yaw + delta

            # ── Hot-swap planner params (planner is always alive) ───────────
            facing = np.array([np.cos(current_target_yaw),
                               np.sin(current_target_yaw), 0.0], dtype=np.float32)
            if phase == "A":
                async_planner.set_planner_params(
                    mode=PlannerOnnx.IDLE, speed=0.0,
                    mvmt_dir=facing, facing_dir=facing,
                )
            elif phase == "B":
                speed, vd = scale_velocity_xy(body_vel_xy_smooth[pub_frame], walk_speed)
                cy, sy = np.cos(current_target_yaw), np.sin(current_target_yaw)
                mvmt = np.array([cy*vd[0] - sy*vd[1],
                                 sy*vd[0] + cy*vd[1], 0.0], dtype=np.float32)
                async_planner.set_planner_params(
                    mode=PLANNER_MODE, speed=speed,
                    mvmt_dir=mvmt, facing_dir=facing,
                )
            else:  # phase C: turn in place toward target yaw
                # Zero movement_dir: any non-zero mvmt_dir (even with speed=0)
                # pulls SLOW_WALK into translating toward it -> arc instead of
                # turning in place.
                async_planner.set_planner_params(
                    mode=PLANNER_MODE, speed=0.0,
                    mvmt_dir=np.zeros(3, dtype=np.float32), facing_dir=facing,
                )

            if pub_frame - last_replan_local >= REPLAN_INTERVAL:
                async_planner.request_replan(pub_frame + MOTION_LOOK_AHEAD_50HZ)
                last_replan_local = pub_frame

            # ── Build encoder window ─────────────────────────────────────────
            if phase == "A":
                jp, jv, bq = build_data_window(qpos_50, quat_50, pub_frame)
            else:
                window = motion_buffer.read_encoder_window(pub_frame, ENCODER_NUM_FRAMES, ENCODER_STEP_50HZ)
                if window is None:
                    time.sleep(0.005)
                    continue
                jp, bq, jv, _ = window
                d_idx = [min(pub_frame + k * ENCODER_STEP_50HZ, n_50 - 1) for k in range(ENCODER_NUM_FRAMES)]
                jp[:, ARM_ISAAC_INDICES] = arms_isaac_50[d_idx]
                arms_v = np.zeros((ENCODER_NUM_FRAMES, len(ARM_ISAAC_INDICES)), dtype=np.float32)
                for k, j in enumerate(d_idx):
                    arms_v[k] = (arms_isaac_50[min(j + 1, n_50 - 1)] - arms_isaac_50[j]) * 50.0
                jv[:, ARM_ISAAC_INDICES] = arms_v
                if wbc is not None:
                    s = wbc.get_state()
                    if s is not None:
                        bq[0] = s["base_quat"].astype(np.float32)

            if hand_50 is not None:
                hand_joints = hand_50[min(pub_frame, len(hand_50) - 1)]

            tok = fsq_quantize(encoder.encode(joint_pos=jp, joint_vel=jv, body_quat=bq)).astype(np.float32)
            if tokens_50 is not None:
                tokens_50[pub_frame] = tok
            pub.publish(hand_joints_14=hand_joints, token_64=tok, body_quat_w=bq[0])
            pub_frame += 1
            n_pub += 1

            sleep = dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

    finally:
        try:
            pub.send_command(start=False, stop=True, planner=False)
        finally:
            if async_planner is not None:
                async_planner.stop()
            # Write the output BEFORE tearing down ZMQ/wbc — some libzmq
            # versions hit an assertion in pub.close() shutdown race that
            # aborts the process; saving the JSON first means the work
            # survives the crash.
            print(f"[main] stopped after {n_pub} tokens")
            if tokens_50 is not None and record_tokens_path is not None:
                # If we exited before consuming all frames, hold the last valid
                # token for the rest (avoids zeroed frames at tail).
                if pub_frame < n_50 and pub_frame > 0:
                    tokens_50[pub_frame:] = tokens_50[pub_frame - 1]
                idx_30_to_50 = np.round(np.arange(n_30) * 5.0 / 3.0).astype(int).clip(0, n_50 - 1)
                tokens_30 = tokens_50[idx_30_to_50]
                write_sonic_json(episode_dir, record_tokens_path, tokens_30)
            pub.close()
            if wbc is not None:
                wbc.close()


def write_sonic_json(episode_dir, out_path, tokens_30):
    """Write data_sonic.json next to original data.json. Per-frame structure
    matches the sonic format expected by replay_token.py:
        timestamp, states:{qpos, quat, hand_joints},
                   actions:{hand_joints, token}, image
    All raw fields come straight from the original data.json (NOT from any
    sol_q/action substitution used internally for token computation):
        states.qpos        = states.leg_state + states.arm_state (29-d)
        states.quat        = states.imu.quaternion (4-d)
        states.hand_joints = states.hand_state (14-d)
        actions.hand_joints = action.left_angles + action.right_angles
                              (fallback to states.hand_state if action missing)
    `tokens_30` length must equal the number of frames in data.json.
    """
    with open(os.path.join(episode_dir, "data.json")) as f:
        d = json.load(f)
    frames = d if isinstance(d, list) else d["frames"]
    n = len(frames)
    if len(tokens_30) != n:
        raise ValueError(f"tokens_30 has {len(tokens_30)} but data has {n} frames")

    out = []
    for i, fr in enumerate(frames):
        s = fr["states"]
        a = fr.get("actions") or {}
        leg = list(s.get("leg_states", s.get("leg_state", [])))[:15]
        arm = list(s.get("arm_states", s.get("arm_state", [])))[:14]
        qpos = leg + arm
        quat = list(s["imu"]["quaternion"])[:4]
        state_hands = list(s.get("hand_state") or [0.0] * 14)[:14]
        if (a.get("left_angles") is not None and a.get("right_angles") is not None
                and len(a["left_angles"]) == 7 and len(a["right_angles"]) == 7):
            action_hands = list(a["left_angles"]) + list(a["right_angles"])
        else:
            action_hands = state_hands

        frame_out = {
            "states": {
                "qpos":        qpos,
                "quat":        quat,
                "hand_joints": state_hands,
            },
            "actions": {
                "hand_joints": action_hands,
                "token":       tokens_30[i].tolist(),
            },
            "image": fr.get("image", ""),
        }
        t = fr.get("time")
        if isinstance(t, (int, float)):
            frame_out["timestamp"] = int(t * 1e9)
        out.append(frame_out)

    # Write to a temp path then atomic-rename, so a crash mid-write never
    # leaves a partial file at out_path (which would falsely pass --skip-existing).
    out_path = Path(out_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(out, f)
    os.replace(tmp_path, out_path)
    print(f"[main] wrote {n} frames → {out_path}")


if __name__ == "__main__":
    main()
